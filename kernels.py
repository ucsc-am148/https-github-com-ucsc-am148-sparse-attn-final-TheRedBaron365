"""My three block-sparse kernels for the final.

  dsd_matmul             A1  block-sparse A (BCSR) @ dense B
  sparse_flash_forward   A2  flash attention forward, mask shortens the key loop
  sparse_flash_backward  A3  flash backward, three kernels

Spec is in ALGORITHMS.md. Shapes/dtypes are fixed by the signatures, the rest is
mine.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


def _tile_for(block):
    # use a 64-wide compute tile, but never bigger than the block, and keep it a
    # divisor of the block so the inner sub-loop lands exactly.
    t = 64
    while block % t != 0:
        t //= 2
    return t


def _attn_meta(d):
    # d=128 needs the extra warps or the fp32 accumulator spills; drop the
    # pipeline to 1 stage there too so the staged tiles fit in 99KB smem.
    if d > 64:
        return dict(num_warps=8, num_stages=1)
    return dict(num_warps=4, num_stages=2)


# ---------------------------------------------------------------- A1: DSD

@triton.jit
def _dsd_kernel(values, row_offsets, column_indices, B, C,
                M, K, N,
                stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # these BLOCK_M output rows sit inside one bcsr block-row (BLOCK_M | BLOCK).
    block_row = (pid_m * BLOCK_M) // BLOCK
    row_in_block = pid_m * BLOCK_M - block_row * BLOCK

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_m = row_in_block + tl.arange(0, BLOCK_M)   # row offset inside the packed block
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    lo = tl.load(row_offsets + block_row)
    hi = tl.load(row_offsets + block_row + 1)
    for idx in range(lo, hi):                          # only A's live k-blocks
        kcol = tl.load(column_indices + idx)
        val_base = values + idx * BLOCK * BLOCK
        for kk in range(0, BLOCK, BLOCK_K):            # strip-mine, else acc spills on 256-blocks
            offs_bk = kk + tl.arange(0, BLOCK_K)
            a = tl.load(val_base + local_m[:, None] * BLOCK + offs_bk[None, :])
            b_rows = kcol * BLOCK + offs_bk
            b = tl.load(B + b_rows[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=mask_n[None, :], other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)      # honest fp32, no tensor cores

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_n[None, :])


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. ALGORITHMS.md sec 1-2.

      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  prefix sum of nnz per block-row
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32
    -> C (M, N) fp32. fp32 throughout, allow_tf32=False.
    """
    M, K, N, block = int(M), int(K), int(N), int(block)
    values = values.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=B.device, dtype=torch.float32)

    # keep the accumulator small. a full 256x256 block in registers spills hard.
    BLOCK_M = block if block <= 64 else 64
    BLOCK_K = block if block < 32 else 32
    BLOCK_N = 64

    grid = (M // BLOCK_M, triton.cdiv(N, BLOCK_N))
    _dsd_kernel[grid](
        values, row_offsets, column_indices, B, C,
        M, K, N,
        B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK=block, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return C


# ---------------------------------------------------------------- A2: forward

@triton.jit
def _fwd_kernel(Q, K, V, O, L,
                q_row_offsets, q_col_indices,
                sm_scale,
                stride_b, stride_t, stride_d,
                stride_lb, stride_lt,
                T,
                BLOCK: tl.constexpr, T_TILE: tl.constexpr,
                BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_qt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    qk_scale = sm_scale * 1.4426950408889634   # fold sigma and log2(e) -> base-2 scores so I can use exp2

    q0 = pid_qt * T_TILE
    iblk = q0 // BLOCK                          # which query block this tile is in
    offs_q = q0 + tl.arange(0, T_TILE)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM                  # pad head dim up to a power of 2
    base = pid_bh * stride_b

    mask_q = (offs_q[:, None] < T) & mask_d[None, :]
    q = tl.load(Q + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=mask_q, other=0.0)

    # running softmax state per query row
    m_i = tl.full((T_TILE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((T_TILE,), dtype=tl.float32)
    acc = tl.zeros((T_TILE, BLOCK_D), dtype=tl.float32)

    lo = tl.load(q_row_offsets + iblk)
    hi = tl.load(q_row_offsets + iblk + 1)
    for idx in range(lo, hi):                   # only the live key blocks for this query
        jblk = tl.load(q_col_indices + idx)
        for kk in range(0, BLOCK, T_TILE):      # block may be 128, tile is 64 -> walk it
            offs_k = jblk * BLOCK + kk + tl.arange(0, T_TILE)
            mask_k = (offs_k[:, None] < T) & mask_d[None, :]
            k = tl.load(K + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                        mask=mask_k, other=0.0)
            v = tl.load(V + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                        mask=mask_k, other=0.0)

            s = tl.dot(q, tl.trans(k)) * qk_scale
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.math.exp2(m_i - m_new)               # rescale old running state
            p = tl.math.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    acc = acc / l_i[:, None]
    L_val = m_i + tl.math.log2(l_i)             # log2 of the denom, the only thing the backward reuses

    o_ptrs = O + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_q)
    tl.store(L + pid_bh * stride_lb + offs_q * stride_lt, L_val, mask=offs_q < T)


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash forward. ALGORITHMS.md sec 1, 3.

      Q, K, V        (B, H, T, d) fp16
      q_row_offsets / q_col_indices   query-block view (live key blocks per query block)
    -> O (B,H,T,d) fp16, L (B,H,T) fp32 (log2 of the softmax denominator)
    """
    Bb, H, T, d = Q.shape
    BH = Bb * H
    block = int(BLOCK_Q)
    Qr = Q.reshape(BH, T, d)                    # fold (B,H) into one batch axis
    Kr = K.reshape(BH, T, d)
    Vr = V.reshape(BH, T, d)

    O = torch.empty((Bb, H, T, d), device=Q.device, dtype=torch.float16)
    L = torch.empty((Bb, H, T), device=Q.device, dtype=torch.float32)
    Or = O.reshape(BH, T, d)
    Lr = L.reshape(BH, T)

    T_TILE = _tile_for(block)
    BLOCK_D = triton.next_power_of_2(d)
    grid = (T // T_TILE, BH)
    _fwd_kernel[grid](
        Qr, Kr, Vr, Or, Lr,
        q_row_offsets, q_col_indices,
        sm_scale,
        Qr.stride(0), Qr.stride(1), Qr.stride(2),
        Lr.stride(0), Lr.stride(1),
        T,
        BLOCK=block, T_TILE=T_TILE, BLOCK_D=BLOCK_D, HEAD_DIM=d,
        **_attn_meta(d),
    )
    return O, L


# ---------------------------------------------------------------- A3: backward

# dO and O are all the D pre-pass needs. D_i = sum_d dO_i * O_i.
# (this is the softmax-jacobian term collapsed down to a length-T vector.)
@triton.jit
def _bwd_d_kernel(O, DO, D,
                  stride_b, stride_t, stride_d,
                  stride_db, stride_dt,
                  T,
                  T_TILE: tl.constexpr, BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_qt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_q = pid_qt * T_TILE + tl.arange(0, T_TILE)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    base = pid_bh * stride_b
    mask = (offs_q[:, None] < T) & mask_d[None, :]

    o = tl.load(O + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                 mask=mask, other=0.0).to(tl.float32)
    d_row = tl.sum(o * do, axis=1)
    tl.store(D + pid_bh * stride_db + offs_q * stride_dt, d_row, mask=offs_q < T)


# dK/dV: key tile is the outer loop so both gradients accumulate locally and I
# never need HBM atomics. I work everything in (key, query) layout so the two
# accumulators come out the right shape without transposing them.
@triton.jit
def _bwd_dkdv_kernel(Q, K, V, DO, L, D, DK, DV,
                     k_row_offsets, k_col_indices,
                     sm_scale,
                     stride_b, stride_t, stride_d,
                     stride_lb, stride_lt,
                     T,
                     BLOCK: tl.constexpr, T_TILE: tl.constexpr,
                     BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_kt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    qk_scale = sm_scale * 1.4426950408889634

    k0 = pid_kt * T_TILE
    jblk = k0 // BLOCK
    offs_k = k0 + tl.arange(0, T_TILE)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    base = pid_bh * stride_b
    mask_k = (offs_k[:, None] < T) & mask_d[None, :]

    k = tl.load(K + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=mask_k, other=0.0)
    v = tl.load(V + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=mask_k, other=0.0)

    dk_acc = tl.zeros((T_TILE, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((T_TILE, BLOCK_D), dtype=tl.float32)

    lo = tl.load(k_row_offsets + jblk)
    hi = tl.load(k_row_offsets + jblk + 1)
    for idx in range(lo, hi):                   # query blocks that hit this key block (mask^T)
        iblk = tl.load(k_col_indices + idx)
        for qq in range(0, BLOCK, T_TILE):
            offs_q = iblk * BLOCK + qq + tl.arange(0, T_TILE)
            mask_q = (offs_q[:, None] < T) & mask_d[None, :]
            q = tl.load(Q + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                        mask=mask_q, other=0.0)
            do = tl.load(DO + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                         mask=mask_q, other=0.0)
            l_i = tl.load(L + pid_bh * stride_lb + offs_q * stride_lt, mask=offs_q < T, other=0.0)
            d_i = tl.load(D + pid_bh * stride_lb + offs_q * stride_lt, mask=offs_q < T, other=0.0)

            # rebuild P (transposed) from the stored L, same as the forward did
            sT = tl.dot(k, tl.trans(q)) * qk_scale
            pT = tl.math.exp2(sT - l_i[None, :])
            dpT = tl.dot(v, tl.trans(do))           # (dO V^T)^T
            dsT = pT * (dpT - d_i[None, :])
            dv_acc += tl.dot(pT.to(do.dtype), do)
            dk_acc += tl.dot(dsT.to(q.dtype), q)

    dk_acc = dk_acc * sm_scale                  # the 1/sqrt(d) rides on dK (and dQ), not dV
    tl.store(DK + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
             dk_acc.to(DK.dtype.element_ty), mask=mask_k)
    tl.store(DV + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
             dv_acc.to(DV.dtype.element_ty), mask=mask_k)


# dQ: opposite loop order. query tile outer, walk its live key blocks. plain
# (query, key) layout this time.
@triton.jit
def _bwd_dq_kernel(Q, K, V, DO, L, D, DQ,
                   q_row_offsets, q_col_indices,
                   sm_scale,
                   stride_b, stride_t, stride_d,
                   stride_lb, stride_lt,
                   T,
                   BLOCK: tl.constexpr, T_TILE: tl.constexpr,
                   BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_qt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    qk_scale = sm_scale * 1.4426950408889634

    q0 = pid_qt * T_TILE
    iblk = q0 // BLOCK
    offs_q = q0 + tl.arange(0, T_TILE)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    base = pid_bh * stride_b
    mask_q = (offs_q[:, None] < T) & mask_d[None, :]

    q = tl.load(Q + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=mask_q, other=0.0)
    do = tl.load(DO + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                 mask=mask_q, other=0.0)
    l_i = tl.load(L + pid_bh * stride_lb + offs_q * stride_lt, mask=offs_q < T, other=0.0)
    d_i = tl.load(D + pid_bh * stride_lb + offs_q * stride_lt, mask=offs_q < T, other=0.0)

    dq_acc = tl.zeros((T_TILE, BLOCK_D), dtype=tl.float32)

    lo = tl.load(q_row_offsets + iblk)
    hi = tl.load(q_row_offsets + iblk + 1)
    for idx in range(lo, hi):
        jblk = tl.load(q_col_indices + idx)
        for kk in range(0, BLOCK, T_TILE):
            offs_k = jblk * BLOCK + kk + tl.arange(0, T_TILE)
            mask_k = (offs_k[:, None] < T) & mask_d[None, :]
            k = tl.load(K + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                        mask=mask_k, other=0.0)
            v = tl.load(V + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                        mask=mask_k, other=0.0)

            s = tl.dot(q, tl.trans(k)) * qk_scale
            p = tl.math.exp2(s - l_i[:, None])
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - d_i[:, None])
            dq_acc += tl.dot(ds.to(k.dtype), k)

    dq_acc = dq_acc * sm_scale
    tl.store(DQ + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
             dq_acc.to(DQ.dtype.element_ty), mask=mask_q)


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,
                          q_row_offsets, q_col_indices,
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash backward. ALGORITHMS.md sec 1, 4.

      Q,K,V,O,dO (B,H,T,d) fp16, L (B,H,T) fp32
      k_* = key-block view (query blocks per key block) -> used by dK/dV
      q_* = query-block view (key blocks per query block) -> used by dQ
    -> dQ, dK, dV (B,H,T,d) fp16
    """
    Bb, H, T, d = Q.shape
    BH = Bb * H
    block = int(BLOCK_Q)
    Qr = Q.reshape(BH, T, d)
    Kr = K.reshape(BH, T, d)
    Vr = V.reshape(BH, T, d)
    Or = O.reshape(BH, T, d)
    dOr = dO.reshape(BH, T, d)
    Lr = L.reshape(BH, T)

    dQ = torch.empty((Bb, H, T, d), device=Q.device, dtype=torch.float16)
    dK = torch.empty((Bb, H, T, d), device=Q.device, dtype=torch.float16)
    dV = torch.empty((Bb, H, T, d), device=Q.device, dtype=torch.float16)
    dQr = dQ.reshape(BH, T, d)
    dKr = dK.reshape(BH, T, d)
    dVr = dV.reshape(BH, T, d)

    D = torch.empty((BH, T), device=Q.device, dtype=torch.float32)

    T_TILE = _tile_for(block)
    BLOCK_D = triton.next_power_of_2(d)
    n_tiles = T // T_TILE
    meta = _attn_meta(d)

    # 1) D pre-pass
    _bwd_d_kernel[(n_tiles, BH)](
        Or, dOr, D,
        Or.stride(0), Or.stride(1), Or.stride(2),
        D.stride(0), D.stride(1),
        T,
        T_TILE=T_TILE, BLOCK_D=BLOCK_D, HEAD_DIM=d,
        num_warps=4,
    )

    # 2) dK, dV  (key tile outer)
    _bwd_dkdv_kernel[(n_tiles, BH)](
        Qr, Kr, Vr, dOr, Lr, D, dKr, dVr,
        k_row_offsets, k_col_indices,
        sm_scale,
        Qr.stride(0), Qr.stride(1), Qr.stride(2),
        Lr.stride(0), Lr.stride(1),
        T,
        BLOCK=block, T_TILE=T_TILE, BLOCK_D=BLOCK_D, HEAD_DIM=d,
        **meta,
    )

    # 3) dQ  (query tile outer)
    _bwd_dq_kernel[(n_tiles, BH)](
        Qr, Kr, Vr, dOr, Lr, D, dQr,
        q_row_offsets, q_col_indices,
        sm_scale,
        Qr.stride(0), Qr.stride(1), Qr.stride(2),
        Lr.stride(0), Lr.stride(1),
        T,
        BLOCK=block, T_TILE=T_TILE, BLOCK_D=BLOCK_D, HEAD_DIM=d,
        **meta,
    )
    return dQ, dK, dV
