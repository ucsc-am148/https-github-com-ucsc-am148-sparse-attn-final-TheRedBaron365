[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WMFvlNly)
# Block-sparse flash attention: final assignment

Block-sparse attention, end to end: a block-sparse matmul (A1), the sparse flash
forward (A2), and the sparse flash backward (A3). You implement three functions
from the spec in `ALGORITHMS.md`. The spec gives you the math and the layout of 
the given inputs.

## The ladder (3 rungs)

You edit one file, `kernels.py`. Only the input and output shapes and dtypes are
fixed (the contract, collected in `ALGORITHMS.md §0.1`); the grader asserts the
returned shapes and dtypes, then checks values against an fp64 reference.
Everything inside is yours: the `@triton.jit` kernels, the grid, the `(B, H)`
flatten, allocation, and tuning.

| Rung | Function | Points | Produces | Spec |
|---|---|---:|---|---|
| A1 | `dsd_matmul` | 10 | block-sparse `A @ B` -> dense `C` | §1, §2 |
| A2 | `sparse_flash_forward` | 20 | sparse flash forward -> `O`, `L` | §1, §3 |
| A3 | `sparse_flash_backward` | 20 | sparse flash backward -> `dQ`, `dK`, `dV` (graded together) | §1, §4 |

## Grading: 100 points

- 50 points, code correctness, owned by this autograder. A1 = 10, A2 = 20,
  A3 = 20; each rung is all-or-nothing across its size battery (A3 across all
  three gradients).
- 50 points, oral exam, offline.

The graded path gates on correctness plus two perf guards. A loose anti-cheese
time floor rejects a correct rung that runs more than ~10x the reference; it
backstops pathologically slow kernels and does not enforce sparsity. A2 and A3
additionally carry a flash-memory requirement (below).

## Files

```
README.md            this file
ALGORITHMS.md        the spec: BCSR layout, what each output is, the backward equations (READ THIS)
requirements.txt     torch>=2.4, triton>=3.6
kernels.py           the three rung-API functions (STUDENT: edit only this)
bcsr.py              host-side BCSR helpers + the dual transpose-view builder (GIVEN)
harness.py           prepare / inputs / run wrappers + fp64 reference + SPARSE_REF toggle (GIVEN)
kernels_golden.py    serves precomputed reference outputs in SPARSE_REF=1 mode (GIVEN)
golden.pt            the precomputed outputs (GIVEN, don't edit)
sanity_check.py      per-rung validation vs the fp64 reference (GIVEN)
.github/workflows/sparse-attn-grader.yml   Classroom autograder trigger (GIVEN)
```

## How to run

From the assignment root:

```bash
pip install -r requirements.txt

# Smoke-test your environment + harness wiring (serves precomputed outputs):
SPARSE_REF=1 python sanity_check.py

# Run your implementation against the live fp64 reference:
python sanity_check.py
```

`SPARSE_REF=1` swaps in `kernels_golden.py`, which looks up frozen reference
outputs from `golden.pt`. It checks that your environment works; it does not check
your kernels. It is keyed by the default cases in `sanity_check.py`; if you change
them you get a `KeyError`, so run the default (`SPARSE_REF=0`) to compare against
the live fp64 reference.

## Precision

Output dtypes are mandated and the grader asserts them: `O`, `dQ`, `dK`, `dV` in
fp16, `L` in fp32 (`ALGORITHMS.md §0.1`). A1 is fp32 with `allow_tf32=False`
(`§2`). Tolerances are shown by `sanity_check.py`.

## Pass criteria

A rung's points are awarded when every size in its battery returns the contracted
shapes and dtypes and is within tolerance against the fp64 reference, its kernel
clears the anti-cheese floor, and (A2/A3) it clears the flash-memory requirement.
A2 checks `O` and `L`; A3 checks `dQ`, `dK`, `dV` together.

## Flash requirement (A2, A3)

A2 and A3 must run in O(T) memory. The grader sweeps the sequence length and
measures peak GPU memory; a forward or backward whose memory grows like the full
attention matrix (O(T^2)) is rejected, even when it is numerically correct. At
fixed density a flash kernel and a materialized one do the same FLOPs, so this is
a memory check, not a timing one: flash keeps the attention matrix on chip and
recomputes it block by block rather than storing it. The recipe is in the lecture
notes.