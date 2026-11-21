# b12x MoE backend — required tree patch

`src/vllm_exl3/b12x_moe.py` (selected only by `VLLM_EXL3_MOE_KERNEL=b12x`)
drives the b12x W4A16 trellis MoE over the live EXL3 arenas. The b12x tree
that ships in the flashinfer experiment does not accept the `mul1` codebook
that our EXL3 packs use, so the backend needs `patches/b12x-route-a.patch`
applied to it first.

What the patch adds:

* `mul1` registered as a codebook (sentinel `0x83DCD12D`, projection flag
  `(False, True)`, bitrates 2/3/4) and treated as LUT-free like `mcg`.
* `packed_dequant_trellis_mul1_to_half2x4` — the fp16 arm, bit-exact to
  `codebook.cuh` `cb == 2`. The pre-existing b12x intrinsic for mul1 finishes
  with `cvt.rn.satfinite.e4m3x2.f16x2`; that rounding is 2.6% rel-L2 on the
  weights, so it is not used for serving.
* the dispatch hook in the W4A16 GEMM decode plus the codebook sets in the
  4 sites that enumerate them, and the joint gate/up projection-major layout.

Apply from the directory that *contains* the `b12x` package (for this host,
`~/b12x-vendor`, or whatever `VLLM_EXL3_B12X_PATH` points at):

```bash
git apply -p3 /path/to/b12x-route-a.patch
# note: -p3, not -p4 — the vendored layout is b12x/_lib/... , not _lib/...
```

Verify against an already-patched tree with
`git apply -R --check -p3 b12x-route-a.patch` (this is how the copy in this
repo was checked: it matches the vendor tree under `~/b12x-vendor/b12x`
exactly).

Measured on the DS41 mia-v1 pack (TP2xPP5, 512k): routed-expert prefill
1.57–1.72x the fused `exl3_moe`, decode unchanged (the backend only takes
batches at or above `VLLM_EXL3_B12X_MIN_M`, default 64, because at 8
rows/expert b12x wins but at decode sizes it is 5–9x slower).

sha256 of the patch as added: `2a227150970d1aacfcaf94f1e13ae6ef349f5471d1ce2abc86a6f961d5e1b6c4`
