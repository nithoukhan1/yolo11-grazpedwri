# Repair status — 2026-11-15

The implementation has now been changed. The original 2026-11-14 reports describe the pre-repair snapshot; this file records the current status. The original 793 files are preserved in `pre_repair_snapshot_2026-11-15.zip`. `REPAIR_DIFF.patch` and `REPAIR_MANIFEST.json` identify the actual source changes.

## Numbered findings

| Finding | Change                                                                                                             | Verification                                                                                                                             |
| ------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1–2     | Snake sampling uses per-image grid sampling; cumulative offsets retain gradients and include both outer positions. | Independent-image equivalence and nonzero finite gradients for both morphs and kernels 3/5; complete models on CPU/GPU.                  |
| 3       | Added an explicitly documented DilateFormer-inspired MSDA adapter; repaired C2MSDA's argument contract.            | Shape/gradient checks; both affected complete models work across all five scales. Exact original architecture remains unidentified.      |
| 4       | Parser inserts the repeat argument exactly once.                                                                   | Complete C3k2_DSConv model construction, loss/backward, inference, all-scale checks.                                                     |
| 5       | Corrected YAML arguments; standalone snake blocks explicitly downsample and project concatenated channels.         | Complete DySnakeConv model checks, including all scales and a GPU optimizer step. This repaired architecture requires fresh experiments. |
| 6       | Corrected routing-attention default head/window order.                                                             | Complete routing-attention model checks and all scales.                                                                                  |
| 7       | Declared `einops` in package dependencies; installed it into workspace-only `.repair_deps` for testing.            | Real local imports and CLI launcher work. Global Python packages were not changed.                                                       |
| 8       | Mask-loss loop iterates over images and selects instance masks inside the loop.                                    | Sparse two-image batch matches direct expected loss and supplies a gradient for the second image.                                        |
| 9       | Disk cache honors the requested channel flag; incompatible existing channel caches are rejected/rebuilt.           | Grayscale cache creation and loading of an incompatible color cache.                                                                     |
| 10      | Partial checkpoint loading checks source-key membership.                                                           | Differently named source backbone loads without KeyError.                                                                                |
| 11      | AMP broadcast retains the tensor receiving the value.                                                              | Executes actual broadcast statements with a synthetic receiving rank; live multi-rank DDP was not run.                                   |
| 12      | SciPy assignment maximizes IoU using negated costs.                                                                | The documented two-match example now returns two true positives.                                                                         |
| 13      | Small-batch crop bounds are clipped to mask dimensions.                                                            | Boundary-crossing example retains the expected nine pixels.                                                                              |
| 14      | Transposed-convolution fusion scales the real output-channel axis, including groups.                               | Fused/unfused equivalence with groups 1/2/4 and bias enabled/disabled.                                                                   |
| 15      | Weight reset restores and freezes fixed DFL bin projections.                                                       | Actual YOLO reset preserves bin values and frozen status.                                                                                |
| 16      | Conv2 distinguishes branch fusion from full BN fusion.                                                             | Numerical equivalence for both fusion stages.                                                                                            |
| 17–18   | Cache identity includes timestamps and annotation contents; grounding passes a path list.                          | Equal-size label edits with preserved mtime and changed-size path identity checks; grounding call sites corrected.                       |
| 19      | Classification splitting rejects populated destinations.                                                           | A second split is rejected without modifying the existing independent memberships.                                                       |
| 20      | RT-DETR filters sorted predictions using their current confidence column.                                          | The 0.9-confidence detection is retained and the 0.1 detection removed.                                                                  |
| 21      | TinyViT reads cached bias only in eval mode and correctly returns self from train().                               | Fresh training, eval, and repeated train transitions with gradients.                                                                     |
| 22      | DDP files serialize Windows model paths with repr.                                                                 | Generated worker file compiles and preserves the absolute path.                                                                          |
| 23      | Comet checks a positive save period before modulo.                                                                 | Metadata helper handles save_period=0 without connecting to Comet.                                                                       |
| 24      | TorchVision video classifier returns nested label/confidence lists to match its renderer.                          | Original postprocessor output can be consumed by the rendering contract.                                                                 |

## Validation completed

- **24 targeted regression tests pass**, including actual model loss/backward/inference for all 13 detection configurations at scale n. Run `python -B repair_regression.py`; captured output is `repair_regression_results.txt`.
- **65 CPU model-scale checks pass**: the baseline plus 12 custom configurations at n/s/m/l/x, using real detection heads and finite predictions.
- **13 GPU FP32 optimizer steps pass** on the available NVIDIA GeForce GTX 1660 Ti.
- **13 AMP cases are skipped by the repository's hardware guard.** Forced AMP initially failed for every model, including the baseline. The existing `check_amp` explicitly disables the 1660 family. Use `amp=False` on this machine; this is not evidence that AMP works on other GPUs. Initial forced-AMP diagnostics are retained separately.
- Three actual JavaScript widget checks pass: empty CSV handling, zero/quote preservation, and event-listener cleanup. Full browser rendering was not tested.
- All current Python source files compile and all YAML/YML files syntax-load.

`repair_matrix_results.json` contains per-model/device/precision results. `repair_validation_matrix.py` reproduces that matrix. These are small synthetic checks, not a full training benchmark or a claim of research accuracy.

## Additional source changes

Also repaired default trainer overrides handling, tuner cleanup of the losing run, missing OpenVINO metadata's batch fallback, pose text coordinates, ECC frame-state advancement and downscale translation, stale region counts, analytics configuration, SAM video overlap suppression/object-ID validation, xView's removed NumPy alias, RF100 YAML saving, discarded geometry assertions, repeated docs head injection, archive Docker builds, unused WCA parameters, and chart CSV/listener handling.

The native examples received ownership/buffer, geometry, and tensor-error-handling repairs. **C++/Rust SDK builds, sanitizers, live Triton/OpenVINO/MNN integration, and native-example output equivalence remain unverified.** Some optional demonstration backends still use class-agnostic NMS; choose that behavior deliberately before adopting those demos for evaluation. Native changes should be reviewed/built with their required SDKs before deployment.

## Research and compatibility limits

Read [MSDA_RESEARCH.md](MSDA_RESEARCH.md) before citing or training the MSDA variants. The new implementation is a documented adaptation, not recovery of the missing original class. The standalone DySnakeConv graph also has explicit new pooling/projection layers. WCA's unused global-pooling parameters were removed while preserving the active forward calculation. Older custom checkpoints are not guaranteed to match these repaired graphs.

No real GRAZPEDWRI-DX dataset, patient split, experiment outputs, or custom weights were supplied. Full training/validation on your data, patient-level independence, research metrics, long-run numerical stability, distributed training, and all export formats remain unverified. Do not treat a smoke-test pass as confirmation that earlier experiments are valid.

## Follow-up custom-module review (2026-11-15)

`ultralytics-main/CUSTOM_MODULE_REVIEW.md` records a separate architecture review
against primary references. Runtime passes do not establish exact research-method
identity: C2MSDA and standalone DySnakeConv are adaptations; ResBlock_CBAM uses
Ultralytics' simplified CBAM; CARAFE has a BN/ReLU predictor adaptation; HetConv
uses an approximate parallel-branch parameterization; DODConv is deformable
convolution rather than a verified DO-DConv implementation; C3k2_DSConv uses
ordinary bottlenecks when c3k=False. The report also records WCA's pooled-axis
kernel parameterization and inactive DSConv orientation parameters. These are
explicit research qualifications, not a claim that the existing runtime tests
failed. Follow-up source edits only clarified docstrings; no functional graph
was changed in this review.

The portable `ultralytics-main/custom_module_checks.py` adds independent numerical
checks for interpolation, neighborhood reassembly, grouped convolution and
attention, plus finite-gradient checks for all eleven main module types.
All **10 tests passed on CPU**, including loss/backward/inference for all 13
n-scale detection configurations at 64x64. All **10 tests also passed on the
local GPU in FP32**, with those 13 complete models checked at **640x640, batch 2**
(Python exit code 0; `custom_module_gpu_results.txt`). These checks do not test
the full training loop, actual Kaggle dependencies/data, multi-GPU execution,
custom-model AMP, or exact paper reproduction.

Copies of this status document and MSDA_RESEARCH.md are included inside
`ultralytics-main` so the qualifications travel with the source checkout.

## Run the local repaired source

From this workspace:

```powershell
python -B run_local.py version
python -B repair_regression.py
```

`run_local.py` uses `ultralytics-main` and workspace `.repair_deps`, avoiding accidental use of another installed Ultralytics version. Supply your real model/data/training arguments through that launcher when ready, with `amp=False` for the available GPU. For a fresh environment, installing this project's declared dependencies is still required.

The older `audit_checks.py`, `extended_audit_checks.py`, and `final_audit_checks.py` were written to demonstrate the original failures; their failure-expecting assertions are historical diagnostics. Use the repair regression suite for the current code.
