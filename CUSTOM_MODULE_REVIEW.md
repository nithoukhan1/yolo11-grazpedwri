# Custom-module implementation review

Reviewed 2026-11-15. Runtime correctness and equivalence to a named research
method are separate questions. The existing model configurations run in the
checks below, but several are adaptations. They must be described as such in
experiments. This review does not certify every argument combination, dataset,
precision mode, export backend, or exact paper reproduction.

## Module verdicts

| Module                  | What is implemented and what remains qualified                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BiLevelRoutingAttention | Routing, gathering, Q/K/V projection and attention follow the official legacy BiFormer calculation, with NCHW input/output adaptation and automatic padding. The single-window numerical check reduces it to dense global attention. Optional differentiable hard routing, fractional pooling and convolutional KV downsampling remain explicitly unsupported, as in the reference. The optional `ret_attn_mask=True` path returns its feature tensor as NHWC, while the normal path returns NCHW. |
| C2MSDA                  | Valid split/process/fuse wrapper around an independently implemented DilateFormer-inspired MSDA adapter. Local attention passes a uniform-weight neighborhood-average check. The original missing implementation remains unknown. Head count and latent width are rounded for YOLO channel compatibility. Read MSDA_RESEARCH.md.                                                                                                                                                                   |
| C3k2_DSConv             | A hybrid architecture: `c3k=True` contains nested snake bottlenecks; `c3k=False` uses the ordinary two-convolution DSCBottleneck and contains no DSConv. The current YAML uses both branches. Do not describe it as replacing every bottleneck with snake convolution or as an exact WCAY DSC-C2f reproduction.                                                                                                                                                                                    |
| CARAFE                  | Pixel-shuffled kernels are softmax-normalized across spatial neighbors; reassembly matches an independent low-resolution neighborhood sum, including zero-padded boundaries. Its compressor/content encoder add batch normalization and an initial ReLU compared with the referenced CARAFEPack. Describe it as a CARAFE adaptation.                                                                                                                                                               |
| DODConv                 | An unmodulated deformable convolution with learned offsets and optional weight groups. Zero offsets match ordinary grouped convolution, and offset predictors receive gradients. Current YAML specifies groups=1: it is not depthwise convolution. It has neither an explicit orientation constraint nor depthwise kernel over-parameterization. The published method intended by the name is unidentified; do not claim DO-DConv reproduction.                                                    |
| DySample                | Sampling lattice, offset scaling, LP/PL ordering and border interpolation follow the official implementation. Zero-offset sampling matches bilinear interpolation for scales 2/3, both styles, and scope enabled/disabled. The extra unused constructor arguments are integration compatibility arguments.                                                                                                                                                                                         |
| DySnakeConv             | Snake-coordinate accumulation and border-clamped sampling match an independent scalar bilinear reference for both orientations, kernels 3/5, rectangular maps and two images. The repaired standalone YOLO block adds pooling/projection for its stride/channel contract. This is an explicit adaptation, requiring fresh experiments.                                                                                                                                                             |
| GAM_Attention           | Implements the paper's core channel-MLP then spatial-convolution sequence, using a grouped/shuffled variant. Shape and finite-gradient checks pass. This is not a demonstrated numerical match to an author checkpoint or a full paper reproduction.                                                                                                                                                                                                                                               |
| HetConv                 | Computes parallel grouped spatial convolution plus dense pointwise convolution. It matches a mathematically fused dense kernel. The extra pointwise center contribution and contiguous group pattern differ from the paper's shifted heterogeneous-filter parameterization and exact parameter/FLOP accounting. Describe it as a HetConv approximation.                                                                                                                                            |
| ResBlock_CBAM           | Valid residual block with Ultralytics' simplified CBAM: channel gating uses average pooling and one convolution. Original CBAM uses average and maximum pooling through a shared reduced MLP. This model runs but is not original CBAM inside a residual block.                                                                                                                                                                                                                                    |
| WCA                     | Horizontal/vertical pooling, learned coordinate weights and final gating follow WCAY's published weighting equations. The pooled-axis transform here is (3,1), whereas the paper describes a 3x3 transform. The pooled tensor has width one: under zero padding only the middle column of a 3x3 kernel would contribute, but parameterization/counts differ. The active calculation is tested for finite gradients; exact author-code/checkpoint equivalence is unverified.                        |

## Additional implementation observations

- DSConv registers both orientation convolution layers but executes only one per
  instance. The inactive layer's weights/bias receive no gradients. This is known
  parameter redundancy; the numerical tests explicitly identify these parameters
  rather than claiming every registered parameter learns. The repository trainer
  uses DDP with `find_unused_parameters=True`; external DDP wrappers must account
  for these unused parameters. No live multi-GPU run has been verified.
- Valid channel divisibility, matching residual shapes and odd kernel sizes are
  necessary for several module constructors. Passing the shipped configurations
  does not imply arbitrary constructor arguments are supported.
- No functional architecture was changed in this follow-up review. Module
  docstrings now make the existing adaptations explicit. Earlier runtime repairs
  are documented in FIX_STATUS.md.

## Reproducible checks

`custom_module_checks.py` imports this checkout, prints the source location,
disables automatic downloads and uses temporary settings. It includes independent
numerical limiting cases for sampling, reassembly, grouped convolution and
attention, finite-gradient checks for all eleven module types, and an optional
full-model check for the baseline plus twelve custom detection configurations.

```bash
# Run from this directory after cloning; install this source and its dependencies.
python -m pip install -e .
python -B custom_module_checks.py

# On Kaggle with a GPU enabled; test all 13 models in FP32 at your image size.
python -B custom_module_checks.py --device cuda --models --imgsz 640
```

The all-model check selects the default n scale, computes loss and backpropagates,
then runs inference using a synthetic two-image batch. It does not run the full
trainer, optimizer/scheduler, actual data loading, validation metrics, AMP or all
five model scales. Full-scale CPU/GPU checks from the preceding repair are
documented separately in FIX_STATUS.md. A numerical check is evidence about a
calculation, not proof of paper identity or expected accuracy.

The CPU run with `--models` passed all ten tests, including all thirteen model
configurations at 64x64. See the follow-up validation section in FIX_STATUS.md
for the GPU run result. Run these checks again in the actual Kaggle environment.
Torch and TorchVision must be a compatible pair with working deformable-convolution
operators. This script tests FP32; AMP needs a separate custom-model check on the
Kaggle GPU. The earlier `amp=False` advice concerned the local GTX 1660 Ti.

## Primary references inspected

- [Official BiFormer legacy attention](https://github.com/rayleizhu/BiFormer/blob/public_release/ops/bra_legacy.py)
- [Official DySample implementation](https://github.com/tiny-smart/dysample/blob/main/dysample.py)
- [Official CBAM implementation](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)
- [CARAFE paper](https://arxiv.org/abs/1905.02188) and [OpenMMLab CARAFEPack](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/carafe.py)
- [HetConv paper](https://arxiv.org/abs/1903.04120)
- [GAM paper](https://arxiv.org/abs/2112.05561)
- [WCAY paper, including WCA equations](https://www.nature.com/articles/s41598-024-77878-6)
- [Official DSCNet project](https://github.com/YaoleiQi/DSCNet)

Primary references help distinguish mechanisms; they do not establish the source
history of this snapshot. In particular, similarly named DO-DConv/DODConv methods
are not interchangeable with TorchVision deformable convolution.
