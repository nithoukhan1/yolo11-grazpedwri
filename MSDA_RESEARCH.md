# MSDA selection and architecture changes

The missing class cannot be recovered from this source snapshot. Its wrapper explicitly describes **Multi-Scale Dilated Attention** and passes `dim`, `num_heads`, and `dilations`. These clues most closely match the mechanism in [DilateFormer, Jiao et al., TMM 2023](https://arxiv.org/abs/2302.01791), with an [official implementation](https://github.com/JIAOJIAYUASD/dilateformer/blob/main/models/dilateformer.py).

This is an inference about the intended mechanism, not proof of the original author's choice. A different paper also uses C2MSDA for defect-aware attention, but that mechanism does not match the dilated-attention description in this repository.

The new `ultralytics/nn/modules/msda.py` is an independent implementation of local query/key/value attention with different dilation rates across groups of heads, followed by projection and a residual pointwise feedforward block. It accepts NCHW feature maps. It was written for this repair; it is **not** the original missing implementation or a claim of exact paper reproduction.

The official implementation requires equal dilation groups with compatible channels and heads. YOLO's scaled widths here often violate those constraints. This adapter rounds the requested head count upward to a multiple of the number of scales, uses `ceil(dim / heads)` channels per head, and projects the resulting latent width back to `dim`. This preserves all configured dilation scales and the surrounding tensor shapes. Layer normalization, residual connections, a GELU feedforward expansion of 2, and the configured dilation tuple are explicit choices in this adapter.

For research, describe this variant as a **YOLO-adapted DilateFormer-inspired MSDA**, record the effective head counts/latent dimensions, and rerun its ablations. Existing unidentified C2MSDA results or weights are not evidence for this new implementation.

The standalone DySnakeConv configuration also needed a declared channel/stride contract. Nested bottlenecks retain three concatenated branches and their existing projection. Standalone parsed blocks now use average-pool downsampling before those branches and a 1x1 projection afterward. This is an explicit repaired architecture that requires fresh validation/training; compatibility with an unspecified previous design is not claimed.
