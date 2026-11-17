"""Offline numerical checks for custom modules; run this file from the cloned source.

Examples:
    python custom_module_checks.py
    python custom_module_checks.py --device cuda --models --imgsz 640

These checks validate stated calculations, not exact paper reproduction or accuracy.
"""

import argparse
import gc
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT.parent / ".repair_deps").is_dir():
    sys.path.insert(1, str(ROOT.parent / ".repair_deps"))
_config = tempfile.TemporaryDirectory(prefix="custom-module-checks-")
os.environ.setdefault("YOLO_CONFIG_DIR", _config.name)
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.block import C2MSDA, BiLevelRoutingAttention, C3k2_DSConv
from ultralytics.nn.modules.conv import (
    CARAFE,
    DSC,
    WCA,
    DODConv,
    DSConv,
    DySample,
    DySnakeConv,
    GAM_Attention,
    HetConv,
    ResBlock_CBAM,
)
from ultralytics.nn.modules.msda import MSDA
from ultralytics.nn.tasks import DetectionModel

OPTIONS = argparse.Namespace(device="cpu", models=False, imgsz=64)


class CustomModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_dysample_zero_offsets_equal_bilinear(self):
        """The initialization lattice must reproduce bilinear upsampling exactly."""
        for style in ("lp", "pl"):
            for scale in (2, 3):
                for scope in (False, True):
                    with self.subTest(style=style, scale=scale, scope=scope):
                        layer = DySample(36, scale, style, dyscope=scope).to(OPTIONS.device)
                        nn.init.zeros_(layer.offset.weight)
                        nn.init.zeros_(layer.offset.bias)
                        x = torch.randn(2, 36, 5, 7, device=OPTIONS.device)
                        expected = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
                        torch.testing.assert_close(layer(x), expected, atol=5e-6, rtol=5e-5)

    def test_deformable_zero_offsets_equal_grouped_convolution(self):
        """DODConv currently implements deformable convolution, including weight groups."""
        for groups in (1, 2, 4):
            for stride in (1, 2):
                layer = DODConv(4, 8, 3, stride, 1, groups, True).to(OPTIONS.device)
                x = torch.randn(2, 4, 7, 9, device=OPTIONS.device, requires_grad=True)
                expected = F.conv2d(x, layer.weight, layer.bias, stride, 1, groups=groups)
                actual = layer(x)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-5)
                actual.square().mean().backward()
                self.assertGreater(layer.offset_conv.weight.grad.abs().sum().item(), 0)

    def test_snake_against_scalar_bilinear_reference(self):
        """Independently accumulate snake offsets and interpolate all four corner values."""
        x = torch.randn(2, 2, 5, 7)
        for kernel in (3, 5):
            for morph in (0, 1):
                offsets = torch.randn(2, 2 * kernel, 5, 7) * 0.3
                actual = (
                    DSC(x.shape, kernel, 1, morph)
                    .deform_conv(x.to(OPTIONS.device), offsets.to(OPTIONS.device), True)
                    .cpu()
                )
                expected = torch.empty_like(actual)
                center = kernel // 2
                for b in range(2):
                    for row in range(5):
                        for col in range(7):
                            for j in range(kernel):
                                lo, hi = (j, center) if j < center else (center + 1, j + 1)
                                delta = sum(float(offsets[b, morph * kernel + t, row, col]) for t in range(lo, hi))
                                y = row + (delta if morph == 0 else j - center)
                                xx = col + (j - center if morph == 0 else delta)
                                y, xx = min(max(y, 0), 4), min(max(xx, 0), 6)
                                y0, x0 = math.floor(y), math.floor(xx)
                                y1, x1 = min(y0 + 1, 4), min(x0 + 1, 6)
                                fy, fx = y - y0, xx - x0
                                value = (
                                    (1 - fy) * (1 - fx) * x[b, :, y0, x0]
                                    + (1 - fy) * fx * x[b, :, y0, x1]
                                    + fy * (1 - fx) * x[b, :, y1, x0]
                                    + fy * fx * x[b, :, y1, x1]
                                )
                                if morph == 0:
                                    expected[b, :, row * kernel + j, col] = value
                                else:
                                    expected[b, :, row, col * kernel + j] = value
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-5)

    def test_carafe_against_low_resolution_neighborhood_sum(self):
        """Reassemble low-resolution neighbors directly, without unfold or upsampling."""
        for scale in (2, 3):
            layer = CARAFE(4, c_mid=8, scale=scale, k_up=3).to(OPTIONS.device).eval()
            x = torch.randn(2, 4, 3, 5, device=OPTIONS.device)
            with torch.no_grad():
                weights = F.pixel_shuffle(layer.enc(layer.comp(x)), scale).softmax(1)
                expected = torch.zeros(2, 4, 3 * scale, 5 * scale, device=OPTIONS.device)
                for row in range(3 * scale):
                    for col in range(5 * scale):
                        for kr in range(3):
                            for kc in range(3):
                                rr, cc = row // scale + kr - 1, col // scale + kc - 1
                                if 0 <= rr < 3 and 0 <= cc < 5:
                                    expected[:, :, row, col] += (
                                        weights[:, kr * 3 + kc, row, col, None] * x[:, :, rr, cc]
                                    )
                torch.testing.assert_close(layer(x), expected, atol=3e-6, rtol=5e-5)

    def test_routing_single_window_equals_global_attention(self):
        """One routing window and no local convolution reduce to dense attention."""
        layer = BiLevelRoutingAttention(16, num_heads=4, n_win=1, topk=1, side_dwconv=0).to(OPTIONS.device)
        x = torch.randn(2, 16, 3, 5, device=OPTIONS.device)
        tokens = x.flatten(2).transpose(1, 2)
        q, k, v = layer.qkv.qkv(tokens).chunk(3, dim=-1)

        def heads(t):
            return t.reshape(2, 15, 4, 4).transpose(1, 2)

        weights = ((heads(q) * (16**-0.5)) @ heads(k).transpose(-2, -1)).softmax(-1)
        attended = (weights @ heads(v)).transpose(1, 2).reshape(2, 15, 16)
        expected = layer.wo(attended).transpose(1, 2).reshape_as(x)
        torch.testing.assert_close(layer(x), expected)

    def test_hetconv_equals_fused_dense_kernel(self):
        """Parallel groupwise and pointwise branches must align spatially and by group."""
        for stride in (1, 2):
            layer = HetConv(8, 12, p=4, stride=stride).to(OPTIONS.device)
            weight = F.pad(layer.pwc.weight, (1, 1, 1, 1)).clone()
            for out in range(12):
                group = out // 3
                weight[out, group * 2 : (group + 1) * 2] += layer.gwc.weight[out]
            x = torch.randn(2, 8, 7, 9, device=OPTIONS.device)
            torch.testing.assert_close(layer(x), F.conv2d(x, weight, stride=stride, padding=1))

    def test_msda_uniform_attention_equals_local_mean(self):
        """Zero query/key makes each dilation group a uniform local average."""
        layer = MSDA(6, num_heads=3, dilations=(1, 2, 3)).to(OPTIONS.device).eval()
        x = torch.randn(2, 6, 4, 5, device=OPTIONS.device)
        with torch.no_grad():
            layer.qkv.weight[: 2 * layer.inner_dim].zero_()
            for p in layer.ffn.parameters():
                p.zero_()
            normalized = layer._norm(x, layer.norm1)
            value = F.conv2d(normalized, layer.qkv.weight[2 * layer.inner_dim :])
            groups = value.chunk(3, dim=1)
            means = []
            for part, dilation in zip(groups, layer.dilations):
                kernel = torch.ones(part.shape[1], 1, 3, 3, device=OPTIONS.device) / 9
                means.append(F.conv2d(part, kernel, padding=dilation, dilation=dilation, groups=part.shape[1]))
            expected = x + layer.proj(torch.cat(means, dim=1))
            torch.testing.assert_close(layer(x), expected)

    def test_all_eleven_module_types_have_finite_gradients(self):
        layers = [
            BiLevelRoutingAttention(16, num_heads=4, n_win=2, topk=2),
            C2MSDA(16, 16),
            C3k2_DSConv(16, 16, c3k=True),
            CARAFE(16, c_mid=8, k_up=3),
            DODConv(16, 16),
            DySample(16),
            DySnakeConv(16, 16),
            GAM_Attention(16, 16),
            HetConv(16, 16),
            ResBlock_CBAM(16, 16),
            WCA(16),
        ]
        for layer in layers:
            with self.subTest(module=type(layer).__name__):
                layer = layer.to(OPTIONS.device).eval()
                x = torch.randn(2, 16, 7, 9, device=OPTIONS.device, requires_grad=True)
                y = layer(x)
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)
                # The historical DSConv class registers both orientation convolutions,
                # but executes only one. Record this known redundancy explicitly.
                inactive = set()
                for sub in layer.modules():
                    if isinstance(sub, DSConv):
                        branch = sub.dsc_conv_y if sub.morph == 0 else sub.dsc_conv_x
                        inactive.update(id(p) for p in branch.parameters())
                for name, p in layer.named_parameters():
                    if p.requires_grad and id(p) not in inactive:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_c3k2_branch_architecture_is_explicit(self):
        plain = C3k2_DSConv(16, 16, c3k=False)
        snake = C3k2_DSConv(16, 16, c3k=True)
        self.assertEqual(sum(isinstance(m, DSConv) for m in plain.modules()), 0)
        self.assertGreater(sum(isinstance(m, DSConv) for m in snake.modules()), 0)

    def test_full_detection_models(self):
        if not OPTIONS.models:
            self.skipTest("pass --models to run loss/backward/inference for all 13 detection configurations")
        directory = ROOT / "ultralytics/cfg/models/11"
        excluded = {"yolo11-cls", "yolo11-cls-resnet18", "yolo11-seg", "yolo11-pose", "yolo11-obb"}
        paths = [
            directory / "yolo11.yaml",
            *[p for p in sorted(directory.glob("yolo11-*.yaml")) if p.stem not in excluded],
        ]
        for path in paths:
            with self.subTest(model=path.name):
                model = DetectionModel(str(path), verbose=False).to(OPTIONS.device)
                model.args = get_cfg()
                model.train()
                images = torch.randn(2, 3, OPTIONS.imgsz, OPTIONS.imgsz, device=OPTIONS.device)
                batch = {
                    "img": images,
                    "batch_idx": torch.tensor([1.0], device=OPTIONS.device),
                    "cls": torch.tensor([[0.0]], device=OPTIONS.device),
                    "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=OPTIONS.device),
                }
                loss, _ = model.loss(batch)
                self.assertTrue(torch.isfinite(loss).all())
                loss.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                with torch.no_grad():
                    model.eval()
                    self.assertTrue(torch.isfinite(model(images)[0]).all())
                print("PASS:", path.name, OPTIONS.device, OPTIONS.imgsz, flush=True)
                del model, loss, batch, images
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--models", action="store_true", help="also test the complete detection models")
    parser.add_argument(
        "--imgsz", type=int, default=64, help="model-check image side; positive multiple of 32, at least 64"
    )
    OPTIONS, remaining = parser.parse_known_args()
    if OPTIONS.imgsz < 64 or OPTIONS.imgsz % 32:
        parser.error("--imgsz must be at least 64 and divisible by 32")
    torch.set_num_threads(2)
    print("Imported local source:", Path(sys.modules["ultralytics"].__file__).resolve(), flush=True)
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
