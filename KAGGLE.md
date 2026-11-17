# Train this source checkout on Kaggle

Enable a GPU and Internet in the Kaggle notebook, and attach your uploaded
dataset. Run these notebook cells from a fresh session:

```python
%cd /kaggle/working
!git clone https://github.com/nithoukhan1/yolo11-grazpedwri.git
%cd /kaggle/working/yolo11-grazpedwri
!python -m pip install -e .
```

If Ultralytics was imported before installation, restart the session and return
to `/kaggle/working/yolo11-grazpedwri` before continuing. Confirm the imported
source belongs to this clone:

```python
from pathlib import Path

import ultralytics

source = Path(ultralytics.__file__).resolve()
print(source)
assert Path("/kaggle/working/yolo11-grazpedwri") in source.parents
```

Check the custom modules and all thirteen default n-scale detection configurations:

```python
!python -B custom_module_checks.py --device cuda --models --imgsz 640
```

These are FP32 synthetic numerical/loss/backward/inference checks. They do not
validate your dataset, the full trainer, mixed precision, or research accuracy.
Torch and TorchVision must be compatible with working deformable-convolution
operators. Read [CUSTOM_MODULE_REVIEW.md](CUSTOM_MODULE_REVIEW.md) before naming
or citing an experimental variant.

Prepare a dataset YAML whose `path`, `train`, and `val` point to the attached
dataset under `/kaggle/input/...`. Verify that its class names and label indices
match, and retain the intended patient-independent split. Keep the dataset in
Kaggle; it is not stored in this GitHub repository.

Start with a short baseline run using your real dataset YAML:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
model.train(
    data="/kaggle/input/YOUR_DATASET/YOUR_DATA_YAML.yaml",  # replace this path
    epochs=3,
    imgsz=640,
    batch=2,
    device=0,
    amp=False,
    project="/kaggle/working/runs",
    name="baseline_smoke",
)
```

The YAML model constructor starts from random initialization. `amp=False` makes
the initial real-data check comparable to the tested FP32 path. Test AMP on the
actual Kaggle GPU before enabling it for a longer custom-model run. Confirm that
training, validation, and checkpoint saving complete before starting full runs.

For an individual custom model, replace the model YAML path, for example:

```python
model = YOLO("ultralytics/cfg/models/11/yolo11-DySample.yaml")
```

Use the same data split and planned initialization/training protocol when
comparing variants. If loading pretrained weights, record which weights loaded;
changed custom graphs do not guarantee compatibility with previous checkpoints.
Save notebook outputs/checkpoints through Kaggle before the session ends.
