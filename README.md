# Embedded and Edge Artificial Intelligence - Radar project

**Project Assignment · Politecnico di Milano**  
Department of Electronics, Information and Bioengineering

---

## Task

Build a **TinyML system** that estimates the **2D positions of up to 4 people** inside a room using raw **UWB radar signals** (Channel Impulse Response - CIR).

| Property | Value |
|---|---|
| Input | Single `.npy` file of shape `(T, 6, 3, 120, 2)` |
| Output | `.jsonl` file - one JSON object per frame |
| Target hardware | ESP32-S3 via TensorFlow Lite for Microcontrollers |
| Max people per frame | 4 (minimum 0) |
| Room dimensions | 4.8 m × 7.2 m |

---

## Dataset

The dataset is published on **Hugging Face** under the private `HAEEAI` organisation  
([multi-person-localization](https://huggingface.co/datasets/HAEEAI/multi-person-localization)).

### Sensor Setup

- **6 fixed UWB radar sensors** (TSRR250) placed around the room perimeter at 1.2 m height
- Each radar has **3 antennas** (L-shape array, 7.5 mm spacing), **120 range bins**, sampled at **25 Hz**
- **1 bin = 0.15 m** — total physical range 18 m
- **Field of view ≈ 120°** (±60° around boresight)
- **First 5 bins** contain direct TX→RX coupling and are typically discarded
- **24 acquisition windows**, each 300 s (7 500 frames) - 180 000 frames total

### Data Format (`.npz` per acquisition window)

| Array | Shape | Description |
|---|---|---|
| `radar_cir_iq` | `(T, 6, 3, 120, 2)` | Complex CIR - T frames × 6 radars × 3 antennas × 120 bins × [I, Q] |
| `people_xy` | `(T, 4, 2)` | Ground-truth [x, y] positions in metres (zero-padded) |
| `people_mask` | `(T, 4)` | Boolean mask — True for active persons |
| `timestamps` | `(T,)` | Acquisition timestamps |

The complex CIR is reconstructed as:

```python
cir = radar_cir_iq[..., 0] + 1j * radar_cir_iq[..., 1]   # shape (T, 6, 3, 120)
```

### Coordinate System

- Origin at the **bottom-left corner** of the room
- **x** axis: positive to the right (0 → 4.8 m)
- **y** axis: positive upward (0 → 7.2 m)
- All positions in **metres**

---

## Input / Output Specification

### Input to `code.py`

A single `.npy` file of shape `(T, 6, 3, 120, 2)` — raw CIR measurements without ground truth.

### Output from `code.py`

A `.jsonl` file with **one JSON object per frame**:

```jsonl
{"frame": 36, "localizations": [[1.2, 3.4], [2.5, 5.1]]}
{"frame": 37, "localizations": [[1.3, 3.5]]}
{"frame": 38, "localizations": []}
```

| Field | Type | Description |
|---|---|---|
| `frame` | int | 0-based frame index |
| `localizations` | list of [x, y] | Predicted positions in metres; empty list `[]` if no person detected |

### How `code.py` is called

```bash
python submission/code.py \
    --input-path  <path/to/input.npy> \
    --output-path <path/to/output.jsonl>
```

### Constraints on `code.py`

Only the packages listed in `requirements.txt` may be imported:

| Package | Version | Permitted use |
|---|---|---|
| `numpy` | ≥ 2.0.0 | All array operations and preprocessing |
| `scipy` | ≥ 1.10.0 | `ndimage`, `signal`, `spatial` submodules |
| `tensorflow` | ≥ 2.13.0 | `tf.lite.Interpreter` only, not `tf.keras` |

Python standard library modules (`argparse`, `json`, `sys`, `pathlib`, `math`, `collections`, `itertools`, …) are always permitted.  
Any other third-party import will cause evaluation failure.

- Model must be in **TFLite format** (`model.tflite`)
- File names must be exactly `code.py` and `model.tflite`

---

## ESP32-S3 Deployment Constraints

Your model will **not** be deployed on real hardware, but it **must meet** these constraints:

| Check | Limit | Notes |
|---|---|---|
| Model file size | **< 800 KB** | Flash budget |
| Activation arena (SRAM) | **< 300 KB** | Model arena only - preprocessing in `code.py` consumes additional SRAM |
| Quantization | INT8 recommended | 4× smaller and faster on ESP32-S3 |
| Forbidden ops | **No LSTM / GRU / RNN / CUSTOM / Flex** | Not supported by TFLM on ESP32-S3 |

Standard supported layers: `Conv2D`, `DepthwiseConv2D`, `Dense`, `BatchNormalization`, `MaxPool2D`, `AveragePooling2D`, `Add`, `ReLU`.

---

## Evaluation Metrics

### F1 Score (primary detection metric)

Hungarian matching with a **1.0 m acceptance threshold**:

- **TP**: predicted position matched to a GT person within 1.0 m
- **FP**: predicted position with no GT match
- **FN**: GT person not detected

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

### Additional Reported Metrics (informational)

- **RMSE**, **MAE**, **Median error**, **P90** - on TP matched pairs only
- **Count MAE**, **Count accuracy** - person count per frame

---

## Testing Locally

Before submitting, run the full pipeline on the provided example.

**Step 1 - Run inference**
```bash
python submission/code.py \
    --input-path  evaluation/example/input_test.npy \
    --output-path evaluation/example/my_output.jsonl
```

**Step 2 - Validate output format and check performance**
```bash
python evaluation/evaluate_performance.py \
    --gt-path   evaluation/example/output_test.jsonl \
    --pred-path evaluation/example/my_output.jsonl
```

**Step 3 - Check ESP32-S3 constraints**
```bash
python evaluation/evaluate_constraint.py \
    --model-path submission/model.tflite
```

> The example input/output is derived from training data and is provided for
> **format verification and sanity checking only**.
> The final evaluation uses a separate acquisition session not seen during training.

---

## Repository Structure

```
submission/
    code.py          ← inference script (required, exact name)
    model.tflite     ← trained TFLite model (required, exact name)

report/
    report.pdf       ← project report

src/
    ...              ← training notebooks, scripts, experiments

evaluation/          ← provided - do not modify
    evaluate_performance.py
    evaluate_constraint.py
    example/
        input_test.npy
        output_test.jsonl

requirements.txt     ← allowed libraries - do not modify
```



