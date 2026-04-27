# Neuron Analysis Tool

Automated pipeline for processing neuron images — skeletonization, branch analysis, and Sholl analysis — with structured CSV output.

---

## Installation

Install all required libraries using:

```bash
pip install -r libraries.txt
```

---

## Repository Structure

```
├── data/           # Sample neuron images (.tif)
├── results/        # Generated output (auto-created)
├── neuron.py       # Main analysis script
└── libraries.txt   # Required dependencies
```

---

## Configuration

Input and output directories are set at the top of `neuron.py` and can be changed to point to your own images:

```python
INPUT_DIR  = Path("./data")
OUTPUT_DIR = Path("./results")
```

> Output folders are created automatically if they don't exist.

---

## Usage

1. Place your `.tif` neuron images in the `data/` folder (or update `INPUT_DIR`).
2. Run the script:

```bash
python neuron.py
```

3. Results will appear in the `results/` folder, organized by image.

---

## Output

For each input image, the tool generates a dedicated folder containing:

| File | Description |
|------|-------------|
| `*_grayscale.tif` | Grayscale version of the input image |
| `*_masked.tif` | Masked image of the input |
| `*_overlay.tif` | Overlay image — **green** = detected neuron sections, **red line** = preliminary skeleton |
| `*_preprocessed.tif` | Preprocessed version of the input |
| `*_skeleton.tif` | Final skeleton map |
| `branch_data.csv` | Per-branch measurements |
| `neuron_data.csv` | Overall neuron measurements |
| `sholl_analysis.csv` | Sholl analysis results |

A **summary CSV** consolidating all results is also written to the root of the output folder.

---

## Notes

- Input images must be `.tif` format.
- This README is a work in progress and will be finalized before the stable release.
