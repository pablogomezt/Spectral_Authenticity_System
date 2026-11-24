# Environment setup

## Requirements

Linux, Conda, Git LFS, and an NVIDIA GPU with a compatible driver. Training explicitly uses CUDA.

```bash
conda env create -f environment.yml
conda activate ramon_env
python -c "import torch, scipy, numpy, torchmetrics; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python TGRS.py --help
```

The original export pins Python 3.10.14, PyTorch 2.7.1, SciPy 1.14.1, torchmetrics 1.4.1 and pip NumPy 2.2.6. It includes exact Linux package builds and both Conda and pip dependencies, so other platforms may require adaptation. The original machine-specific prefix was removed.

Follow the [dataset guide](datasets/README.md), then run from the repository root:

```bash
python TGRS.py --dataset salinas --epochs 2 --max_signatures 500
```

This is a short training check, not the full paper evaluation. CPU-only execution requires code changes. If GPU memory is exhausted, reduce signature selection; full-cube restoration also uses GPU memory. Invalid MAT files may be unresolved LFS pointers.

```bash
conda deactivate
```
