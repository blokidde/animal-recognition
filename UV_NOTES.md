# uv setup notes

`uv` is installed globally, not inside this folder. This project now has:

- `pyproject.toml`: dependencies for the project
- `.python-version`: asks uv for Python 3.12
- `uv.lock`: generated after `uv lock` / `uv sync`
- `.venv/`: generated after `uv sync`

## YOLO first

For YOLO training/inference, the useful packages from the old `D:\venvs\tfds_env` are:

- `ultralytics`
- `torch`
- `torchvision`
- `opencv-python`
- `pillow`
- `numpy`
- `pandas`
- `pyyaml`
- `tqdm`
- `matplotlib`
- `scikit-learn`

Also kept because existing helper scripts use them:

- `imagehash`
- `python-dotenv`
- `google-images-search`

The old venv also had a lot of TensorFlow, TFDS, Apache Beam, SpeciesNet, Roboflow, SAHI, JAX, and cloud packages. Those are not needed for a first YOLO training run, so they are not in the base environment.

## Commands

Create/sync the environment:

```powershell
cd D:\MachineLearning\new_animal_model
uv sync
```

Check YOLO and CUDA:

```powershell
uv run python -c "import torch; from ultralytics import YOLO; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
uv run yolo version
```

Install the old TensorFlow stack only when you need the older Keras/TFLite scripts:

```powershell
uv sync --extra tensorflow
```
