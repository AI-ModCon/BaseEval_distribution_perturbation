# distribution_perturbation

A library for **first-order distribution perturbation** of text, image, and numeric data, with examples of **second-order perturbation** (perturbing a precursor to produce perturbed input data).

- **First-order**: perturbations applied directly to input data (e.g. word replacement, pixel noise, additive Gaussian)
- **Second-order**: perturbations applied to a data source or generation process, producing perturbed datasets (e.g. augmented benchmark variants for robustness evaluation)

> Note: the first/second-order terminology is still being refined.

## Installation

```bash
git clone <repo-url>
cd distribution_perturbation
git submodule update --init --recursive

pip install -e .            # core library (text + numeric)
pip install -e ".[image]"   # adds image perturbers (Pillow, scipy, imagecorruptions)
```

For the NeMo Skills example only:
```bash
pip install -e /path/to/perlmutter_nemo-skills-main
pip install -e examples/perlmutter_nemo_skills/nemo-custom-benchmark
```

A pinned `requirements.txt` is also provided:
```bash
pip install -r requirements.txt
```

## Notebook

`notebooks/dist_pert_usage.ipynb` contains runnable examples for all three data types. To use it, register the environment as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name dist_pert_env --display-name "dist_pert_env"
jupyter lab
```

## Python API

All perturbers share the same interface: instantiate with config, then call on a `list`.

```python
from dist_pert.text import ContextualWordPerturber
from dist_pert.image import GaussianNoisePerturber, SaltPepperPerturber, ImageCorruptionPerturber
from dist_pert.numeric import AdditiveGaussianPerturber, MultiplicativeNoisePerturber
import numpy as np

# Text — replaces tokens with contextually similar alternatives (BERT)
t = ContextualWordPerturber(aug_p=0.2)
result = t(["The storm produced rotating columns of air."])

# Image — pixel noise
img = GaussianNoisePerturber(sigma=25)
noisy = img([np.zeros((224, 224, 3), dtype=np.uint8)])

# Image — ImageNet-C style corruptions (gaussian_blur, jpeg_compression, shot_noise, brightness, contrast)
corr = ImageCorruptionPerturber(corruption="gaussian_blur", severity=3)
corrupted = corr([np.zeros((224, 224, 3), dtype=np.uint8)])

# Numeric
num = AdditiveGaussianPerturber(sigma=0.5)
perturbed = num([np.array([1.0, 2.0, 3.0])])
```

## CLI

```bash
dist-perturb \
  --input data/tasks.jsonl \
  --config perturbation.yaml \
  --field question \
  --output output/aug.jsonl
```

Config YAML format:
```yaml
type: text.contextual_word.ContextualWordPerturber
aug_p: 0.1
model_path: google-bert/bert-base-cased
model_type: bert
```

## Second-Order Example: TBD
