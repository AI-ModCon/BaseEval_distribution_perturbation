# Coherent Scattering Example

## Overview

This example is based on the paper **"Data-driven discovery of dynamics from time-resolved coherent scattering"** by Nina Andrejevic et al. ([arXiv:2311.14196](https://arxiv.org/abs/2311.14196)).

The paper introduces `dynamiCXS`, a framework that combines physics simulations of dynamical systems with coherent X-ray scattering (CXS) measurements. The core idea is that time-resolved CXS patterns — speckle images in reciprocal (Fourier) space — encode information about the real-space dynamics of a sample, and that a neural ODE can be trained to recover those dynamics directly from the scattering signal, without access to real-space images.

The `dynamiCXS` library (`dynamiCXS/dynamicxs/`) provides:
- **`ode.py`**: PyTorch-based ODE systems (`Kuramoto`, `Swarm`, `LotkaVolterra`, `GrayScott`, `Turing`, and graph-based variants) with `torchdiffeq` integration.
- **`cxs.py`**: Forward models (`CXSGrid`, `CXSPoint`) that compute the coherent scattering intensity pattern for a given real-space state, either on a pixel grid or as a point cloud.

The experimental dataset for the ptychographic scan example is in `data/` (sourced from [Zenodo:10204976](https://zenodo.org/doi/10.5281/zenodo.10204976)).

---

## Data Files (`data/`)

The three synthetic case studies (Kuramoto, Swarm, Lotka-Volterra) generate all data at runtime. Only the experimental ptychographic scan notebook (`trajectory.ipynb`) uses the files below.

The directory has two subdirectories:

```
data/
├── numeric_data/   # ptychographic reconstruction outputs (object, probe, archive)
└── raw/            # raw experimental measurements (diffraction frames, scan positions, beam stop mask)
```

### `data/numeric_data/`

| File | Shape | dtype | Description |
|---|---|---|---|
| `run0.npz` | — | — | NumPy archive containing `probe` `(2, 512, 512)` complex64, `obj` `(915, 915)` complex64, and `pixelsize` scalar float64 (`6.824e-9 m`). Convenience bundle of the three CSVs below. |
| `run0_object_0.csv` | `(915, 915)` | `complex64` | Complex transmission function of the scanned specimen, reconstructed on a `915×915` pixel grid. Magnitude encodes absorption; phase encodes optical path length. Used by `trajectory.ipynb` via `TorchRegularGridInterpolator` to evaluate the object at arbitrary sub-pixel probe positions. |
| `run0_probe_0.csv` | `(512, 512)` | `complex64` | Single-mode probe wavefront — the reconstructed complex amplitude of the focused X-ray beam at the detector plane. |
| `run0_probes_0.csv` | `(1024, 512)` | `complex64` | Two-mode probe stack stored as `(n_modes × height, width)` — two `512×512` slices concatenated row-wise. Loaded and reshaped to `(-1, 512, 512)` in the notebook. The two-mode sum models partial coherence of the beam. |
| `probes.csv` | `(1024, 512)` | `complex64` (string-encoded) | Same content as `run0_probes_0.csv`. Note: row 0 is treated as a column header by `pd.read_csv` by default, reducing effective shape to `(1023, 512)`. Load with `header=None` to recover all 1024 rows. |
| `pattern.png` | `540×544` | RGBA | Reference figure from the paper showing an example scattering pattern. Not loaded by any notebook. |

### `data/raw/`

| File | Shape | dtype | Description |
|---|---|---|---|
| `image_000000.bin` … `image_000962.bin` | `(512, 512)` each | `float32` | 963 measured diffraction intensity frames, one per scan position. Each binary file is a flat array of 262,144 float32 values read with `np.fromfile(..., dtype=np.float32).reshape(512, 512)`. Intensity values range from 0 to ~61,000 counts. These are the training targets for `trajectory.ipynb`. |
| `positions.csv` | `(963, 2)` | `float64` | Nominal scan positions in meters (x, y) for each of the 963 diffraction frames. Values are on the order of nanometers to micrometers. Multiplied by `1e6` in the notebook to convert to µm. |
| `beamstopMask.h5` | `(512, 512)` | `float32` | HDF5 file (`entry/data/data`) containing a binary mask of the beam stop — detector pixels blocked by the physical beam stop. Used to zero out the center of diffraction patterns during processing. |

### How the data flows through `trajectory.ipynb`

1. **Positions** (`raw/positions.csv` → `(963, 2)` in µm): defines the discrete scan grid. The neural ODE's predicted continuous `(x, y)` trajectory is projected to the nearest scan point at each step to index into the diffraction data.

2. **Object** (`numeric_data/run0_object_0.csv` → `(915, 915)` complex): wrapped in a `TorchRegularGridInterpolator`. At each training step the object is evaluated at the predicted probe position by bilinear interpolation to produce a local `512×512` complex exit wave.

3. **Probe modes** (`numeric_data/run0_probes_0.csv` → `(2, 512, 512)` complex): multiplied against the interpolated object window. Intensities from both modes are summed to form the simulated diffraction pattern.

4. **Simulated CXS** (on the fly): `fftshift(|FFT2(ifftshift(exit_wave))|²)` → `512×512` float intensity per scan point.

5. **Measured frames** (`raw/image_XXXXXX.bin` → `(963, 512, 512)` float32): L1 loss between simulated and measured pattern trains the trajectory neural ODE. Training uses frames 0–79; frames 80–162 are held out for extrapolation evaluation.

---

## The Four Case Studies

Each notebook in `dynamiCXS/dynamicxs/systems/` simulates a dynamical system, computes time-resolved CXS patterns from it, trains a neural ODE on those patterns, and evaluates how well the learned dynamics match ground truth.

### 1. Kuramoto — Locally-coupled moments (`systems/kuramoto/kuramoto.ipynb`)

A 2D lattice of `N×N = 80×80` phase oscillators evolving under the Kuramoto model with a Laplacian-of-Gaussian coupling kernel. The CXS signal is computed via `CXSGrid` using a phase-dependent form factor (orientation-sensitive magnetic scattering). The neural ODE learns the convolutional coupling kernel from CXS patterns alone.

**Data:** Fully synthetic — no files loaded. `Kuramoto.init_state(M=100)` generates 100 random initial phase fields; `Kuramoto.solve(t)` integrates them to produce a `(T=101, M=100, 1, 6400)` state tensor. `CXSGrid` converts each frame to a `42×42` scattering intensity pattern.

**Key parameters:** `N=80`, `L=2.0`, `v=0` (frequency), `K=20.0` (coupling strength), `s=1.5` (kernel length scale)

### 2. Swarm — Self-organizing particles (`systems/swarm/swarm.ipynb`)

A collection of `N=500` point particles interacting via a short-range pairwise potential with spatial and phase coupling. CXS is computed via `CXSPoint` (spherical form factor). The neural ODE is a graph neural network that learns the interaction potential from scattering patterns.

**Data:** Fully synthetic — no files loaded. `Swarm.init_state(M=50)` generates 50 initial configurations of 500 particles each; `Swarm.solve(t)` integrates to `(T=101, M=50, 500, 3)` (x, y, angle per particle). `CXSPoint` converts each frame to a `64×64` intensity pattern.

**Key parameters:** `N=500`, `L=5.0`, `R=0.15` (particle radius), `rc=0.75` (interaction cutoff), `D=2` (spatial dimensions)

### 3. Lotka-Volterra — Fluctuating source (`systems/fluctuation/lotka.ipynb`)

A point cloud of `N=400` particles whose center-of-mass evolves according to Lotka-Volterra predator-prey dynamics. CXS is computed via `CXSPoint` against a random speckle background. The neural ODE recovers the mean-field trajectory.

**Data:** Fully synthetic — no files loaded. `LotkaVolterra.init_state(M=100)` places 400 particles uniformly at random; `LotkaVolterra.solve(t)` integrates to `(T=9, M=100, 400, 2)` over 16 time units. `CXSPoint` produces `56×56` intensity patterns. The predator-prey oscillation is visible as periodic motion of the particle centroid in the scattering signal.

**Key parameters:** `N=400`, `L=2.0`, `R=0.07`, `alpha=1/3`, `beta=2/3`, `gamma=0.5`, `delta=0.5`

### 4. Ptychographic scan — Probe trajectory (`systems/experiment/trajectory.ipynb`)

Uses real experimental CXS data from an X-ray ptychographic scan. The neural ODE learns the 3D probe trajectory (x, y, focal depth z) from the sequence of diffraction patterns, then extrapolates beyond the training window.

**Data:** Loads from `data/numeric_data/` and `data/raw/`:
- `numeric_data/run0_object_0.csv` → `(915, 915)` complex object transmission function, wrapped in a `TorchRegularGridInterpolator` for sub-pixel probe positioning.
- `numeric_data/run0_probe_0.csv` → `(512, 512)` complex single-mode probe wavefront, used as the illumination function.
- `numeric_data/run0_probes_0.csv` → `(2, 512, 512)` two-mode probe stack (concatenated row-wise as `(1024, 512)`), used for multi-mode CXS simulation.
- `raw/image_000000.bin` … `raw/image_000962.bin` → 963 measured `512×512` float32 diffraction frames used as training targets.
- `raw/positions.csv` → nominal scan positions in meters, converted to µm during loading.

At each training step, the model predicts `(x, y, z)` → the object is interpolated at `(x, y)` → multiplied by each probe mode → FFT2 → intensity summed across modes → L1 loss against the measured frame.

**Key parameters:** `D=3` (trajectory dimensions), `max_time=80` (training frames out of 163 total), pixel size `p=6.82×10⁻³ µm`

---

## Robustness Evaluation in the Paper

The paper's robustness evaluation is limited to **generalization in time**: all four case studies train on a subset of time steps and then assess whether the learned dynamics extrapolate correctly beyond the training window. There is no systematic noise sweep or held-out condition variation — the notebooks contain dormant Poisson noise infrastructure (`lmax` parameter, `PoissonNLLLoss` import) but every notebook runs with `lmax = 0` (ideal, noiseless scattering) throughout training and evaluation.

### What was evaluated and how

| Case study | Training data | Evaluation | Reported result |
|---|---|---|---|
| Kuramoto | 100 trajectories × 101 time steps, `lmax=0` (noiseless CXS) | Predicted vs. ground-truth CXS patterns and learned coupling kernel | Kernel shape and spatial scale recovered from scattering alone |
| Swarm | 50 trajectories × 101 time steps, `lmax=0` | Predicted clustering dynamics and learned interaction potential | Interaction potential shape reproduced; cluster connectivity tracked |
| Lotka-Volterra | 100 trajectories × 9 time steps, `lmax=0` | Predicted vs. ground-truth vector field; trajectory extended to T=500 | Predator-prey limit cycle recovered from CXS centroid motion |
| Ptychographic scan | Frames 0–79 (of 163), real measured diffraction | Trajectory extrapolated to frames 80–162; angular error computed | Angular error 0.348 rad (5.5% of full rotation); position error ≲ ±0.75 µm |

For the three synthetic cases the loss is L1 between predicted and ground-truth CXS intensities. For the experimental case the loss is L1 between the simulated CXS (computed from the predicted probe position via the forward model) and the measured diffraction frame at the nearest scan point.

### What robustness testing is absent

The notebooks do not test:
- Models trained on noiseless data evaluated on Poisson-noisy patterns (the `lmax` infrastructure exists but is never activated)
- Any variation of ODE parameters between training and test
- Different initial conditions at test time
- Degraded or corrupted detector data

This is the gap that `dist_pert` perturbations can address.

---

## First-Order Perturbations

First-order perturbations act directly on the CXS intensity patterns — the model's input. These are the most straightforward to apply and test whether a trained neural ODE degrades gracefully under realistic measurement imperfections.

### Poisson photon-counting noise (built-in)

The existing `lmax` mechanism in the NODE class scales patterns to a target max photon count and applies `torch.poisson`. It is already wired into training; activating it is a matter of setting `lmax > 0`. This is the most physically motivated perturbation for X-ray diffraction.

```python
# In any synthetic notebook: set lmax before instantiating the NODE
lmax = 1000   # low flux → noisy; 1e5 → near-ideal
# lmax > 15 → L1Loss; lmax ≤ 15 → PoissonNLLLoss (selected automatically)
```

### Gaussian detector readout noise

Additive Gaussian noise models electronic readout noise independent of photon count. Apply with `dist_pert` directly to the `(n, n)` intensity array at evaluation time:

```python
import numpy as np
from dist_pert.image import GaussianNoisePerturber

perturber = GaussianNoisePerturber(sigma=15)

# cxs(y[t]) returns a flat tensor; reshape to (n, n) uint8 for the perturber
pattern = cxs(y[t]).reshape(n, n).cpu().numpy()
pattern_uint8 = (255 * pattern / pattern.max()).astype(np.uint8)
noisy = perturber([pattern_uint8])[0]
```

### Beam stop masking / center occlusion

The `beamstopMask.h5` file in `data/raw/` contains the physical beam stop mask for the experimental data. For synthetic cases, `CXS.__init__` accepts an `f_mask` argument that zeros a circular central region. Varying `f_mask` (fraction of `n` masked) tests sensitivity to low-q information loss:

```python
# Wider beam stop → more low-q information removed
cxs_masked = CXSGrid(N, n, L=L, dq=dq, f_mask=0.1)   # nominal
cxs_masked = CXSGrid(N, n, L=L, dq=dq, f_mask=0.2)   # perturbed
```

### ImageNet-C style corruptions

For testing sensitivity to detector artifacts or optical degradation, `ImageCorruptionPerturber` covers compression artifacts, blur, and contrast shifts:

```python
from dist_pert.image import ImageCorruptionPerturber

# jpeg_compression simulates lossy detector encoding artifacts
perturber = ImageCorruptionPerturber(corruption="jpeg_compression", severity=3)
corrupted = perturber([pattern_uint8])[0]
```

---

## Second-Order Perturbations

Second-order perturbations change the *generation process* that produces CXS patterns rather than the patterns themselves. For the synthetic cases that means perturbing ODE parameters or initial conditions; for the experimental case it means perturbing the forward model inputs (the reconstructed object and probe).

### Synthetic cases: ODE parameter perturbation

Perturbing a physical parameter (e.g. coupling strength `K`, interaction cutoff `rc`, or predator-prey rate `alpha`) shifts the entire distribution of simulated CXS patterns. The same trained model can then be evaluated on patterns from a physically different system, testing whether it recovers the right dynamics under distribution shift.

```python
from dist_pert.numeric import MultiplicativeNoisePerturber

perturber = MultiplicativeNoisePerturber(sigma=0.05)

# Kuramoto: perturb coupling strength
K_nominal = np.array([20.0])
K_perturbed = perturber([K_nominal])[0]
kuramoto_perturbed = Kuramoto({'N': 80, 'L': 2., 'v': 0., 'K': float(K_perturbed), 's': 1.5})

# Swarm: perturb interaction cutoff radius
rc_nominal = np.array([0.75])
rc_perturbed = perturber([rc_nominal])[0]
```

| System | Parameter | Physical meaning | Expected CXS effect |
|---|---|---|---|
| Kuramoto | `K` | Oscillator coupling strength | Shifts synchronization timescale; changes correlation ring width |
| Kuramoto | `s` | Coupling kernel length scale | Shifts dominant spatial frequency in scattering |
| Swarm | `rc` | Interaction cutoff radius | Changes typical cluster size and inter-particle spacing |
| Swarm | `N` | Particle count | Changes overall scattering intensity and speckle statistics |
| LotkaVolterra | `alpha`/`gamma` | Growth/decay rates | Shifts oscillation period of centroid motion |

### Synthetic cases: initial condition perturbation

`init_state` is seeded, so using a different seed or adding noise to `y0` produces a different draw from the same distribution. This is a minimal perturbation that tests sensitivity to initial state without changing the underlying dynamics:

```python
from dist_pert.numeric import AdditiveGaussianPerturber

perturber = AdditiveGaussianPerturber(sigma=0.1)
y0_nominal = ode.y0.detach().numpy()        # shape (M, 1, N*N)
y0_perturbed = perturber([y0_nominal[0]])[0]
```

### Experimental case: probe wavefront perturbation

The reconstructed probe in `numeric_data/run0_probe_0.csv` is itself an estimated quantity with reconstruction uncertainty. Perturbing it shifts the forward model — every simulated diffraction pattern changes — without touching the raw measured frames. This tests whether the trajectory model is sensitive to probe reconstruction error.

The probe is a `(512, 512)` complex array. The real and imaginary parts can be perturbed independently:

```python
from dist_pert.numeric import AdditiveGaussianPerturber
import numpy as np

perturber = AdditiveGaussianPerturber(sigma=0.01)  # small relative to probe amplitude range ~[-1.8, 2.1]

probe = np.genfromtxt('data/numeric_data/run0_probe_0.csv', delimiter=',', dtype=np.complex64)
probe_real_perturbed = perturber([probe.real])[0]
probe_imag_perturbed = perturber([probe.imag])[0]
probe_perturbed = probe_real_perturbed + 1j * probe_imag_perturbed
```

**What additional information is needed:** a reasonable estimate of the probe reconstruction uncertainty (e.g. from running ptychographic reconstruction with multiple random initializations or different algorithm hyperparameters). Without this, the perturbation magnitude is arbitrary. The probe amplitude ranges from roughly −1.8 to +2.1, so `sigma=0.01–0.05` is a plausible starting range.

### Experimental case: object transmission function perturbation

Similarly, `run0_object_0.csv` is a reconstructed quantity. Perturbing the `(915, 915)` complex object changes which exit wave the model sees at each predicted position, effectively simulating uncertainty in the specimen reconstruction.

**What additional information is needed:** the same as for the probe — reconstruction uncertainty bounds. Additionally, because the object is spatially structured (phase varies significantly across the field of view), spatially correlated noise (e.g. smoothed Gaussian noise) would be more physically realistic than i.i.d. pixel noise. `dist_pert` does not currently have a spatially-correlated numeric perturber; this would need to be implemented or the image blur perturbers applied to the real/imaginary channels separately.

### Experimental case: scan position jitter

`raw/positions.csv` gives the nominal probe positions. In practice, piezo stage drift and vibration introduce position error. Perturbing the positions shifts which object region is sampled at each scan point:

```python
from dist_pert.numeric import AdditiveGaussianPerturber

perturber = AdditiveGaussianPerturber(sigma=0.005)  # in µm; stage repeatability is typically ~1–10 nm

positions = pd.read_csv('data/raw/positions.csv', header=None).values * 1e6  # convert to µm
positions_perturbed = np.stack([perturber([positions[:, i]])[0] for i in range(2)], axis=1)
```

**What additional information is needed:** the stage positioning repeatability specification for the instrument used in experiment 811. Without this, a reasonable default is 10–50 nm (0.01–0.05 µm), which is typical for synchrotron nanoprobe stages.

---

## Installation

Create a dedicated virtual environment for this example — the dynamiCXS dependency
pins (`torch==1.13.1`, `numpy==1.22.4`) conflict with more recent versions.

```bash
# From examples/coherent_scattering_example/
python -m venv env
source env/bin/activate                         # Windows: env\Scripts\activate

# Install all pinned deps (dynamiCXS + torch-geometric + ipykernel)
pip install -r requirements.txt

# Install dist_pert with image support from the repo root
pip install -e "../../.[image]"

# torch-scatter and torch-sparse must be installed separately;
# the wheel URL depends on your CUDA version (use +cpu for CPU-only):
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

# Register as a Jupyter kernel
python -m ipykernel install --user \
    --name coherent_scattering_env \
    --display-name "coherent_scattering_env"
```

After registering the kernel, open any notebook and select
**coherent_scattering_env** from the kernel picker.

> **Note:** `ode.py` imports `torch_geometric` at the module level, so
> `torch-geometric` is required even if you only run the Kuramoto notebook.
> `torch-scatter` and `torch-sparse` are build-time dependencies of
> `torch-geometric` that cannot be resolved via PyPI alone — the `-f` flag
> above points `pip` to the prebuilt wheels for torch 1.13.1.
