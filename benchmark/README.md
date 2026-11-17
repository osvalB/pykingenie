# Parameter-recovery benchmark

These notebooks check that pykingenie recovers the true parameters of each
surface-based kinetics model. They are AI generated.

For each model and each experiment layout, a notebook:

1. writes the model's rate equations from first principles and integrates them
   with SciPy (`solve_ivp`, Radau method). No pykingenie function is used to make
   the data;
2. adds Gaussian noise (standard deviation 0.1 RU, with Rmax = 50 RU);
3. fits the data with pykingenie, with one Smax (Rmax) shared by all curves;
4. compares the fitted parameters with the true ones. A parameter counts as
   recovered when its relative error is at most 5%.

There is one simulated dataset per model and layout (noise seed 0).

## Layouts

- `Multi-cycle-kinetics`: each concentration is injected on a regenerated surface
  for 300 s, followed by 600 s of buffer.
- `Single-cycle-kinetics`: five concentrations are injected one after the other
  without regeneration, 120 s each with 60 s of buffer in between, and 900 s of
  buffer after the last one.

Data are sampled every second. The first dissociation point is the last association
point of the same cycle, where the injection ends.

## Models

| Notebook | Model | pykingenie fit |
|---|---|---|
| `01_one_to_one` | 1:1 binding | `fit_one_site_assoc_and_disso` |
| `02_one_to_one_mass_transport` | 1:1 with mass transport (two-compartment model, simulated with a thin layer of finite volume) | `fit_one_site_assoc_and_disso(fit_ktr=True)` |
| `03_induced_fit` | Induced fit | `fit_one_site_if_assoc_and_disso` |
| `04_induced_fit_signal_ratio` | Induced fit, induced complex giving 1.5× the signal | `fit_one_site_if_assoc_and_disso(fit_signal_ratio=True)` |
| `05_two_site` | Two identical, independent sites on the ligand | `fit_two_site_assoc_and_disso` |
| `06_two_site_cooperative` | Two sites with cooperativity (σ = 4) | `fit_two_site_assoc_and_disso(fit_sigma=True)` |
| `07_heterogeneous_ligand` | Two independent ligand populations | `fit_two_site_heterogeneous_ligand_assoc_and_disso` |

## Results

Largest relative error over the fitted parameters of each model:

| Model | Multi-cycle | Single-cycle |
|---|---|---|
| 1:1 | +0.09% (Kd) | +0.05% (k_off) |
| 1:1 with mass transport | −0.27% (k_tr) | −0.66% (k_tr) |
| Induced fit | +0.48% (k_off) | −0.41% (k_off) |
| Induced fit, signal ratio | +0.48% (k_off) | −0.44% (k_off) |
| Two sites | −0.26% (Rmax_PL) | −0.38% (Rmax_PL) |
| Two sites, cooperative | −0.24% (sigma) | −0.39% (Rmax_PL) |
| Heterogeneous ligand | −0.37% (Kd2) | −0.19% (k_off2) |

All parameters are recovered within 1% in all 14 notebooks.

## Running

Run a notebook from its own folder, so that `benchmark_helpers.py` can be imported
(the notebooks add the parent folder to the path):

```bash
uv run --extra dev jupyter nbconvert --to notebook --execute --inplace benchmark/Multi-cycle-kinetics/01_one_to_one.ipynb
```

`benchmark_helpers.py` holds the injection schedules, the noise, the comparison
table and the plots. It uses NumPy, SciPy, pandas and Matplotlib only.
