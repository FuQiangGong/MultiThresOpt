# MultiThresOpt-ASE

MultiThresOpt-ASE performs atomic and cell geometry optimisation with ASE,
a DeepMD machine-learned interatomic potential, and multi-criteria
convergence checks. In addition to the usual maximum atomic force criterion,
an optimisation is considered converged only when the stress, per-step
displacement, and per-step energy change also satisfy their thresholds.

The optimiser supports one structure or a directory of structures and can run
batch jobs in parallel. It also records machine-readable status files and can
resume an unfinished optimisation from its previous geometry.

## Requirements

- Python 3.9 or newer
- [ASE](https://ase-lib.org/)
- [DeepMD-kit](https://docs.deepmodeling.com/projects/deepmd/) with the
  `deepmd.calculator.DP` calculator
- NumPy
- A compatible DeepMD model, named `frozen_model.pth` by default

DeepMD-kit installation depends on the target CPU/GPU environment. Follow the
official installation instructions for a wheel that matches your hardware,
then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ase numpy deepmd-kit
```

## Installation

```bash
git clone https://github.com/FuQiangGong/MultiThresOpt.git
cd MultiThresOpt
```

Place the DeepMD model at the root of the project:

```text
MultiThresOpt/
|-- frozen_model.pth
`-- opt.py
```

Alternatively, set `MODEL_PATH` near the top of `opt.py` to another model
location.

## Configuration

The script is configured through variables at the top of `opt.py`:

```python
MODEL_PATH = "frozen_model.pth"

FORCE_MAX = 0.05    # eV/A
STRESS_MAX = 0.01   # eV/A^3
DISP_MAX = 0.001    # A
ENERG_MAX = 1e-6    # eV

RANDOM_GENERATED = "./structures"
MAX_STEPS = 200
N_WORKERS = 10

LOG_FILE = "opt.txt"
REMAINING_FILE = "remaining-0.txt"
```

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | Path to the DeepMD model. |
| `FORCE_MAX` | Maximum permitted atomic force, in eV/A. |
| `STRESS_MAX` | Maximum permitted absolute stress component, in eV/A^3. |
| `DISP_MAX` | Maximum per-step atomic displacement, in A. |
| `ENERG_MAX` | Maximum per-step change in potential energy, in eV. |
| `RANDOM_GENERATED` | One ASE-readable structure file, or a directory containing structures. |
| `MAX_STEPS` | Maximum number of LBFGS optimisation steps. |
| `N_WORKERS` | Number of parallel workers used in batch mode. |
| `LOG_FILE` | Summary table for completed optimisation runs. |
| `REMAINING_FILE` | File to which the number of unconverged structures is appended. |

Set `RANDOM_GENERATED` to a file to run one structure:

```python
RANDOM_GENERATED = "./example_single/cof_10_P2_m_0KNSbmXE.res"
```

Set it to a directory to process every non-hidden file in that directory:

```python
RANDOM_GENERATED = "./example_multi"
```

ASE reads the input structure files, and formats that the installed ASE build
cannot read are reported and skipped. For batch runs, `N_WORKERS` is capped by
the available computing resources and should be chosen to fit the model's CPU
or GPU memory requirements.

## Running

```bash
python opt.py
```

Each structure is optimised using ASE's LBFGS algorithm with a
`FrechetCellFilter`, allowing both atomic positions and cell parameters to
relax. A structure is written as converged only if all enabled criteria pass.
The displacement and energy checks compare the current step with the previous
step.

## Outputs

For an input named `structure.res`, the optimiser creates a directory named
`structure/` in the current working directory:

| File | Description |
| --- | --- |
| `structure/opt.cif` | Optimised geometry and cell. |
| `structure/opt.traj` | ASE optimisation trajectory. |
| `structure/opt.log` | Per-step energy, force, stress, displacement, and energy-change values. |
| `structure/opt_conv.txt` | Human-readable final convergence table. |
| `structure/opt_status.json` | Final metrics, thresholds, and convergence status in JSON format. |

At the run level, `opt.txt` contains one summary row per completed optimisation
and `REMAINING_FILE` records the number of structures that still require
optimisation. The final metrics are:

- `fmax`: maximum atomic force vector magnitude
- `stress_max`: maximum absolute component of the stress tensor
- `disp_max`: maximum atomic displacement since the previous step
- `energy_diff`: absolute potential-energy change since the previous step
- `final_energy`: final potential energy

An `opt_status.json` file has this structure:

```json
{
  "converged": true,
  "thresholds": {
    "fmax": 0.05,
    "stress_fmax": 0.01,
    "disp_fmax": 0.001,
    "energy_fmax": 1e-06
  },
  "final_values": {
    "fmax": 0.0028,
    "stress_max": 0.000081,
    "disp_max": 0.000363,
    "energy_diff": 9.536743e-07,
    "final_energy": -7703.791326
  }
}
```

## Restarting

When `structure/opt.cif` exists, it is used as the starting geometry instead of
the original input file. If `structure/opt_status.json` exists, has matching
thresholds, and records `"converged": true`, that structure is skipped.

Because the status and restart geometry are stored in the output directory,
keep that directory intact between runs. Change `LOG_FILE` and
`REMAINING_FILE` for separate production campaigns if you need to preserve
their histories.

## Repository layout

```text
.
|-- opt.py
|-- example_single/    # Single-structure example run and results
`-- example_multi/     # Parallel batch example run and results
```

The example outputs illustrate successful convergence and the available log
formats. The DeepMD model is not included in the repository.
