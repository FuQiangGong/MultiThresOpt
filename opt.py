"""
Geometry optimisation with ASE + MLIP + multi-criteria convergence.

Adds stress, displacement, and energy convergence thresholds on top of
the standard force-based fmax check.
"""
import os
import json
import math
from pathlib import Path
import numpy as np
from datetime import datetime
from glob import glob
from ase.io import read, write
from deepmd.calculator import DP as DPCalculator
from ase.filters import FrechetCellFilter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multithresopt import LBFGS_Multi

# MLIP model path 
MODEL_PATH = "frozen_model.pth"

# Convergence thresholds
FORCE_MAX = 0.05    # eV/A
STRESS_MAX = 0.01   # eV/A^3
DISP_MAX = 0.001    # A 
ENERG_MAX = 1e-6    # eV

RANDOM_GENERATED = "./csp-stcs-0"       # path to an ASE-readable file (single structure) or a directory (batch)
MAX_STEPS = 200              # maximum opt steps
N_WORKERS = 10               # parallel number

# Log files
LOG_FILE = "opt.txt"             # recording the final optimisation status for each structure
REMAINING_FILE = "remaining-0.txt" # recording the number of unconverged structures which can be further used to restart batch optimisation

# Running
dp = DPCalculator(MODEL_PATH)
if os.path.isfile(RANDOM_GENERATED):
    # Single structure mode
    fs = [RANDOM_GENERATED]
    print(f"Single structure mode: {RANDOM_GENERATED}")
elif os.path.isdir(RANDOM_GENERATED):
    # Batch mode
    fs = sorted([
        os.path.join(RANDOM_GENERATED, f)
        for f in os.listdir(RANDOM_GENERATED)
        if os.path.isfile(os.path.join(RANDOM_GENERATED, f))
        and not f.startswith('.')
    ])
    print(f"Found {len(fs)} candidate files in {RANDOM_GENERATED}")
else:
    print(f"Warning: '{RANDOM_GENERATED}' not found. No structures to process.")
    fs = []

def process_file(f):
    start_time = datetime.now()
    out_root = Path(f).stem
    status_file = f"{out_root}/opt_status.json"
    if os.path.exists(status_file):
        with open(status_file) as fc:
            prev = json.load(fc)
        t = prev.get("thresholds", {})
        thresholds_match = (
            t.get("fmax") == FORCE_MAX
            and t.get("stress_fmax") == STRESS_MAX
            and t.get("disp_fmax") == DISP_MAX
            and t.get("energy_fmax") == ENERG_MAX
        )
        if thresholds_match and prev.get("converged"):
            return "done"
    if os.path.exists(f"{out_root}/opt.cif"):
        atoms = read(f"{out_root}/opt.cif")
    else:
        try:
            atoms = read(f)
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)}: cannot read this file format ({e})")
            return "skip"
        os.makedirs(out_root, exist_ok=True)
    atoms.calc = dp
    ecf = FrechetCellFilter(atoms)
    dyn = LBFGS_Multi(
        ecf,
        trajectory=f"{out_root}/opt.traj",
        append_trajectory=True,
        logfile=f"{out_root}/opt.log",
    )
    dyn.set_additional_criteria(
        stress_fmax=STRESS_MAX,   # eV/Å³
        disp_fmax=DISP_MAX,       # Å
        energy_fmax=ENERG_MAX,   # eV
    )
    dyn.run(fmax=FORCE_MAX, steps=MAX_STEPS)
    summary = dyn.convergence_summary()
    with open(f"{out_root}/opt_conv.txt", "w") as fc:
        fc.write(summary)
    print(summary)
    write(f"{out_root}/opt.cif", atoms)
    end_time = datetime.now()
    dur_sec = (end_time - start_time).total_seconds()
    forces = atoms.get_forces()
    maxif = np.max(np.abs(forces))
    try:
        stress = atoms.get_stress()
        stress_max = np.max(np.abs(stress))
    except Exception:
        stress_max = float("nan")
    disp_max = dyn._log_disp_max if dyn._log_disp_max is not None else float("nan")
    energy_diff = dyn._log_energy_diff if dyn._log_energy_diff is not None else float("nan")
    try:
        final_energy = atoms.get_potential_energy()
    except Exception:
        final_energy = float("nan")
    _converged = (
        maxif <= FORCE_MAX
        and (math.isnan(stress_max) or stress_max <= STRESS_MAX)
        and (math.isnan(disp_max) or disp_max <= DISP_MAX)
        and (math.isnan(energy_diff) or energy_diff <= ENERG_MAX)
    )
    _clean = lambda v: None if (isinstance(v, (float, np.floating)) and math.isnan(float(v))) else float(v) if isinstance(v, np.floating) else v
    with open(f"{out_root}/opt_status.json", "w") as fjs:
        json.dump({
            "converged": bool(_converged),
            "thresholds": {
                "fmax": FORCE_MAX,
                "stress_fmax": STRESS_MAX,
                "disp_fmax": DISP_MAX,
                "energy_fmax": ENERG_MAX,
            },
            "final_values": {
                "fmax": _clean(maxif),
                "stress_max": _clean(stress_max),
                "disp_max": _clean(disp_max),
                "energy_diff": _clean(energy_diff),
                "final_energy": _clean(final_energy),
            },
        }, fjs, indent=2)
    return (out_root, dur_sec, maxif,
            stress_max, disp_max, energy_diff, final_energy)


def _handle_result(result):
    """Write result to LOG_FILE and return whether structure is still unconverged."""
    if result in ("done", "skip"):
        return False
    (fname, dur_sec, maxif,
     stress_max, disp_max, energy_diff, final_energy) = result

    unconverged = not (
        maxif <= FORCE_MAX
        and (math.isnan(stress_max) or stress_max <= STRESS_MAX)
        and (math.isnan(disp_max) or disp_max <= DISP_MAX)
        and (math.isnan(energy_diff) or energy_diff <= ENERG_MAX)
    )

    with open(LOG_FILE, "a") as ft:
        ft.write(
            f"{fname}\t{dur_sec}\t{maxif:.4f}\t"
            f"{stress_max:.6f}\t{disp_max:.6f}\t"
            f"{energy_diff:.6e}\t{final_energy:.6f}\n"
        )
    return unconverged


if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as ft:
            ft.write("file_name\tduration\tfmax\tstress_max\tdisp_max\t|dE|\tfinal_energy\n")
    remaining = 0
    if len(fs) == 1:
        # Single structure — process directly
        result = process_file(fs[0])
        remaining += _handle_result(result)
    elif fs:
        # Multiple structures — parallel
        n_workers = N_WORKERS
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in fs}
            for future in as_completed(futures):
                try:
                    remaining += _handle_result(future.result())
                except Exception as e:
                    print(f"Error: {futures[future]}: {e}")
    with open(REMAINING_FILE, "a") as f:
        f.write(f"\n{remaining}")
