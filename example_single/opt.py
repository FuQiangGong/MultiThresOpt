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
import time
from datetime import datetime
from glob import glob
from ase.io import read, write
from ase.optimize import LBFGS
from deepmd.calculator import DP as DPCalculator
from ase.filters import FrechetCellFilter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# MLIP model path 
MODEL_PATH = "/Users/f.gong@bham.ac.uk/Desktop/frozen_model.pth"

# Convergence thresholds
FORCE_MAX = 0.05    # eV/A
STRESS_MAX = 0.01   # eV/A^3
DISP_MAX = 0.001    # A 
ENERG_MAX = 1e-6    # eV

RANDOM_GENERATED = "./cof_10_P2_m_0KNSbmXE.res"       # path to an ASE-readable file (single structure) or a directory (batch)
MAX_STEPS = 200              # maximum opt steps
N_WORKERS = 1               # parallel number

# Log files
LOG_FILE = "single_opt.txt"             # recording the final optimisation status for each structure
REMAINING_FILE = "single_remaining.txt" # recording the number of unconverged structures which can be further used to restart batch optimisation

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

class LBFGS_Multi(LBFGS):
    """
    LBFGS with force, stress, displacement & energy convergence checks.
    """

    def __init__(self, atoms, **kwargs):
        super().__init__(atoms, **kwargs)
        self.t_start = time.time()       # ensure compat across ASE versions
        self.stress_fmax = None
        self.disp_fmax = None
        self.energy_fmax = None
        self._bare_atoms = atoms.atoms if hasattr(atoms, 'atoms') else atoms
        self._log_stress_max = None
        self._log_disp_max = None
        self._log_energy_diff = None
        self._log_prev_pos = None
        self._log_prev_energy = None
        self._log_written_this_iter = False

    def set_additional_criteria(self, stress_fmax=None, disp_fmax=None, energy_fmax=None):
        self.stress_fmax = stress_fmax
        self.disp_fmax = disp_fmax
        self.energy_fmax = energy_fmax

    def step(self, f=None):
        self._log_prev_pos = self.atoms.get_positions().copy()
        try:
            self._log_prev_energy = self.atoms.get_potential_energy()
        except Exception:
            self._log_prev_energy = None
        self._log_written_this_iter = False
        super().step(f)

    def converged(self, forces=None):
        """Check ALL convergence criteria."""
        if self.nsteps == 0 and (self.disp_fmax is not None or self.energy_fmax is not None):
            return False
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1)).max() if forces.ndim > 1 else np.abs(forces).max()
        return self._check_all_criteria(fmax)

    def _update_log_values(self):
        forces = self.atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1)).max() if forces.ndim > 1 else np.abs(forces).max()
        if self.stress_fmax is not None:
            stress = self._bare_atoms.get_stress()
            self._log_stress_max = np.max(np.abs(stress)) if stress is not None else None
        curr_pos = self.atoms.get_positions()
        if self._log_prev_pos is not None:
            self._log_disp_max = np.sqrt(((curr_pos - self._log_prev_pos) ** 2).sum(axis=1)).max()
        else:
            self._log_disp_max = 0.0
        try:
            curr_e = self.atoms.get_potential_energy()
            if self._log_prev_energy is not None:
                self._log_energy_diff = abs(curr_e - self._log_prev_energy)
            else:
                self._log_energy_diff = None
        except Exception:
            self._log_energy_diff = None
        return fmax

    def _check_all_criteria(self, fmax_from_forces):
        if fmax_from_forces > self.fmax:
            return False
        if self.stress_fmax is not None:
            stress = self._bare_atoms.get_stress()
            if stress is not None and np.max(np.abs(stress)) > self.stress_fmax:
                return False
        if self.disp_fmax is not None and self._log_prev_pos is not None:
            curr = self.atoms.get_positions()
            dmax = np.sqrt(((curr - self._log_prev_pos) ** 2).sum(axis=1)).max()
            if dmax > self.disp_fmax:
                return False
        if self.energy_fmax is not None and self._log_prev_energy is not None:
            try:
                curr_e = self.atoms.get_potential_energy()
                if abs(curr_e - self._log_prev_energy) > self.energy_fmax:
                    return False
            except Exception:
                pass
        return True

    def dump(self, forces=None, *args, **kwargs):
        """Return convergence metric."""
        if isinstance(forces, np.ndarray):
            if not getattr(self, '_log_written_this_iter', False):
                self._write_log_line(forces)
            self._log_written_this_iter = False
            real_fmax = np.abs(forces).max()
        else:
            real_fmax = np.abs(self.atoms.get_forces().ravel()).max()
        self._log_fmax = real_fmax
        if self._check_all_criteria(real_fmax):
            return real_fmax
        return 999.9


    def _write_log_line(self, forces):
        fmax = self._update_log_values()
        e = self.atoms.get_potential_energy()
        t = time.time() - self.t_start
        if self.nsteps == 0:
            self.logfile.write(
                "{:<8s}  {:>12s}  {:>18s}  {:>10s}  {:>12s}  {:>10s}  {:>14s}\n".format(
                    "Step","Time","Energy","fmax","stress_max","disp_max","|dE|"))
            self.logfile.write("-" * 96 + "\n")
            self.logfile.flush()
        _smax = f"{self._log_stress_max:.6f}" if self._log_stress_max is not None else "    n/a"
        _dmax = f"{self._log_disp_max:.6f}"   if self._log_disp_max is not None else "    n/a"
        _de   = f"{self._log_energy_diff:.6e}" if self._log_energy_diff is not None else "    n/a"
        self.logfile.write(
            "{:<8d}  {:>12.6f}  {:>18.10f}  {:>10.4f}  {:>12s}  {:>10s}  {:>14s}\n".format(
                self.nsteps, t, e, fmax, _smax, _dmax, _de))
        self.logfile.flush()
        self._log_fmax = fmax

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        if self.logfile is not None:
            self.dump(forces)
        self._log_written_this_iter = True

    def convergence_summary(self):
        forces = self.atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1)).max() if forces.ndim > 1 else np.abs(forces).max()
        stress = self._bare_atoms.get_stress()
        stress_max = np.max(np.abs(stress)) if stress is not None else None
        try:
            energy = self.atoms.get_potential_energy()
        except Exception:
            energy = None
        lines = ["\n" + "=" * 72, "  Convergence summary", "=" * 72]
        lines.append(f"  {'Criteria':<20s}  {'Value':>16s}  {'Tolerance':>12s}  {'Status':>8s}")
        lines.append("  " + "-" * 60)
        def _status(val, tol):
            if tol is None:
                return "  n/a"
            try:
                if math.isnan(val):
                    return "  n/a"
            except TypeError:
                pass
            return "  OK  " if val <= tol else "FAILED"
        lines.append(f"  {'Force max':<20s}  {fmax:>16.6f}  {self.fmax:>12.4f}  {_status(fmax, self.fmax):>8s}")
        if self.stress_fmax is not None and stress_max is not None:
            lines.append(f"  {'Stress max':<20s}  {stress_max:>16.6f}  {self.stress_fmax:>12.4f}  {_status(stress_max, self.stress_fmax):>8s}")
        if self.disp_fmax is not None and self._log_disp_max is not None:
            lines.append(f"  {'Displ. max':<20s}  {self._log_disp_max:>16.6f}  {self.disp_fmax:>12.4f}  {_status(self._log_disp_max, self.disp_fmax):>8s}")
        if self.energy_fmax is not None and self._log_energy_diff is not None:
            lines.append(f"  {'|dE|':<20s}  {self._log_energy_diff:>16.6e}  {self.energy_fmax:>12.1e}  {_status(self._log_energy_diff, self.energy_fmax):>8s}")
        lines.append(f"  {'Final energy':<20s}  {energy:>16.6f}" if energy is not None else "")
        lines.append("=" * 72 + "\n")
        return "\n".join(lines)

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
