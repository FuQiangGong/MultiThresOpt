"""Shared ASE optimiser with force, stress, displacement, and energy checks."""

import math
import time

import numpy as np
from ase.optimize import LBFGS


class LBFGS_Multi(LBFGS):
    """LBFGS with configurable multi-criteria convergence checks."""

    def __init__(self, atoms, **kwargs):
        super().__init__(atoms, **kwargs)
        self.t_start = time.time()
        self.stress_fmax = None
        self.disp_fmax = None
        self.energy_fmax = None
        self._bare_atoms = atoms.atoms if hasattr(atoms, "atoms") else atoms
        self._log_stress_max = None
        self._log_disp_max = None
        self._log_energy_diff = None
        self._log_prev_pos = None
        self._log_prev_energy = None
        self._log_written_this_iter = False

    def set_additional_criteria(
        self,
        stress_fmax=None,
        disp_fmax=None,
        energy_fmax=None,
    ):
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
        """Return True only when every active convergence criterion passes."""
        if self.nsteps == 0 and (
            self.disp_fmax is not None or self.energy_fmax is not None
        ):
            return False
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = _force_max(forces)
        return self._check_all_criteria(fmax)

    def _update_log_values(self):
        forces = self.atoms.get_forces()
        fmax = _force_max(forces)
        if self.stress_fmax is not None:
            stress = self._bare_atoms.get_stress()
            self._log_stress_max = (
                np.max(np.abs(stress)) if stress is not None else None
            )
        curr_pos = self.atoms.get_positions()
        if self._log_prev_pos is not None:
            self._log_disp_max = np.sqrt(
                ((curr_pos - self._log_prev_pos) ** 2).sum(axis=1)
            ).max()
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
        if fmax_from_forces is None or not np.isfinite(fmax_from_forces):
            return False
        if fmax_from_forces > self.fmax:
            return False
        if self.stress_fmax is not None:
            stress = self._bare_atoms.get_stress()
            if stress is not None:
                if not np.all(np.isfinite(stress)):
                    return False
                if np.max(np.abs(stress)) > self.stress_fmax:
                    return False
        if self.disp_fmax is not None and self._log_prev_pos is not None:
            curr = self.atoms.get_positions()
            if (
                not np.all(np.isfinite(curr))
                or not np.all(np.isfinite(self._log_prev_pos))
            ):
                return False
            dmax = np.sqrt(
                ((curr - self._log_prev_pos) ** 2).sum(axis=1)
            ).max()
            if dmax > self.disp_fmax:
                return False
        if self.energy_fmax is not None and self._log_prev_energy is not None:
            try:
                curr_e = self.atoms.get_potential_energy()
                if (
                    not np.isfinite(curr_e)
                    or not np.isfinite(self._log_prev_energy)
                ):
                    return False
                if abs(curr_e - self._log_prev_energy) > self.energy_fmax:
                    return False
            except Exception:
                pass
        return True

    def dump(self, forces=None, *args, **kwargs):
        """Return the convergence metric used by ASE's logging machinery."""
        if isinstance(forces, np.ndarray):
            if not getattr(self, "_log_written_this_iter", False):
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
        energy = self.atoms.get_potential_energy()
        elapsed = time.time() - self.t_start
        if self.nsteps == 0:
            self.logfile.write(
                "{:<8s}  {:>12s}  {:>18s}  {:>10s}  {:>12s}  "
                "{:>10s}  {:>14s}\n".format(
                    "Step",
                    "Time",
                    "Energy",
                    "fmax",
                    "stress_max",
                    "disp_max",
                    "|dE|",
                )
            )
            self.logfile.write("-" * 96 + "\n")
            self.logfile.flush()
        stress_max = (
            f"{self._log_stress_max:.6f}"
            if self._log_stress_max is not None
            else "    n/a"
        )
        displacement_max = (
            f"{self._log_disp_max:.6f}"
            if self._log_disp_max is not None
            else "    n/a"
        )
        energy_diff = (
            f"{self._log_energy_diff:.6e}"
            if self._log_energy_diff is not None
            else "    n/a"
        )
        self.logfile.write(
            "{:<8d}  {:>12.6f}  {:>18.10f}  {:>10.4f}  {:>12s}  "
            "{:>10s}  {:>14s}\n".format(
                self.nsteps,
                elapsed,
                energy,
                fmax,
                stress_max,
                displacement_max,
                energy_diff,
            )
        )
        self.logfile.flush()
        self._log_fmax = fmax

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        if self.logfile is not None:
            self.dump(forces)
        self._log_written_this_iter = True

    def convergence_summary(self):
        """Return a formatted summary of the final convergence metrics."""
        forces = self.atoms.get_forces()
        fmax = _force_max(forces)
        stress = self._bare_atoms.get_stress()
        stress_max = np.max(np.abs(stress)) if stress is not None else None
        try:
            energy = self.atoms.get_potential_energy()
        except Exception:
            energy = None

        def status(value, tolerance):
            if tolerance is None:
                return "  n/a"
            try:
                if math.isnan(value):
                    return "  n/a"
            except TypeError:
                pass
            return "  OK  " if value <= tolerance else "FAILED"

        lines = [
            "\n" + "=" * 72,
            "  Convergence summary",
            "=" * 72,
            "  {:<20s}  {:>16s}  {:>12s}  {:>8s}".format(
                "Criteria", "Value", "Tolerance", "Status"
            ),
            "  " + "-" * 60,
            f"  {'Force max':<20s}  {fmax:>16.6f}  "
            f"{self.fmax:>12.4f}  {status(fmax, self.fmax):>8s}",
        ]
        if self.stress_fmax is not None and stress_max is not None:
            lines.append(
                f"  {'Stress max':<20s}  {stress_max:>16.6f}  "
                f"{self.stress_fmax:>12.4f}  "
                f"{status(stress_max, self.stress_fmax):>8s}"
            )
        if self.disp_fmax is not None and self._log_disp_max is not None:
            lines.append(
                f"  {'Displ. max':<20s}  {self._log_disp_max:>16.6f}  "
                f"{self.disp_fmax:>12.4f}  "
                f"{status(self._log_disp_max, self.disp_fmax):>8s}"
            )
        if self.energy_fmax is not None and self._log_energy_diff is not None:
            lines.append(
                f"  {'|dE|':<20s}  {self._log_energy_diff:>16.6e}  "
                f"{self.energy_fmax:>12.1e}  "
                f"{status(self._log_energy_diff, self.energy_fmax):>8s}"
            )
        if energy is not None:
            lines.append(f"  {'Final energy':<20s}  {energy:>16.6f}")
        lines.append("=" * 72 + "\n")
        return "\n".join(lines)


def _force_max(forces):
    if forces.ndim > 1:
        return np.sqrt((forces**2).sum(axis=1)).max()
    return np.abs(forces).max()
