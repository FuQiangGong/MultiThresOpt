#!/usr/bin/env python3
"""MLIP phonon calculation with multi-criteria convergence.

Workflow:
  1. Read an ASE- or pymatgen-readable structure.
  2. Optimise it with LBFGS and force/stress/displacement/energy criteria
  3. Build the finite-displacement supercell, evaluate all displacement
     forces in parallel with worker-local MLIP calculators.
  4. Assemble the dynamical matrix and write DOS, Gamma-point modes,
     eigenvectors, and crystal thermodynamics.

Usage:
    python phonon_mlip.py --model path/to/model --struct structure.cif
        --workers 8 --fmax 0.05 --stress-max 0.01
        --supercell 2 2 2 --temperatures 100,200,298.15,300
"""

import argparse
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.phonons import Phonons
from ase.thermochemistry import CrystalThermo
from ase.units import invcm
from deepmd.calculator import DP as DPCalculator
from multithresopt import LBFGS_Multi


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


_DP_WORKER = None
_BASE_ATOMS_WORKER = None
_OFFSET_WORKER = None
_BASE_POS_WORKER = None


def _init_worker(model_path, base_atoms, offset, base_pos, device):
    global _DP_WORKER, _BASE_ATOMS_WORKER, _OFFSET_WORKER, _BASE_POS_WORKER
    if device == "cpu":
        os.environ["DEVICE"] = "cpu"
    else:
        os.environ["DEVICE"] = "cuda"
    _DP_WORKER = DPCalculator(model_path)
    _BASE_ATOMS_WORKER = base_atoms.copy()
    _OFFSET_WORKER = offset
    _BASE_POS_WORKER = base_pos


def _compute_force(job):
    a, i, sign, delta = job
    atoms = _BASE_ATOMS_WORKER.copy()
    atoms.positions[_OFFSET_WORKER + a, i] = \
        _BASE_POS_WORKER[a, i] + sign * delta
    atoms.calc = _DP_WORKER
    return atoms.get_forces()


class ParallelPhonons(Phonons):
    def run_parallel(self, model_path, workers, device):
        atoms_N = self.atoms * self.supercell
        natoms = len(self.atoms)
        offset = natoms * self.offset
        pos = atoms_N.positions[offset: offset + natoms].copy()

        jobs = [(-1, -1, 0, 0.0)]  # equilibrium structure
        for a in self.indices:
            for i in range(3):
                for sign in (-1, 1):
                    jobs.append((a, i, sign, self.delta))

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(model_path, atoms_N, offset, pos, device),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {pool.submit(_compute_force, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                forces = future.result()
                if job[0] == -1:
                    disp = self._eq_disp()
                else:
                    a, i, sign, delta = job
                    disp = self._disp(a, i, sign)
                with self.cache.lock(disp.name) as handle:
                    if handle is not None:
                        handle.save({"forces": forces})
        self.comm.barrier()


def read_structure(path):
    try:
        return read(path)
    except Exception as ase_err:
        try:
            from pymatgen.core import Structure
            from pymatgen.io.ase import AseAtomsAdaptor
            structure = Structure.from_file(str(path))
            return AseAtomsAdaptor().get_atoms(structure)
        except Exception as pymatgen_err:
            raise RuntimeError(
                f"Cannot read '{path}' with ASE ({ase_err}) "
                f"or pymatgen ({pymatgen_err})"
            ) from pymatgen_err


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MLIP phonon calculation with multi-criteria convergence."
    )
    p.add_argument("--model", required=True,
                   help="Path or identifier for the MLIP model")
    p.add_argument("--struct", required=True,
                   help="ASE- or pymatgen-readable structure file")
    p.add_argument("--outdir", default="phonon_out",
                   help="Output directory (default: phonon_out)")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                   help="Number of parallel force workers (default: CPU count)")
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu",
                   help="Run the MLIP calculator on CPU or GPU "
                        "(default: gpu; falls back "
                        "to CPU automatically if CUDA is unavailable)")

    # Convergence criteria (same names as opt-0.py)
    p.add_argument("--fmax", type=float, default=0.05, help="Force max (eV/A)")
    p.add_argument("--stress-max", type=float, default=0.01,
                   help="Stress max (eV/A^3)")
    p.add_argument("--disp-max", type=float, default=0.001,
                   help="Displacement max (A)")
    p.add_argument("--energy-max", type=float, default=1e-6,
                   help="Energy change max (eV)")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Maximum optimisation steps")
    p.add_argument("--no-opt", action="store_true",
                   help="Skip optimisation and use the input as-is")
    p.add_argument("--allow-unconverged", action="store_true",
                   help="Continue with phonons even if optimisation did not "
                        "meet all convergence criteria")

    # Phonon settings
    p.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 2],
                   help="Supercell repeats (default: 2 2 2)")
    p.add_argument("--delta", type=float, default=0.01,
                   help="Finite-displacement size in A (default: 0.01)")
    p.add_argument("--dos-kpts", nargs=3, type=int, default=[40, 40, 40],
                   help="DOS q-point mesh (default: 40 40 40)")
    p.add_argument("--dos-npts", type=int, default=3000,
                   help="DOS energy grid points (default: 3000)")
    p.add_argument("--dos-width", type=float, default=5e-4,
                   help="DOS Gaussian width in eV (default: 5e-4)")
    p.add_argument("--dos-xmin", type=float, default=0.0,
                   help="DOS lower energy in eV (default: 0.0)")
    p.add_argument("--formula-units", type=int, default=1,
                   help="Formula units in the cell for thermochemistry")
    p.add_argument("--temperatures", default="298.15",
                   help="Comma-separated temperatures in K")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["DEVICE"] = "cpu" if args.device == "cpu" else "cuda"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    temperatures = [float(t) for t in args.temperatures.split(",") if t.strip()]
    if not temperatures:
        temperatures = [298.15]

    print(f"Reading structure: {args.struct}")
    atoms = read_structure(args.struct)
    print(f"  {len(atoms)} atoms, formula "
          f"{atoms.get_chemical_formula()}")

    dp = DPCalculator(args.model)

    if not args.no_opt:
        atoms.calc = dp
        ecf = FrechetCellFilter(atoms)
        dyn = LBFGS_Multi(ecf, logfile=str(outdir / "opt.log"))
        dyn.set_additional_criteria(
            stress_fmax=args.stress_max,
            disp_fmax=args.disp_max,
            energy_fmax=args.energy_max,
        )
        dyn.run(fmax=args.fmax, steps=args.max_steps)
        summary = dyn.convergence_summary()
        print(summary)
        (outdir / "opt_conv.txt").write_text(summary)
        atoms.write(outdir / "opt.cif")

        # Explicit convergence check: do not silently call it "optimised".
        forces = atoms.get_forces()
        fmax = (np.sqrt((forces ** 2).sum(axis=1)).max()
                if forces.ndim > 1 else np.abs(forces).max())
        stress = atoms.get_stress()
        stress_max = np.max(np.abs(stress)) if stress is not None else float("nan")
        disp_max = dyn._log_disp_max if dyn._log_disp_max is not None else float("nan")
        energy_diff = (dyn._log_energy_diff
                       if dyn._log_energy_diff is not None else float("nan"))
        converged = (
            fmax <= args.fmax
            and (math.isnan(stress_max) or stress_max <= args.stress_max)
            and (math.isnan(disp_max) or disp_max <= args.disp_max)
            and (math.isnan(energy_diff) or energy_diff <= args.energy_max)
        )
        if not converged and not args.allow_unconverged:
            print(
                "ERROR: optimisation did not meet all convergence criteria; "
                "rerun with more steps or use --allow-unconverged to force "
                "the phonon calculation.",
                file=sys.stderr,
            )
            return 2
        if not converged:
            print(
                "WARNING: continuing with an unconverged structure "
                "(--allow-unconverged).",
                file=sys.stderr,
            )
    else:
        atoms.calc = dp
        atoms.write(outdir / "input.cif")

    potential_energy = atoms.get_potential_energy()
    print(f"Optimised potential energy: {potential_energy:.6f} eV")

    # Phonon force constants, parallel over displacements
    ph = ParallelPhonons(
        atoms,
        dp,
        supercell=tuple(args.supercell),
        delta=args.delta,
        name="phonon",
    )
    print(f"Running {len(atoms) * 6 + 1} displacement calculations "
          f"with {args.workers} workers ...")
    ph.run_parallel(args.model, args.workers, args.device)
    ph.read(acoustic=True)
    ph.clean()

    # Gamma-point modes and eigenvectors.
    # ph.get_force_constant() returns the un-mass-weighted C_N (eV/Ang^2);
    # mass-weight it before diagonalising, otherwise the frequencies and
    # eigenvectors are wrong (frequencies come out far too high).
    masses = atoms.get_masses()[ph.indices]
    m_inv_x = np.repeat(masses ** -0.5, 3)
    D_N = ph.get_force_constant().copy() * np.outer(m_inv_x, m_inv_x)
    dm = ph.compute_dynamical_matrix(np.array([0.0, 0.0, 0.0]), D_N)
    eigvals, eigvecs = np.linalg.eigh(dm)
    freqs_cm = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * 521.471
    np.savetxt(outdir / "phonon.mode.dat", freqs_cm,
               header="mode_frequency_cm-1", fmt="%.6f")
    np.savetxt(outdir / "phonon.eigenvectors.real.dat", eigvecs.real,
               fmt="%.8f", header="rows=modes columns=atom*3+cartesian")
    np.savetxt(outdir / "phonon.eigenvectors.imag.dat", eigvecs.imag,
               fmt="%.8f", header="rows=modes columns=atom*3+cartesian")
    print(f"  Gamma modes -> {outdir / 'phonon.mode.dat'}")
    print(f"  Eigenvectors -> {outdir / 'phonon.eigenvectors.real.dat'}, "
          f"{outdir / 'phonon.eigenvectors.imag.dat'}")

    # DOS and thermochemistry
    dos = ph.get_dos(kpts=tuple(args.dos_kpts)).sample_grid(
        npts=args.dos_npts, width=args.dos_width, xmin=args.dos_xmin
    )
    phonon_energies = dos.get_energies()      # eV
    phonon_dos = dos.get_weights()
    np.savetxt(
        outdir / "phonon.dos.dat",
        np.vstack((phonon_energies / invcm, phonon_dos * invcm)).T,
        header="energy_cm-1 dos_per_cm-1",
        fmt="%.6f",
    )

    thermo = CrystalThermo(
        phonon_energies=phonon_energies,
        phonon_DOS=phonon_dos,
        potentialenergy=potential_energy,
        formula_units=args.formula_units,
    )
    omega_e = np.asarray(phonon_energies)
    dos_e = np.asarray(phonon_dos)
    if omega_e[0] == 0.0:
        omega_e = omega_e[1:]
        dos_e = dos_e[1:]
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    zpe = trapezoid(omega_e / 2.0 * dos_e, omega_e)
    if args.formula_units:
        zpe /= args.formula_units
    with open(outdir / "phonon.thermo.dat", "w") as f:
        f.write("# T_K   ZPE_eV   U_eV   S_eV_per_K   F_eV\n")
        for T in temperatures:
            U = thermo.get_internal_energy(T)
            S = thermo.get_entropy(T)
            F = thermo.get_helmholtz_energy(T)
            f.write(f"{T:.6f}\t{zpe:.6f}\t{U:.6f}\t{S:.6e}\t{F:.6f}\n")
    print(f"  DOS -> {outdir / 'phonon.dos.dat'}")
    print(f"  Thermo -> {outdir / 'phonon.thermo.dat'}")
    print(f"  ZPE = {zpe:.6f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
