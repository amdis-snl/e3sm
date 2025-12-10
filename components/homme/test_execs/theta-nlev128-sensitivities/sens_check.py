#!/usr/bin/env python3

import sys
import pathlib
import argparse

import xarray as xr
import numpy as np

###############################################################################
def parse_command_line(args):
###############################################################################
    parser = argparse.ArgumentParser(
        usage=f"""\n{pathlib.Path(args[0]).name} <ARGS> [--verbose]
OR
{pathlib.Path(args[0]).name} --help

Compute Finite-Difference sensitivities and compare against computed sensitivities

The args -b and -p provide file names perfixes for the base and perturbed simulations.
Along with the -t arg and the values passed to -d, we can compute the filenames:
  base run: ${{base}}_${{test_case}}1.nc and ${{base}}_${{test_case}}2.nc, where the latter is assumed
            to store the sensitivities
  perturbed runs: ${{perturbed}}_${{delta}}_${{test_case}}1.nc, where ${{delta}} is replaced with
                  each of the values provided via -d

We compute FD sensitividies via fd_sens=(perturbed-base)/delta, for each of the
variables names provided via the -n flag. We also assume that the file
${{base}}_${{test_case}}2.nc contains the sensitivity for each requested variable,
under the name '${{varname}}_sens'.

"""
)

    # Variables names
    parser.add_argument("-v","--var-names",nargs='+', default=[],
            help="Name the list of variables for which to check sensitivities")
    # Values of delta parameter
    parser.add_argument("-d","--deltas",nargs='+', default=[],
                        help="Values of deltas used")

    parser.add_argument("-p","--perturbed-prefix",type=str,
                        help="Prefix for the perturbed runs files")

    parser.add_argument("-b","--base-prefix",type=str,
                        help="Filename prefix for the unperturbed runs file")

    parser.add_argument("-t","--test-case",type=str,
                        help="Test case that was used to generate the output")

    parser.add_argument("--tol",type=float,default=-1,
                        help="Tolerance for comparison")
    return parser.parse_args(args[1:])

###############################################################################
def run_check(var_names,deltas,perturbed_prefix,base_prefix,test_case,tol):
###############################################################################

    print ("** Comparing computed sensitivities against Finite Difference approximation **")
    print (f" variable names: {var_names}")
    print (f" rel perturb values: {deltas}")
    print (f" unperturbed run variables file: {base_prefix}_{test_case}1.nc")
    print (f" unperturbed run sensitivities file: {base_prefix}_{test_case}2.nc")
    print (f" perturbed run variables files: {perturbed_prefix}_<perturb_value>_{test_case}1.nc")
    if tol>=0:
        print (f" tolerance: {tol}")

    base_ds = xr.open_dataset(f'{base_prefix}_{test_case}1.nc',engine='netcdf4',decode_timedelta=False)
    sens_ds = xr.open_dataset(f'{base_prefix}_{test_case}2.nc',engine='netcdf4',decode_timedelta=False)

    # Extract values of computed sensitivities as well as fields from base
    sens = {}
    base = {}
    last_base = base_ds.isel(time=-1)
    last_sens = sens_ds.isel(time=-1)
    fd_sens = {}
    for n in var_names:
        base[n] = last_base.variables[n]
        sens[n] = last_sens.variables[f'{n}_sens']
        fd_sens[n] = []

    # Open the perturbed datasets, and compute sensitivities via FD 
    for d in deltas:
        pert_ds = xr.open_dataset(f'{perturbed_prefix}_{d}_{test_case}1.nc',engine='netcdf4',decode_timedelta=False)
        last_pert = pert_ds.isel(time=-1)
        for n in var_names:
            pert_var = last_pert.variables[n]
            fd_sens[n].append((pert_var.values - base[n].values) / float(d))

    # Compare
    approx_good = True
    for n in var_names:
        fd_abs_errors = []
        fd_rel_errors = []
        for idx,d in enumerate(deltas):
            fd_abs_errors.append(np.max(np.abs(fd_sens[n][idx]-sens[n])))
            fd_rel_errors.append(np.max(np.abs(fd_sens[n][idx]-sens[n]))/np.max(np.abs(sens[n])))

        print(f"For var={n}:")
        print(f"  FD approx abs errors: {[float(e) for e in fd_abs_errors]}")
        print(f"  FD approx rel errors: {[float(e) for e in fd_rel_errors]}")

        if tol>0 and max(float(np.min(fd_abs_errors)),float(np.min(fd_rel_errors)))>=tol:
            print (f"  -> NOT within tol={tol}")
            approx_good = False

    return approx_good

###############################################################################
if (__name__ == "__main__"):
    success = run_check (**vars(parse_command_line(sys.argv)))

    sys.exit(0 if success else 1)
