#!/usr/bin/env python3

import numpy as np
from glob import glob
from CifFile import ReadCif
from itertools import product, combinations, combinations_with_replacement
import math
from atomic_property_dict import apd
from datetime import datetime
import shutil
import argparse
import subprocess
import os

# By default, save the csv in the current working directory
script_path, script_name = os.path.split(os.path.realpath(__file__))
cwd = os.getcwd()

# Desired distance bins (in this case with linear increase in bin size from 2 to 30 A)
bins = np.arange(113, dtype=np.float64)
bins[0] = 2.0
step = 0.004425

for i in range(1, 113):
    bins[i] = bins[i - 1] + step
    step += 0.004425

n_bins = len(bins)


def calc_rdfs(name, props, smooth, factor):
    mof = ReadCif(name)
    mof = mof[mof.visible_keys[0]]

    elements = mof["_atom_site_type_symbol"]
    n_atoms = len(elements)

    super_cell = np.array(list(product([-1, 0, 1], repeat=3)), dtype=float)

    # Make sure the property exists for the particular element
    prop_list = [apd[name] for name in props]
    for atom in set(elements):
        for count, prop in enumerate(prop_list):
            if atom not in prop:
                print( "\n***WARNING***",
                       "Property '{}' does not exist for atom {}!".format(prop_names[count], atom),
                       "ML model cannot do predictions without this property. Exiting...")
                exit()

    n_props = len(props)
    prop_dict = {}


    for a1, a2 in combinations_with_replacement(set(elements), 2):
        prop_arr = [prop[a1] * prop[a2] for prop in prop_list]
        prop_dict[(a1, a2)] = prop_arr
        if a1 != a2:
            prop_dict[(a2, a1)] = prop_arr

    la = float(mof["_cell_length_a"])
    lb = float(mof["_cell_length_b"])
    lc = float(mof["_cell_length_c"])
    aa = np.deg2rad(float(mof["_cell_angle_alpha"]))
    ab = np.deg2rad(float(mof["_cell_angle_beta"]))
    ag = np.deg2rad(float(mof["_cell_angle_gamma"]))
    # If volume is missing from .cif, calculate it.
    try:
        cv = float(mof["_cell_volume"])
    except KeyError:
        cv = la * lb * lc * math.sqrt(1 - (math.cos(aa)) ** 2 -
                (math.cos(ab)) ** 2 - (math.cos(ag)) ** 2 +
                (2 * math.cos(aa) * math.cos(ab) * math.cos(ag)))


    frac2cart = np.zeros([3, 3], dtype=float)
    frac2cart[0, 0] = la
    frac2cart[0, 1] = lb * np.cos(ag)
    frac2cart[0, 2] = lc * np.cos(ab)
    frac2cart[1, 1] = lb * np.sin(ag)
    frac2cart[1, 2] = lc * (np.cos(aa) - np.cos(ab)*np.cos(ag)) / np.sin(ag)
    frac2cart[2, 2] = cv / (la * lb * np.sin(ag))

    frac = np.array([
        mof["_atom_site_fract_x"],
        mof["_atom_site_fract_y"],
        mof["_atom_site_fract_z"],
    ], dtype=float).T

    apw_rdf = np.zeros([n_props, n_bins], dtype=np.float64)
    for i, j in combinations(range(n_atoms), 2):
        cart_i = frac2cart @ frac[i]
        cart_j = (frac2cart @ (super_cell + frac[j]).T).T
        dist_ij = min(np.linalg.norm(cart_j - cart_i, axis=1))
        rdf = np.exp(smooth * (bins - dist_ij) ** 2)
        rdf = rdf.repeat(n_props).reshape(n_bins, n_props)
        apw_rdf += (rdf * prop_dict[(elements[i], elements[j])]).T
    apw_rdf = np.round(apw_rdf.flatten() * factor / n_atoms, decimals=12)

    return apw_rdf.tolist()

def calc_geo_props(name, zeo_exe="network", discard_geo=True):
    path = "{}/geo_props".format(cwd)
    commands = ["-sa 1.86 1.86 2000 {}/ASA.txt".format(path),
                "-vol 1.86 1.86 50000 {}/AVA.txt".format(path),
                "-res {}/PD.txt".format(path)]
    if not os.path.exists(path):
        os.mkdir(path)
        # We will only remove the directory if we made it (safest)
        made_directory = True
    # Do the necessary calculations
    else:
        made_directory = False
    for command in commands:
        p = subprocess.Popen([zeo_exe,
                              '-ha',
                              *[arg for arg in command.split(" ")],
                              name], stdout=subprocess.PIPE)
        out, err = p.communicate()
        if err is not None:
            print("Warning: Zeo++ encountered an error for the calculation of {}".format(
                                                                  command.split()[0]))
    with open("{}/ASA.txt".format(path), "r") as f:
        lines = f.readlines()
        density = float(lines[0].strip().split("Density: ")[1].split()[0])
        vASA = float(lines[0].strip().split("ASA_m^2/cm^3: ")[1].split()[0])
        gASA = float(lines[0].strip().split("ASA_m^2/g: ")[1].split()[0])
    f.close

    with open("{}/AVA.txt".format(path), "r") as f:
        lines = f.readlines()
        AVAf = float(lines[0].strip().split("AV_Volume_fraction: ")[1].split()[0])
        AVAg = float(lines[0].strip().split("AV_cm^3/g: ")[1].split()[0])

    with open("{}/PD.txt".format(path), "r") as f:
        lines = f.readlines()
        Di = float(lines[0].strip().split()[1])
        Df = float(lines[0].strip().split()[2])
        Dif = float(lines[0].strip().split()[3])

    geo_props = [Di, Df, Dif, density, vASA, gASA, AVAf, AVAg]
    
    if discard_geo and made_directory:
        shutil.rmtree(path)

    return geo_props

