# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:08:31 2023

@author: lucie.thompson
"""



#%% Loading modules and global variables


import pandas as pd
## pandas needs to be version 1.5.1 to read the npy pickled files with np.load() 
## created under this version 
## this can be checked using pd.__version__ and version 1.5.1 can be installed using
## pip install pandas==1.5.1 (how I did it on SLURM - 26/03/2024)
## if version 2.0 or over yields a module error ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'
## I think to solve this, would need to run the code and save files under version > 2.0 of pandas
## or manually create this pandas.core.index.numerical function see issue in https://github.com/pandas-dev/pandas/issues/53300

import matplotlib.pyplot as plt 

import numpy as np # Numerical objects

import os

import seaborn as sb

import re

import pickle




os.chdir('D:/TheseSwansea/Patch-Models/Script')
import FunctionsAnalysis as fn
import FunctionsPatchModel as fnPM

f = "D:/TheseSwansea/PatchModels/outputs/S100_C10/StableFoodWebs_55persist_Stot100_C10_t10000000000000.npy"
stableFW = np.load(f, allow_pickle=True).item()


Stot = 100 # initial pool of species
P = 10 # number of patches
C = 0.1 # connectance
tmax = 10**12 # number of time steps
d = 1e-8
# FW = fn.nicheNetwork(Stot, C)

# S = np.repeat(Stot,P) # local pool sizes (all patches are full for now)
S = np.repeat(round(Stot*1/3),P) # initiate with 50 patches




#%%%% 9 PATCHES - donut landscape

P = 9
os.chdir('D:/TheseSwansea/Patch-Models/outputs/9Patches/Homogeneous/donut-landscape-test')
init_9P_files = ['D:/TheseSwansea/Patch-Models/outputs/9Patches/Homogeneous/donut-landscape-test/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]

# run summary statistics function (above)
res9_init, FW9_init = fn.summarise_initial_pop_dynamics(list_files=init_9P_files, nb_patches=9)

FW9_init['simulation_length_years'] = FW9_init['simulation_length']/(60*60*24*365)

res9_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/9Patches/donut-landscape-test/ResultsInitial-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW9_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/9Patches/donut-landscape-test/ResultsInitial-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
