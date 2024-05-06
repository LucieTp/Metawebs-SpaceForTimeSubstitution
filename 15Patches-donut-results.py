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




#%%%% 15 PATCHES - donut landscape

P = 15
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3')
init_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]

# run summary statistics function (above)
res15_init, FW15_init = fn.summarise_initial_pop_dynamics(list_files=init_15P_files, nb_patches=P)

FW15_init['simulation_length_years'] = FW15_init['simulation_length']/(60*60*24*365)

res15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')



# %% Load datasets

P = 15
## 10P homogeneous
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')

res15_init = pd.read_csv(f'ResultsInitial-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_init = pd.read_csv(f'ResultsInitial-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_init['landscape'] = 'donut'
FW15_init['landscape'] = 'donut'






