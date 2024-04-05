# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:08:31 2023

@author: lucie.thompson
"""


###############################################################################
### SIMPLE 3 patches simulation - O - O - O
## 1 - No protected area O - O - O
## 2 - One protected area in the middle O - (O) - O

import pandas as pd

from scipy.integrate import odeint, solve_ivp # Numerical integration function from the scientific programming package

import matplotlib.pyplot as plt 

import numpy as np # Numerical objects

import time

import networkx as nx

import igraph as ig

from argparse import ArgumentParser # for parallel runnning on Slurm

import pickle



#%% Looking at the results

import os

os.chdir('D:/TheseSwansea/SFT/test/')

with open('InitialPopDynamics_homogeneous_sim7_20Patches_Stot100_C10_t1000000000000r.pkl', 'rb') as f:
    sol20P = pickle.load(f)
f.close()

### runtime?
print(round(sol20P['sim_duration'] / (60*60)), ' hours and ', round((sol20P['sim_duration'] % (60*60)) / (60)), ' minutes') ## 3 hours and


with open('InitialPopDynamics_homogeneous_sim7_15Patches_Stot100_C10_t1000000000000r.pkl', 'rb') as f:
    sol15P = pickle.load(f)
f.close()
    
print(round(sol15P['sim_duration'] / (60*60)), ' hours and ', round((sol15P['sim_duration'] % (60*60)) / (60)), ' minutes') ## 3 hours and


### plotting dynamics
sol = sol15P
nb_patches = 15



solT = sol['t'] ## time
solY = sol['y'] ## biomasses

FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
Stot_new = FW_new['Stot']

# mean biomass across the last 5% of the time steps
thresh_high = solT[-1] # get higher time step boundary (tfinal)
thresh_low = (solT[-1] - (solT[-1] - solT[1])*0.05) # get lower time step boundary (tfinal - 5% * tfinal)

## index of those time steps between low and high boundaries
index_bf = np.where((solT > thresh_low) & (solT <= thresh_high))[0]
## biomasses over those 10% time steps
Bf1 = np.mean(solY[index_bf], axis = 0).reshape(nb_patches,Stot_new)


for p in range(nb_patches):
   
    ind = p + 1
    
    surviving_sp = np.where(Bf1[p]>0)[0]
    local_FW = FW_new['M'][Bf1[p]>0,:][:,Bf1[p]>0]

    prop_local = Bf1[p][Bf1[p]>0]/np.sum(Bf1[p][Bf1[p]>0])
    
    print(pd.DataFrame({'S_local':len(surviving_sp), 'C_local': np.sum(local_FW),
                                         'MeanGen_local': np.mean(np.sum(local_FW, axis = 0)),
                                         'MeanVul_local': np.mean(np.sum(local_FW, axis = 1)),
                                         'MeanTL_local': np.mean(FW_new['TL'][surviving_sp]), # mean trophic level of surviving species on patch p
                                          ## diversity measure
                                         'alpha_diversity':-np.sum(prop_local*np.log(prop_local)), # shannon diversity (-sum(pi*ln(pi)))
                                         
                                         'simulation_length':sol['t'][-1]
                                         }, index=[ind]))
    
    plt.subplot(4, 5, ind)
    plt.tight_layout()
    
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solY.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)])
   
plt.title("With events")
plt.show()



    
