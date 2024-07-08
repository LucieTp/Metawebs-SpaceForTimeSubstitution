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

import colorcet as cc ## to have more than 12 categorical colours




os.chdir('D:/TheseSwansea/SFT/Script')
import FunctionsAnalysisRestorationDonut as fn
os.chdir('D:/TheseSwansea/Patch-Models/Script')
import FunctionsPatchModel as fnPM

f = "D:/TheseSwansea/PatchModels/outputs/S100_C10/StableFoodWebs_55persist_Stot100_C10_t10000000000000.npy"
stableFW = np.load(f, allow_pickle=True).item()


Stot = 100 # initial pool of species
P = 15 # number of patches
C = 0.1 # connectance
tmax = 10**12 # number of time steps
d = 1e-8
# FW = fn.nicheNetwork(Stot, C)

# S = np.repeat(Stot,P) # local pool sizes (all patches are full for now)
S = np.repeat(round(Stot*1/3),P) # initiate with 50 patches


## create landscape 
extentx = [0,0.4]
radius_max = 0.1

## surface of the circle
((radius_max**2)*np.pi)/(0.5*0.5)

seed_index = 3
seed = np.arange(P*seed_index, P*(seed_index + 1))
coords_seed3 = fn.create_landscape(P, extentx, radius_max, 5, seed = seed)

palette_colors = sb.color_palette(cc.glasbey, 15)
sb.palplot(palette_colors)
plt.show()


sb.scatterplot(data = coords_seed3, x = 'x',y = 'y', hue = 'Patch', style = 'position', palette = palette_colors)
plt.scatter(x = np.mean(extentx), y = np.mean(extentx), s = 100, marker = 'X', c = 'black') # center of the landscape
plt.legend(bbox_to_anchor = [1,1.1])

plt.savefig('D:/TheseSwansea/SFT/Figures/seed3-Landscape.png', dpi = 400, bbox_inches = 'tight')


# %% writing file names to csv file

import os
import csv

# Define the directory you want to list files from
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow')
init_15P_files = [i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]


# Define the CSV file name
csv_file_name = 'D:/TheseSwansea/Patch-Models/outputs/15Patches/init_files_15Patches_narrow.csv'

# Write the file names to a CSV file
with open(csv_file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header
    csvwriter.writerow(['File Name'])
    
    # Write the file namesf
    for file_name in init_15P_files:
        csvwriter.writerow([file_name])

# Control
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/narrow')
control_15P_files = [i for i in os.listdir() if '.pkl' in i and 'CONTROL' in i]

csv_file_name = 'D:/TheseSwansea/Patch-Models/outputs/15Patches/control_files_15Patches_narrow.csv'

with open(csv_file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header
    csvwriter.writerow(['File Name'])
    
    # Write the file namesf
    for file_name in control_15P_files:
        csvwriter.writerow([file_name])


## restored
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow')
het_15P_files_invasion = [i for i in os.listdir() if '.pkl' in i and 'patchImproved' in i]

csv_file_name = 'D:/TheseSwansea/Patch-Models/outputs/15Patches/improved_files_15Patches_narrow.csv'

with open(csv_file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header
    csvwriter.writerow(['File Name'])
    
    # Write the file namesf
    for file_name in het_15P_files_invasion:
        csvwriter.writerow([file_name])


# %% plotting dynamics

#%%%% Initial dynamics


os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3')
init_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]


for f in init_15P_files[0]:
    
    print(f)
    
    # load file with population dynamics
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses

    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']
    
    for p in range(3):
   
        ind = p + 1
                
        plt.subplot(1, 3, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)])
                
    plt.title("Invasion - control")
    
    plt.savefig(f'D:/TheseSwansea/Patch-Models/Figures/PopDynamics.png', dpi = 400, bbox_inches = 'tight')

    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10)])
    plt.show()



#%%%% Controls

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3')
control_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/'+i for i in os.listdir() if 'invasion' not in i and ('.pkl' in i or '.npy' in i)]


for f in control_15P_files:
    
    print(f)
    
    # load file with population dynamics
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses

    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']
    
    for p in range(P):
   
        ind = p + 1
                
        plt.subplot(3, 5, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)])
                
    plt.title("Invasion - control")
    plt.show()
    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10)])
    plt.show()





#%%%% Controls - invasion  - corner patch

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3')
control_invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/'+i for i in 
                              os.listdir() if '.pkl' in i and 'invasion' in i and '-10000-' not in i and 'CornerPatch' in i]


for f in control_invasion_15P_files:
    
    print(f)
    
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses
    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']


    for p in range(P):
   
        ind = p + 1
        
        invaders = FW_new['invaders'][p]
        
        plt.subplot(3, 5, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,~invaders])
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                   solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,invaders], linestyle="dotted")

        
    plt.title("Invasion - control")
    plt.show()
    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,~FW_new['invaders'].reshape(P*Stot)])
    plt.title(f'{f}')
    plt.show()
    
    # species who invaded middle patches 
    # species who had zero biomass (FW_new['y0']) in centre patches before invasion
    solY_centre_patches = solY[:,np.repeat(FW_new['coords']['position'] == 'center',Stot)]
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY_centre_patches[np.arange(0,solT.shape[0],10),:][:,(sol['FW_new']['y0'][FW_new['coords']['position'] == 'center'].reshape(5*Stot) == 0)])
    plt.title(f'invaders into middle patches')
    plt.show()
    
    # species who weren't initialised in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) == 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dotted')
    plt.title(f'y0 == 0 in initial period')
    plt.show()
    
    # species who went extinct in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) != 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dashed')
    plt.title(f'went extinct in initial period')
    plt.show()
    
    '''
    
    It seems that even among the species that were initialised and went extinct during the initial run
    species can still invade once dynamics have stabilised, in the absence of restoration
    
    '''
    
    
    
#%%%% invasion - restoration - corner patch

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch')
invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/'+i for i in 
                              os.listdir() if '.pkl' in i and 'invasion' in i]


for f in invasion_15P_files:
    
    print(f)
    
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses
    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']


    for p in range(P):
   
        ind = p + 1
        
        invaders = FW_new['invaders'][p]
        
        plt.subplot(3, 5, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,~invaders])
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                   solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,invaders], linestyle="dotted")

        
    plt.title(f'')
    plt.show()
    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,~FW_new['invaders'].reshape(P*Stot)])
    plt.title(f'{f}')
    plt.show()
    
    # species who invaded middle patches 
    # species who had zero biomass (FW_new['y0']) in centre patches before invasion
    solY_centre_patches = solY[:,np.repeat(FW_new['coords']['position'] == 'center',Stot)]
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY_centre_patches[np.arange(0,solT.shape[0],10),:][:,(sol['FW_new']['y0'][FW_new['coords']['position'] == 'center'].reshape(5*Stot) == 0)])
    plt.title(f'invaders into middle patches')
    plt.show()
    
    # species who weren't initialised in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) == 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dotted')
    plt.title(f'y0 == 0 in initial period')
    plt.show()
    
    # species who went extinct in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) != 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dashed')
    plt.title(f'went extinct in initial period')
    plt.show()
    
    '''
    
    It seems that even among the species that were initialised and went extinct during the initial run
    species can still invade once dynamics have stabilised, in the absence of restoration
    
    '''



#%% Create summary stats



# %%% normal landscape 

# intial population dynamics - normal
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow')
init_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]

# run summary statistics function (above)
res15_init, FW15_init = fn.summarise_initial_pop_dynamics(list_files=init_15P_files, nb_patches=P)

FW15_init['simulation_length_years'] = FW15_init['simulation_length']/(60*60*24*365)

res15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-narrow_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-narrow-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')


# %%%% corner patch being invaded 

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow')
init_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]


# Control
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/narrow')
control_invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/narrow/'+i for i in os.listdir() if '.pkl' in i and 'invasion' in i]

res15, FW15 = fn.summarise_pop_dynamics(list_files=control_invasion_15P_files, nb_patches=P, 
                                        initial_files=init_15P_files)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-narrow-CornerPatch-handmade_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-narrow-CornerPatch-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C.csv')


# restoration
P = 15
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow')
het_15P_files_invasion = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow/'+i for i in os.listdir() if '.pkl' in i and 'patchImproved' in i]



# run summary statistics function (above)
res15, FW15 = fn.summarise_pop_dynamics(list_files=het_15P_files_invasion, nb_patches=P, 
                                        initial_files=init_15P_files)

FW15['simulation_length_years'] = FW15['simulation_length']/(60*60*24*365)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C.csv')


# %% Load datasets

P = 15
## 10P homogeneous
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')



# %%% normal landscape

res15_init_normal = pd.read_csv(f'ResultsInitial-narrow_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_init_normal = pd.read_csv(f'ResultsInitial-narrow-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_init_normal['landscape'] = 'normal'
FW15_init_normal['landscape'] = 'normal'
res15_init_normal['stage'] = 'init'
FW15_init_normal['stage'] = 'init'


# %%%% Corner Patch



os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')

res15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C.csv')
res15_invasion['landscape'] = 'normal'
FW15_invasion['landscape'] = 'normal'
res15_invasion['stage'] = 'restored'
FW15_invasion['stage'] = 'restored'


res15_control_invasion_normal = pd.read_csv(f'Control-Invasion-seed3-narrow-CornerPatch-handmade_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15_control_invasion_normal = pd.read_csv(f'Control-Invasion-seed3-narrow-CornerPatch-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C.csv')
res15_control_invasion_normal['landscape'] = 'normal'
FW15_control_invasion_normal['landscape'] = 'normal'
res15_control_invasion_normal['stage'] = 'control'
FW15_control_invasion_normal['stage'] = 'control'

res15_invasion_normal = pd.concat([res15_control_invasion_normal, res15_invasion[np.isin(res15_invasion['sim'], np.unique(res15_control_invasion_normal['sim']))]])
FW15_invasion_normal = pd.concat([FW15_control_invasion_normal, FW15_invasion[np.isin(FW15_invasion['sim'], np.unique(FW15_control_invasion_normal['sim']))]])

res15_invasion_normal = pd.concat([res15_invasion_normal, res15_init_normal])
FW15_invasion_normal = pd.concat([FW15_invasion_normal, FW15_init_normal])





# %% Plots

x = res15_invasion_normal.groupby(['sim','restoration_type','nb_improved','landscape_seed']).agg({'B_final':['mean','count', np.std]})


# %%% Corner landscape

## Landscape-level:
    
## Landscape-level species richness
ax = sb.stripplot(data = FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0)], y = 'S_local', 
                  hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0)], y = 'S_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '-', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0)], y = 'S_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '--', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])


'''
The landscape is scattered enough that the improvements to the middle patches only 
reach the periphery when more high quality patches are added

Does increasing the density of the cluster of middle patches (i.e making them as one
                                                              big connected patch) 
increase how far the benefits of restoration span?

If the introduction comes from too far away, the recolonisers cannot reach the improved patches

As we improve patches, the species richness increases
 
Is this due to an increase in the nb of invaders? Or to an increase in species richness 

If the distance between patches is too small, then all the patches are tightly coupled
and the dynamics synchronise across the landscape (example of what happened in the 
                                                   donut landscape)

Increasing the biomass of species is probably a way to make them travel further bc
it increases the immigration portion of the population
 
'''

FW15_init_normal[(FW15_init_normal['patch'] == 9) & (FW15_init_normal['sim'] == 1)]['S_local']

sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['patch'] == 9], y = 'S_local', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = palette_colors,
              errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])


## Number of invasions
ax = sb.stripplot(data = FW15_invasion_normal, y = 'nb_invaders_initial_pop', hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0)], 
             y = 'nb_invaders_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0)], 
             y = 'nb_invaders_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, 
             palette = palette_colors, linestyles = '--', ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])

FW15_invasion_normal['nb_extinct_initial_pop'] = np.nan_to_num(FW15_invasion_normal['nb_extinct_initial_pop'], nan = -1)

## Number of invasions
ax = sb.stripplot(data = FW15_invasion_normal, y = 'nb_extinct_initial_pop', hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0)], 
             y = 'nb_extinct_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0)], 
             y = 'nb_extinct_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, 
             palette = palette_colors, linestyles = '--', ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])



## Landscape-level biomass
ax = sb.stripplot(data = res15_invasion_normal, y = 'B_final', 
                  hue = 'restoration_type', x = 'nb_improved', palette = palette_colors, dodge = True)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])



## Landscape-level mean TL
ax = sb.stripplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                              ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0))], y = 'MeanTL_local', 
                  hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                              ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0))], y = 'MeanTL_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '-', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'MeanTL_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '--', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])



## mfcl
ax = sb.stripplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                              ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0))], y = 'Mfcl_local', 
                  hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'Mfcl_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '-', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'Mfcl_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '--', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])



## mean body mass
ax = sb.stripplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                              ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0))], y = 'MeanBodyMass_local', 
                  hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'MeanBodyMass_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '-', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'MeanBodyMass_local', 
             hue = 'patch', x = 'nb_improved', linestyles = '--', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])


## nb_top
## nb of int, hrb and plants reamined basically constant. Only nb of top species icmreased
ax = sb.stripplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                              ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0))], y = 'nb_top', 
                  hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'nb_top', 
             hue = 'patch', x = 'nb_improved', linestyles = '-', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'nb_top', 
             hue = 'patch', x = 'nb_improved', linestyles = '--', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])


## nb_top
## nb of int, hrb and plants reamined basically constant. Only nb of top species icmreased
ax = sb.stripplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                              ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                              (FW15_invasion_normal['nb_improved'] == 0))], y = 'nb_int', 
                  hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'clustered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'nb_int', 
             hue = 'patch', x = 'nb_improved', linestyles = '-', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
sb.pointplot(data =  FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') & 
                                          ((FW15_invasion_normal['restoration_type'] == 'scattered') | 
                                          (FW15_invasion_normal['nb_improved'] == 0))], y = 'nb_int', 
             hue = 'patch', x = 'nb_improved', linestyles = '--', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])

# %%% Change in species composition
s = 8

for p in range(P):

    ind = p + 1
        
    S_local = np.unique(FW15_control_invasion_normal['S_local'][(FW15_control_invasion_normal['sim'] == s) & (FW15_control_invasion_normal['patch'] == p)])
    plt.subplot(3, 5, ind)
    plt.pie(data = res15_control_invasion_normal[(res15_control_invasion_normal['sim'] == s) & (res15_control_invasion_normal['B_final'] > 0) & (res15_control_invasion_normal['patch'] == p)], 
               x = 'B_final', labels = "sp_ID", colors = sb.color_palette('pastel'), autopct='%.0f%%',
               textprops={'fontsize': 5})
    plt.title(f"{p} - {S_local}")

plt.savefig('D:/TheseSwansea/SFT/Figures/CommunityCompo.png', dpi = 400, bbox_inches = 'tight')


for p in range(P):

    ind = p + 1
        
    S_local = np.unique(FW15_init_normal['S_local'][(FW15_init_normal['sim'] == s) & (FW15_init_normal['patch'] == p)])
    plt.subplot(3, 5, ind)
    plt.pie(data = res15_init_normal[(res15_init_normal['sim'] == s) & (res15_init_normal['B_final'] > 0) & (res15_init_normal['patch'] == p)], 
               x = 'B_final', labels = "sp_ID", colors = sb.color_palette('pastel'), autopct='%.0f%%',
               textprops={'fontsize': 5})
    plt.title(f"{p} - {S_local}")

plt.savefig('D:/TheseSwansea/SFT/Figures/CommunityCompo-beforeInvasion.png', dpi = 400, bbox_inches = 'tight')



# %%% Difference inside outside

fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 6))
sb.boxplot(data = FW15_invasion_normal, y = 'MeanTL_local', x = 'stage', ax = ax1)
ax1.tick_params(axis='x', rotation=45)
ax1.set_title('Before/after')

## showing the mean trophic level inside and outside higher quality patches
sb.boxplot(data = FW15_invasion_normal[FW15_invasion_normal['stage'] == 'restored'], y = 'MeanTL_local', x = 'deltaR', ax = ax2)
ax2.set_title('inside/outside')
ax2.tick_params(axis='x', rotation=45)


## mfcl
fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 6))
sb.boxplot(data = FW15_invasion_normal, y = 'Mfcl_local', x = 'stage', ax = ax1)
ax1.tick_params(axis='x', rotation=45)
ax1.set_title('Before/after')

## showing the mean trophic level inside and outside higher quality patches
sb.boxplot(data = FW15_invasion_normal[FW15_invasion_normal['stage'] == 'restored'], y = 'Mfcl_local', x = 'deltaR', ax = ax2)
ax2.set_title('inside/outside')
ax2.tick_params(axis='x', rotation=45)
