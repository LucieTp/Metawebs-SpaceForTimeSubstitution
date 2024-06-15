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
extentx = [0,0.55]
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


# %%% donut landscape 

## no invasion
P = 15
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3')
het_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/'+i for i in os.listdir() if '.pkl' in i and 'patchImproved' in i]

# run summary statistics function (above)
res15, FW15 = fn.summarise_pop_dynamics(list_files=het_15P_files, nb_patches=P)

FW15['simulation_length_years'] = FW15['simulation_length']/(60*60*24*365)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')


# controls
# initial invasion biomass /10 000
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3')
control_invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/'+i for i in os.listdir() if '.pkl' in i and 'invasion' in i and '-10000-' in i]

res15, FW15 = fn.summarise_pop_dynamics(list_files=control_invasion_15P_files, nb_patches=P)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-10000-seed3-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-10000-seed3-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')



# invasion and restoraton
P = 15
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/donut')
het_15P_files_invasion = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/donut/'+i for i in os.listdir() if '.pkl' in i and 'patchImproved' in i]

# run summary statistics function (above)
res15, FW15 = fn.summarise_pop_dynamics(list_files=het_15P_files_invasion, nb_patches=P)

FW15['simulation_length_years'] = FW15['simulation_length']/(60*60*24*365)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-invasion-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-invasion-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')


# %%% normal landscape 

# intial population dynamics - normal
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3')
init_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]

# run summary statistics function (above)
res15_init, FW15_init = fn.summarise_initial_pop_dynamics(list_files=init_15P_files, nb_patches=P)

FW15_init['simulation_length_years'] = FW15_init['simulation_length']/(60*60*24*365)

res15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')



# %%%% peripheric patches being invaded 

# control

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3')
control_invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/'+i for i in os.listdir() if '.pkl' in i and 'invasion' in i and '-10000-' not in i]

res15, FW15 = fn.summarise_pop_dynamics(list_files=control_invasion_15P_files, nb_patches=P)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')


#### invasion and restoration
P = 15
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion')
het_15P_files_invasion = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/'+i for i in os.listdir() if '.pkl' in i and 'patchImproved' in i]

# run summary statistics function (above)
res15, FW15 = fn.summarise_pop_dynamics(list_files=het_15P_files_invasion, nb_patches=P)

FW15['simulation_length_years'] = FW15['simulation_length']/(60*60*24*365)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-invasion_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-invasion-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C.csv')

# %%%% corner patch being invaded 


# Control
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3')
control_invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/'+i for i in os.listdir() if '.pkl' in i and 'invasion' in i and '-10000-' not in i and 'CornerPatch' in i]

res15, FW15 = fn.summarise_pop_dynamics(list_files=control_invasion_15P_files, nb_patches=P, 
                                        initial_files=init_15P_files)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-CornerPatch-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-CornerPatch-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')


# restoration
P = 15
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch')
het_15P_files_invasion = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/'+i for i in os.listdir() if '.pkl' in i and 'patchImproved' in i]

# run summary statistics function (above)
res15, FW15 = fn.summarise_pop_dynamics(list_files=het_15P_files_invasion, nb_patches=P, 
                                        initial_files=init_15P_files)

FW15['simulation_length_years'] = FW15['simulation_length']/(60*60*24*365)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-invasion-CornerPatch_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-invasion-CornerPatch-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C.csv')


# %% Load datasets

P = 15
## 10P homogeneous
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')

# %%% donut 

res15_init = pd.read_csv(f'ResultsInitial-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_init = pd.read_csv(f'ResultsInitial-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_init['landscape'] = 'donut'
FW15_init['landscape'] = 'donut'
res15_init['stage'] = 'init'
FW15_init['stage'] = 'init'


res15 = pd.read_csv(f'ResultsHeterogeneous-seed3-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15 = pd.read_csv(f'ResultsHeterogeneous-seed3-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15['landscape'] = 'donut'
FW15['landscape'] = 'donut'
res15['stage'] = 'restored'
FW15['stage'] = 'restored'

res15 = pd.concat([res15, res15_init])
FW15 = pd.concat([FW15, FW15_init])

############################

res15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-invasion-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-invasion-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_invasion['landscape'] = 'donut'
FW15_invasion['landscape'] = 'donut'
res15_invasion['stage'] = 'restored'
FW15_invasion['stage'] = 'restored'

res15_control_invasion = pd.read_csv(f'Control-Invasion-seed3-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_control_invasion = pd.read_csv(f'Control-Invasion-seed3-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_control_invasion['landscape'] = 'donut'
FW15_control_invasion['landscape'] = 'donut'
res15_control_invasion['stage'] = 'before'
FW15_control_invasion['stage'] = 'before'

res15_invasion_donut = pd.concat([res15_control_invasion, res15_invasion[np.isin(res15_invasion['sim'], np.unique(res15_control_invasion['sim']))]])
FW15_invasion_donut = pd.concat([FW15_control_invasion, FW15_invasion[np.isin(FW15_invasion['sim'], np.unique(FW15_control_invasion['sim']))]])


res15_control_invasion_10000 = pd.read_csv(f'Control-Invasion-10000-seed3-donut-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_control_invasion_10000 = pd.read_csv(f'Control-Invasion-10000-seed3-donut-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')

res15_control_invasion_10000['invader_pop'] = '10000'
FW15_control_invasion_10000['invader_pop'] = '10000'

res15_control_invasion_10000 = res15_control_invasion_10000[np.isin(res15_control_invasion_10000['sim'], np.unique(res15_control_invasion['sim']))]
res15_control_invasion = res15_control_invasion[np.isin(res15_control_invasion['sim'], np.unique(res15_control_invasion_10000['sim']))]

np.unique(res15_control_invasion['sim'])
np.unique(res15_control_invasion_10000['sim'])

res15_control_invasion['invader_pop'] = '100'
FW15_control_invasion['invader_pop'] = '100'

res15_control = pd.concat([res15_control_invasion, res15_control_invasion_10000])
FW15_control = pd.concat([FW15_control_invasion, FW15_control_invasion_10000])

palette_colors = sb.color_palette("colorblind", 10)
sb.palplot(palette_colors)


# %%% normal landscape

res15_init_normal = pd.read_csv(f'ResultsInitial-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_init_normal = pd.read_csv(f'ResultsInitial-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_init_normal['landscape'] = 'normal'
FW15_init_normal['landscape'] = 'normal'
res15_init_normal['stage'] = 'init'
FW15_init_normal['stage'] = 'init'


# %%%% Peripheric patches



os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')

res15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-invasion_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-invasion-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C.csv')
res15_invasion['landscape'] = 'donut'
FW15_invasion['landscape'] = 'donut'
res15_invasion['stage'] = 'restored'
FW15_invasion['stage'] = 'restored'


res15_control_invasion_normal = pd.read_csv(f'Control-Invasion-seed3-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_control_invasion_normal = pd.read_csv(f'Control-Invasion-seed3-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_control_invasion_normal['landscape'] = 'donut'
FW15_control_invasion_normal['landscape'] = 'donut'
res15_control_invasion_normal['stage'] = 'before'
FW15_control_invasion_normal['stage'] = 'before'

res15_invasion_normal = pd.concat([res15_control_invasion_normal, res15_invasion[np.isin(res15_invasion['sim'], np.unique(res15_control_invasion_normal['sim']))]])
FW15_invasion_normal = pd.concat([FW15_control_invasion_normal, FW15_invasion[np.isin(FW15_invasion['sim'], np.unique(FW15_control_invasion_normal['sim']))]])



# %%%% Corner Patch


### corner patch- control

res15_control_invasion_corner = pd.read_csv(f'Control-Invasion-seed3-CornerPatch-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
FW15_control_invasion_corner = pd.read_csv(f'Control-Invasion-seed3-CornerPatch-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')
res15_control_invasion_corner['landscape'] = 'donut'
FW15_control_invasion_corner['landscape'] = 'donut'
res15_control_invasion_corner['stage'] = 'before'
FW15_control_invasion_corner['stage'] = 'before'


### corner patch- invasion

res15_invasion_corner = pd.read_csv(f'ResultsHeterogeneous-seed3-invasion-CornerPatch_sim_{P}Patches_{Stot}sp_{C}C.csv')
FW15_invasion_corner = pd.read_csv(f'ResultsHeterogeneous-seed3-invasion-CornerPatch-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C.csv')
res15_invasion_corner['landscape'] = 'donut'
FW15_invasion_corner['landscape'] = 'donut'
res15_invasion_corner['stage'] = 'restored'
FW15_invasion_corner['stage'] = 'restored'

res15_invasion_corner = pd.concat([res15_control_invasion_corner, res15_invasion_corner[np.isin(res15_invasion_corner['sim'], np.unique(res15_control_invasion_corner['sim']))]])
FW15_invasion_corner = pd.concat([FW15_control_invasion_corner, FW15_invasion_corner[np.isin(FW15_invasion_corner['sim'], np.unique(FW15_control_invasion_corner['sim']))]])










## biomass
ax = sb.stripplot(data = res15_control[res15_control['Invaders']], y = 'B_final', x = 'sim', hue = 'invader_pop', dodge = True, palette = palette_colors)
sb.pointplot(data = res15_control[res15_control['Invaders']], y = 'B_final', x = 'invader_pop', join=False, dodge = .8, scale = 0.5, palette = ['black'], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])





# %% Plots




## biomass
ax = sb.stripplot(data = res15, y = 'B_final', x = 'patch', hue = 'nb_improved', dodge = True, palette = palette_colors,
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5])
sb.pointplot(data = res15, y = 'B_final', x = 'patch', hue = 'nb_improved', join=False, dodge = .8, scale = 0.5, palette = ['black'],
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])

plt.savefig('D:/TheseSwansea/SFT/Figures/BiomassNbImproved.png', dpi = 400, bbox_inches = 'tight')


## species richness
ax = sb.stripplot(data = FW15, y = 'S_local', x = 'patch', hue = 'nb_improved', dodge = True, palette = palette_colors,
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5])
sb.pointplot(data = FW15, y = 'S_local', x = 'patch', hue = 'nb_improved', join=False, dodge = .8, scale = 0.5, palette = ['black'],
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])

plt.savefig('D:/TheseSwansea/SFT/Figures/SpeciesRichnessNbImproved.png', dpi = 400, bbox_inches = 'tight')


## mean TL
ax = sb.stripplot(data = FW15, y = 'MeanTL_local', x = 'patch', hue = 'nb_improved', dodge = True, palette = palette_colors,
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5])
sb.pointplot(data = FW15, y = 'MeanTL_local', x = 'patch', hue = 'nb_improved', join=False, dodge = .8, scale = 0.5, palette = ['black'],
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])

plt.savefig('D:/TheseSwansea/SFT/Figures/MeanTLNbImproved.png', dpi = 400, bbox_inches = 'tight')




### landscape level

## species richness
ax = sb.stripplot(data = FW15, y = 'S_local', x = 'nb_improved', hue = 'deltaR', dodge = True, palette = palette_colors,
             hue_order=[0.5,1.5], order = [0,1,2,3,4,5])
sb.pointplot(data = FW15, y = 'S_local', x = 'nb_improved', hue = 'deltaR', join=False, dodge = .3, scale = 0.5, palette = ['black'],
             hue_order=[0.5,1.5], order = [0,1,2,3,4,5], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])


## biomass
ax = sb.stripplot(data = res15, y = 'B_final', x = 'nb_improved', hue = 'deltaR', dodge = True, palette = palette_colors,
             hue_order=[0.5,1.5], order = [0,1,2,3,4,5])
sb.pointplot(data = res15, y = 'B_final', x = 'nb_improved', hue = 'deltaR', join=False, dodge = .3, scale = 0.5, palette = ['black'],
             hue_order=[0.5,1.5], order = [0,1,2,3,4,5], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])


# %%% plots invasion

res15_invasion.groupby(['sim','nb_improved']).agg({'B_final':['mean', np.std]})

## biomass
ax = sb.stripplot(data = res15_invasion, y = 'B_final', x = 'patch', hue = 'nb_improved', dodge = True, palette = palette_colors,
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5])
sb.pointplot(data = res15_invasion, y = 'B_final', x = 'patch', hue = 'nb_improved', join=False, dodge = .8, scale = 0.5, palette = ['black'],
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])



## species richness
ax = sb.stripplot(data = FW15_invasion, y = 'S_local', x = 'patch', hue = 'nb_improved', dodge = True, palette = palette_colors,
             order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5])
# sb.pointplot(data = FW15_invasion, y = 'S_local', x = 'patch', hue = 'nb_improved', join=False, dodge = .8, scale = 0.5, palette = ['black'],
#              order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], hue_order = [0,1,2,3,4,5], ax = ax)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])







## Landscape-level - normal landscape
    
## Landscape-level species richness
ax = sb.stripplot(data = FW15_invasion_normal, y = 'S_local', hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data = FW15_invasion_normal, y = 'S_local', hue = 'patch', x = 'nb_improved', scale = 0.5,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])



## mean TL
ax = sb.stripplot(data = FW15_invasion, y = 'MeanTL_local', hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data = FW15_invasion, y = 'MeanTL_local', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = ['black'],
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,0.8])

sb.pointplot(data = FW15_invasion, y = 'MeanTL_local', hue = 'patch', x = 'nb_improved', scale = 0.5,
              errorbar = None)
plt.legend(bbox_to_anchor = [1,0.8])


## species compoisiton across patches
plt.yscale('log')
sb.barplot(data = res15_control_invasion_normal[(res15_control_invasion_normal['sim'] == 0) & (res15_control_invasion_normal['B_final'] > 0)], 
           y = "B_final", x = "sp_ID", hue = "patch")


plt.yscale('log')
sb.barplot(data = res15_control_invasion_normal[(res15_control_invasion_normal['sim'] == 2) & (res15_control_invasion_normal['B_final'] > 0)], 
           y = "B_final", x = "sp_ID", hue = "patch")




plt.yscale('log')
sb.barplot(data = res15_control_invasion[(res15_control_invasion['sim'] == 2) & (res15_control_invasion['B_final'] > 0)], 
           y = "B_final", x = "sp_ID", hue = "patch")



# %%% Corner landscape

## Landscape-level:
    
## Landscape-level species richness
ax = sb.stripplot(data = FW15_invasion_corner, y = 'S_local', hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data = FW15_invasion_corner, y = 'S_local', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = palette_colors,
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

sb.pointplot(data = FW15_invasion_corner[FW15_invasion_corner['patch'] == 9], y = 'S_local', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = palette_colors,
              errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])


## Number of invasions
ax = sb.stripplot(data = FW15_invasion_corner, y = 'nb_invaders', hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data = FW15_invasion_corner, y = 'nb_invaders', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])


## Number of invasions
ax = sb.stripplot(data = FW15_invasion_corner, y = 'nb_extinct', hue = 'patch', x = 'nb_improved', palette = palette_colors)
sb.pointplot(data = FW15_invasion_corner, y = 'nb_extinct', hue = 'patch', x = 'nb_improved', style = 'position', scale = 0.5, palette = palette_colors,
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])


100 - FW15_invasion_corner['nb_extinct']


## Number of invasions - relative to initial population
ax = sb.stripplot(data = FW15_invasion_corner, y = 'nb_invaders_initial_pop', hue = 'patch', x = 'nb_improved', palette = sb.color_palette(cc.glasbey, 15))
sb.pointplot(data = FW15_invasion_corner, y = 'nb_invaders_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = sb.color_palette(cc.glasbey, 15),
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])

## Number of extinctions - relative to initial population
ax = sb.stripplot(data = FW15_invasion_corner, y = 'nb_extinct_initial_pop', hue = 'patch', x = 'nb_improved', palette = sb.color_palette(cc.glasbey, 15))
sb.pointplot(data = FW15_invasion_corner, y = 'nb_extinct_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = sb.color_palette(cc.glasbey, 15),
              ax = ax, errorbar = None)
plt.setp(ax.lines, zorder=100) # to have the pointplot on top
plt.setp(ax.collections, zorder=100)

ax.legend(bbox_to_anchor = [1,1])

## invaded patch

## here the invaded patch gained two species, with 5 extinctions and 7 invasions
FW15_invasion_corner[FW15_invasion_corner['patch'] == 9]['S_local']

sb.pointplot(data = FW15_invasion_corner[FW15_invasion_corner['patch'] == 9], y = 'nb_extinct_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = sb.color_palette(cc.glasbey, 15),
              errorbar = None)
sb.pointplot(data = FW15_invasion_corner[FW15_invasion_corner['patch'] == 9], y = 'nb_invaders_initial_pop', hue = 'patch', x = 'nb_improved', scale = 0.5, palette = sb.color_palette(cc.glasbey, 15),
              errorbar = None)


# %%% Change in species composition
s = 3

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
        
    S_local = np.unique(FW15_control_invasion_corner['S_local'][(FW15_control_invasion_corner['sim'] == s) & (FW15_control_invasion_corner['patch'] == p)])
    plt.subplot(3, 5, ind)
    plt.pie(data = res15_control_invasion_corner[(res15_control_invasion_corner['sim'] == s) & (res15_control_invasion_corner['B_final'] > 0) & (res15_control_invasion_corner['patch'] == p)], 
               x = 'B_final', labels = "sp_ID", colors = sb.color_palette('pastel'), autopct='%.0f%%',
               textprops={'fontsize': 5})
    plt.title(f"{p} - {S_local}")

plt.savefig('D:/TheseSwansea/SFT/Figures/CommunityCompo-CornerPatch.png', dpi = 400, bbox_inches = 'tight')


for p in range(P):

    ind = p + 1
        
    S_local = np.unique(FW15_init_normal['S_local'][(FW15_init_normal['sim'] == s) & (FW15_init_normal['patch'] == p)])
    plt.subplot(3, 5, ind)
    plt.pie(data = res15_init_normal[(res15_init_normal['sim'] == s) & (res15_init_normal['B_final'] > 0) & (res15_init_normal['patch'] == p)], 
               x = 'B_final', labels = "sp_ID", colors = sb.color_palette('pastel'), autopct='%.0f%%',
               textprops={'fontsize': 5})
    plt.title(f"{p} - {S_local}")

plt.savefig('D:/TheseSwansea/SFT/Figures/CommunityCompo-beforeInvasion.png', dpi = 400, bbox_inches = 'tight')


