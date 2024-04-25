# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:08:31 2023

@author: lucie.thompson
"""


# %% Load modules
###############################################################################
### SIMPLE 3 patches simulation - O - O - O
## 1 - No protected area O - O - O
## 2 - One protected area in the middle O - (O) - O

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

import random

import re

os.chdir('D:/TheseSwansea/Patch-Models/Script')
import FunctionsAnalysis as fn
import FunctionsPatchModel as fnPM # functions to run simulations


#%% Investigating the role of PA configuration 

## does PA configuration (i.e. having the PA grouped in the middle for example) affects
## the rest of the landscape


## these are the results starting off from a homogeneous landscape and restoring part of the landscape
os.chdir("D:\TheseSwansea\Patch-Models\outputs\9Patches\HighQualityConfig")

res9_quality = pd.DataFrame()
FW9_quality = pd.DataFrame()

P = 9

for rdseed in np.arange(10):
    
    files_quality = [i for i in os.listdir() if '.npy' in i and f"seed{rdseed}" in i]
    
    extentx = [0,0.4]
    coords = pd.DataFrame()
    for i in range(P):
        np.random.seed(i)
        coords = pd.concat((coords, pd.DataFrame({'Patch':[i],'x':[np.random.uniform(extentx[0], extentx[1])],
                                  'y':[np.random.uniform(extentx[0], extentx[1])]})))

    random.seed(int(rdseed)) # set seed for repeatability
    high_qual_patches = random.sample(range(P),3)
    
    print(high_qual_patches)

    deltaR = np.repeat(0.5,P)
    deltaR[high_qual_patches] = 1
    
    coords['deltaR'] = deltaR
    
    sb.scatterplot(coords, x = 'x', y = 'y', hue = 'deltaR')
    plt.title(high_qual_patches)
    plt.show()
    
    
    if len(files_quality)>0:
        
        res9_temp, FW9_temp = fn.summarise_initial_pop_dynamics(list_files=files_quality, nb_patches=9)
        res9_temp['seed'] = rdseed
        FW9_temp['seed'] = rdseed
        
        res9_quality = pd.concat([res9_quality, res9_temp])
        FW9_quality = pd.concat([FW9_quality, FW9_temp])
        
        
    
        
        
        
        
FW9_quality.groupby(['sim','type','seed']).agg({'MeanTL_local':['mean', np.std]})




## each seed corresponds to a random positioning of the high quality patches
sb.boxplot(data = FW9_quality, y = 'MeanTL_local', x = 'deltaR', hue = 'seed')
plt.title('9 Patches')
plt.ylabel('Mean trophic level')
plt.xlabel('Patch quality')
plt.legend(title='seed')
plt.show()

# average per patch type
sb.boxplot(data = FW9_quality, y = 'MeanTL_local', x = 'deltaR')
plt.title('9 Patches')
plt.ylabel('Mean trophic level')
plt.xlabel('Patch quality')
plt.show()

sb.boxplot(data = FW9_quality, y = 'S_local', x = 'deltaR')
plt.title('9 Patches')
plt.ylabel('Mean trophic level')
plt.xlabel('Patch quality')
plt.show()

sb.boxplot(data = FW9_quality, y = 'S_local', x = 'deltaR', hue = 'seed')
plt.title('9 Patches')
plt.ylabel('Species richness')
plt.xlabel('Patch quality')
plt.legend(title='seed')
plt.show()


# %%% look at the changes that occur in the restored patches

res9_quality['ratio'] = res9_quality['B_final']/res9_quality['B_init']
res9_quality['ratio'].describe()

## if all zeros then can proceed
res9_quality[np.isnan(res9_quality['ratio'])]['B_final'].describe()
res9_quality[np.isnan(res9_quality['ratio'])]['B_init'].describe()

res9_quality['ratio'] = res9_quality['ratio'].fillna(1)
## there are infs which are species that were absent in the initial community but were able to invade after 
## for those species, we change the 'zero' biomass to a very small value 0.0000001 so as to avoid division by zero
res9_quality.loc[res9_quality['ratio'].isin([np.inf]),"ratio"] = res9_quality['B_final'][res9_quality['ratio'].isin([np.inf])]/(res9_quality['B_init'][res9_quality['ratio'].isin([np.inf])] + 0.0000001)


plt.yscale('log')
sb.boxplot(data = res9_quality, y = 'ratio', x = 'deltaR', hue = 'sim')
plt.title('9 Patches')
plt.ylabel('Ratio of biomass change')
plt.xlabel('Patch quality')
plt.legend(title='seed')
plt.show()

plt.yscale('log')
sb.boxplot(data = res9_quality, y = 'ratio', x = 'deltaR')
plt.title('9 Patches')
plt.ylabel('Ratio of biomass change')
plt.xlabel('Patch quality')
plt.show()


# %%% look at the role of centrality or distance to edges

############## SHORTEST DISTANCE
init_files = []
# HOMOGENEOUS
os.chdir('D:/TheseSwansea/Patch-Models/outputs/9Patches/Homogeneous')
init_files = init_files + ['D:/TheseSwansea/Patch-Models/outputs/9Patches/Homogeneous/' + i for i in os.listdir() if '.npy' in i and 'InitialPopDynamics' in i and 
                           ('sim0_' in i or 'sim1_' in i or 'sim2_' in i or 'sim3_' in i or 'sim4_' in i or 'sim5_' in i)]

### calculate shortest distance to higher quality patches
P = 9
C = 0.1
d = 1e-8
Stot = 100

P = 9
extentx = [0,0.4]
coords9P = fn.create_landscape(P=9,extent=extentx)

dist9P = fn.get_distance(coords9P)

disp = np.repeat(d, Stot*P).reshape(P,Stot)
shortest_distance9P = pd.DataFrame()
for f in init_files:
    
    for rdseed in np.arange(10):
        
        high_qual_patches = random.sample(range(P),3)
        
        sol_init = np.load(f, allow_pickle=True).item()
        FW = sol_init['FW']
        Stot_new, FW_new, disp_new = fnPM.reduce_FW(FW, FW['y0'], P, disp)
        deltaR = sol_init['deltaR']
        
        quality_ratio = min(deltaR)/max(deltaR)
    
        Bf = np.zeros(shape = (P,Stot_new))
        for patch in range(P):
            
            p_index = patch + 1
            
            # sol_ivp_k1["y"][sol_ivp_k1["y"]<1e-20] = 0         
            solT = sol_init['t']
            solY = sol_init['y']
            ind = patch + 1
            
            Bf[patch] = solY[-1,range(Stot_new*ind-Stot_new,Stot_new*ind)]
                
        Stot_init, FW_init, disp_init = fnPM.reduce_FW(FW_new, Bf, P, disp_new)
        shortest_distance_temp_0 = fn.calculate_shortest_path(nb_patch=9, start_patch=high_qual_patches[0], dist=dist9P, FW=FW_init)
        shortest_distance_temp_1 = fn.calculate_shortest_path(nb_patch=9, start_patch=high_qual_patches[1], dist=dist9P, FW=FW_init)
        shortest_distance_temp_3 = fn.calculate_shortest_path(nb_patch=9, start_patch=high_qual_patches[2], dist=dist9P, FW=FW_init)
    
        shortest_distance_temp = pd.concat([shortest_distance_temp_0, shortest_distance_temp_1])
        shortest_distance_temp = pd.concat([shortest_distance_temp, shortest_distance_temp_3])
    
        shortest_distance_temp = shortest_distance_temp.groupby(['Patch']).agg({'shortest_distance':'min',
                                                                                'path_length':'min'})
        shortest_distance_temp = shortest_distance_temp.reset_index(['Patch'])
    
        shortest_distance_temp['file_init'] = f
        shortest_distance_temp['rdseed'] = rdseed
        shortest_distance_temp['deltaR'] = deltaR
        shortest_distance_temp['Stot_init'] = Stot_init
        shortest_distance_temp['sim'] = re.search(r'sim(\d+)', f).group(1)
        shortest_distance_temp['quality_ratio'] = quality_ratio
        
        shortest_distance9P = pd.concat([shortest_distance9P, shortest_distance_temp])


coords9P = pd.merge(coords9P, shortest_distance9P, on = 'Patch')
# (4) Euclidian distance to mean coordinate (x = 0.22; y = 0.25)
coords9P['distance'] = round(((coords9P['x'] - np.mean(coords9P['x']))**2 + (coords9P['y'] - np.mean(coords9P['y']))**2)**0.5, 2)



## add distance info to FW and res 
FW9_quality['sim'] = FW9_quality['sim'].astype(int)
coords9P['sim'] = coords9P['sim'].astype(int)

FW9_quality['seed'] = FW9_quality['seed'].astype(int)
coords9P['rdseed'] = coords9P['rdseed'].astype(int)

FW9_quality_dist = FW9_quality.merge(coords9P[['sim','rdseed', 'Patch', 'x', 'y', 'distance', 'shortest_distance']], 
                                left_on = ['sim','patch','seed'], right_on = ['sim', 'Patch','rdseed'])

res9_quality_dist = res9_quality.merge(coords9P[['sim','rdseed', 'Patch', 'x', 'y', 'distance', 'shortest_distance']], 
                                  left_on = ['sim', 'patch','seed'], right_on = ['sim', 'Patch','rdseed'])


sb.scatterplot(x='shortest_distance', y = 'S_local', hue = "deltaR", data=FW9_quality_dist) # remove the points' default edges 

sb.boxplot(x='distance', y = 'S_local', hue = "deltaR", data=FW9_quality_dist) # remove the points' default edges 
plt.savefig('D:/TheseSwansea/Patch-Models/Figures/DsitanceToEdge-PAConfig-quality.png', dpi = 400, bbox_inches = 'tight')






