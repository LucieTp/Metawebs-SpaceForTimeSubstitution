# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:36:41 2024

@author: lucie.thompson

Exploration of space for time substitution in the context of metafoodwebs

"""




#%% Loading modules and global variables

import pandas as pd # data frames

import matplotlib.pyplot as plt  # plotting

import numpy as np # Numerical objects

import os # changing directory

import seaborn as sb # plotting

import re # regular expressions

os.chdir('D:/TheseSwansea/Patch-Models/Script')
import FunctionsAnalysis as fn # functions from Patch-Models project 
import FunctionsPatchModel as fnPM # functions to run simulations

Stot = 100
C = 0.1
d = 1e-8


# %% 9 PATCH SCENARIO

P = 9
## 10P homogeneous
os.chdir('D:/TheseSwansea/Patch-Models/outputs/9Patches/Homogeneous')

res9_sp_homogeneous = pd.read_csv(f'ResultsDisturbed-handmade_sim_{P}Patches_{Stot}sp_{C}C_homogeneous_04032024.csv')
FW9_metrics_homogeneous = pd.read_csv(f'ResultsDisturbed-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_homogeneous_04032024.csv')

res9_init_homogeneous = pd.read_csv(f'ResultsInitial-handmade_sim_{P}Patches_{Stot}sp_{C}C_homogeneous_04032024.csv')
FW9_init_homogeneous = pd.read_csv(f'ResultsInitial-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_homogeneous_04032024.csv')


# 10P heteroegeneous
os.chdir('D:/TheseSwansea/Patch-Models/outputs/9Patches/Heterogeneous')

res9_sp_heterogeneous = pd.read_csv(f'ResultsDisturbed-handmade_{P}Patches_{Stot}sp_{C}C_Heterogeneous_04032024.csv')
FW9_metrics_heterogeneous = pd.read_csv(f'ResultsDisturbed-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_Heterogeneous_04032024.csv')

res9_init_heterogeneous = pd.read_csv(f'ResultsInitial-handmade_sim_{P}Patches_{Stot}sp_{C}C_heterogeneous_04032024.csv')
FW9_init_heterogeneous = pd.read_csv(f'ResultsInitial-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_heterogeneous_04032024.csv')



# (1) MERGE HOMOGENEOUS AND HETEROEGENEOUS RESULTS

res9 = pd.concat([res9_sp_homogeneous, res9_sp_heterogeneous])
res9_init = pd.concat([res9_init_heterogeneous, res9_init_homogeneous])

# (2) set negative biomasses to 0 (extinct species)
res9.loc[res9['B_final'] < 0, 'B_final'] = 0
res9_init.loc[res9_init['B_final'] < 0, 'B_final'] = 0

# (2.1) some species had negative initial biomass (went extinct immediately) - need to check why
# we remove them from further analysis as they should not have been part anyway
res9 = res9.loc[res9['B_init'] >= 0,:]

# (3) create 'ratio' column to describe the change in biomass before and after disturbance
res9['ratio'] = res9['B_final']/res9['B_init']
res9['ratio'].describe() 
## NANs come from instances where the species was absent both before and after (0/0), 
## we replace those with ones (biomass did not change):
res9['ratio'] = res9['ratio'].fillna(1)
## there are infs which are species that were absent in the initial community but were able to invade after 
## for those species, we change the 'zero' biomass to a very small value 0.0000001 so as to avoid division by zero
res9.loc[res9['ratio'].isin([np.inf]),"ratio"] = res9['B_final'][res9['ratio'].isin([np.inf])]/(res9['B_init'][res9['ratio'].isin([np.inf])] + 0.0000001)

# (8) remove null ratios (as we are logging, this messes up the plots)
res9['ratio_plot'] = res9['ratio'] + 1e-10 # so as to have no null values

# (9) keep simulations that we were able to run both for the 10 patch and 3 patch scenari
# res3 = res3[np.isin(res3['sim'], np.unique(res10['sim']))] 
# res3_init = res3_init[np.isin(res3_init['sim'], np.unique(res10['sim']))] 

# check
res9.groupby(['sim','patch_disturbed','type']).agg({'ratio':['mean', np.std]})

# (2) combine homogeneous and heterogeneous scenarios
FW9 = pd.concat([FW9_metrics_homogeneous, FW9_metrics_heterogeneous])
FW9_init = pd.concat([FW9_init_homogeneous, FW9_init_heterogeneous])



############## patch coordinates
P = 9
extentx = [0,0.4]
coords9P = fn.create_landscape(P=9,extent=extentx)

dist9P = fn.get_distance(coords9P)

############## SHORTEST DISTANCE
init_files = []
# HOMOGENEOUS
os.chdir('D:/TheseSwansea/Patch-Models/outputs/9Patches/Homogeneous')
init_files = init_files + ['D:/TheseSwansea/Patch-Models/outputs/9Patches/Homogeneous/' + i for i in os.listdir() if '.npy' in i and 'InitialPopDynamics' in i]

# HETEROGENEOUS
os.chdir('D:/TheseSwansea/Patch-Models/outputs/9Patches/Heterogeneous')
init_files = init_files + ['D:/TheseSwansea/Patch-Models/outputs/9Patches/Heterogeneous/' + i for i in os.listdir() if '.npy' in i and 'InitialPopDynamics' in i]


### calculate shortest distance
P = 9
Stot = 100
disp = np.repeat(d, Stot*P).reshape(P,Stot)
shortest_distance9P = pd.DataFrame()
for f in init_files:
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
    shortest_distance_temp_0 = fn.calculate_shortest_path(nb_patch=9, start_patch=0, dist=dist9P, FW=FW_init)
    shortest_distance_temp_1 = fn.calculate_shortest_path(nb_patch=9, start_patch=1, dist=dist9P, FW=FW_init)
    shortest_distance_temp_3 = fn.calculate_shortest_path(nb_patch=9, start_patch=3, dist=dist9P, FW=FW_init)

    shortest_distance_temp = pd.concat([shortest_distance_temp_0, shortest_distance_temp_1])
    shortest_distance_temp = pd.concat([shortest_distance_temp, shortest_distance_temp_3])

    shortest_distance_temp = shortest_distance_temp.groupby(['Patch']).agg({'shortest_distance':'min',
                                                                            'path_length':'min'})
    shortest_distance_temp = shortest_distance_temp.reset_index(['Patch'])

    shortest_distance_temp['file_init'] = f
    shortest_distance_temp['Stot_init'] = Stot_init
    shortest_distance_temp['sim'] = re.search(r'sim(\d+)', f).group(1)
    shortest_distance_temp['quality_ratio'] = quality_ratio
    
    shortest_distance9P = pd.concat([shortest_distance9P, shortest_distance_temp])


coords9P = pd.merge(coords9P, shortest_distance9P, on = 'Patch')
# (4) Euclidian distance to disturbed patch (9) and scatter distance measure to look better when plotting
coords9P['distance'] = round(((coords9P['x'] - coords9P['x'].iloc[9])**2 + (coords9P['y'] - coords9P['y'].iloc[9])**2)**0.5, 2)


plt.scatter(coords9P['y'], coords9P['x'], c = coords9P['shortest_distance'])
plt.colorbar()

## add distance info to FW and res 
FW9['sim'] = FW9['sim'].astype(int)
coords9P['sim'] = coords9P['sim'].astype(int)
FW9 = FW9.merge(coords9P, left_on = ['sim', 'patch','quality_ratio'], right_on = ['sim', 'Patch','quality_ratio'])
FW9_init = FW9_init.merge(coords9P, left_on = ['sim', 'patch','quality_ratio'], right_on = ['sim', 'Patch','quality_ratio'])


res9 = res9.merge(coords9P, left_on = ['sim', 'patch','quality_ratio'], right_on = ['sim', 'Patch','quality_ratio'])
res9_init = res9_init.merge(coords9P, left_on = ['sim', 'patch','quality_ratio'], right_on = ['sim', 'Patch','quality_ratio'])


res9 = res9.loc[res9['sim'].isin([0,1,2,3,7,9]),:] ## keep only simulations where we have heterogeneous and homogeneous results
FW9 = FW9.loc[FW9['sim'].isin([0,1,2,3,7,9]),:] ## keep only simulations where we have heterogeneous and homogeneous results
FW9_init = FW9_init.loc[FW9_init['sim'].isin([0,1,2,3,7,9]),:] ## keep only simulations where we have heterogeneous and homogeneous results
res9_init = res9_init.loc[res9_init['sim'].isin([0,1,2,3,7,9]),:] ## keep only simulations where we have heterogeneous and homogeneous results



#%% Comparing inside/outside and before/after

## here we compare independant simulations for the before/after scenario,
## need to rerun by restoring the landscape post initial condiitons so that 
## species from the original regional pool can reinvade etc


#%%% mean trophic level 

## showing the mean trophic level before and after disturbance in disturbed 
## although really here there the simulations are independant

FW9['quality_ratio_labels'] = FW9['quality_ratio']
FW9['quality_ratio_labels'] = FW9['quality_ratio_labels'].map({1.0: 'Homogeneous', 0.5: 'Intermediate heterogeneity', 0.3: 'Max heterogeneity'})

FW9['deltaR_labels'] = FW9['deltaR']
FW9['deltaR_labels'] = FW9['deltaR_labels'].map({1.0: 'High quality patch', 0.5: 'Low quality patch'})


res9['quality_ratio_labels'] = res9['quality_ratio']
res9['quality_ratio_labels'] = res9['quality_ratio_labels'].map({1.0: 'Homogeneous', 0.5: 'Intermediate heterogeneity', 0.3: 'Max heterogeneity'})

res9['deltaR_labels'] = res9['deltaR']
res9['deltaR_labels'] = res9['deltaR_labels'].map({1.0: 'High quality patch', 0.5: 'Low quality patch'})


fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 6))
sb.boxplot(data = FW9[np.in1d(FW9['patch'],[0,1,3])], y = 'MeanTL_local', x = 'quality_ratio_labels', ax = ax1)
ax1.tick_params(axis='x', rotation=45)
ax1.set_title('Before/after')

## showing the mean trophic level inside and outside higher quality patches
sb.boxplot(data = FW9[FW9['quality_ratio'] == 0.5], y = 'MeanTL_local', x = 'deltaR_labels', ax = ax2)
ax2.set_title('inside/outside')
ax2.tick_params(axis='x', rotation=45)

# plt.savefig('D:/TheseSwansea/SFT/Figures/MeanTL_InitialDynamics-local-3P-10P.png', dpi = 400, bbox_inches = 'tight')


#%%% mean biomass

fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 6))
ax1.set_yscale('log')
sb.boxplot(data = res9[np.in1d(res9['patch'],[0,1,3])], y = 'B_final', x = 'quality_ratio_labels', ax = ax1)
ax1.set_title('Before/after')
ax1.tick_params(axis='x', rotation=45)

ax2.set_yscale('log')
sb.boxplot(data = res9[res9['quality_ratio'] == 0.5], y = 'B_final', x = 'deltaR_labels', ax = ax2)
ax2.set_title('inside/outside')
ax2.tick_params(axis='x', rotation=45)
# plt.savefig('D:/TheseSwansea/SFT/Figures/MeanTL_InitialDynamics-local-3P-10P.png', dpi = 400, bbox_inches = 'tight')

#%%% mean connectance

fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 6))
sb.boxplot(data = FW9[np.in1d(FW9['patch'],[0,1,3])], y = 'S_local', x = 'quality_ratio', order = [1.0, 0.5, 0.3], ax = ax1)
ax1.set_title('Before/after')

sb.boxplot(data = FW9[FW9['quality_ratio'] == 0.5], y = 'S_local', x = 'deltaR', ax = ax2)
ax2.set_title('inside/outside')

# plt.savefig('D:/TheseSwansea/SFT/Figures/MeanTL_InitialDynamics-local-3P-10P.png', dpi = 400, bbox_inches = 'tight')



#%% Look at the shape of food web metrics as you go further from the high quality patches

## need to get distance to high quality patches [0,1,3]

FW9_init['quality_ratio_labels'] = FW9_init['quality_ratio']
FW9_init['quality_ratio_labels'] = FW9_init['quality_ratio_labels'].map({1.0: 'Homogeneous', 0.5: 'Intermediate heterogeneity', 0.3: 'Max heterogeneity'})

FW9_init['deltaR_labels'] = FW9_init['deltaR']
FW9_init['deltaR_labels'] = FW9_init['deltaR_labels'].map({1.0: 'High quality patch', 0.5: 'Low quality patch'})


## it seems that regardless of heterogeneity, patch isolation plays a strong role in how
## many species can survive in a patch
sb.boxplot(x='shortest_distance', y = 'S_local', hue = "quality_ratio_labels", data=FW9_init) # remove the points' default edges 

plt.savefig('D:/TheseSwansea/SFT/Figures/SpeciesRichness-DistanceHighQualityPatches.png', dpi = 400, bbox_inches = 'tight')


sb.scatterplot(x='shortest_distance', y = 'MeanTL_local', hue = "quality_ratio", data=FW9_init,
              edgecolor='none', s = 15, hue_order = [0.5, 1.0]) # remove the points' default edges 

plt.savefig('D:/TheseSwansea/SFT/Figures/TrophicLevel-DistanceHighQualityPatches.png', dpi = 400, bbox_inches = 'tight')

### biomass change as you get further from high quality central patches?
sb.scatterplot(x='shortest_distance', y = 'B_final', hue = "quality_ratio", data=res9_init,
              edgecolor='none', s = 15, hue_order = [0.5, 1.0]) # remove the points' default edges 

plt.savefig('D:/TheseSwansea/SFT/Figures/FinalBiomass-DistanceHighQualityPatches.png', dpi = 400, bbox_inches = 'tight')


