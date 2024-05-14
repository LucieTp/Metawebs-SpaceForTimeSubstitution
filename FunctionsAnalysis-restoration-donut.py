# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:49:08 2024

@author: lucie.thompson
"""


import pandas as pd

import numpy as np

import re

# import teneto

import matplotlib.pyplot as plt 

import pickle


Stot = 100 # initial pool of species
d = 1e-8



def create_landscape(P, extent):
    
    '''
    Create landscape (x and y coordinates of each patch)
    
    Input the number of patch and extent of the landscape (consider that all lanscapes are 
                                                           square of dimensions LxL, thus extent 
                                                           corresponds to length of one edge of the square (= L))
    

    Randomly draws x and y coordinates from a uniform distribution bounded by the extent given. 
    We seed the random number generation to have the same landscape configuration at each run. 
        
    Returns pandas dataframe with x and y columns and patch ID
    
    '''
    
    coords = pd.DataFrame()
    for i in range(P):
        np.random.seed(i)
        coords = pd.concat((coords, pd.DataFrame({'Patch':[i],'x':[np.random.uniform(extent[0], extent[1])],
                                  'y':[np.random.uniform(extent[0], extent[1])]})))
    
    return(coords)

def get_distance(coords, extent):
    
    '''
    Calculate euclidian distance between patches whose coordinates are 
    recorded in coords (should have an 'x' and 'y' column with coordinates).
    
    We take the smallest distance between the regular euclidian distance and the 
    distance to points on the other side of the landscape (to create a continuous
                                                           landscape which traverses
                                                           edges)
    
    Returns a PxP matrix for all possible pairwise distances.
    
    '''
    
    P = coords.shape[0]
    dist = np.zeros((P,P))
    for p1 in range(P): # patch 1
        for p2 in range(P): # patch 2
            # normal distance
            dist_inside = ((coords['x'].iloc[p2] - coords['x'].iloc[p1])**2 + (coords['y'].iloc[p2] - coords['y'].iloc[p1])**2)**(1/2)
            
            # we calculate the distance to the edge of the landscape for each point to see if they are closer from the 'other side'
            mean_x = np.mean(coords['x'].iloc[[p1,p2]])
            mean_y = np.mean(coords['y'].iloc[[p1,p2]])
            
            dist_horizontal_p1 = (min((coords['x'].iloc[p1] - extent[0])**2,(coords['x'].iloc[p1] - extent[1])**2) + (coords['y'].iloc[p1] - mean_y)**2)**(1/2)
            dist_vertical_p1 = ((coords['x'].iloc[p1] - mean_x)**2 + min((coords['y'].iloc[p1] - extent[0])**2,(coords['y'].iloc[p1] - extent[1])**2))**(1/2)
            dist_p1 = min(dist_horizontal_p1, dist_vertical_p1)
            
            dist_horizontal_p2 = (min((coords['x'].iloc[p2] - extent[0])**2,(coords['x'].iloc[p2] - extent[1])**2) + (coords['y'].iloc[p2] - mean_y)**2)**(1/2)
            dist_vertical_p2 = ((coords['x'].iloc[p2] - mean_x)**2 + min((coords['y'].iloc[p2] - extent[0])**2,(coords['y'].iloc[p2] - extent[1])**2))**(1/2)
            dist_p2 = min(dist_horizontal_p2, dist_vertical_p2)
            
            dist_outside = dist_p1 + dist_p2
            
            dist[p1,p2] = min(dist_inside, dist_outside)
            
    return(dist)




def summarise_pop_dynamics(list_files, nb_patches): ## coords = DataFrame with one 'Patch' col with patch IDs, and coordinates under 'x' and 'y' cols
    
    ## initialise DataFrames to save summary statistics
    FW_metrics = pd.DataFrame()
    res_sp = pd.DataFrame()
    ## initialise index for plotting
    index = 0
    
    for f in list_files:
        
        print(f)
        
        # load file with population dynamics
        if '.npy' in f:
            sol_disturbed = np.load(f, allow_pickle=True).item()
        elif ".pkl" in f:
            with open(f, 'rb') as file: 
                sol_disturbed = pickle.load(file)  
            file.close()
            
        s = sol_disturbed['sim'] ## simulation number
        # FW_og = sol_disturbed['FW'] ## regional food web (this goes with sp_ID)
        FW_new = sol_disturbed['FW_new'] ## subset food web (this goes with sp_ID_)
        Stot_new = FW_new['Stot']
        coords = sol_disturbed['FW_new']['coords'] # get patch coordinates

        patch_improved = sol_disturbed['patch_to_improve']
        
        deltaR = sol_disturbed['deltaR'] # patch qualities
        ratio = np.min(deltaR)/np.max(deltaR) # quality ratio 
        y0 = FW_new['y0'].reshape(nb_patches*Stot_new) # intial population biomass (pre-disturbance)
        
        if ratio == 1:
            scenario_type='homogeneous'
        else:
            scenario_type='heterogeneous'
        
        # Bf1 = np.zeros(shape = (nb_patches,Stot_new)) # post-disturbance pop biomasses
        
        ## go through each patch in turn and summarise food web characteristics and species'
        ## characteristics
        
        solT = sol_disturbed['t'] ## time
        solY = sol_disturbed['y'] ## biomasses
        
        # mean biomass across the last 5% of the time steps
        thresh_high = solT[-1] # get higher time step boundary (tfinal)
        thresh_low = (solT[-1] - (solT[-1] - solT[1])*0.05) # get lower time step boundary (tfinal - 5% * tfinal)

        ## index of those time steps between low and high boundaries
        index_bf = np.where((solT >= thresh_low) & (solT < thresh_high))[0]
        ## biomasses over those 10% time steps
        Bf1 = np.mean(solY[index_bf], axis = 0).reshape(nb_patches,Stot_new)
        
        ## food web metrics for the final food web
        # array of True of False, whether each species went extinct during the experiment
        extinct2 = np.repeat(False,Stot_new) # in any patch
        extinct1 = np.repeat(False,Stot_new*nb_patches) # in a given patch
        if ((FW_new['y0'] > 0) & (Bf1 == 0)).any():
            for j in range(nb_patches):
                extinct2 = extinct2 | (FW_new['y0'][j] > 0) & (Bf1[j] == 0)
                extinct1[j*Stot_new:(j*Stot_new + Stot_new)] = (FW_new['y0'][j] > 0) & (Bf1[j] == 0)
        
        
        prop_regional = Bf1[Bf1 > 0]/np.sum(Bf1[Bf1 > 0])
        S_regional = Bf1.shape[1]
        prop_local_all = Bf1/np.repeat(np.sum(Bf1, axis = 1),S_regional).reshape(nb_patches,S_regional)

    
        for p in range(nb_patches):
       
            ind = p + 1
            
            # Bf1[p] = solY[-1,range(Stot_new*ind-Stot_new,Stot_new*ind)]
            prop_local = Bf1[p][Bf1[p]>0]/np.sum(Bf1[p][Bf1[p]>0])
            
            ### get local food web characteristics
            
            surviving_sp = np.where(Bf1[p]>0)[0]
            local_FW = FW_new['M'][Bf1[p]>0,:][:,Bf1[p]>0]
                    
            
            # summary table at the patch level (9 rows or 3 rows per simulation depending on the landscape)
            FW_metrics = pd.concat([FW_metrics, 
                                   pd.DataFrame({'file':f, 'sim':s, 'patch':p, 'patch_improved':patch_improved, 
                                                 'type':scenario_type, 'quality_ratio':ratio,
                                                 'S_regional':S_regional,
                                                 'S_local':len(surviving_sp), 'C_local': np.sum(local_FW),
                                                 'MeanGen_local': np.mean(np.sum(local_FW, axis = 0)),
                                                 'MeanVul_local': np.mean(np.sum(local_FW, axis = 1)),
                                                 'MeanTL_local': np.mean(FW_new['TL'][surviving_sp]), # mean trophic level of surviving species on patch p
                                                 
                                                 'simulation_length':sol_disturbed['t'][-1],
                                                  
                                                 ## diversity measures

                                                 # landscape level:
                                                 'gamma_diversity_shannon':-np.sum(prop_regional*np.log(prop_regional)), # regional diversity
                                                 'mean_alpha_diversity_shannon':np.mean(-np.sum(np.nan_to_num(prop_local_all*np.log(prop_local_all)), axis = 1)), # mean (landscape-level) alpha diversity 
                                                 'beta_diversity_shannon':(-np.sum(prop_regional*np.log(prop_regional)))/np.mean(-np.sum(np.nan_to_num(prop_local_all*np.log(prop_local_all)), axis = 1)), # gamma/alpha ()

                                                 # patch level
                                                 'alpha_diversity_shannon':-np.sum(prop_local*np.log(prop_local)), # shannon diversity (-sum(pi*ln(pi)))
                                                 
                                                 ## quality ratio
                                                 'deltaR':deltaR[p]
                                                 }, index=[index])])
            index += 1
            
        #     plt.subplot(2, 5, ind)
        #     plt.tight_layout()
            
        #     plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
        #                 solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)])
        #     plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
        #                 solY[np.arange(0,solT.shape[0],10),sp+ Stot_new*(ind-1)], linestyle="dotted")
            
        # plt.title("With events")
        # plt.show()
        
        # ## plotting all dynamics together
        # plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
        #             solY[np.arange(0,solT.shape[0],10),:])
        # plt.show()
        
                
                
        with np.errstate(divide='ignore', invalid='ignore'): # suppress 
        #'D:\TheseSwansea\Patch-Models\Script\FunctionsAnalysis.py:303: RuntimeWarning: divide by zero encountered in log
        # D:\TheseSwansea\Patch-Models\Script\FunctionsAnalysis.py:303: RuntimeWarning: invalid value encountered in multiply'
        
            # species level summary
            res_sp = pd.concat([res_sp, pd.DataFrame({
                # record all the characterstics of the simulation
                'file':f,
                'sim':s,'d':d,'Stot':Stot,'Stot_new':Stot_new,
                "S_regional":S_regional,'P':nb_patches,
                'type':scenario_type, 'quality_ratio':ratio,
                'patch_improved':patch_improved, 
                'patch':np.repeat(coords['Patch'],Stot_new), # patch ID
                
                ## coordinates
                'x':np.repeat(coords['x'],Stot_new),
                'y':np.repeat(coords['y'],Stot_new),
                
                'sp_ID':np.tile(FW_new['sp_ID'],nb_patches),
                # "TL_sp_ID":np.tile(FW_og["TL"][FW_new['sp_ID']],nb_patches),
                # "TP_sp_ID":np.tile(FW_og["TP"][FW_new['sp_ID']],nb_patches),
                
                ## mean characteristics of extinct species
                ## nan if no extinct species
                'mean_TL_extinct':np.mean(FW_new["TL"][extinct2]),
                'mean_TP_extinct':np.mean(FW_new["TP"][FW_new["sp_ID"][extinct2]]),
                'mean_C_extinct':np.mean(np.sum(FW_new["M"][:,extinct2], axis = 0) + np.sum(FW_new["M"][extinct2,:], axis = 1)),
                'mean_Vul_extinct':np.mean(np.sum(FW_new["M"][extinct2,:], axis = 1)),
                'mean_Gen_extinct':np.mean(np.sum(FW_new["M"][:,extinct2], axis = 0)),
                'nb_extinct':sum(extinct2),
                'extinct_any_patch':np.tile(extinct2,nb_patches), # whether the species went extinct or not in at least one patch
                'extinct':extinct1,
                                
                ## quality ratio
                'deltaR':np.repeat(deltaR,Stot_new),
        
                # record biomass 
                'B_final':Bf1.reshape(Stot_new*nb_patches),
                "B_init":y0,
                
                # raw species characteristics
                'TL': np.tile(FW_new["TL"], nb_patches),
                'TP': np.tile(FW_new["TP"][FW_new['sp_ID']], nb_patches),
                'C': np.tile(np.sum(FW_new["M"], axis = 0) + np.sum(FW_new["M"], axis = 1), nb_patches),
                'Vul': np.tile(np.sum(FW_new["M"], axis = 1), nb_patches),
                'Gen': np.tile(np.sum(FW_new["M"], axis = 0), nb_patches)
    
                })])
        
    return(res_sp, FW_metrics)


## summary statistics for initial population dynamics (before applying disturbance)
def summarise_initial_pop_dynamics(list_files, nb_patches): ## coords = DataFrame with one 'Patch' col with patch IDs, and coordinates under 'x' and 'y' cols
    
    
    ## initialise DataFrames to save summary statistics
    FW_metrics = pd.DataFrame()
    res_sp = pd.DataFrame()
    ## initialise index for plotting
    index = 0
    
    for f in list_files:
        print(f)
        # load file with population dynamics
        
        if '.npy' in f:
            sol_disturbed = np.load(f, allow_pickle=True).item()
        elif ".pkl" in f:
            with open(f, 'rb') as file: 
                sol_disturbed = pickle.load(file)  
            file.close()
            
        s = sol_disturbed['sim'] ## simulation number
        # FW_og = sol_disturbed['FW'] ## regional food web (this goes with sp_ID)
        FW_new = sol_disturbed['FW_new'] ## subset food web (this goes with sp_ID_)
        Stot_new = int(sol_disturbed['y'].shape[1]/nb_patches)
        coords = sol_disturbed['FW_new']['coords'] # get patch coordinates
        
        deltaR = sol_disturbed['deltaR'] # patch qualities
        ratio = np.min(deltaR)/np.max(deltaR) # quality ratio 
        
        if ratio == 1:
            scenario_type='homogeneous'
        else:
            scenario_type='heterogeneous'
            
        y0 = FW_new['y0'].reshape(nb_patches*Stot_new) # intial population biomass (pre-disturbance)
        
        
        # Bf1 = np.zeros(shape = (nb_patches,Stot_new)) # post-disturbance pop biomasses
        
        ## go through each patch in turn and summarise food web characteristics and species'
        ## characteristics
        
        solT = sol_disturbed['t'] ## time
        solY = sol_disturbed['y'] ## biomasses
        
        # mean biomass across the last 5% of the time steps
        thresh_high = solT[-1] # get higher time step boundary (tfinal)
        thresh_low = (solT[-1] - (solT[-1] - solT[1])*0.05) # get lower time step boundary (tfinal - 5% * tfinal)

        ## index of those time steps between low and high boundaries
        index_bf = np.where((solT > thresh_low) & (solT <= thresh_high))[0]
        ## biomasses over those 10% time steps
        Bf1 = np.mean(solY[index_bf], axis = 0).reshape(nb_patches,Stot_new)
        
        prop_regional = Bf1[Bf1 > 0]/np.sum(Bf1[Bf1 > 0])
        S_regional = Bf1.shape[1]
        prop_local_all = Bf1/np.repeat(np.sum(Bf1, axis = 1),S_regional).reshape(nb_patches,S_regional)

        
        for p in range(nb_patches):
       
            ind = p + 1
            
            prop_local = Bf1[p][Bf1[p]>0]/np.sum(Bf1[p][Bf1[p]>0])
            
            ### get local food web characteristics
            
            surviving_sp = np.where(Bf1[p]>0)[0]
            local_FW = FW_new['M'][Bf1[p]>0,:][:,Bf1[p]>0]
                    
            FW_metrics = pd.concat([FW_metrics, 
                                   pd.DataFrame({'file':f, 'sim':s, 'patch':p, 
                                                 'type':scenario_type, 'quality_ratio':ratio,
                                                 'S_regional':S_regional, 
                                                 'S_local':len(surviving_sp), 'C_local': np.sum(local_FW),
                                                 'MeanGen_local': np.mean(np.sum(local_FW, axis = 0)),
                                                 'MeanVul_local': np.mean(np.sum(local_FW, axis = 1)),
                                                 'MeanTL_local': np.mean(FW_new['TL'][surviving_sp]), # mean trophic level of surviving species on patch p
                                                  ## diversity measures

                                                  # landscape level:
                                                  'gamma_diversity_shannon':-np.sum(prop_regional*np.log(prop_regional)), # regional diversity
                                                  'mean_alpha_diversity_shannon':np.mean(-np.sum(np.nan_to_num(prop_local_all*np.log(prop_local_all)), axis = 1)), # mean (landscape-level) alpha diversity 
                                                  'beta_diversity_shannon':(-np.sum(prop_regional*np.log(prop_regional)))/np.mean(-np.sum(np.nan_to_num(prop_local_all*np.log(prop_local_all)), axis = 1)), # gamma/alpha ()

                                                  # patch level
                                                  'alpha_diversity_shannon':-np.sum(prop_local*np.log(prop_local)), # shannon diversity (-sum(pi*ln(pi)))
                                                                                       ## quality ratio
                                                 'deltaR':deltaR[p],
                                                 
                                                 'simulation_length':sol_disturbed['t'][-1]
                                                 }, index=[index])])
            index += 1
            
        #     plt.subplot(2, 5, ind)
        #     plt.tight_layout()
            
        #     plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
        #                 solY[np.arange(0,solY.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)])
           
        # plt.title("With events")
        # plt.show()
        
        ## food web metrics for the final food web
        
        extinct2 = np.repeat(False,Stot_new) # in any patch
        extinct1 = np.repeat(False,Stot_new*nb_patches) # in a given patch
        if ((FW_new['y0'] > 0) & (Bf1 == 0)).any():
            for j in range(nb_patches):
                extinct2 = extinct2 | (FW_new['y0'][j] > 0) & (Bf1[j] == 0)
                extinct1[j*Stot_new:(j*Stot_new + Stot_new)] = (FW_new['y0'][j] > 0) & (Bf1[j] == 0)
        
        
        
                
        with np.errstate(divide='ignore', invalid='ignore'): # suppress 
        # 'mean_Vul_extinct', 'mean_Gen_extinct'
        #'D:\TheseSwansea\Patch-Models\Script\FunctionsAnalysis.py:303: RuntimeWarning: divide by zero encountered in log
        # D:\TheseSwansea\Patch-Models\Script\FunctionsAnalysis.py:303: RuntimeWarning: invalid value encountered in multiply'

            res_sp = pd.concat([res_sp, pd.DataFrame({
                # record all the characterstics of the simulation
                'file':f,
                'sim':s,'d':d,'Stot':Stot,'Stot_new':Stot_new,'P':nb_patches,
                'type':scenario_type, 'quality_ratio':ratio,
                'patch':np.repeat(coords['Patch'],Stot_new), # patch ID
                
                ## coordinates
                'x':np.repeat(coords['x'],Stot_new),
                'y':np.repeat(coords['y'],Stot_new),
                
                'sp_ID':np.tile(FW_new['sp_ID'],nb_patches),
                # "TL_sp_ID":np.tile(FW_og["TL"][FW_new['sp_ID']],nb_patches),
                # "TP_sp_ID":np.tile(FW_og["TP"][FW_new['sp_ID']],nb_patches),
                
                ## mean characteristics of extinct species
                'mean_TL_extinct':np.mean(FW_new["TL"][extinct2]),
                'mean_TP_extinct':np.mean(FW_new["TP"][FW_new["sp_ID"][extinct2]]),
                'mean_C_extinct':np.mean(np.sum(FW_new["M"][:,extinct2], axis = 0) + np.sum(FW_new["M"][extinct2,:], axis = 1)),
                'mean_Vul_extinct':np.mean(np.sum(FW_new["M"][extinct2,:], axis = 1)),
                'mean_Gen_extinct':np.mean(np.sum(FW_new["M"][:,extinct2], axis = 0)),
                'nb_extinct':sum(extinct2),
                'extinct_any_patch':np.tile(extinct2,nb_patches), # whether the species went extinct or not in at least one patch
                'extinct':extinct1,
                       
                ## gamma diversity (regional diversity)
                'gamma_diversity':-np.sum(prop_regional*np.log(prop_regional)), # regional diversity
                'mean_alpha_diversity':np.mean(-np.sum(np.nan_to_num(prop_local_all*np.log(prop_local_all)), axis = 1)), # mean alpha diversity (at the patch level)
                'beta_diversity':(-np.sum(prop_regional*np.log(prop_regional)))/np.mean(-np.sum(np.nan_to_num(prop_local_all*np.log(prop_local_all)), axis = 1)), # gamma/alpha ()
                
                ## quality ratio
                'deltaR':np.repeat(deltaR,Stot_new),
        
                # record biomass 
                'B_final':Bf1.reshape(Stot_new*nb_patches),
                "B_init":y0,
                
                # raw species characteristics
                'TL': np.tile(FW_new["TL"], nb_patches),
                'TP': np.tile(FW_new["TP"][FW_new['sp_ID']], nb_patches),
                'C': np.tile(np.sum(FW_new["M"], axis = 0) + np.sum(FW_new["M"], axis = 1), nb_patches),
                'Vul': np.tile(np.sum(FW_new["M"], axis = 1), nb_patches),
                'Gen': np.tile(np.sum(FW_new["M"], axis = 0), nb_patches)
                })])
        
    return(res_sp, FW_metrics)


## calculate shortest distance with effective species composition


def calculate_shortest_path(nb_patch, start_patch, dist, FW, quantile = 0.50):
    
    '''
    Function to calculate shortest path between disturbed patch (9 for the 10P scenario and 
    0 for the 3P scenario) and all others
    this is done by considering that only patches within dmax of the best disperser
    are connected. Dmax is chosen as the median number of patch species' can disperse to
    Thus, the shortest distance for patches directly connected (within dmax of patch 9)
    to patch 9 are their euclidian distance to patch 9, but all other patches 
    are only connected to patch 9 via one of these direct neighbours etc. 
    We thus look for the shortest path from 9 > stepping stone > patch n trying all
    possible stepping stone patches (path_length >= 2). Path length is measured as the sum of the 
    euclidian distance between patch 9 to stepping stone + stepping stone to destination. 
    For example, if we want to connect A to D, but dist(AD) > dmax, then the shortest distance
    might be shortest_path(AD) = dist(AB) + dist(BC) + dist(CD)
    '''
    
    ## dataframe to save the shortest distances (nb_patch rows, including starting patch)
    shortest_distance = pd.DataFrame({'shortest_distance':[0], 'Patch':[start_patch], 'path_length':[0], 'start_patch':start_patch}) 
    #  nb_patch - 1 destination patches (at t + 1) - is used to record which patches 
    # have not yet been reached
    patch_list = np.array([i for i in range(nb_patch) if i != start_patch]) 
    # save a permanent version of patch_list that will never be modified:
    patch_list_init = patch_list.copy() 

    start_list = [start_patch] # starting patches (t) - just one patch at t0 but then can be any number of patches to try and compare distances

    ## one column j is populated with the steps (patch IDs) taken by a species to get to patch j 
    ## thus start_history[0,j] is the first stepping stone patch in time step 0, 
    ## then species moves to start_history[1,j] until it patch j is reached
    start_history = np.zeros((nb_patch,nb_patch)) # list to record stepping stone patches (cols = max possible path length x rows = nb of patches)
    start_history[:,0] = start_patch # the first stepping stone patch for everyone is the starting patch at t0 (9 or 0)

    path_length = 0 # number of stepping stone patches - start with step zero

    while(len(patch_list)>0): # while there are still some unreached destination patches (t + 1)
                
        # matrix to record the distance between the starting patch (start_patch) to each unreached patch (patch_list)
        # Used to record all distances and keep only minimum (shortest) path length
        distance = np.array(np.repeat(100, len(patch_list)*len(start_list)), dtype = 'float').reshape(len(patch_list), len(start_list))
        # increment number of steps
        path_length+=1
    
        for start_ind in range(len(start_list)): # loop across destination patches 
            
            # current patch (t)
            start = start_list[start_ind]
            # accessible patches from current patch (t)
            # nb_accessible = np.sum(FW['access'][start], axis = 0) # number of accessible pacthes from patch 9 for each species
            dmax = FW['dmax'][FW['sp_ID_']]
            dmedian = np.quantile(dmax, quantile)
            
            
            # Here we differenciate ID from index. ID is the patch's ID (0-9 or 0-3),
            # whilst index is the index of the patch ignoring the starting patch (see below)
            # and can thus be different from the ID. It is used to know the patch's position
            # inside numpy arrays that have (nb_patch - 1) rows.
            # Here the indexing goes e.g. for start = 7: [0,1,2,3,4,5,6,8,9] - only 9 indexes, but 10 patches
            index_accessible_patches = [i if i < start else i + 1 for i in np.arange(9)] 
            
            # get the list of patches accessible to the "median" species (species that have access to the median number of patches from patch "start")
            # 0 = patch inaccessible to the "median" species, 1 = patch accessible
            accessible_patches = np.sum(FW['access'][start][:,dmax <= dmedian], axis = 1)
            accessible_patches = np.array([1 if i > 0 else 0 for i in accessible_patches]) # set all to 1
            # keep only patches that have not been reached yet (oi.e. inside patch_list)
            accessible_patches = accessible_patches[np.intersect1d(index_accessible_patches, patch_list, return_indices = True)[1]]
            
            # distance from current (t) patch to possible future patches in patch_list(t + 1)
            distance_current_patch = dist[start,patch_list]
            # distance from previous paths (t0 to t - 2 etc) + from start (t-1) to current patch (t) 
            # this is the distance effectively covered by the median species before it could reach patches in patch_list
            distance_start_current = 0
            if path_length > 1:
                
                step = start_history[start,:] # look at which patch IDs were reached (possible stepping stones) at this time step
                step = step.astype(int) # need to be integer to use as index
                
                for i in range(path_length-1): # loop accross previous steps
                    # record distance at between patch at t-1 and t
                    distance_start_current = distance_start_current + dist[step[i], step[i+1]]
                distance_start_current+= dist[start,step[i+1]] # add current step
            else: 
                distance_start_current = 0
            
            
            if all(accessible_patches == 0): # if no more accessible patches
                print("No accessible patches from patch:", start, 'at path length ', path_length)
                ## use simple euclidian distance - this means that these patches are too far for the median species
                distance[:,start_ind] = (distance_current_patch + distance_start_current) # if accessible (1 in accessible patches, then add distance from t-1 to t)
            else: # at least one patch is within reach of median species from start
                # record distance 
                distance[:,start_ind] = accessible_patches * (distance_current_patch + distance_start_current) # if accessible (1 in accessible patches, then add distance from t-1 to t)
    
        # now we compare the distances from different starting points and keep the smallest, 
        # then update patch_list for unreached patches
        distance[np.where(distance == 0)] = 100 # disconnected patches
        distance_min = np.min(distance, axis = 1) # keep only minimun path length
        accessible_patch_id = patch_list[distance_min != 100] # record ID of patches that are reached
        
        start_history[:,path_length][np.intersect1d(patch_list_init, accessible_patch_id, return_indices = True)[1]] = [start_list[i] for i in np.argmin(distance[distance_min != 100,:], axis = 1)] # update latest step
        
        distance_min = distance_min[distance_min != 100] # only keep distance for patches that were reached
    
        patch_list = np.setdiff1d(patch_list, accessible_patch_id) # remove any patches that might have not been reached, so that we can start the loop again
        start_list = accessible_patch_id # update start_list to patches that were not reached
        
        # save distance for patches that were reached
        shortest_distance = pd.concat([shortest_distance, 
                                      pd.DataFrame({'shortest_distance':distance_min,
                                                   'Patch':accessible_patch_id,
                                                   'path_length': path_length,
                                                   'start_patch':start_patch})])
        
    return(shortest_distance)



def create_temporal_network(access, P):
    '''
    
    Creating temporal network with succession of accessible patches, starting 
    from disturbed patch 9 for each species (which all have different dispersal
    capabilities). 
    
    For example, starting from patch 9, if you are a species with max dispersal 
    distance dmax = 3, then you can reach all patches within dmax of patch 9 in 
    timestep 0. 
    Next time step, you will start from those stepping stone patches, and
    disperse further etc. 
    
    At each time step we record in A 10x10 matrix which which patches can be 
    reached from the starting patches. Thus, for the first time 
    step, we start at patch 9 so all other rows have value zero. Then positive
    values in row 9 indicate the the species has a positive success rate when 
    dispersing to these patches (ID from column index). 
    

    Parameters
    ----------
    FW : Dictionnary with 'access' entry 
        Food web used for simulation from which we take species composition and 
        dispersal capabilities.
    P : int
        Number of patches.

    Returns
    ---------
    
    Network G, a collection of nb_sp 10 x 10 x nb_timesteps matrices that
    can be 'fed' to teneto to get shortest_temporal_path (see function 
                                                          'get_shortest_path').
    
    '''
    
    access = 1 - access # dispersal success
    access[access == 1] = 0
    start_patch = 9
    nb_patch = 10
    nb_tsteps = 15
    G = np.zeros(shape = (P, P, nb_tsteps, access.shape[2])) # (start, destinations, t, sp)

    for sp in range(access.shape[2]):
        
        print('Species : ', sp)
        
        stop = False
        start_list = [start_patch] # starting patches (t) - just one patch at t0 but then can be any number of patches to try and compare distances

        start_history = np.zeros((nb_patch - 1,10)) # list to record stepping stone patches
        start_history[:,0] = start_patch # the first stepping stone patch for everyone is the starting patch at t0 (9 or 0)
        
        patch_reached = np.array([])
        t = 1 # number of stepping stone patches - start with step zero
        
        ## doing things in terms of dispersal success complicates things I think,
        ## because then we get 
        ## easier in terms of euclidian distance
        
        np.sum(access[:,:,sp], axis = 1)
        accessible_patches_history = [np.array([-2,-1]), np.array([-1,-2])]
        
        while(len(patch_reached) < 9 and not stop): # while there are still some unreached destination patches (t + 1)
                    
            accessible_patches = np.array([])
            for start_ind in range(len(start_list)): # loop across destination patches 
                
                # current patch (t)
                start = start_list[start_ind]
                
                patch_list = np.array([i for i in range(nb_patch) if i != start]) 
            
                G[start,patch_list,t-1,sp] = access[start,:,sp]
              
                accessible_patches = np.unique(np.append(accessible_patches, np.array(patch_list)[access[start,:,sp] > 0]))
                print(start,accessible_patches)
            
            accessible_patches_history = accessible_patches_history + [accessible_patches]
            patch_reached = np.unique(np.append(patch_reached, accessible_patches)) # remove any patches that might have not been reached, so that we can start the loop again
            print(patch_reached)
                
            start_list = accessible_patches.astype(int) # update start_list to patches that were not reached

            t+=1     
            if (len(accessible_patches_history[t]) == len(accessible_patches_history[t - 1])):
                if (all(accessible_patches_history[t] == accessible_patches_history[t-1])):
                    
                    stop = True
            else:
                stop = False
                
    return(G)

# def get_shortest_path(G, sp, dist):
    
#     '''
#     Here we use teneto to get the shortest paths from patch 9 to other patches,
#     and then calculate total success rate and euclidian distances.
#     '''
    
#     # create network to feed to teneto (nb_patches, nb_patches, time_step, species)
#     network = G[:,:,:,sp]
#     # remove extra time steps (after we have been through the whole network)
#     network = network[:,:,np.sum(np.sum(network, axis = 0), axis = 0)>0]
#     # some plants have too small dispersal distance to go anywhere, thus their 
#     # network is empty, we skip these species
#     if network.shape[2] == 0:
#         # exit and return empty dataframe
#         return(pd.DataFrame({'from':np.nan, 'to':np.nan, 't_start':np.nan, 
#                              'temporal-distance':np.nan, 'topological-distance':np.nan,
#                              'path includes':np.nan, 'euclidian_dist':np.nan, 'success_rate':np.nan, 
#                              'max_success_rate':np.nan,'mean_success_rate':np.nan, 
#                              'min_euclidian_dist':np.nan, 'mean_euclidian_dist':np.nan, 'sp':sp}, index=[0]))
    
    
    
    
#     fig, ax = plt.subplots(1)
#     teneto.plot.slice_plot(network, ax, cmap='Set2')
    
#     '''
#     https://direct.mit.edu/netn/article/1/2/69/5394/From-static-to-temporal-network-theory
    
#     'The shortest path is the minimum number of edges (or sum of edge weights) 
#     that it takes for a node to communicate with another node. In temporal 
#     networks, a similar measure can be derived. Within temporal networks, we 
#     can quantify the time taken for one node to communicate with another node. 
#     This is sometimes called the “shortest temporal distance” or “waiting time.” 
#     Temporal paths can be measured differently by calculating either how many 
#     edges are traveled or how many time steps are taken (see Masuda & Lambiotte, 2016); 
#     here we quantify the time steps taken.'
    
#     '''
    
    
#     shortest_path = teneto.networkmeasures.shortest_temporal_path(network, i=9, it = 0)
    
#     ## create empty columns to save euclidian distance
#     shortest_path['euclidian_dist'] = pd.Series([[] for i in range(shortest_path.shape[0])])
#     shortest_path['success_rate'] = pd.Series([[] for i in range(shortest_path.shape[0])])
    
#     shortest_path['max_success_rate'] = 0
#     shortest_path['mean_success_rate'] = 0
    
#     shortest_path['min_euclidian_dist'] = 0
#     shortest_path['mean_euclidian_dist'] = 0
    
#     shortest_path['sp'] = sp
    
#     ## calculate distance to travel from patch 9 to destination
#     for i in range(9): # travel through each species
        
#         paths = shortest_path.loc[i,'path includes'] # all possible paths from 9 to i
        
#         nest = 0
#         paths_sub = paths
        
#         # check if a path exists (topological distance is not nan)
#         if np.isnan(shortest_path.loc[i,'topological-distance']): 
#             # if it is, set distance to nan too
#             shortest_path.loc[i,'euclidian_dist'].append(shortest_path.loc[i,'topological-distance'])
#             shortest_path.loc[i,'success_rate'].append(shortest_path.loc[i,'topological-distance'])
#         else:
#             # loop through each possible path
#             while(isinstance(paths_sub[0], list)):
#                 paths_sub = paths_sub[0]
#                 nest+=1
                
#             # check if patch i is directly connected to patch 9
#             if shortest_path.loc[i,'topological-distance'] == 1:
#                 # in which case the distance is just the euclidiant distance between patch 9 and i
#                 shortest_path.loc[i,'euclidian_dist'].append(dist[paths[0][0],paths[0][1]]) 
#                 shortest_path.loc[i,'success_rate'].append(network[paths[0][0],paths[0][1],0]) 
                
#             # finally if there are several steps to go from 9 to i:
#             elif nest == 1:
#                 for j in range(len(paths)): 
#                     eucl_dist_save = 0 # to sum the distances 
#                     success_rate_save = 0
#                     paths_sub = paths[j] # current path
#                     # loop through the each step for this path and add distance 
#                     # (ex: path_sub = [[A,B],[B,C]] then eucl_dist = dist(AB) + dist(BC))
#                     for k in range(len(paths_sub)):
#                         if k == 0 and paths_sub[0] != 9: # sometimes teh first step is inverted (teneto mistake?) so we interchange the coordinates so they are in the right order. 
#                         # the subsequent coordinates seem in order. 
#                             paths_sub[1] = paths_sub[0] # exchange 1 and 0
#                             paths_sub[0] = 9 # set first coordinate to 9
                            
#                         eucl_dist_save+= dist[paths_sub[k],paths_sub[k]]
#                         success_rate_save+= network[paths_sub[k],paths_sub[k],0]
#                     shortest_path.loc[i,'euclidian_dist'].append(eucl_dist_save)
#                     shortest_path.loc[i,'success_rate'].append(success_rate_save)
                
#             elif nest == 2:
#                 for j in range(len(paths)): 
#                     eucl_dist_save = 0 # to sum the distances 
#                     success_rate_save = 0
#                     paths_sub = paths[j] # current path
#                     # loop through the each step for this path and add distance 
#                     # (ex: path_sub = [[A,B],[B,C]] then eucl_dist = dist(AB) + dist(BC))
#                     for k in range(len(paths_sub)):
#                         if k == 0 and paths_sub[k][0] != 9: # sometimes teh first step is inverted (teneto mistake?) so we interchange the coordinates so they are in the right order. 
#                         # the subsequent coordinates seem in order. 
#                             paths_sub[k][1] = paths_sub[k][0] # exchange 1 and 0
#                             paths_sub[k][0] = 9 # set first coordinate to 9
                            
#                         eucl_dist_save+= dist[paths_sub[k][0],paths_sub[k][1]]
#                         success_rate_save+= network[paths_sub[k][0],paths_sub[k][1],0]
#                     shortest_path.loc[i,'euclidian_dist'].append(eucl_dist_save)
#                     shortest_path.loc[i,'success_rate'].append(success_rate_save)
    
#         shortest_path.loc[i,'min_euclidian_dist'] = min(shortest_path.loc[i,'euclidian_dist'])
#         shortest_path.loc[i,'mean_euclidian_dist'] = np.mean(shortest_path.loc[i,'euclidian_dist'])
        
#         shortest_path.loc[i,'max_success_rate'] = max(shortest_path.loc[i,'success_rate'])
#         shortest_path.loc[i,'mean_success_rate'] = np.mean(shortest_path.loc[i,'success_rate'])
        
#     return(shortest_path)
    
    
    