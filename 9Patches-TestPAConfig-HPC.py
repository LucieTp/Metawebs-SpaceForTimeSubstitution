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
## pandas needs to be version 1.5.1 to read the npy pickled files with np.load() 
## created under this version 
## this can be checked using pd.__version__ and version 1.5.1 can be installed using
## pip install pandas==1.5.1 (how I did it on SLURM - 26/03/2024)
## if version 2.0 or over yields a module error ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'
## I think to solve this, would need to run the code and save files under version > 2.0 of pandas
## or manually create this pandas.core.index.numerical function see issue in https://github.com/pandas-dev/pandas/issues/53300

from scipy.integrate import odeint, solve_ivp # Numerical integration function from the scientific programming package

import matplotlib.pyplot as plt 

import numpy as np # Numerical objects

import time

import networkx as nx

import igraph as ig

from argparse import ArgumentParser # for parallel runnning on Slurm


### this function implements the niche model by Williams and Martinez (2000) 
### given two params: S and C it yields food webs based on the distance between 
### species' niches
def nicheNetwork(S,C):
    
    x = 0
    y = S
    dup_sp = False
    
    while x < S or y > 1 or dup_sp :
         dup_sp = False
         M = np.zeros(shape = (S,S))
         
         # first we obtain the n values for all the species 
         # (ordered so we know the first sp is a basal species)
         n = np.sort(np.random.uniform(0,1,S))
         
         #we then obtain the feeding range for each of the species, drawn from a beta
         #distribution with beta value = (1/(2*C)) - 1
         beta = (1/(2*C)) - 1
         r = np.random.beta(1, beta, S) * n
         
         #we enforce the species with the lowest niche value to be a basal species
         r[0] = 0
         
         #the centre of the feeding range is drawn from a uniform distribution
         #between ri/2 and min(ni, 1-ri/2)
         c = np.zeros(S)
         for i in range(S):
             c[i] = np.random.uniform(r[i]/2,min(n[i], 1-r[i]/2))
             
             offset = r[i]/2
             upper_bound = c[i] + offset
             lower_bound = c[i] - offset
             
             for j in range(S):
                 if n[j] > lower_bound and n[j] < upper_bound:
                     M[j,i] = 1
                     
         #we verify that the network (i) is connected and (2) does not have cycles with no external energy input
         M_temp = M.copy()
         np.fill_diagonal(M_temp,0)
    
         graf = ig.Graph.Adjacency(M)
         y = graf.components(mode='weak')
         y = y._len
         if y > 1: # number of clusters
             next
    
         clts = graf.components(mode='strong')
         x = clts._len # number of clusters
         if x < S:
             clts_nos = [i for i in range(len(clts.sizes())) if clts.size(i) > 1]
             cycles_no_input = False
             
             for cn in clts_nos:
                 members = [i for i in range(len(clts.membership)) if clts.membership[i] == cn]
                 cluster_ok = False
            
                 for m in members:
                    prey = graf.neighbors(m, mode='in')
                    
                    if len(np.intersect1d(prey,members))  > len(members) + 1:
                        ## if this happens, this cycle/cluster has external energy input
                        cluster_ok = True
                        break
                
                    if not cluster_ok:
                       #print("NicheNetwork :: the network has cycles with no external energy input...");
                       cycles_no_input = True
                       break
       
      
             if cycles_no_input:
                 next
             else:
                 x = S
      
    
         #and we also verify that there are not duplicate species
         for i in range(S):
             if dup_sp:
                 break
          
             preys_i = M[:,i]
             predators_i = M[i,:]
             for j in range(S):
                if i == j:  
                    next
                sim_prey = preys_i == M[:,j]
                sim_preds = predators_i == M[j,:]
                
                if sum(sim_prey) == S and sum(sim_preds) == S:
                  dup_sp <- True
                  #print("NicheNetwork :: there is a duplicate species");
                  break
    
          
              #as long as we are traversing the network we might as well check
              #for basal species with cannibalistic links...
          
             if sum(preys_i) == 1 and M[i,i] == 1:
                 #print("NicheNetwork :: a cannibalistic link was found in a basal species... removing it");
                 M[i,i] = 0
        
    #we keep a reference of the original niche as a way of idetifying the species in the network:
    #each species is going to be identified by the index of its niche value within the 
    #original_niche array.
    original_niche = n
  
  
    ########################################################################################
  ## c: centre of the feeding range, r: distance from centre of the feeding range
  ## n: niche of the species

    return {'Stot':len(M), 'C':M.sum()/(len(M)**2), 'M':M, 'Niche':n, 'Centre':c, 'radius':r, 'Original_C':C, 'original_niche':original_niche}



###############################################################################
## ADDING BODY SIZE BASED ON TROPHIC POSITION 

## heavily inspired by Sentis et al 2020 (warming and invasion in food webs)
## r code at https://github.com/mlurgi/temperature-dependent-invasions

def NormaliseMatrix(M):
    colsum = sum(M)
    colsum[colsum == 0] = 1
    return M/colsum


def TrophicLevels(FW, TP):
    S = FW['Stot']
    
    if S<2:
        return FW
    
    TL = np.repeat(-1, S)
    
    M_temp = FW['M']
    np.fill_diagonal(M_temp,0) 
    
    ## species w/ TP 1 are basal species
    TL[np.array(TP) == 1] = 0
    
    for i in range(S):
        if TL[i] == -1:
            herb = True
            top = True
            
            # 1 - if the species only has prey in TL 0 then it is a herbivore
            if sum(TL[M_temp[:,i] != 0]) != 0:
                herb = False
            
            # 2 - if it has any predators then not a top pred
            if sum(M_temp[i,:]) > 0:
                top = False
            
            if herb:
                TL[i] = 1
            elif top:
                TL[i] = 3
            else:
                TL[i] = 2
    
    return TL


##### prey-averaged trophic level metric:
##### Williams and Martinez (2004). American Naturalist.
##### Limits to trophic levels and omnivory in complex food webs: theory and data.

def trophicPositions(FW):
    S = FW['Stot']
    
    if S<3: 
        return FW
    
    M = NormaliseMatrix(FW['M'])
    
    ## solves (I - A) * X = B
    # this means that species with no prey get TP = 1
    if np.linalg.det(np.diag(np.repeat(1,S)) - np.transpose(M)) != 0: 
        TP = np.linalg.solve(np.diag(np.repeat(1,S)) - np.transpose(M), np.repeat(1,S))
    
    else:
        tmp = np.diag(np.repeat(1,S))
        for i in range(9):
            tmp = np.matmul(tmp,np.transpose(M)) + np.diag(np.repeat(1,S))
        TP = np.matmul(tmp, np.repeat(1,S))
    W = M
    TL = TrophicLevels(FW, TP)
    
    FW.update({'TL':TL,'TP':TP,'NormM':W})
    return FW

    
def obtainBodySize(FW):
    
    FW = trophicPositions(FW)
    # average predator prey mass ratio
    ppmr = 100
    
    BS = 0.01*(ppmr**(FW['TP'] - 1))
    
    FW.update({'BS':BS})
    return FW



def MeanFoodChainLength(FW):
    
    ## calculates the mean number of nodes between all species 
    
    fcl = []
    G = nx.DiGraph(FW)
    if G.number_of_nodes() == 0:
        mfcl = 0
    else:
        for i in range(G.number_of_nodes()): # loop across all source nodes
            for j in range(i,G.number_of_nodes()): # and across all target nodes 
                if nx.has_path(G,i,j): # check that a path exists between the two nodes
                    paths = nx.all_simple_paths(G, source=i, target=j) # record the number of nodes between source and target
                    for p in paths:
                        fcl.append(len(p))
        mfcl = sum(fcl)/len(fcl)
        
    return(mfcl)            
          
params = {## attack rate parameters - alphas (Binzer et al)
    "ab":0.25, # scale to body mass of resource species
    "ad":np.exp(-13.1),
    "ac":-0.8, # scale to body mass of consumer species
    
    ## handling time parameters - th (Binzer et al)
    "hd":np.exp(9.66),
    "hb":-0.45, # scale to body mass of ressource species
    "hc":0.47, # scale to body mass of consumer species
    
    # growth rate params
    "rd":np.exp(-15.68), "rb":-0.25,
    
    # metabolic rate parameters
    "xd":np.exp(-16.54), "xb":-0.31,
    
    # carrying capacity params
    "Kb":0.28, "Kd":5, 
    
    ## maximum dispersal distance (Häussler et al 2021)
    "d0": 0.1256, "eps":0.05} ## might need to change those paramaters if we want them to range exactly from 0.158 to 0.5 as for Johanna's study

      
def getSimParams(FW,S,P,coords,params=params):
    
    FW = obtainBodySize(FW)
    
    Stot = FW['Stot']
    M = FW['M']
    BS = FW['BS']
    
    sp_ID = np.arange(Stot)
        
    # intialise matrixes 
    a = np.zeros((Stot,Stot)) # attack rates
    h = np.zeros((Stot,Stot)) # handling time
    
    # BSr = np.zeros((Stot,Stot)) # just for plotting
    
    # mfcl = MeanFoodChainLength(M)
         
    # # allometric constants
    # ar = 1
    # ax = 0.88 # metabolic rate for ectotherms
    BSp = 0.01 # body mass of producer
    
    ## body mass dependant metabolic rate of predators, 0.138 (Ryser et al. 2021) for basal species aka plants
    # x = np.array([0.138 if sum(a[:,i]) == 0 else ax*(BS[i]**(-0.25))/(ar*(BSp**(-0.25))) for i in range(len(a))])
    
    ## body mass dependant metabolic rate for consumers only
    x = np.array([params['xd']*BS[i]**params['xb'] if sum(M[:,i]) > 0 else 0 for i in range(Stot)])
    # plt.scatter(x,[math.log(i) for i in BS])
    # plt.show() 
    
    ## growth rate (only for plants)
    r = [params['rd']*BSp**params['rb'] if sum(M[:,i]) == 0 else 0 for i in range(Stot)]
    # equal patch quality
    r = np.tile(r,P).reshape(P,Stot)
    
    K = np.array([params["Kd"]*BSp**params["Kb"] if sum(M[:,i]) == 0 else 0 for i in range(Stot)])
    
    # initial conditions - first we randomly allocate the species present in each patch
    y0 = [[1]*Sp + [0]*(Stot-Sp) for Sp in S]
    for init in range(len(y0)):
        np.random.seed(init)
        np.random.shuffle(y0[init])
        
    # then we adjust the initial biomass (=K for producers, K/8 for consumers)
    y0 = y0*np.array([k if k!=0 else K[K!=0].mean()/8 for k in K])
    
    # calculate attack rates and handling times that scale with body size
    # j: predator, i: prey
    # for i in range(Stot):
    #     for j in range(Stot):
    #         if M[i,j]>0: # j eats i
    #             # BSr[i,j] = BS[j]/BS[i]
    #             # a[i,j] = params['ad']*BS[j]**params['ab']*BS[i]**params['ac'] # attack rate of consumer j on prey i
    #             # h[i,j] = params['hd']*BS[j]**params['hb']*BS[i]**params['hc'] # handling time of consumer j of prey i
                
    #             a[i,j] = params['ad']*BS[j]**params['ab']*BS[i]**params['ac'] # attack rate of consumer j on prey i
    #             h[i,j] = params['hd']*BS[j]**params['hb']*BS[i]**params['hc'] # handling time of consumer j of prey i


    a = params['ad']*BS.reshape(Stot,1)**params['ab']*BS**params['ac']*FW["M"]
    h = params['hd']*BS.reshape(Stot,1)**params['hb']*BS**params['hc']*FW["M"]

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.scatter(np.divide(BS,BS.reshape(Stot,1)), a)
    
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.scatter(np.divide(BS,BS.reshape(Stot,1)), h)
    # plt.title("handling time")

    
    scatter = plt.scatter(coords['x'],coords['y'], c = np.arange(P), label = np.arange(P))
    plt.legend(*scatter.legend_elements(), title = "Patches")
    plt.title("Patch map")
    plt.show()

    # then calculate distances
    distances = np.zeros(P*P).reshape(P,P)
    for i in range(len(distances)):
        for j in range(len(distances)):
            distances[i,j] = ((coords.iloc[i,:]['x'] - coords.iloc[j,:]['x'])**2 + (coords.iloc[i,:]['y'] - coords.iloc[j,:]['y'])**2)**(1/2)

    
    ## maximum dispersal distance (Häussler et al. 2021)
    ## scales with body size for animals and uniformly drawn from uniform distribution for plants
    dmax = np.array([params["d0"]*BS[i]**params["eps"] if sum(M[:,i]) > 0 else np.random.uniform(0,0.5) for i in range(Stot)])
    
    # and create matrix of patch accessibility for each species depending on their starting point
    access = np.repeat(np.zeros(shape = (P-1, Stot)), P).reshape(P,P-1,Stot) # create empty array
    dd = np.repeat(np.zeros(shape = (P-1, Stot)), P).reshape(P,P-1,Stot)
    for patch in range(P):
        # neighbouring patches
        n = [j for j in range(P) if j!=patch] # index of neighbouring patches
        dn = distances[:,patch][n] # distance of neighbouring patches
        acc = np.array([dmax > j for j in dn])*1 # accessible patches for each species
        access[patch] = acc
        
        ddnz = np.array([i/dmax for i in dn])*acc # dispersal death while travelling from n to z (patch)
        dd[patch] = ddnz
        
    FW.update({'sp_ID':sp_ID, 'x':x,'r':r,'a':a,'h':h,'y0':y0,'K':K,"dmax":dmax,
               "dd":dd,"access":access,"distances":distances,"coords":coords})
    return(FW)
                    




###############################################################################
### two patches dynamics with predation



def TwoPatchesPredationBS_ivp(y0,q,P,S,FW,d,connectivity,deltaR,h,tfinal):
        
# Function defining the model. Inputs initial conditions, parameters and maxtime, outputs time and trajectories arrays
    def System(t,y,q,P,S,FW,d,connectivity,deltaR,h): # Function defining the system of differential equations
        
    
        a = FW['a'] # attack rate
        h = FW['h'] # handling time
        x = FW['x'] # metabolic rate
        r = FW['r'] # growth rate
        K = FW['K'] + 0.00001 # just to prevent division by zero
        dmax = FW["dmax"] # maximum dispersal distance 
        
        # empty population size
        dN = np.zeros(S*P).reshape(P,S)
        
        y = np.array(y).reshape(P,S)
        y[y < 1e-8] = 0
        
        for p in range(P):
            # current patch
            N = y[p]  # previous population Np(t)
            dNp = np.zeros(S)  # initialising Np(t + 1)
            
            # neighbouring patches
            n = [j for j in range(P) if j!=p] # index of neighbouring patches
            # Nn = [y[j] for j in n]
            
            Nn = y[n,:] # biomass in neighbouting patches
            # Nn = Nn*access[p] # filter species' biomass from patches that are within their maximum disp distance
            
            # (1) PREDATION
            # resources in rows, consumers in columns
            
            # Functionnal response: resource dependant feeding rate of consumer j on prey i influenced by j's k (other?) resources 
            # Fij = aij*Bi/(1+sum(aik*hik*Nk)) 
            # q = 1.4 # hill exponent (Sentis et al - between type II (hyperbolic q = 1) and type III (sigmoidal q = 2))
            # increased from 1.2 to 1.4 after suggestion of Benoit Gauzens for increasing persistence. Could also think about adding interference compeition.
            # who wrote ATNr package for allometric food webs
            
            F = a*(N.reshape(S,1)**(1 + q))
            low = 1 + np.matmul(np.transpose(a*h), N**(1 + q)) + 0.2*N 
            
            F = np.divide(F,low)
            
            predationGains = N*F.sum(axis = 0)*0.85 # xi*e*Ni*sum(Fik)
            predationLosses = np.matmul(N,F.T) # sum(xj*Nj*Fki)
            
            # print(F,"gain",predationGains, "loss",predationLosses)
            
            # (2) DISPERSAL 
            emigration = -d[p]*N # biomass that leaves the source patch (z)
            
            # En = Nn*d[n,:] # emigrants leaving neighbouring patches 
            
            # low_d = np.sum(1 - dd[n,:,:], axis = 0)
            # Nim = np.sum(En * (1 -  dd[p]) * (1 -  dd[p])/low_d, axis = 0)
            
            Nim = np.zeros(S) 
            for ind in range(len(n)): # loop across neighbours 
                Nim+= (d[p]/(P-1))*Nn[ind]*connectivity[p,n[ind]] # biomass of each species in neighbouring patches that successfully disperse
            
            
            immigration = Nim # biomass not lost to the matrix during dispersal
            
            # print('immigration',immigration,'emigration', emigration)
            # print(N, 'Gains', predationGains, 'loss', predationLosses)
            
            # growth rate*logistic growth (>0 for plants only) - metabolic losses + predation gains (>0 for animals only) - predation losses + dispersal
            dNp = N*(r[p]*deltaR[p]*(1 - N/K) - x - h) + predationGains - predationLosses + immigration + emigration
            # print(dNp)
            
            # dNp[N<1e-8] = 0 
            dN[p] = dNp
            
        return np.array(dN).reshape(P*S)
    
    t_span = (0,tfinal)
    sol_ivp = solve_ivp(fun = System, t_span = t_span, y0 = y0, args=(q,P,S,FW,d,connectivity,deltaR,h), method='LSODA') #, t_eval=t) # sol_ivp is the newer version of odeint https://danielmuellerkomorowska.com/2021/02/16/differential-equations-with-scipy-odeint-or-solve_ivp/
    sol_ivp['y'][sol_ivp['y']<1e-8] = 0
    
    return sol_ivp



# choice of timestep:
# if I want 10**13 to be 100 years, and one time step to be one day then
# one day = 10**13/(100*365)

from csv import writer
import time


def TwoPatchesPredationBS_ivp_Kernel(y0,q,P,S,FW,d,deltaR,harvesting,tfinal,tinit,s,extinct = 1e-14):
        
    # create an extinction event:
    def extinction(t,y,q,P,S,FW,d,deltaR,harvesting,s):
        
        # extinction
        if ((y <= extinct) & (y != 0)).any():
            
            index = np.where(((y <= extinct) & (y != 0)))
            print(y[index])
            print(index)
            
            ## /!\ be careful here to not cause conflict with other 
            np.savetxt(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/Sp_extinct_event_{s}.csv",index)
            
            return 0 ## will terminate the ode solver
        
        # no extinction
        else: 
            return 1 ## solver can continue
                
    extinction.terminal = True # creates a termination event instead of just recordning the time step and biomass
    # extinction.direction = -1 # only species going down will trigger event, so invading species can invade
        
    
    def System(t,y,q,P,S,FW,d,deltaR,harvesting,s): # Function defining the system of differential equations
        
        a = FW['a'] # attack rate
        ht = FW['h'] # handling time
        x = FW['x'] # metabolic rate
        r = FW['r'] # growth rate
        K = FW['K'] + 0.00001 # just to prevent division by zero
        # dmax = FW["dmax"] # maximum dispersal distance 
        dd = FW["dd"]
        access = FW["access"]
        
        # empty population size
        dN = np.zeros(S*P).reshape(P,S)
        y = np.array(y).reshape(P,S)
        y[y < 0] = 0
        
        for p in range(P):
            # current patch
            N = y[p]  # previous population Np(t)
            dNp = np.zeros(S)  # initialising Np(t + 1)
            
            # neighbouring patches
            n = [j for j in range(P) if j!=p] # index of neighbouring patches
            # Nn = [y[j] for j in n]
            
            Nn = y[n,:] # biomass in neighbouting patches
            Nn = Nn*access[p] # filter species' biomass from patches that are within their maximum disp distance
            
            # =================================================================
            # (1) PREDATION
            # resources in rows, consumers in columns
            
            # Functionnal response: resource dependant feeding rate of consumer j on prey i influenced by j's k (other?) resources 
            # Fij = aij*Bi/(1+sum(aik*hik*Nk)) 
            # q = 1 # hill exponent (Sentis et al - between type II (hyperbolic q = 1) and type III (sigmoidal q = 2))
            # increased from 1.2 to 1.4 after suggestion of Benoit Gauzens for increasing persistence. Could also think about adding interference compeition.
            # who wrote ATNr package for allometric food webs
            
            F = a*(N.reshape(S,1)**(1 + q))
            low = 1 + np.matmul(np.transpose(a*ht), N**(1 + q)) 
            
            F = np.divide(F,low)
            
            predationGains = N*F.sum(axis = 0)*0.85 # xi*e*Ni*sum(Fik)
            predationLosses = np.matmul(N,F.T) # sum(xj*Nj*Fki)
            
            # =================================================================
            # (2) DISPERSAL 
            emigration = -d[p]*N # biomass that leaves the source patch (z)
            
            En = Nn*d[n,:] # emigrants leaving neighbouring patches 
            
            low_d = np.sum(1 - dd[n,:,:], axis = 0)
            Nim = np.sum(En * (1 -  dd[p]) * (1 -  dd[p])/low_d, axis = 0)
            
            # Nim = np.zeros(S) 
            # for ind in range(len(n)): # loop across neighbours 
            #     Nim+= (d[p]/(P-1))*Nn[ind]*connectivity[p,n[ind]] # biomass of each species in neighbouring patches that successfully disperse
            
            
            immigration = Nim # biomass not lost to the matrix during dispersal
            immigration[immigration<extinct] = 0
                  
            
            # print('immigration',immigration,'emigration', emigration)
            # print(N, 'Gains', predationGains, 'loss', predationLosses)
            
            # =================================================================
            
            # growth rate*logistic growth (>0 for plants only) - metabolic losses + predation gains (>0 for animals only) - predation losses + dispersal
            dNp = N*(r[p]*deltaR[p]*(1 - N/K) - x - harvesting[p]) + predationGains - predationLosses + immigration + emigration

                    
            # print(dNp)
            dNp[(N + dNp) < extinct] = 0 
            dN[p] = dNp
        # print(dN[0][33],dN[1][33],dN[2][33])
        return np.array(dN).reshape(P*S)
        
    
    # choose the start and end time
    t_span = (tinit,tfinal)
    # run solver
    return solve_ivp(fun = System, t_span = t_span, y0 = y0, args=(q,P,S,FW,d,deltaR,harvesting,s), 
                     method='LSODA', rtol = 1e-8, atol = 1e-8, events = extinction, t_eval = np.arange(tinit,tfinal,60*60*24*31)) # choose when extinctions and dispersal will be evaluated
    ### END FUNCTION


# =============================================================================
# run_dynamics()
#
# Runs simulations and re starts the dynamics when species go extinct after 
# setting their biomasses to zero and incrementing one timestep.
# It also checks if the simulations have stabilised (last 10% of biomasses have a cv < 1e-5) 
# and stops the simulations when that conditions is reached
# =============================================================================

def run_dynamics(y0,tinit,runtime,q,P,Stot,FW,disp,deltaR,harvesting,patch,sp,disturbance,s,sol_save={}):
    status = 10
    tstart = tinit
    stabilised = False
    count = 0
    
    start = time.time()    
    
    if min(deltaR) == max(deltaR):
        ty = 'Homogeneous'
    else:
        ty = 'Heterogeneous'

        
    while not stabilised:
        
        count+=1
        
        # print(tinit)
        sol = TwoPatchesPredationBS_ivp_Kernel(y0,q,P,Stot,FW,disp,deltaR,harvesting,tinit+runtime,tinit,s) # Call the model
        sol.y = sol.y.T
        
        print(tinit+runtime) # changed so that tmax extends with each extinction

        if tinit == tstart:
            sol_save = sol.copy()
            
        else:
            for i in ["y","t"]:
                sol_save[i] = np.concatenate((sol_save[i], sol[i])) 
                
        
        status = sol.status # update status of the solver (has it reach tfinal yet?)
        
        ## Checking if the dynamics have stablised
        if sol_save['y'].shape[0] > 1000: ## initial burn in period to let the dynamics unfold
            
            solY = sol_save['y']
            solT = sol_save['t']
            
            ### sliding window of variation of the mean biomass for each species
            mean_df = np.zeros((5,solY.shape[1])) ## initiate empty matrix to store results
            ind = 0 # counter to loop through the mean_df matrix
            for i in np.flip([0,1,2,3,4]): # we subset the last 25% timesteps into 5 equal length subsections [[75-80%],[80-85%],[85-90%],[90-95%],[95-100%]]
                
                thresh_high = (solT[-1] - (solT[-1] - solT[1])*(0.05*i)) # get higher time step boundary (tfinal - X% * tfinal)
                thresh_low = (solT[-1] - (solT[-1] - solT[1])*(0.05*(i + 1))) # get lower time step boundary (tfinal - (X + 1)% * tfinal)
        
                ## index of those time steps between low and high boundaries
                index = np.where((solT >= thresh_low) & (solT < thresh_high))[0]
                ## biomasses over those 10% time steps
                Bsub = solY[index]
                
                # save mean biomass per species across time
                mean_df[ind,:] = Bsub.mean(axis=0)
                ind+=1 # increment index
                
            # calculate coefficient of variation of each species's mean biomass 
            # across 25% of the simulation 
            cv = mean_df.std(axis=0)/mean_df.mean(axis=0)
           
            if (cv[~np.isnan(cv)] < 1e-2).all() : # if all the coefficient of variation are small enough we stop the simulation
                print('Stabilised')
                stabilised = True
                
                # plt.plot(mean_df)
                # plt.savefig('/lustrehome/home/s.lucie.thompson/Metacom/Figures/stabilisation.png', dpi = 400, bbox_inches = 'tight')
        
                break
                
        
        tinit = sol.t[-1].copy() + 1 # update the initial time step to the last time step of the latest run (keep the previous time step in and restart at t+1)
        
        
        # if the biomasses haven't stabilised:
        
        if ((status == 0) and (not stabilised)): # if it has reached tfinal (didn't end because of an extinction), then we extend the simulation
            
            print('extended')
            # tmax = tmax + (tmax - tstart)*0.10
            # y0 = sol.y[-1]
        
        
        if ((status == 1) and (not stabilised)):
            
            # get ID of species that went extinct and are to be set to zero Biomass
            ID = np.loadtxt(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/Sp_extinct_event_{s}.csv")
            np.savetxt(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/Sp_extinct_event_{s}.csv",[]) # erase information from file
    
            ID = ID.astype(int)
            print(ID)

            # update initial conditions, set to zero all extinct species      
            y0 = sol.y[-1].copy() # get latest vector of biomass
            y0[ID] = 0
            
            
            if sp*(patch+1) in ID:
                harvesting[patch][sp] = 0 # can't harvest an extinct species (otherwise yields negative biomasses)
            else:
                harvesting[patch][sp] = disturbance
        
        if count%200 == 0: # save progress every 200 extinctions
        
            sol_save_temp = sol_save.copy()
            sol_save_temp['y'] = sol_save_temp['y'][np.arange(0,sol_save_temp['y'].shape[0],10),:]
            sol_save_temp['t'] = sol_save_temp['t'][np.arange(0,sol_save_temp['t'].shape[0],10)]
            
            print('SAVING -- y = ',sol_save_temp['y'].shape,'t = ', sol_save_temp['t'].shape)
    
            ## save results
            sol_save_temp.update({'FW_new':FW, 'type':ty, "sim":s,"disp":disp,"harvesting":harvesting,"deltaR":deltaR,'q':q})
            np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/{ty}/sim{s}/temp_Patch{patch}_PopDynamics_{ty}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}_disturbance{disturbance}_{sp}SpDisturbed.npy',sol_save_temp, allow_pickle = True)
        
        ## simulation is taking too long we stop it - save it as a seperate file 
        ## need to investigate what is going on 
        if time.time() - start > 60*60*12:
            print(f'Took more than 12 hours - skipping {patch}, sp {sp}, {ty}')
            
            ## save results
            sol_save_temp.update({'FW_new':FW, 'type':ty, "sim":s,"disp":disp,"harvesting":harvesting,"deltaR":deltaR,'q':q})
            np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/{ty}/sim{s}/NotStabilised_Patch{patch}_PopDynamics_{ty}_sim{s}_{P}Patches_Stot100_C{int(C*100)}_disturbance{disturbance}_{sp}SpDisturbed.npy',sol_save_temp, allow_pickle = True)

            return 'Did not stabilise after 12 hours'
        
    return sol_save


# =============================================================================
# reduce_FW()
#
# subset the food web to only species that are present in the landscape
# =============================================================================
 
def reduce_FW(FW, y0, P, disp):
    
    Stot = FW['Stot']
    
    # subset only species that are present in the landscape
    boo = np.repeat(False, Stot)
    for i in range(P):
        boo = ((boo) | (y0[i]!=0))
    # get their index from the original food web
    ID_original = FW['sp_ID'][boo]  
    present = np.where(boo)[0]
    # realised species richness    
    Stot_new = len(present)
    # updated initial conditions
    y0_new = y0.reshape(Stot*P,)[np.tile(present, P) + np.repeat(np.arange(P)*Stot, Stot_new)].reshape(P,Stot_new)
    # check - should be none
    # print(np.where((y0_new[0]==0) & (y0_new[1]==0) & (y0_new[2]==0))[0])
    
    ## subset the food webs and all its parameters
    FW_new = FW.copy()
    FW_new['Stot'] = Stot_new
    FW_new['M'] = FW_new['M'][np.ix_(present,present)]
    FW_new['a'] = FW_new['a'][np.ix_(present,present)]
    FW_new['h'] = FW_new['h'][np.ix_(present,present)]
    FW_new['r'] = FW_new['r'].reshape(Stot*P,)[np.tile(present, P) + np.repeat(np.arange(P)*Stot, Stot_new)].reshape(P,Stot_new)
    FW_new['x'] = FW_new['x'][present]
    FW_new['K'] = FW_new['K'][present]
    FW_new['TL'] = FW_new['TL'][present]
    
    access = np.repeat(np.zeros(shape = (P-1, Stot_new)), P).reshape(P,P-1,Stot_new)
    dd = np.repeat(np.zeros(shape = (P-1, Stot_new)), P).reshape(P,P-1,Stot_new)
    for i in range(P):
        access[i] = FW_new['access'][i][:,present]
        dd[i] = FW_new['dd'][i][:,present]
    
    FW_new['access'] = access
    FW_new['dd'] = dd
    
    # keep in memory which species we kept from the regional food web
    FW_new['sp_ID'] = ID_original
    FW_new['sp_ID_'] = present # sp ID for the subsetted food web 
    FW_new['y0'] = y0_new
    disp_new = disp.reshape(Stot*P,)[np.tile(present, P) + np.repeat(np.arange(P)*Stot, Stot_new)].reshape(P,Stot_new)
    
    return Stot_new, FW_new, disp_new







# %% initial run homogeneous

## Need to create 3 folders where to save simulation outputs
## these should be named ~/{P}Patches then ~/{P}Patches/Homogeneous and ~/{P}Patches/Heterogeneous
## inside each Homogeneous and Heterogeneous folder, create subfolders called sim1 to sim10
## this can be done through in git bash through the simple bash command:
    ## (1) cd to the desired folder (e.g cd ~/{P}Patches/Heterogeneous) 
    ## (2) then run :
    ## for i in {0..10}; do
    ##    mkdir "sim$i"
    ## done


f = "/lustrehome/home/s.lucie.thompson/Metacom/Init_test/StableFoodWebs_55persist_Stot100_C10_t10000000000000.npy"
stableFW = np.load(f, allow_pickle=True).item()


Stot = 100 # initial pool of species
P = 9 # number of patches
C = 0.1 # connectance
tmax = 10**12 # number of time steps

# FW = nicheNetwork(Stot, C)

# S = np.repeat(Stot,P) # local pool sizes (all patches are full for now)
S = np.repeat(round(Stot*1/3),P) # initiate with 50 patches

parser = ArgumentParser()
parser.add_argument('SimNb')
args = parser.parse_args()
s = int(args.SimNb)
print(args, s, flush=True)

# s = 3

import os 

## create file to load extinction events
path = f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/Sp_extinct_event_{s}.csv"
if not os.path.exists(path):
    file = open(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/Sp_extinct_event_{s}.csv","x") # create a file
    file.close()
    
    
print('simulation',s, flush=True)

k = [k for k in stableFW.keys()]
k = k[s]
FW = stableFW[k]

## create three patches distant enough that all species can't cross all the way through
extentx = [0,0.4]
coords = pd.DataFrame()
for i in range(P):
    np.random.seed(i)
    coords = pd.concat((coords, pd.DataFrame({'Patch':[i],'x':[np.random.uniform(extentx[0], extentx[1])],
                              'y':[np.random.uniform(extentx[0], extentx[1])]})))


plt.scatter(coords['x'], coords['y'], c = coords['Patch'])
plt.colorbar()

FW = getSimParams(FW, S, P, coords)

## calculate distance between patches
dist = np.zeros((P,P))

for p1 in range(P): 
    for p2 in range(P):
        dist[p1,p2] = ((coords['x'].iloc[p2] - coords['x'].iloc[p1])**2 + (coords['y'].iloc[p2] - coords['y'].iloc[p1])**2)**(1/2)

np.median(dist) ## should be close to np.median(FW['dmax'])

plt.xscale("log")
plt.scatter(FW["BS"], FW["dmax"])
plt.ylabel("dispersal distance")
plt.xlabel("Logged body size")
plt.show()

# dispersal rate
d = 1e-8
disp = np.repeat(d, Stot*P).reshape(P,Stot)

q = 0.1 # hill exponent - type II functionnal response (chosen following Ryser et al 2021)


Stot_new, FW_new, disp_new = reduce_FW(FW, FW['y0'], P, disp)
# no havresting for the 'warm up' period
harvesting = np.zeros(shape = (P, Stot_new))




###############################################################################
###############################################################################
### HETEROGENEOUS
###############################################################################
###############################################################################


# %% Habitat restoration test


# path = f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Homogeneous/sim{s}/InitialPopDynamics_homogeneous_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.npy'


# ## load initial dynamics with all low patch quality
# sol_homogeneous = np.load(path, allow_pickle = True).item()

# ## get final species composition in order to subset only surviving species from the 
# ## regional pool
# Bf_homogeneous = np.zeros(shape = (P,Stot_new))
# for patch in range(P):
    
#     p_index = patch + 1
    
#     # sol_ivp_k1["y"][sol_ivp_k1["y"]<1e-20] = 0         
#     solT = sol_homogeneous['t']
#     solY = sol_homogeneous['y']
#     ind = patch + 1
    
#     Bf_homogeneous[patch] = solY[-1,range(Stot_new*ind-Stot_new,Stot_new*ind)]
    
# Bf_homogeneous_restored = Bf_homogeneous.copy()
# tstart = sol_homogeneous["t"][-1].copy()
# tinit = tstart.copy()
# runtime = 1e11

# ty = 'Homogeneous'

# # Get reduced space
# Stot_homogeneous_new, FW_homogeneous_new, disp_homogeneous_new = reduce_FW(FW_new, Bf_homogeneous_restored, P, disp_new)


# res_sim_sp_homogeneous = pd.DataFrame()

# ## create file to load extinction events
# path = f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/Sp_extinct_event_{s}.csv"
# if not os.path.exists(path):
#     file = open(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/Sp_extinct_event_{s}.csv","x") # create a file
#     file.close()
    
    
    


# import random
# for rdseed in range(10): # TO TEST DIFFERENT LANDSCAPE CONFIGURATIONS

#     random.seed(rdseed) # set seed for repeatability
#     high_qual_patches = random.sample(range(P),3)

#     deltaR = np.repeat(0.5,P)
#     deltaR[high_qual_patches] = 1
    
#     # plt.scatter(coords['x'], coords['y'], c = deltaR)
    
#     # for deltaR in deltaR: # if want to do different ratios
    
#     ratio = np.min(deltaR)/np.max(deltaR)
    
#     harvesting = np.zeros(shape = (P, Stot_homogeneous_new))
    
#     path = f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/InitialPopDynamics_seed{rdseed}_heterogeneous_ratio{ratio}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.npy'
#     if not os.path.exists(path):
    
#         print('Heterogeneous - initial run',s, flush=True)
        
#         y0 = FW_homogeneous_new['y0'].reshape(P*Stot_homogeneous_new)
        
#         start = time.time()   
#         sol_restored = run_dynamics(y0, tstart, tinit + runtime, q, P, Stot_homogeneous_new, FW_homogeneous_new, disp_homogeneous_new, deltaR, harvesting, -1, -1, 0, s)
#         stop = time.time()   
#         print(stop - start)

#         sol_restored.update({'FW':FW, 'y0':y0, 'type':'heterogeneous', 'Stot_new':Stot_homogeneous_new, 'FW_new':FW_homogeneous_new, 'disp_new':disp_homogeneous_new,"sim":s,
#                                   'FW_ID':k,"FW_file":f,"disp":disp,"harvesting":harvesting,"deltaR":deltaR,'ratio':ratio,
#                                   'tstart':0, 'tmax':tmax,'q':q, 'seed':rdseed})
#         np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/InitialPopDynamics_seed{rdseed}_heterogeneous_ratio{ratio}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.npy',sol_restored, allow_pickle = True)
    
 
    
 
    
 
# %% High habitat quality from the start (no restoration - similar to the patch model simulation)


import random
for rdseed in range(10): # TO TEST DIFFERENT LANDSCAPE CONFIGURATIONS

    random.seed(rdseed) # set seed for repeatability
    high_qual_patches = random.sample(range(P),3)

    deltaR = np.repeat(0.5,P)
    deltaR[high_qual_patches] = 1
    
    # plt.scatter(coords['x'], coords['y'], c = deltaR)
    
    # for deltaR in deltaR: # if want to do different ratios
    
    ratio = np.min(deltaR)/np.max(deltaR)
    
    harvesting = np.zeros(shape = (P, Stot_new))
    
    path = f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/InitialPopDynamics_seed{rdseed}_heterogeneous_ratio{ratio}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.npy'
    if not os.path.exists(path):
    
        print('Heterogeneous - initial run',s, flush=True)
        
        
        start = time.time()    
        sol = run_dynamics(FW_new['y0'].reshape(Stot_new*P,),0,tmax,q,P,Stot_new,FW_new,disp_new,deltaR,harvesting,-1,-1,0,s)
        stop = time.time()   
        print(stop - start)

        sol.update({'FW':FW, 'y0':FW_new['y0'].reshape(Stot_new*P,), 'type':'heterogeneous', 'Stot_new':Stot_new, 'FW_new':FW_new, 'disp_new':disp_new,"sim":s,
                                  'FW_ID':k,"FW_file":f,"disp":disp,"harvesting":harvesting,"deltaR":deltaR,'ratio':ratio,
                                  'tstart':0, 'tmax':tmax,'q':q, 'seed':rdseed})
        np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/TestPAConfig/InitialPopDynamics_seed{rdseed}_heterogeneous_fromZero_ratio{ratio}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.npy',sol, allow_pickle = True)
    
 