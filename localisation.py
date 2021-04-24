# !/usr/bin/env python
# 
# 'localisation.py' has a simple implementation of calculation for
#  disorder-averaged amplitudes of a 1d-tight binding model with 
#  the nearest neighbor couplings being perturbed by disorder. 
#
#
# MIT License. Copyright (c) 2020 Vijay Mocherla
#
# Source code at 
# <htts://github.com/vijaymocherla/Localisation-in-Open-quantum-systems.git>

# Importing a few packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from numpy import linalg
from scipy import integrate
import time as Tclock

def Hamiltonian(N,V):
    """ Generates the 'N'-dimensional Hamiltonian matrix of
        a 1-d tight binding model for an array nearest neighbor site couplings 'V'
    """
    H = np.zeros((N,N))
    H[0][1] = V[0]
    H[N-1][N-2] = V[N-1]
    for i in range(1,N-1):
        H[i][i+1] = V[i]
        H[i][i-1] = V[i]
    H = np.eye(N) + H
    return(H)


def conf_run(N,seed_loc,params):
    """  This function returns the matrix object at every time point for a certain realisation of disorder """
    K,T,tSteps = params
    x = 1
    time = np.linspace(0,T,tSteps)
    V = 2*np.random.random(N) + (x-1)
    vals,vecs = linalg.eigh(Hamiltonian(N,V))
    trans_prob = lambda t : (np.abs( np.array( [ vecs[i][seed_loc-1]*vecs[i]*np.exp(1j*vals[i]*t) for i in range(N)] ) )**2).sum(axis=0)
    data = np.array([trans_prob(t) for t in time])
    return(data)

def disorder_avg(N,seed_loc,params):
    K,T,tSteps = params
    time = np.linspace(0,T,tSteps)
    davg_tps = np.zeros((tSteps,N))
    st1 = Tclock.time()
    # In the following loop, we generation random outcomes K times and add to davg_tps.
    # davg_tps is then averaged by total no. cycles.
    for i in range(K):
        #st2 = Tclock.time()
        davg_tps+=conf_run(N,seed_loc,params)
        #e2 = Tclock.time()    
        #print(e2-st2)

    davg_tps = davg_tps/K
    
    e1 = Tclock.time()
    print('time per cycle : ',(e1-st1)/K )
    print('time entire run in mins : ', (e1-st1)/60 )
    return(davg_tps,time)

def main(N,params):
    K,T,tSteps = params
    data,time = disorder_avg(N,int(N/2 +1),params)
    data,time = disorder_avg(N,int(6),params) 
    sort = [data[:,i:i+1].flat for i in range(N)]
    time_avg  = [np.trapz(array,time)/params[1] for array in sort]
    print('Total Probability for consistency check, (this should be 1.0):',sum(time_avg))
    #population_data[(N,100)] = time_avg
    np.savetxt('N_'+str(N)+'_conf_'+str(K)+'_T_'+str(T)+'data',data)
    fig = plt.figure()
    plt.plot(range(1,N+1),time_avg,':',marker='o')
    plt.xlabel('n')
    plt.ylabel(r'$\langle P(n) \rangle_{\Theta,\tau} $')
    plt.title('N ='+str(N)+' averaged over '+str(K)+' configurations')
    fig.savefig('N_'+str(N)+'_conf_'+str(K)+'_T_'+str(T)+'.png')
    return(time_avg)

if __name__ == '__main__':
    N = int(sys.argv[2])
    K = int(sys.argv[4])
    T = int(sys.argv[6])
    #population_data = dict()
    tSteps = int(sys.argv[8])
    params = (K,T,tSteps)
    print('works!',N,params)
    data = main(N,params)
    print(data)

    
