# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:10:48 2022

@author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
@email: josalggui@i3m.upv.es
"""
import numpy as np

def getIndex(etl=1, nPH=1, sweepMode=1):
    
    ##########################################################################
    # Create 'ind' array that give you the order to sweep the k-space phase 
    # lines along an echo train length.
    # sweepMode = 0: -kMax to kMax
    # sweepMode = 1: 0 to kMax
    # sweepMode = 2: kMax to 0
    # sweepMode = 3: Niquist modulated method
    ##########################################################################
    
    n2ETL=int(nPH/2/etl)
    ind = []
    if nPH==1:
         ind = 0
    else: 
        if sweepMode==0:   # Sequential for T2 contrast
            for ii in range(int(nPH/etl)):
                ind = np.concatenate((ind, np.linspace(ii, nPH+ii, num=etl, endpoint=False)), axis=0)
            ind = ind[::-1]
        elif sweepMode==1: # Center-out for T1 contrast
            if etl==nPH:
                ind = np.zeros(nPH)
                ind[0::2] = np.linspace(int(nPH/2), nPH, num=int(nPH/2), endpoint=False)
                ind[1::2] = np.linspace(int(nPH/2)-1, -1, num=int(nPH/2),  endpoint=False)
            else:
                for ii in range(n2ETL):
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)+ii, nPH+ii, num=etl, endpoint=False)), axis=0)
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)-ii-1, -ii-1, num=etl, endpoint=False)), axis=0)
        elif sweepMode==2: # Out-to-center for T2 contrast
            if etl==nPH:
                ind = np.zeros(nPH)
                ind[0::2] = np.linspace(int(nPH/2), nPH, num=int(nPH/2), endpoint=False)
                ind[1::2] = np.linspace(int(nPH/2)-1, -1, num=int(nPH/2),  endpoint=False)
            else:
                for ii in range(n2ETL):
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)+ii, nPH+ii, num=etl, endpoint=False)), axis=0)
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)-ii-1, -ii-1, num=etl, endpoint=False)), axis=0)
            ind = ind[::-1]
        elif sweepMode==3:  # Niquist modulated to reduce ghosting artifact
            if etl==nPH:
                ind = np.arange(0, nPH, 1)
            else:
                for ii in range(int(n2ETL)):
                    ind = np.concatenate((ind, np.arange(0, nPH, 2*n2ETL)+2*ii), axis=0)
                    ind = np.concatenate((ind, np.arange(nPH-1, 0, -2*n2ETL)-2*ii), axis=0)

    return np.int32(ind)