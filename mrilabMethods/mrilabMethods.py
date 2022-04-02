# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:10:48 2022

@author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
@email: josalggui@i3m.upv.es
"""
import numpy as np
from configs.hw_config import Gx_factor
from configs.hw_config import Gy_factor
from configs.hw_config import Gz_factor

##############################################################
##############################################################
##############################################################


def getIndex(etl=1, nPH=1, sweepMode=1):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Create 'ind' array that give you the order to sweep the k-space phase lines along an echo train length.
    sweepMode = 0: -kMax to kMax
    sweepMode = 1: 0 to kMax
    sweepMode = 2: kMax to 0
    sweepMode = 3: Niquist modulated method
    """
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


##############################################################
##############################################################
##############################################################


def reorganizeGfactor(axes):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Create an array of 3 elements with the gradient conversion factor corresponding to the given axes order
    """
    
    # Set the normalization factor for readout, phase and slice gradient
    gFactor = np.array([0., 0., 0.])
    for ii in range(3):
        if axes[ii]==0:
            gFactor[ii] = Gx_factor
        elif axes[ii]==1:
            gFactor[ii] = Gy_factor
        elif axes[ii]==2:
            gFactor[ii] = Gz_factor
    
    return(gFactor)


##############################################################
##############################################################
##############################################################


def fixEchoPosition(echoes, data0):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Oversampled data obtained with a given echo train length and readout gradient only is used here to determine the true position of k=0.
    After getting the position of k = 0 for each gradient-spin-echo, it shift the sampled data to place k = 0 at the center of each acquisition window.
    """
    
    etl = np.size(echoes, axis=0)
    n = np.size(echoes, axis=1)
    idx = np.argmax(np.abs(echoes), axis=1)
    idx = idx-int(n/2)
    data1 = data0*0
    for ii in range(etl):
        if idx[ii]>0:
            idx[ii] = 0
        echoes[ii, -idx[ii]::] = echoes[ii, 0:n+idx[ii]]
        data1[:, ii, -idx[ii]::] = data0[:, ii, 0:n+idx[ii]]
    return(data1)
