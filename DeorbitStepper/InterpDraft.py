#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:09:47 2026

@author: faunw
"""

import newPyglow
import matplotlib.pyplot as plt
from datetime import datetime, UTC

import time as tm
import scipy.constants as Cons
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# def returnData(date,altitudes)

def MakeInterpolator(date,altmin,altmax):
    #needs to take a selected number of points from across the earth at a range of altitudes 
    #for one time point (initially planning to recalculate monthly)
    #should also be fast and return the right values
    #
    # latRes=7
    # lonRes=5
    # altRes=5
    latRes=3
    lonRes=3
    altRes=5
    lats=np.linspace(-90,90,latRes)
    lons=np.linspace(-180,180,lonRes)
    if altmin<200:
        altmin=200
    if altmax>1000:
        altmax=1000
    alts=np.linspace(altmin,altmax,altRes)
    shape = (len(lats), len(lons), len(alts))
    RhoM = np.empty(shape)
    RhoNI=np.empty(shape)
    AvgMassI=np.empty(shape)
    # LAT, LON, ALT = np.meshgrid(lats, lons, alts, indexing='ij')
    # X = f(LAT, LON, ALT)  # your function — must accept arrays
    T1=tm.time()
    for i, j, k in np.ndindex(shape):
        # X[i, j, k] = f(lats[i], lons[j], alts[k])
        print('____')
        X1=newPyglow.DensitySweep(np.array([lats[i]]), np.array([lons[j]]), np.array([alts[k]]),date)
        # print(X1)
        RhoM[i, j, k] = X1['M_n'].iloc[0]
        RhoNI[i, j, k] = X1['N_i'].iloc[0]
        AvgMassI[i, j, k] = X1['Av_M'].iloc[0]
    T2=tm.time()
    
    interpM = RegularGridInterpolator((lats, lons, alts),RhoM,method='linear',bounds_error=False,fill_value=np.nan)
    interpI = RegularGridInterpolator((lats, lons, alts),RhoNI,method='linear',bounds_error=False,fill_value=np.nan)   
    interpAvgM = RegularGridInterpolator((lats, lons, alts),AvgMassI,method='linear',bounds_error=False,fill_value=np.nan)   
    T3=tm.time()
    print('Scan time: '+str(T2-T1)+'s')
    print('Interpolation time: '+str(T3-T2)+ 's')
    # print(X)
    return interpM,interpI,interpAvgM
# dn=datetime(2022, 8, 22, 18, 0).astimezone(UTC)

# IT=MakeInterpolator(dn,400,600)
# print(IT([0,0,500]))
    
    
# def TestInterp():
#     ITP= MakeInterpolator(Time("2017-04-05"), 661-250,661+50)
    
    # return 0