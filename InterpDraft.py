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
    latRes=6
    lonRes=6
    altRes=10
    lats=np.linspace(-90,90,latRes)
    lons=np.linspace(-180,180,lonRes)
    if altmin<30:
        altmin=30
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
        # print('rhom: '+str(RhoM[i,j,k]))
        RhoNI[i, j, k] = X1['N_i'].iloc[0]
        AvgMassI[i, j, k] = X1['Av_M'].iloc[0]
    T2=tm.time()
    
    interpM = RegularGridInterpolator((lats, lons, alts),RhoM   ,method='linear',fill_value=np.nan)
    interpI = RegularGridInterpolator((lats, lons, alts),RhoNI,method='linear',bounds_error=False,fill_value=np.nan)   
    interpAvgM = RegularGridInterpolator((lats, lons, alts),AvgMassI,method='cubic',bounds_error=False,fill_value=np.nan)   
    T3=tm.time()
    print('Scan time: '+str(T2-T1)+'s')
    print('Interpolation time: '+str(T3-T2)+ 's')
    # print(X)
    return interpM,interpI,interpAvgM
def TestAccuracy():
    dt=datetime(2022, 8, 22, 18, 0).astimezone(UTC)
    alts=np.linspace(200,700,5)
    interpArr=MakeInterpolator(dt, 200,700)
    InterpRhoM=interpArr[0]
    interpNI=interpArr[1]
    interpAM=interpArr[2]
    lat=40.9
    lon=174
    lat=-90
    lon=-180
    Alts2=np.linspace(200,700,20)

    X1=newPyglow.DensitySweepAlt(lat, lon, np.array(Alts2),dt)
    
    print(X1)
    rhomn =interpNI([lat,lon,500])
    rhomn =InterpRhoM([lat,lon,500])

    # NQ=float(interpNI([lat,lon,height])[0])
    # AVM=float(interpAM([lat,lon,height])[0])
    print(rhomn)
    Alts=[i for i in X1['alts']]
    Avm1=[i for i in X1['Av_M']]
    Mn1=[i for i in X1['M_n']]
    Ni1=[i for i in X1['N_i']]
    # MNARR=[InterpRhoM([lat,lon,i]) for i in Alts]
    Alts2=np.linspace(200,700,20)
    MNARR=[interpAM([[lat,lon,i]]) for i in Alts2]
    # X2=newPyglow.DensitySweepAlt(lat, lon, np.array(Alts2),dt)
    
    print(MNARR)
    print(interpNI([[0,0,300]]))

    plt.figure()
    plt.scatter(Alts2,MNARR,label='Interpolated')
    plt.scatter(Alts,Avm1,label='Calculated')
    plt.xlabel('Alt (km)')
    plt.legend()
    # plt.yscale('log')
    plt.ylabel('AvM')
    plt.show()
    print(Alts)
# TestAccuracy()
# dn=datetime(2022, 8, 22, 18, 0).astimezone(UTC)

# IT=MakeInterpolator(dn,400,600)
# print(IT([0,0,500]))
    
    
# def TestInterp():
#     ITP= MakeInterpolator(Time("2017-04-05"), 661-250,661+50)
    
    # return 0