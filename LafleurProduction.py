#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:14:34 2026

@author: faunw
"""

from ssapy import *
import numpy as np
import scipy.constants as Cons
import matplotlib.pyplot as plt
from functools import lru_cache

from ssapy.constants import EARTH_MU, EARTH_RADIUS
from ssapy.utils import norm, sunPos, _gpsToTT, ntw_to_r
from ssapy.ellipsoid import Ellipsoid
import ModifiedAccel
import time as tm
try:
    import erfa
except ImportError:
    # Let this raise
    import astropy._erfa as erfa
def SaveFile(TA,AA,SatDict): #rw,phi,satnum,Alt):
    Init_Alt=SatDict['Alt']
    Name=SatDict['Name']
    r_w=SatDict['r_w']
    phi_B=SatDict['Phi']
    filename = f"StepperCSV/Deorbit_{Name}_{Init_Alt}km_rw{r_w}_phi{phi_B}.npz"
    np.savez(filename, Times=TA, Alts=AA, Alt=Alt, rw=rw, phi=phi)

def loadFile(SatDict):
    Init_Alt=SatDict['Alt']
    Name=SatDict['Name']
    r_w=SatDict['r_w']
    phi_B=SatDict['Phi']
    filename = f"StepperCSV/Deorbit_{Name}_{Init_Alt}km_rw{r_w}_phi{phi_B}.npz"
    data = np.load(filename)
    Times = data['Times']
    Alts = data['Alts']
    return [Times,Alts]
def plotDeorbit(TA,AA):
    from scipy.ndimage import gaussian_filter1d
    plt.figure()
    TA=TA[:len(AA)]
    y_smooth = gaussian_filter1d(AA, sigma=100)


    plt.plot(TA,AA)
    plt.plot(TA,y_smooth)

    plt.xlabel('Time(days)')
    plt.ylabel('Altitude (km)')
    plt.legend
    plt.show()
    
def StandardOrbitChargedTest(SatDict):
    t0 = Time("2000-10-03")
    Init_Alt=SatDict['Alt']
    Name=SatDict['Name']
    r_w=SatDict['r_w']
    l_w=SatDict['l_w']
    phi_B=SatDict['Phi']
    Area_Max=SatDict['Max_Area']

    RE=6370000
    a = RE+Init_Alt*1000 #constants.RGEO
    e = 0.0013
    i = np.radians(57.0)
    pa = np.radians(130)#argument of perigee?
    raan = np.radians(247)
    ta = np.radians(15)
    Frequency=4 #Minute
    
    sat_kwargs = dict(
        mass=3.5,  # [kg]
        area=0.01,  # 
        CD=2.2,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
        CC=2,
        lw=l_w,
        rw=r_w,
        phi=phi_B,
        name=Name,
        RSI=int(24*60*20/Frequency) #every 20 days
    )
    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    sun = get_body("Sun")
    Earth = get_body("Earth", model="EGM2008")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)

    ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)

    accel = aEarth  +ADrag
    prop = SciPyPropagator(accel)
    times1 = utils.get_times(duration=(1, 'day'), freq=(Frequency, 'minute'), t0=t0)
    T1=tm.time()
    r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
    T2=tm.time()

    TArr1=[(j-t0).to_value('sec')/3600/24 for j in times1]
    RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in r1]
    print(TArr1)
    print(RArr1)
    SaveFile(TArr1, RArr1, SatDict)
    print('Sim time:'+str((T2-T1)/60)+" min")
def ProductionLoop(satdict):
    StandardOrbitChargedTest(TestSat)
    Data=loadFile(TestSat)
    plotDeorbit(Data[0], Data[1])
TestSat=dict(Alt=400,Name='TestSat',r_w=0.05,l_w=40,Phi=30,Max_Area=40)
StandardOrbitChargedTest(TestSat)
Data=loadFile(TestSat)

