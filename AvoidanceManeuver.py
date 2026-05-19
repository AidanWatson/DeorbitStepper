#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:33:41 2026

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

def itrs_to_ric(r_chief, v_chief, r_deputy):
    """
    Convert ITRS Cartesian positions to RIC (Radial, In-track, Cross-track)
    relative coordinates.
 
    Parameters
    ----------
    r_chief : array-like, shape (N, 3)
        ITRS position vectors of the chief satellite [m or km].
    v_chief : array-like, shape (N, 3)
        ITRS velocity vectors of the chief satellite [m/s or km/s].
    r_deputy : array-like, shape (N, 3)
        ITRS position vectors of the deputy (chaser) satellite [m or km].
 
    Returns
    -------
    ric : np.ndarray, shape (N, 3)
        Relative position in the RIC frame [same units as input].
        Columns: [Radial, In-track, Cross-track]
    """
    r_chief = np.asarray(r_chief, dtype=float)
    v_chief = np.asarray(v_chief, dtype=float)
    r_deputy = np.asarray(r_deputy, dtype=float)
 
    if r_chief.ndim == 1:
        r_chief  = r_chief[np.newaxis, :]
        v_chief  = v_chief[np.newaxis, :]
        r_deputy = r_deputy[np.newaxis, :]
 
    N = r_chief.shape[0]
    ric = np.empty((N, 3))
 
    for i in range(N):
        r = r_chief[i]
        v = v_chief[i]
        delta = r_deputy[i] - r          # relative position in ITRS
 
        # --- Build the RIC unit vectors ---
        r_hat = r / np.linalg.norm(r)    # Radial: outward from Earth centre
 
        h = np.cross(r, v)               # angular momentum vector
        c_hat = h / np.linalg.norm(h)    # Cross-track: normal to orbital plane
 
        i_hat = np.cross(c_hat, r_hat)   # In-track: completes right-hand frame
 
        # Rotation matrix (rows = RIC axes expressed in ITRS)
        R = np.array([r_hat, i_hat, c_hat])
 
        ric[i] = R @ delta               # project relative position onto RIC axes
 
    return ric
def SingleOrbitTest(SatDict):
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
        Max_Area=Area_Max,
        RSI=int(0*24*60*20/Frequency) #every 20 days #fudge factor of 20 added in because i realised it calls the function more than once per timestep
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
    print('Final Alt: '+str(RArr1[-1]))
    # print(TArr1)
    print(r1)
    print('Sim time:'+str((T2-T1)/60)+" min")

def TestDivergence(SatDict1,SatDict2):
    t0 = Time("2000-10-03")
    Init_Alt1=SatDict1['Alt']
    Name1=SatDict1['Name']
    r_w1=SatDict1['r_w']
    l_w1=SatDict1['l_w']
    phi_B1=SatDict1['Phi']
    Area_Max1=SatDict1['Max_Area']
    Init_Alt2=SatDict2['Alt']
    Name2=SatDict2['Name']
    r_w2=SatDict2['r_w']
    l_w2=SatDict2['l_w']
    phi_B2=SatDict2['Phi']
    Area_Max2=SatDict2['Max_Area']
    RE=6370000
    a = RE+Init_Alt1*1000 #constants.RGEO
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
        lw=l_w1,
        rw=r_w1,
        phi=phi_B1,
        name=Name1,
        Max_Area=Area_Max1,
        RSI=int(0*24*60*20/Frequency) #every 20 days #fudge factor of 20 added in because i realised it calls the function more than once per timestep
    )
    sat_kwargs2 = dict(
        mass=3.5,  # [kg]
        area=0.01,  # 
        CD=2.2,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
        CC=2,
        lw=l_w2,
        rw=r_w2,
        phi=phi_B2,
        name=Name2,
        Max_Area=Area_Max2,
        RSI=int(0*24*60*20/Frequency) #every 20 days #fudge factor of 20 added in because i realised it calls the function more than once per timestep
    )
    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    sun = get_body("Sun")
    Earth = get_body("Earth", model="EGM2008")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
    ADrag1=ModifiedAccel.AccelDrag2(**sat_kwargs)
    ADrag2=ModifiedAccel.AccelDrag2(**sat_kwargs2)

    accel1 = aEarth  +ADrag1
    accel2 = aEarth  +ADrag2

    prop1 = SciPyPropagator(accel1)
    prop2 = SciPyPropagator(accel2)

    times1 = utils.get_times(duration=(1, 'hour'), freq=(Frequency, 'minute'), t0=t0)
    T1=tm.time()
    
    r1, v1 = rv(orbit=orbit, time=times1, propagator=prop1)
    r2, v2 = rv(orbit=orbit, time=times1, propagator=prop2)

    T2=tm.time()

    TArr1=[(j-t0).to_value('sec')/3600/24 for j in times1]

    RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in r1]
    RArr2=[(np.linalg.norm(j)-6370000)/1000 for j in r2]
    plt.figure()
    plt.plot(TArr1,RArr1)
    plt.plot(TArr1,RArr2)
    plt.show()
    plt.figure()
    plt.plot(TArr1,[RArr2[i]-RArr1[i] for i in range(len(RArr1))] )
    # plt.plot(TArr1,RArr2)
    plt.show()
    # plotUtils.ground_track_plot(r1, times1)
    # plotUtils.ground_track_plot(r2, times1)
    # plotUtils.ground_track_plot(r1, times1, 1)
    plotUtils.ground_track_plot2(r1,r2, times1)

    print('Final Alt: '+str(RArr1[-1]))
    # print(TArr1)
    print(r1)
    print(r2)
    # SaveFile(TArr1, RArr1, SatDict)
    print('Sim time:'+str((T2-T1)/60)+" min")
    return TArr1,r1, r2, v1
# def ProductionLoop(satdict):
#     StandardOrbitChargedTest(satdict)
#     Data=loadFile(satdict)
#     plotDeorbit(Data[0], Data[1])

TestSat=dict(Alt=600,Name='TestSat',r_w=0.005,l_w=4.8,Phi=0.1,Max_Area=40)
F1=dict(Alt=600,Name='F1',r_w=0.005,l_w=4.8,Phi=250,Max_Area=1.98) #650-629->605->549->360
F1Pair=dict(Alt=600,Name='F1',r_w=0.005,l_w=20.8,Phi=250,Max_Area=100.98) #650-629->605->549->360

F1N=dict(Alt=457,Name='F1N',r_w=0.005,l_w=2.4,Phi=0.01,Max_Area=1.98) #650->637->621->602->576->528->457

# ProductionLoop(F1N)
# SingleOrbitTest(F1)
# TestFunc(TestSat)
# SetupLafleurTest(F5)w
# StandardOrbitChargedTest(TestSat)
# Data=loadFile(TestSat)
T,R1,R2,V1=TestDivergence(F1,F1Pair)
RIC=itrs_to_ric(R1,V1, R2)
print(RIC)
plt.figure()
plt.plot(T,[RIC[i][0] for i in range(len(RIC))])
plt.show()
