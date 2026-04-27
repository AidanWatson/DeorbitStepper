#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:40:29 2026

@author: faunw
"""


# print('hello world')
# import ssapy
from ssapy import *
import numpy as np
import scipy.constants as Cons
import matplotlib.pyplot as plt
from functools import lru_cache

from ssapy.constants import EARTH_MU, EARTH_RADIUS
from ssapy.utils import norm, sunPos, _gpsToTT, ntw_to_r
from ssapy.ellipsoid import Ellipsoid
import ModifiedAccel
try:
    import erfa
except ImportError:
    # Let this raise
    import astropy._erfa as erfa

def TestFrequency():
    t0 = Time("2024-1-1")
    
    RE=6370000
    a = RE+450*1000 #constants.RGEO
    # a=constants.RGEO
    e = 0.00065
    i = np.radians(51)
    pa = np.radians(130)#argument of perigee?
    raan = np.radians(247)
    ta = np.radians(325)
    
    # e = 0.00065
    # i = np.radians(0)
    # pa = np.radians(0)#argument of perigee?
    # raan = np.radians(0)
    # ta = np.radians(325)
    # sat_kwargs = dict(
    #         mass=4e4,  # [kg]
    #         area=1000,  # [m^2]
    #         CD=2.3,  # Drag coefficient
    #         CR=1.3,  # Radiation pressure coefficient
    #         CC=2
    # )
    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    
    
    # moon = get_body("moon")
    sun = get_body("Sun")
    
    Earth = get_body("Earth", model="EGM2008")
    
    Frequency=[1,2,4,8,16,32,64,128,256,312]
    # Frequency=[1,8]
    times=[]
    RArr=[] 
    for i in range(len(Frequency)):
        sat_kwargs = dict(
                mass=4e4,  # [kg]
                area=1000,  # [m^2]
                CD=2.3,  # Drag coefficient
                CR=1.3,  # Radiation pressure coefficient
                CC=2,
                RSI=int((24*60)/Frequency[i]) #once every 24 hours
        )
        aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
    
        ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)
        accel = aEarth  +ADrag
        prop = SciPyPropagator(accel)
        
        times1 = utils.get_times(duration=(3, 'day'), freq=(Frequency[i], 'minute'), t0=t0)
        r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
        times.append(times1)
        RArr.append(r1)
    plt.figure()
    for i in range(len(Frequency)):
        TArr1=[(j-t0).to_value('sec')/3600 for j in times[i]]
        RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in RArr[i]]
        plt.plot(TArr1,RArr1,label=str(Frequency[i])+" min")
    plt.xlabel('Time(Hours)')
    plt.ylabel('Altitude (km)')
    plt.legend
    
    plt.show()
    
    plt.figure()
    for i in range(len(Frequency)):
        # TArr1=[(j-t0).to_value('sec')/60 for j in Times[i]]
        RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in RArr[i]]
        plt.scatter(Frequency[i],RArr1[-1])
    plt.xlabel('Sampling rate (mins)')
    plt.ylabel('Altitude (km)')
    plt.legend
    
    plt.show()
    # times2 = utils.get_times(duration=(3, 'day'), freq=(2, 'minute'), t0=t0)
    # times3 = utils.get_times(duration=(3, 'day'), freq=(4, 'minute'), t0=t0)

    # r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
    # r2, v2 = rv(orbit=orbit, time=times1, propagator=prop)
    # r3, v3 = rv(orbit=orbit, time=times1, propagator=prop)

    TArr1=[(i-t0).to_value('sec')/60 for i in times1]
    RArr1=[(np.linalg.norm(i)-6370000)/1000 for i in r1]


    # plt.figure()
    # plt.plot(TArr1,RArr1)
    # plt.plot(TArr2,RArr2)
    # plt.plot(TArr3,RArr3)

    # plt.xlabel('Time(Hours)')
    # plt.ylabel('Altitude (km)')
    
    # plt.show()
    # print('Iterations:'+str(len(TArr)))
    # plotUtils.globe_plot(r, times).
    # plotUtils.orbit_plot(r, times, frame="lunar", show=True)
    # plotUtils.ground_track_plot(r, times)


TestFrequency()