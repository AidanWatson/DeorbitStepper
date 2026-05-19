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
import time as tm
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
    
    # Frequency=[1,2,4,8,16,32,64,128,256,312]
    Frequency=[1,2,4,8,16]
    times=[]
    RArr=[] 
    runtimes=[]
    for i in range(len(Frequency)):
        T1=tm.time()
        sat_kwargs = dict(
                mass=4e4,  # [kg]
                area=1000,  # [m^2]
                CD=2.3,  # Drag coefficient
                CR=1.3,  # Radiation pressure coefficient
                CC=2,
                RSI=int(24*60)
                # RSI=int((24*60)/Frequency[i]) #once every 24 hours
        )
        aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
    
        ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)
        accel = aEarth  +ADrag
        prop = SciPyPropagator(accel)

        
        times1 = utils.get_times(duration=(1, 'day'), freq=(Frequency[i], 'minute'), t0=t0)
        r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
        T2=tm.time()
        times1 = utils.get_times(duration=(1, 'day'), freq=(Frequency[i], 'minute'), t0=t0)
        r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
        T2=tm.time()
        runtimes.append((T2-T1))
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
    
    plt.figure()
    for i in range(len(Frequency)):
        # TArr1=[(j-t0).to_value('sec')/60 for j in Times[i]]
        RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in RArr[i]]
        plt.scatter(Frequency[i],runtimes[i]/60)
    plt.xlabel('Sampling rate (mins)')
    plt.ylabel('runtime(mins)')
    plt.legend
    
    plt.show()


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
def TestDeorbit():
    t0 = Time("2024-1-1")
    
    RE=6370000
    a = RE+450*1000 #constants.RGEO
    # a=constants.RGEO
    e = 0.00065
    i = np.radians(51)
    pa = np.radians(130)#argument of perigee?
    raan = np.radians(247)
    ta = np.radians(325)
    Frequency=1 #Minute
    T1=tm.time()
    sat_kwargs = dict(
        mass=4e4,  # [kg]
        area=1000,  # [m^2]
        CD=2.3,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
        CC=2,
        RSI=int(24*60*10/Frequency) #every 10 days
    )
    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    
    
    # moon = get_body("moon")
    sun = get_body("Sun")
    
    Earth = get_body("Earth", model="EGM2008")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)

    ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)
    accel = aEarth  +ADrag
    prop = SciPyPropagator(accel)
    times1 = utils.get_times(duration=(50, 'day'), freq=(Frequency, 'minute'), t0=t0)
    r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
    plt.figure()
    TArr1=[(j-t0).to_value('sec')/3600/24 for j in times1]
    RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in r1]
    plt.plot(TArr1,RArr1)
    plt.xlabel('Time(days)')
    plt.ylabel('Altitude (km)')
    plt.legend
    
    plt.show()

def TestNanosail():
    t0 = Time("2011-01-20")
    # t0 = Time("2024-1-1")

    RE=6370000
    # a = RE+654*1000 #constants.RGEO
    a = RE+560*1000 #constants.RGEO

    # a=6969*1000
    # a=constants.RGEO
    e = 0.00
    i = np.radians(71.9)
    pa = np.radians(130)#argument of perigee?
    raan = np.radians(247)
    # ta = np.radians(325)

    # raan = np.radians(15)

    ta = np.radians(15)
    Frequency=2 #Minute
    T1=tm.time()
    sat_kwargs = dict(
        mass=4.00,  # [kg]
        area=4.00,  # [m^2
        CD=2.2,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
        CC=2,
        RSI=int(24*60*10/Frequency) #every 10 days
    )
    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    
    
    # moon = get_body("moon")
    sun = get_body("Sun")
    
    Earth = get_body("Earth", model="EGM2008")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)

    ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)
    accel = aEarth  +ADrag
    prop = SciPyPropagator(accel)
    # T1=tm.time()
    times1 = utils.get_times(duration=(40, 'day'), freq=(Frequency, 'minute'), t0=t0)
    # times1 = utils.get_times(duration=(10, 'day'), freq=(Frequency, 'minute'), t0=t0)
    T1=tm.time()
    r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
    T2=tm.time()

    plt.figure()
    TArr1=[(j-t0).to_value('sec')/3600/24 for j in times1]
    RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in r1]
    # y_smooth = np.convolve(RArr1, np.ones(50)/50, mode='same')
    from scipy.ndimage import gaussian_filter1d

    y_smooth = gaussian_filter1d(RArr1, sigma=100)


    plt.plot(TArr1,RArr1)
    plt.plot(TArr1,y_smooth)

    plt.xlabel('Time(days)')
    plt.ylabel('Altitude (km)')
    plt.legend
    
    plt.show()
    print('Sim time:'+str((T2-T1)/60)+" min")

def TestOderacs():
    t0 = Time("1994-10-03")
    # t0 = Time("2024-1-1")

    RE=6370000
    # a = RE+654*1000 #constants.RGEO
    a = RE+337*1000 #constants.RGEO

    # a=6969*1000
    # a=constants.RGEO
    e = 0.0013
    i = np.radians(57.0)
    pa = np.radians(130)#argument of perigee?
    raan = np.radians(247)
    # ta = np.radians(325)

    # raan = np.radians(15)

    ta = np.radians(15)
    Frequency=4 #Minute
    T1=tm.time()
    sat_kwargs = dict(
        mass=1.482,  # [kg]
        area=0.0081,  # [m^2
        CD=1.93,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
        CC=2,
        RSI=int(24*60*10/Frequency) #every 10 days
    )
    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    
    
    # moon = get_body("moon")
    sun = get_body("Sun")
    
    Earth = get_body("Earth", model="EGM2008")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)

    # ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)
    ADrag=ModifiedAccel.AccelDrag(**sat_kwargs)

    accel = aEarth  +ADrag
    prop = SciPyPropagator(accel)
    # T1=tm.time()
    times1 = utils.get_times(duration=(200, 'day'), freq=(Frequency, 'minute'), t0=t0)
    T1=tm.time()
    r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
    T2=tm.time()

    # plt.figure()
    TArr1=[(j-t0).to_value('sec')/3600/24 for j in times1]
    RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in r1]
    print(TArr1)
    print(RArr1)
    SaveFile(TArr1, RArr1, 0, 0, 'ODERACS', 337)
    
    # y_smooth = np.convolve(RArr1, np.ones(50)/50, mode='same')
    # from scipy.ndimage import gaussian_filter1d
    # TArr1=TArr1[:len(RArr1)]
    # y_smooth = gaussian_filter1d(RArr1, sigma=100)


    # plt.plot(TArr1,RArr1)
    # plt.plot(TArr1,y_smooth)

    # plt.xlabel('Time(days)')
    # plt.ylabel('Altitude (km)')
    # plt.legend
    
    # plt.show()
    print('Sim time:'+str((T2-T1)/60)+" min")
def TestDrag():
    t0 = Time("2024-1-1")
    
    RE=6370000
    a = RE+450*1000 #constants.RGEO
    # a=constants.RGEO
    e = 0.00065
    i = np.radians(51)
    pa = np.radians(130)#argument of perigee?
    raan = np.radians(247)
    ta = np.radians(325)
    Frequency=1 #Minute
    T1=tm.time()
    sat_kwargs = dict(
        mass=4e4,  # [kg]
        area=1000,  # [m^2]
        CD=2.3,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
        CC=2,
        RSI=int(24*60*10/Frequency), #every 10 days
        lw=2,
        rw=0.005,
        phi=40,
        maxArea=4
        
    )
    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    
    
    # moon = get_body("moon")
    sun = get_body("Sun")
    
    Earth = get_body("Earth", model="EGM2008")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)

    ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)
    accel = aEarth  +ADrag
    prop = SciPyPropagator(accel)
    times1 = utils.get_times(duration=(10, 'day'), freq=(Frequency, 'minute'), t0=t0)
    r1, v1 = rv(orbit=orbit, time=times1, propagator=prop)
    plt.figure()
    TArr1=[(j-t0).to_value('sec')/3600/24 for j in times1]
    RArr1=[(np.linalg.norm(j)-6370000)/1000 for j in r1]
    plt.plot(TArr1,RArr1)
    plt.xlabel('Time(days)')
    plt.ylabel('Altitude (km)')
    plt.legend
    
    plt.show()
def SaveFile(TA,AA,rw,phi,satnum,Alt):
    filename = f"StepperCSV/Deorbit_{satnum}_{Alt}km_rw{rw}_phi{phi}.npz"
    np.savez(filename, Times=TA, Alts=AA, Alt=Alt, rw=rw, phi=phi)

def loadFile(rw,phi,satnum,Alt):
    filename = f"StepperCSV/Deorbit_{satnum}_{Alt}km_rw{rw}_phi{phi}.npz"
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

# TestFrequency()
# TestDrag()
TestOderacs()
Arrs=loadFile( 0, 0, 'ODERACS', 337)
plotDeorbit(Arrs[0],Arrs[1])