#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:39:44 2026

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
# import ModifiedAccel
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


import time as tm
try:
    import erfa
except ImportError:
    # Let this raise
    import astropy._erfa as erfa


def smooth_combined(combined_alt, combined_time, window=51, poly_order=3):
    """
    Smooth the combined series using a Savitzky-Golay filter.
    
    Parameters
    ----------
    window     : int  — number of points in the smoothing window (must be odd)
                        larger = smoother, but risks losing real features
    poly_order : int  — polynomial order (3 or 4 works well for orbital decay)
    """
    # Data must be evenly spaced for savgol — interpolate onto uniform alt grid first
    alt_uniform = np.linspace(combined_alt.max(), combined_alt.min(), len(combined_alt))
    f = interp1d(combined_alt, combined_time, bounds_error=False, fill_value=np.nan)
    time_uniform = f(alt_uniform)

    time_smoothed = savgol_filter(time_uniform, window_length=window, polyorder=poly_order)
    return alt_uniform, time_smoothed
def get_overlap(alt1, alt2):
    """Return the altitude range where two series overlap."""
    overlap_min = max(alt1.min(), alt2.min())
    overlap_max = min(alt1.max(), alt2.max())
    if overlap_min < overlap_max:
        return (overlap_min, overlap_max)
    return None
def sigmoid_weight(alt_grid, alt_min, alt_max):
    """Weights from 1→0 across the overlap (1 = favour upper series, 0 = favour lower)."""
    midpoint = (alt_min + alt_max) / 2
    scale = (alt_max - alt_min) / 8      # increase denominator for a gentler transition
    return 1 / (1 + np.exp(-(alt_grid - midpoint) / scale))
def blend_overlap(alt1, time1, alt2, time2, overlap, n_points=300):
    """Interpolate both series onto a common grid and sigmoid-blend them."""
    alt_grid = np.linspace(overlap[1], overlap[0], n_points)  # high → low

    f1 = interp1d(alt1, time1, bounds_error=False, fill_value=np.nan)
    f2 = interp1d(alt2, time2, bounds_error=False, fill_value=np.nan)

    t1_interp = f1(alt_grid)
    t2_interp = f2(alt_grid)

    w = 0.5#sigmoid_weight(alt_grid, overlap[0], overlap[1])
    blended_time = w * t1_interp + (1 - w) * t2_interp

    return alt_grid, blended_time
def combine_orbital_series(Alts, Times, n_points=5):
    """
    Combine a list of altitude/time decay series into one.
    - If consecutive series overlap in altitude: sigmoid-blend the overlap.
    - If no overlap: join smoothly using a cubic Hermite bridge.

    Parameters
    ----------
    Alts     : list of np.ndarray  — altitude series, each sorted high → low
    Times    : list of np.ndarray  — corresponding time series
    n_points : int                 — resolution of blended/bridged regions

    Returns
    -------
    combined_alt, combined_time : np.ndarray
    """
    combined_alt  = []
    combined_time = []

    for i in range(len(Alts) - 1):
        alt1, time1 = Alts[i],   Times[i]
        alt2, time2 = Alts[i+1], Times[i+1]

        overlap = get_overlap(alt1, alt2)

        if overlap:
            # --- Normal case: sigmoid-blend the overlapping region ---
            upper_mask = alt1 > overlap[1]
            combined_alt.append(alt1[upper_mask])
            combined_time.append(time1[upper_mask])

            alt_blend, time_blend = blend_overlap(alt1, time1, alt2, time2, overlap, n_points)
            combined_alt.append(alt_blend)
            combined_time.append(time_blend)

        else:
            # --- Gap case: cubic Hermite bridge between endpoints ---

            # Endpoints and their neighbours for estimating derivatives
            alt1_end,  time1_end  = alt1[-1],  time1[-1]   # bottom of series i
            alt2_start,time2_start= alt2[0],   time2[0]    # top of series i+1

            # Estimate dt/dalt gradients at each endpoint using last/first 2 points
            grad1 = (time1[-1] - time1[-2]) / (alt1[-1] - alt1[-2])
            grad2 = (time2[1]  - time2[0])  / (alt2[1]  - alt2[0])

            # Bridge altitude grid across the gap
            alt_bridge = np.linspace(alt1_end, alt2_start, n_points)

            # Normalised parameter t: 0 at end of series 1, 1 at start of series 2
            t = np.linspace(0, 1, n_points)

            # Cubic Hermite basis functions
            h00 = 2*t**3 - 3*t**2 + 1   # weight for p0 (value at start)
            h10 = t**3  - 2*t**2 + t    # weight for m0 (gradient at start)
            h01 = -2*t**3 + 3*t**2      # weight for p1 (value at end)
            h11 = t**3  - t**2          # weight for m1 (gradient at end)

            # Scale gradients from dt/dalt to dt/dt_normalised
            alt_span = alt2_start - alt1_end
            m0 = grad1 * alt_span
            m1 = grad2 * alt_span

            time_bridge = h00*time1_end + h10*m0 + h01*time2_start + h11*m1

            # Append full series i, then the bridge
            combined_alt.append(alt1)
            combined_time.append(time1)
            combined_alt.append(alt_bridge)
            combined_time.append(time_bridge)

    # --- Always append the final series in full ---
    combined_alt.append(Alts[-1])
    combined_time.append(Times[-1])

    combined_alt  = np.concatenate(combined_alt)
    combined_time = np.concatenate(combined_time)

    # Ensure final result is sorted high → low
    sort_idx = np.argsort(combined_alt)[::-1]
    return combined_alt[sort_idx], combined_time[sort_idx]
def crop_and_combine(Alts, Times):
    """
    Crop each series to remove altitudes already covered by the series above it,
    then concatenate into one clean combined series.

    Parameters
    ----------
    Alts  : list of np.ndarray — altitude series, each sorted high → low
    Times : list of np.ndarray — corresponding time series

    Returns
    -------
    combined_alt, combined_time : np.ndarray
    """
    cropped_alts  = []
    cropped_times = []

    for i, (alt, time) in enumerate(zip(Alts, Times)):
        if i == 0:
            # First series: keep everything
            cropped_alts.append(alt)
            cropped_times.append(time)
            current_floor = alt.min()  # lowest altitude claimed so far
        else:
            # Crop this series to only the altitudes below the current floor
            mask = alt < current_floor
            cropped_alts.append(alt[mask])
            cropped_times.append(time[mask])
            current_floor = alt[mask].min()

    combined_alt  = np.concatenate(cropped_alts)
    combined_time = np.concatenate(cropped_times)

    # Ensure sorted high → low
    sort_idx = np.argsort(combined_alt)[::-1]
    return combined_alt[sort_idx], combined_time[sort_idx]
def loadFile(SatDict,alt):
    Init_Alt=alt
    Name=SatDict['Name']
    r_w=SatDict['r_w']
    phi_B=SatDict['Phi']
    filename = f"StepperCSV/Deorbit_{Name}_{Init_Alt}km_rw{r_w}_phi{phi_B}.npz"
    data = np.load(filename)
    Times = data['Times']
    Alts = data['Alts']
    return [Times,Alts]
#should take the arrays for time vs alt, load them all and splice them together end to end
def plotCombined():
    F1_01=dict(Name='F1',r_w=0.005,l_w=4.8,Phi=0.1,Max_Area=1.98)
    AltsInit_01=[650,641,631,619,606,600,581,554,508,367]
    # AltsInit_01=[650,641]

    Times_01=[loadFile(F1_01, i)[0] for i in AltsInit_01]
    Alts_01=[loadFile(F1_01, i)[1] for i in AltsInit_01]
    Times_01=[(Times_01[i][:len(Alts_01[i])]+365*i)/365 for i in range(len(Times_01))]
    Alts_Altered=[gaussian_filter1d(Alts_01[i], sigma=1000) for i in range(len(Alts_01))]
    combined_alt, combined_time =crop_and_combine(Alts_Altered, Times_01)
    print(Times_01)
    print(Alts_01)
    print(combined_time)
    print(combined_alt)
    plt.figure()
 
    plt.plot(Times_01[0],Alts_01[0],label='raw')
    plt.plot(combined_time, combined_alt,label='Normal')

    plt.plot(combined_time, gaussian_filter1d(combined_alt,sigma=2000),label='Smoothed')
    # SaveDeets('Neutral650_F1',combined_time,gaussian_filter1d(combined_alt,sigma=2000))
    plt.legend()
    plt.show()
    times,alts=LoadDeetz('Neutral650_F1')
    # plt.figure()
 
    # plt.plot(times,alts,label='Normal')

    # plt.legend()
    # plt.show()
def SaveDeets(fname,Times,Alts):
    np.savez('StepperCSV/DeorbitTotal_{fname}.npz',times=Times, alts=Alts)
    return 0
def LoadDeetz(fname):
    data=np.load('StepperCSV/DeorbitTotal_{fname}.npz')
    times=data['times']
    alts=data['alts']
    return times, alts
def BasicDeorbitPlot():
    F1_01=dict(Name='F1',r_w=0.005,l_w=4.8,Phi=0.1,Max_Area=1.98)
    F1_30=dict(Name='F1',r_w=0.005,l_w=4.8,Phi=30,Max_Area=1.98)
    F1_10=dict(Name='F1',r_w=0.005,l_w=4.8,Phi=10,Max_Area=1.98)
    F1_100=dict(Name='F1',r_w=0.005,l_w=4.8,Phi=100,Max_Area=1.98)
    F1_250=dict(Name='F1',r_w=0.005,l_w=4.8,Phi=250,Max_Area=1.98)
    F1_N=dict(Name='F1N',r_w=0.005,l_w=2.4,Phi=0.01,Max_Area=1.98)

    AltsInit_01=[650,641,631,619,606,600,581,554,508,367]
    AltsInit_30=[650,637,623,608,600,567,527,444]
    AltsInit_10=[650,640,629,617,602,584,560,523,452]
    AltsInit_100=[650,632,610,585,551,496]
    AltsInit_250=[650,629,605,549,360]
    AltsInit_F=[650,637,621,602, 576, 538,457]

    Times_01=[loadFile(F1_01, i)[0] for i in AltsInit_01]
    Alts_01=[loadFile(F1_01, i)[1] for i in AltsInit_01]
    Times_30=[loadFile(F1_30, i)[0] for i in AltsInit_30]
    Alts_30=[loadFile(F1_30, i)[1] for i in AltsInit_30]
    Times_10=[loadFile(F1_10, i)[0] for i in AltsInit_10]
    Alts_10=[loadFile(F1_10, i)[1] for i in AltsInit_10]
    Times_100=[loadFile(F1_100, i)[0] for i in AltsInit_100]
    Alts_100=[loadFile(F1_100, i)[1] for i in AltsInit_100]
    Times_250=[loadFile(F1_250, i)[0] for i in AltsInit_250]
    Alts_250=[loadFile(F1_250, i)[1] for i in AltsInit_250]
    Times_N=[loadFile(F1_N, i)[0] for i in AltsInit_F]
    Alts_N=[loadFile(F1_N, i)[1] for i in AltsInit_F]
    print(Times_01)
    print(Alts_01)
    # TElapsed=0
    plt.figure()
    for i in range(len(Times_01)):
        # TElapsed=
        plt.plot((Times_01[i][:len(Alts_01[i])]+365*i)/365,Alts_01[i])
        plt.plot((Times_01[i][:len(Alts_01[i])]+365*i)/365,gaussian_filter1d(Alts_01[i], sigma=100),color='black')
    for i in range(len(Times_10)):
        plt.plot((Times_10[i][:len(Alts_10[i])]+365*i)/365,Alts_10[i])
        plt.plot((Times_10[i][:len(Alts_10[i])]+365*i)/365,gaussian_filter1d(Alts_10[i], sigma=100),color='black')
    for i in range(len(Times_30)):
        plt.plot((Times_30[i][:len(Alts_30[i])]+365*i)/365,Alts_30[i])
        plt.plot((Times_30[i][:len(Alts_30[i])]+365*i)/365,gaussian_filter1d(Alts_30[i], sigma=100),color='black')
    for i in range(len(Times_100)):
        plt.plot((Times_100[i][:len(Alts_100[i])]+365*i)/365,Alts_100[i])
        plt.plot((Times_100[i][:len(Alts_100[i])]+365*i)/365,gaussian_filter1d(Alts_100[i], sigma=100),color='black')
    for i in range(len(Times_250)):
        plt.plot((Times_250[i][:len(Alts_250[i])]+365*i)/365,Alts_250[i])
        plt.plot((Times_250[i][:len(Alts_250[i])]+365*i)/365,gaussian_filter1d(Alts_250[i], sigma=100),color='black')
    for i in range(len(Times_N)):
        plt.plot((Times_N[i][:len(Alts_N[i])]+365*i)/365,Alts_N[i])
        plt.plot((Times_N[i][:len(Alts_N[i])]+365*i)/365,gaussian_filter1d(Alts_N[i], sigma=100),color='red')

    plt.xlabel('Time (yrs)')
    plt.ylabel('Alt (km)')
    plt.show()
    plt.figure()
    
    for i in range(len(Times_250)):
        plt.plot((Times_250[i][:len(Alts_250[i])]+365*i)/365,Alts_250[i])
        plt.plot((Times_250[i][:len(Alts_250[i])]+365*i)/365,gaussian_filter1d(Alts_250[i], sigma=100),color='black')

        
        
    # print(Times_01)
BasicDeorbitPlot()
# plotCombined()

