# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

print('swag')
t0 = Time("2024-1-1")
print(t0)

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
sat_kwargs = dict(
        mass=4e4,  # [kg]
        area=1000,  # [m^2]
        CD=2.3,  # Drag coefficient
        CR=1.3,  # Radiation pressure coefficient
        CC=2,
        RSI=300 #number of iterations between recalculation
)
kElements = [a, e, i, pa, raan, ta]
orbit = Orbit.fromKeplerianElements(*kElements, t=t0)


# moon = get_body("moon")
sun = get_body("Sun")

Earth = get_body("Earth", model="EGM2008")
Mars = get_body("Mars")
Jupiter = get_body("Jupiter")
Saturn = get_body("Saturn")
Uranus = get_body("Uranus")
Neptune = get_body("Neptune")
#need to define better acceleration

aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
# aSun = AccelThirdBody(sun)
# aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
aSolRad = AccelSolRad(**sat_kwargs)
aEarthRad = AccelEarthRad(**sat_kwargs)
ADrag=ModifiedAccel.AccelDrag2(**sat_kwargs)
accel = aEarth  + aSolRad + aEarthRad+ADrag

prop = SciPyPropagator(accel)

times = utils.get_times(duration=(3, 'day'), freq=(1, 'minute'), t0=t0)
r, v = rv(orbit=orbit, time=times, propagator=prop)

TArr=[(i-t0).to_value('sec')/60 for i in times]
RArr=[(np.linalg.norm(i)-6370000)/1000 for i in r]
plt.figure()
plt.scatter(TArr,RArr)
plt.xlabel('Time(Hours)')
plt.ylabel('Altitude (km)')
plt.show()
print('Iterations:'+str(len(TArr)))
# plotUtils.globe_plot(r, times).
# plotUtils.orbit_plot(r, times, frame="lunar", show=True)
plotUtils.ground_track_plot(r, times)