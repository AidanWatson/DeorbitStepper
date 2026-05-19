#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:01:10 2026

@author: faunw
"""
import numpy as np
from ssapy import *

import scipy.constants as Cons
import ssapy
from ssapy.constants import EARTH_MU, EARTH_RADIUS
from ssapy.utils import norm, sunPos, _gpsToTT, ntw_to_r
from ssapy.ellipsoid import Ellipsoid
from astropy.time import Time
from iri20py import Iri2020, alt_grid
from msis21py import NrlMsis21
import newPyglow as NPG
import InterpDraft as ID
import LafleurInterp as LI
try:
    import erfa
except ImportError:
    # Let this raise
    import astropy._erfa as erfa
def DataStandin(lat,lon,alt,dn):
    Data=NPG.DensitySweep(np.array([lat]), np.array([lon]), np.array([alt]), dn)
    return Data['M_i'], Data['M_n'],Data['N_i'],Data['Av_M']
    
def FindDensity(lat,lon,alt,dn): #placeholder currently. final version should recalculate pyglow array once per (day? month?)
    rhoMI=0
    rhoMN=4e11*Cons.m_p
    rhoNI=0
    avg_ion_mass=0
    rhoMI,rhoMN,rhoNI,avg_ion_mass=DataStandin(lat,lon,alt,dn)

    return rhoMI, rhoMN, rhoNI, avg_ion_mass

class AccelDrag(Accel):
    """Acceleration due to atmospheric drag.

    This class uses the Harris-Priester density model, which includes diurnal
    variation in the atmospheric bulge, but omits longer period seasonal
    variations.

    The acceleration also depends on a drag coefficient, which is hard to
    determine a priori, but takes on typical values around ~2 to ~2.3 for most
    satellites.

    See Section 3.5 of Montenbruck and Gill for more details.

    Parameters
    ----------
    recalc_threshold : float, optional
        Number of seconds past which the code will recompute the
        precession/nutation matrix.  Default: 86400*30  (30 days)
    defaultkw : dict
        default parameters for kwargs passed to __call__,
        (area, mass, CR)
    """
    def __init__(self, recalc_threshold=86400 * 30, **defaultkw):
        # from . import _ssapy
        from ssapy import _ssapy
        self.recalc_threshold = recalc_threshold
        self._t = None
        super().__init__()
        self.defaultkw = defaultkw
        ellip = Ellipsoid()
        self.atm = _ssapy.HarrisPriester(ellip, n=6.0)

    def __call__(self, r, v, t, _T=None, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters in GCRF frame.
        v : array_like, shape(3, )
            Velocity in meters per second in GCRF frame.
        t : float
            Time as GPS seconds
        area : float
            Area in meters^2.
        mass : flat
            Mass in kg.
        CD : float
            Drag coefficient.  Typical values are ~ 2 - 2.3

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        kw = dict()
        kw.update(self.defaultkw)
        kw.update(kwargs)
        mjd_tt = _gpsToTT(t)
        if _T is None:
            if self._t is None or np.abs(t - self._t) > self.recalc_threshold:
                self._t = t
                self._T = erfa.pnm80(2400000.5, mjd_tt)
            _T = self._T
        r_sun = sunPos(t)
        r_tod = _T @ r
        
        v_tod = _T @ v

        v_rel = v_tod - np.cross([0, 0, 7.2921159e-5], r_tod)  # MG (3.98)
        ra_sun = np.arctan2(r_sun[1], r_sun[0])
        dec_sun = np.arctan(r_sun[2] / np.hypot(r_sun[0], r_sun[1]))
        r_sun = sunPos(t)
        r_tod = _T @ r 
        r_ecef = tod_to_ecef(r_tod, t)
        lat, lon, height = ecef_to_latlon(r_ecef)
        if height>110:
            density = self.atm.density(
                *r_tod,
                ra_sun,
                dec_sun
            )
            if not np.isfinite(density):
                print(f"r_tod = {r_tod}")
                print(f"ra_sun = {ra_sun}")
                print(f"dec_sun = {dec_sun}")
                raise ValueError("non finite density")
        else:
            density=0
        # if not np.isfinite(density):
        #     print(f"r_tod = {r_tod}")
        #     print(f"ra_sun = {ra_sun}")
        #     print(f"dec_sun = {dec_sun}")
        #     raise ValueError("non finite density")
        a_tod = -0.5 * kw['CD'] * kw['area'] / kw['mass'] * density * v_rel * norm(v_rel)
        print('mjd_tt '+str(mjd_tt))
        # print('t '+str(t))
        print('density: '+str(density))
        print('a_tod: '+str(a_tod))
        print('Alt: '+str(height))


        return _T.T @ a_tod

    def __hash__(self):
        return hash((
            "AccelDrag",
            frozenset(self.defaultkw.items())
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelDrag):
            return False
        return self.defaultkw == rhs.defaultkw

def tod_to_ecef(r_tod, t_gps):
    """
    Convert position from TOD (True of Date) frame to ECEF frame.
    
    Parameters
    ----------
    r_tod : array_like, shape(3,)
        Position in TOD frame (meters)
    t_gps : float
        Time in GPS seconds
    
    Returns
    -------
    r_ecef : array_like, shape(3,)
        Position in ECEF frame (meters)
    """
    # GPS -> TT (fixed offset)
    GPS_TO_TT = 51.184
    mjd_tt = 44244.0 + (t_gps + GPS_TO_TT) / 86400.0

    # GPS -> UTC (fixed offset, 18 leap seconds as of 2017)
    GPS_LEAP_SECONDS = 18
    t_unix = t_gps - 315964800 + GPS_LEAP_SECONDS
    mjd_utc = t_unix / 86400.0 + 40587.0

    # UTC -> UT1 (approximate, dut1 < 1 second)
    dut1 = 0.0
    mjd_ut1 = mjd_utc + dut1 / 86400.0

    # Compute Greenwich Apparent Sidereal Time
    gast = erfa.gst94(2400000.5, mjd_ut1)

    # Rotate TOD -> ECEF by GAST around z-axis
    cos_g = np.cos(gast)
    sin_g = np.sin(gast)
    R = np.array([
        [ cos_g, sin_g, 0],
        [-sin_g, cos_g, 0],
        [0,      0,     1]
    ])
    # print('R_ecef: '+str(R @ r_tod))
    return R @ r_tod
def ecef_to_latlon(r_ecef):
    """
    Convert ECEF position to geodetic latitude, longitude, height.
    
    Returns
    -------
    lat : float, degrees
    lon : float, degrees  
    height : float, meters
    """
    # erfa.gc2gd body 1 = WGS84
    # print('lat, lon, alt: '+str(erfa.gc2gd(1, r_ecef)))
    # _, lon, lat, height = erfa.gc2gd(1, r_ecef)
    lon, lat, height = erfa.gc2gd(1, r_ecef)
    # print('lat, lon, alt: '+str(erfa.gc2gd(1, r_ecef)))
    # print('lat: '+str(np.degrees(lat))+' lon:'+str(np.degrees(lon))+' alt: '+str(height/1000))

    return np.degrees(lat), np.degrees(lon), height/1000
class AccelDrag2(Accel):
    """Acceleration due to atmospheric drag.

    This class uses the Harris-Priester density model, which includes diurnal
    variation in the atmospheric bulge, but omits longer period seasonal
    variations.

    The acceleration also depends on a drag coefficient, which is hard to
    determine a priori, but takes on typical values around ~2 to ~2.3 for most
    satellites.

    See Section 3.5 of Montenbruck and Gill for more details.

    Parameters
    ----------
    recalc_threshold : float, optional
        Number of seconds past which the code will recompute the
        precession/nutation matrix.  Default: 86400*30  (30 days)
    defaultkw : dict
        default parameters for kwargs passed to __call__,
        (area, mass, CR)
    """
    def __init__(self, recalc_threshold=86400 * 30, **defaultkw):
        self.stepCount=0
        # from . import _ssapy
        from ssapy import _ssapy
        self.recalc_threshold = recalc_threshold
        self._t = None
        super().__init__()
        self.defaultkw = defaultkw
        ellip = Ellipsoid()
        # _ssapy.HarrisPriester(1,11)
        self.atm = _ssapy.HarrisPriester(ellip, n=6.0)
        
        self.steps = 0
        self.interpDay=0
        self.ncalls=1000000 #should be the time over which an average is taken, divided by the frequency
        #rn 1 month divided by 10 second iterations
        kw = dict()
        kw.update(self.defaultkw)#kwargs are basic ballistic coefficients largely
        # kw.update(kwargs)
        self.Charged=True
        if self.Charged:
            self.lw=kw['lw']#Wire Length
            self.phi=kw['phi']
            self.rw=kw['rw']
            self.MaxArea=kw['Max_Area']
            self.LInt=LI.LoadOrCreateInterpolator(self.rw, self.phi)
            
        

    def __call__(self, r, v, t, _T=None, **kwargs):
        """Evaluate acceleration at particular place/moment.

        Parameters
        ----------
        r : array_like, shape(3, )
            Position in meters in GCRF frame.
        v : array_like, shape(3, )
            Velocity in meters per second in GCRF frame.
        t : float
            Time as GPS seconds
        area : float
            Area in meters^2.
        mass : flat
            Mass in kg.
        CD : float
            Drag coefficient.  Typical values are ~ 2 - 2.3

        Returns
        -------
        accel : array_like, shape(3,)
            Acceleration in meters per second^2
        """
        self.stepCount=self.stepCount+1
        print("Iteration Number:"+str(self.stepCount))
        kw = dict()
        kw.update(self.defaultkw)#kwargs are basic ballistic coefficients largely
        kw.update(kwargs)
        mjd_tt = _gpsToTT(t)
        self.Recalc_Int=kw['RSI']
        # Charged=True


        if _T is None: #precession/nutation rotation matrix (from gcrf to true of date)
            if self._t is None or np.abs(t - self._t) > self.recalc_threshold:
                self._t = t
                self._T = erfa.pnm80(2400000.5, mjd_tt)
            
            _T = self._T
        
        # print('mjd_tt '+str(mjd_tt))
        # print('t '+str(t))

        #Need to convert to ECEF for lat/long
        r_sun = sunPos(t)
        r_tod = _T @ r 
        r_ecef = tod_to_ecef(r_tod, t)
        lat, lon, height = ecef_to_latlon(r_ecef)
        # self.heightrange
        # r_ecef = R @ r_tod
        # print('r_tod: '+str(r_tod))
        v_tod = _T @ v

        v_rel = v_tod - np.cross([0, 0, 7.2921159e-5], r_tod)  # MG (3.98) relativ e velocity
        ra_sun = np.arctan2(r_sun[1], r_sun[0])
        dec_sun = np.arctan(r_sun[2] / np.hypot(r_sun[0], r_sun[1]))
        dt = Time(mjd_tt, format='mjd', scale='tt').to_datetime()
        
        # rhomi, rhomn,rhoni,avgm =FindDensity(lat,lon,height,dt)
        # print('Steps: '+str( self.steps ))
        
        if self.steps==0 or self.steps==self.Recalc_Int:
            self.steps=0
            self.HMax=height+120
            self.HMin=height-400#formerly270
            interpArr=ID.MakeInterpolator(dt, self.HMin,self.HMax)
            self.InterpRhoM=interpArr[0]
            self.interpNI=interpArr[1]
            self.interpAM=interpArr[2]
        self.steps=self.steps+1
        H2=max([50, height])
        print('H2: '+str(H2))
        print('HMax: '+str(self.HMax))
        print('HMin: '+str(self.HMin))

        rhomn =self.InterpRhoM([lat,lon,H2])
        NQ=max([0,(float(self.interpNI([lat,lon,H2])[0]))])
        AVM=float(self.interpAM([lat,lon,H2])[0])

        # rhomi, rhomn,rhoni,avgm =FindDensity(lat,lon,height,dt)
        print('mjd_tt '+str(mjd_tt))
        print('t '+str(t))
        # print('Step count: '+str(self.steps))
        # print('RSI: '+str(self.Recalc_Int))
        print('Alt: '+str(height))
        print('lat: '+str(lat))
        print('lon: '+str(lon))

            
            # raise ValueError("non finite density")
        
        density=float(rhomn[0])
        print('Density: '+str(density))
            # NQ=float(rhomn[1])
            # AVM=float(rhomn[2])
        print(f"Ion Number Density: {NQ:.4e}  m^(-3)")
        print('Average Ion Mass: '+str(AVM)+' (amu)')
    # f"{num:.2e}"
            # if not np.isfinite(density):
            #     raise ValueError("non finite density")
            # print(kw['CC'])#CC is succesfully passed, may just specify charge and wire length this way
            # CC=2
        print('Density: '+str(density))
            # print('Second Density: '+str(rhomn))
            # print(str(type(rhomn)))
            # print('Second Density: '+str(float(rhomn)))
        if not np.isfinite(density) :
            print(f"r_tod = {r_tod}")
            print(f"ra_sun = {ra_sun}")
            print(f"dec_sun = {dec_sun}")
            density=0
            NQ=0
            AVM=1
        if self.Charged:
            print('_______-')
                # print(np.log10(NQ))
                # print(AVM)
                # print(self.LInt)
                # print(self.LInt([12,15]))
            if height>100:
                CC=float(self.LInt([np.log10(NQ),AVM])[0])
            else:
                CC=2
                density = 1e-7#self.atm.density(*r_tod,ra_sun,dec_sun)
                NQ=1e-7
                AVM=1
            print('CC: '+str(CC))

            if height>50:
                FnA=0.5 * kw['CD'] * (kw['area']+(kw['lw']*kw['rw']*2))*density+NQ*(AVM*Cons.m_p)*0.5 * kw['CD'] * kw['area']
                ChargedArea=CC * kw['lw']*kw['rw']*2
                if ChargedArea>2*self.MaxArea:
                    print('Effective Area:'+str(ChargedArea/2))
                    print('Updated Area:' +str(self.MaxArea))
                    ChargedArea=2*self.MaxArea #sets maximum drag force
                    
                FiA=NQ*AVM*Cons.m_p*0.5 * ChargedArea
                Ft=FnA+FiA
                print('Total Force/v2: '+str(Ft)+' (N)')
                print('Neutral Force/v2: '+str(FnA)+' (N)')
                print('Charged Force/v2: '+str(FiA)+' (N)')

                print('Drag Ratio: '+str(FnA/FiA))
                a_tod=-1*Ft/ kw['mass']* v_rel * norm(v_rel)
            else: 
                print('Below Threshold Altitude')
                a_tod=0*1/kw['mass']* v_rel * norm(v_rel)
            # a_tod=0
        # else:
            # density=0
        
        # a_tod = -0.5 * kw['CD'] * (kw['area']+(kw['lw']*kw['rw']*2)) / kw['mass'] * density * v_rel * norm(v_rel)
        return _T.T @ a_tod

    def __hash__(self):
        return hash((
            "AccelDrag",
            frozenset(self.defaultkw.items())
        ))

    def __eq__(self, rhs):
        if not isinstance(rhs, AccelDrag):
            return False
        return self.defaultkw == rhs.defaultkw