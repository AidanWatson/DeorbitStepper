#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:10:20 2026

@author: faunw
"""

from __future__ import annotations

from msis21py import NrlMsis21, alt_grid
from datetime import datetime, UTC
import matplotlib.pyplot as plt
from datetime import datetime, UTC
import matplotlib
from iri20py import Iri2020, alt_grid
import numpy as np
from msis21py import NrlMsis21
import time as tm
import scipy.constants as Cons
import pandas as pd
import seaborn as sns
msis21 = NrlMsis21()
iri20 = Iri2020()
from joblib import Parallel, delayed

# import time
def CheckPoint(date,lat,long,alt):
    # print(len(alt))
    # print('a')
    ds20 = msis21.evaluate(date, lat, long, alt)
    _, IR20 = iri20.evaluate(date, lat, long, alt)
    msNames=['O2','N2','NO','Ar']
    IRNames=['N+','O+','O2+','NO+','He+','H+','Ne','Te','alt_km']
    # print(float(IR20['H+'][0]))
    dicto={}
    for n in msNames:
        dicto[n]=float(ds20[n][0])
    for n in IRNames:
        dicto[n]=float(IR20[n][0])
    # print(dicto)
    return dicto

def DensitySweep(lats, longs, alts, dn): #takes input of 3 arrays and a datetime value, and outputs ions for each combination of these
    ni_O=[]
    ni_H=[]
    ni_He=[]
    ni_O2=[]
    ni_NO=[]
    ni_N=[]
    nn_O2=[]
    nn_O=[]
    nn_N2=[]
    nn_H=[]
    nn_N=[]    
    nn_He=[]
    ne=[]
    M_n=[]
    M_i=[]
    N_i=[]
    N_n=[]
    Alts=[]
    Lats=[]
    Longs=[]
    rhoM=[]
    M_Ratio=[]
    Av_Mass=[]
    Te=[]
    for lon in longs:
        for lat in lats:

            for alt in alts:
                print("Computing alt=%3.1f km..." % (alt))
                ds20 = msis21.evaluate(dn, lat, lon, np.array([alt]))
                _, IR20 = iri20.evaluate(dn, lat, lon, np.array([alt]))

                Alts.append(alt)
                Lats.append(lat)
                Longs.append(lon)
                # print(alt)
                nn_O2.append(float(ds20['O2'][0])*1e6)
                nn_O.append(float(ds20['O'][0])*1e6)
                nn_N2.append(float(ds20['N2'][0])*1e6)
                #ni_He.append(pt.ni['HE+'])
                
                nn_H.append(float(ds20['H'][0])*1e6)  #
                nn_N.append(float(ds20['N'][0])*1e6)  #
                nn_He.append(float(ds20['He'][0])*1e6)  #
                # rhoM.append((float(ds20['O'][0])*16+float(ds20['O2'][0])*32+float(ds20['N2'][0])*28+float(ds20['Ar'][0])*18+float(ds20['H'][0])+float(ds20['He'][0])*2)*Cons.m_p*1e6)


                ni_O.append(float(IR20['O+'][0])*1e6)
                ni_He.append(float(IR20['H+'][0])*1e6)
                ni_O2.append(float(IR20['O2+'][0])*1e6)
                ni_NO.append(float(IR20['NO+'][0])*1e6)
                Te.append(float(IR20['Te'][0]))
                
                ni_H.append(float(IR20['H+'][0])*1e6)
                ne.append(float(IR20['Ne'][0])*1e6)
                # rhoM.append((float(ds20['O'][0])*16+float(ds20['O2'][0])*32+float(ds20['N2'][0])*28+float(ds20['Ar'][0])*18+float(ds20['H'][0])+float(ds20['He'][0])*2+float(IR20['H+'][0])+float(IR20['NO+'][0])*30+float(IR20['O+'][0]))*Cons.m_p*1e6)

    for i in range(len(nn_O)):
        M_n.append(Cons.m_p*(nn_O2[i]*32+nn_O[i]*16+nn_N2[i]*28+nn_N[i]*14+nn_He[i]*2+nn_H[i]))
        M_i.append(Cons.m_p*(ni_H[i]+ni_O[i]*16+ni_O2[i]*32+ni_He[i]*2+ni_NO[i]*30))
        N_n.append((nn_O2[i]+nn_O[i]+nn_N2[i]+nn_N[i]+nn_He[i]+nn_H[i]))
        N_i.append((ni_H[i]+ni_O[i]+ni_O2[i]+ni_He[i]+ni_NO[i]))
        M_Ratio.append(1/((nn_O2[i]*32+nn_O[i]*16+nn_N2[i]*28+nn_N[i]*14+nn_He[i]*2+nn_H[i])/(ni_H[i]+ni_O[i]*16+ni_O2[i]*32+ni_He[i]*2+ni_NO[i]*30)))
        Av_Mass.append((ni_H[i]+ni_O[i]*16+ni_O2[i]*32+ni_He[i]*2+ni_NO[i]*30)/(ni_H[i]+ni_O[i]+ni_O2[i]+ni_He[i]+ni_NO[i]))
        #nn_O2.append( (rhoM[i]-Cons.m_p*(nn_O[i]*16+nn_N2[i]*28+nn_N[i]*14+nn_He[i]*2+nn_Ar[i]*39+nn_H[i]) )/(Cons.m_p*32))
#print(ni_O)
    dicto={'M_Ratio':M_Ratio,'N_i':N_i,'N_n':N_n,'Av_M':Av_Mass,'Te':Te,'O+':ni_O,'H+':ni_H,'O':nn_O,'N2':nn_N2,'lats':Lats,'Longs':Longs, 'O2':nn_O2,'alts':Alts,'M_n':M_n,'M_i':M_i,'e-':ne,'He':nn_He,'H':nn_H,'N':nn_N,'He+':ni_He,'O2+':ni_O2,'NO+':ni_NO}
    Data=pd.DataFrame.from_dict(dicto)
    return Data

def DensitySweepAlt(lat, long, alts, dn): #takes input of 3 arrays and a datetime value, and outputs ions for each combination of these
    ni_O=[]
    ni_H=[]
    ni_He=[]
    ni_O2=[]
    ni_NO=[]
    ni_N=[]
    nn_O2=[]
    nn_O=[]
    nn_N2=[]
    nn_H=[]
    nn_N=[]    
    nn_He=[]
    nn_Ar=[]
    ne=[]
    M_n=[]
    M_i=[]
    N_i=[]
    N_n=[]
    Alts=[]
    Lats=[]
    Longs=[]
    rhoM=[]
    M_Ratio=[]
    Av_Mass=[]
    Te=[]
    # for lon in longs:
        # for lat in lats:
            # nn=[[1],[2],[3]]*4
            
            # print("Computing alt=%3.1f km..." % (alts))
    ds20 = msis21.evaluate(dn, lat, long, alts)
    _, IR20 = iri20.evaluate(dn, lat, long, alts)

            # Lats.append(lat)
            # Longs.append(lon)
            # print(alt)
    # print(ds20['O2'])
    # print([float(i) for i in ds20['O2']])

    nn_O2=(np.array([float(i)*1e6 for i in ds20['O2']]))
    nn_O=(np.array([float(i)*1e6 for i in (ds20['O'])]))

    nn_N2=(np.array([float(i)*1e6 for i in (ds20['N2'])]))
                
    nn_H=(np.array([float(i)*1e6 for i in (ds20['H'])]) ) #
    nn_N=(np.array([float(i)*1e6 for i in (ds20['N2'])]))  #
    nn_He=(np.array([float(i)*1e6 for i in (ds20['He'])]))  #
    nn_Ar=(np.array([float(i)*1e6 for i in (ds20['Ar'])]))  #


    ni_O=(np.array([float(i)*1e6 for i in (IR20['O+'])]))
    ni_He=(np.array([float(i)*1e6 for i in (IR20['He+'])]))
    ni_O2=(np.array([float(i)*1e6 for i in (IR20['O2+'])]))
    ni_NO=(np.array([float(i)*1e6 for i in (IR20['NO+'])]))
    
    Te=(np.array([float(i) for i in (IR20['Te'])]))
                
    ni_H=(np.array([float(i)*1e6 for i in (IR20['H+'])]))
    ne=(np.array([float(i)*1e6 for i in (IR20['Ne'])]))
    Alts=alts

    for i in range(len(nn_O)):
        # print(i)
        M_n.append(float(Cons.m_p*(nn_O2[i]*32+nn_O[i]*16+nn_N2[i]*28+nn_N[i]*14+nn_He[i]*2+nn_H[i]+nn_Ar[i]*40)))
        # print(M_n)
        # print(nn_O2)
        # print(Alts)
        M_i.append(float(Cons.m_p*(ni_H[i]+ni_O[i]*16+ni_O2[i]*32+ni_He[i]*2+ni_NO[i]*30)))
        N_n.append(float((nn_O2[i]+nn_O[i]+nn_N2[i]+nn_N[i]+nn_He[i]+nn_H[i]+nn_Ar[i])))
        N_i.append(float((ni_H[i]+ni_O[i]+ni_O2[i]+ni_He[i]+ni_NO[i])))
        M_Ratio.append(float(1/((nn_O2[i]*32+nn_O[i]*16+nn_N2[i]*28+nn_N[i]*14+nn_He[i]*2+nn_H[i]+nn_Ar[i]*40)/(ni_H[i]+ni_O[i]*16+ni_O2[i]*32+ni_He[i]*2+ni_NO[i]*30))))
        Av_Mass.append(float((ni_H[i]+ni_O[i]*16+ni_O2[i]*32+ni_He[i]*2+ni_NO[i]*30)/(ni_H[i]+ni_O[i]+ni_O2[i]+ni_He[i]+ni_NO[i])))
        #nn_O2.append( (rhoM[i]-Cons.m_p*(nn_O[i]*16+nn_N2[i]*28+nn_N[i]*14+nn_He[i]*2+nn_Ar[i]*39+nn_H[i]) )/(Cons.m_p*32))
#print(ni_O)
    
    dicto={'M_Ratio':M_Ratio,'N_i':N_i,'N_n':N_n,'Av_M':Av_Mass,'Te':Te,'O+':ni_O,'H+':ni_H,'O':nn_O,'N2':nn_N2, 'O2':nn_O2,'alts':Alts,'M_n':M_n,'M_i':M_i,'e-':ne,'He':nn_He,'H':nn_H,'N':nn_N,'He+':ni_He,'O2+':ni_O2,'NO+':ni_NO,'Ar':nn_Ar,'M_n':M_n}
    Data=pd.DataFrame.from_dict(dicto)
    D1=Data.copy()
    return D1
def CheckAltTimes():
    T1=tm.time()
    alts=np.array(np.linspace(200,400,100))
    date = time = datetime(2022, 3, 22, 18, 0).astimezone(UTC)
    glat = 42.6
    glon = -71.2
    sett=[]
    for alt in alts:
        sett.append(CheckPoint(date,glat,glon,np.array([alt])))
    # print(alts)
    T2=tm.time()
    ds20 = msis21.evaluate(date, glat, glon, alts)
    _, IR20 = iri20.evaluate(date, glat, glon, alts)
    T3=tm.time()
    altArr=[arr['alt_km'] for arr in sett]
    OArr=[arr['O+'] for arr in sett]
    plt.figure()
    plt.plot(OArr,altArr,label='singular')
    plt.plot(IR20['O+'],IR20['alt_km'],label='grouped',linestyle='--')
    plt.xscale('log')
    plt.legend()
    plt.show()
    print('Time for individual calls: '+str(T2-T1))
    print('Time for group calls: '+str(T3-T2))

    return 0
def PlotAltSweep(date,lat,lon,alts):
    Data=DensitySweepAlt(lat, lon, alts, date)
    plt.figure()
    ax=sns.lineplot(data=Data,x='alts',y='N_n')
    ax=sns.lineplot(data=Data,x='alts',y='N_i')
    plt.xlabel('Altitude (km)')
    plt.ylabel(r'Ion density $(m^{-3})$')
    plt.yscale('log')
    plt.title(date)
    plt.show()
# CheckAltTimes()

# date  = datetime(2022, 3, 22, 18, 0).astimezone(UTC)
glat = 42.6
glon = -71.2
galt=np.array(np.linspace(200,1000,50))

def PlotAltSweep2():
    # Data1=DensitySweepAlt(42.6, -71.2, np.array(np.linspace(100,1000,500)), datetime(2016, 8, 22, 18, 0).astimezone(UTC))
    Data1=DensitySweepAlt(42.6, -1.2, np.array(np.linspace(100,1000,500)), datetime(2022, 8, 22, 18, 0).astimezone(UTC))
    Data2=DensitySweepAlt(42.6, -71.2, np.array(np.linspace(100,1000,500)), datetime(2022, 8, 22, 18, 0).astimezone(UTC))

    plt.figure()
    # ax=sns.lineplot(data=Data1,x='O+',y='alts')
    plt.plot(Data1['M_n'],Data1['alts'],label='2016')
    plt.plot(Data2['M_n'],Data2['alts'],label='2022')
    plt.legend()
    # ax=sns.lineplot(data=Data1,x='alts',y='N_i')
    # print(Data2)
    # ax=sns.lineplot(data=Data2,x='alts',y='N_i')
    plt.ylabel('Altitude (km)')
    plt.xlabel(r'Mass density $(arb.)$')
    plt.xscale('log')
    # plt.title(date)
    plt.show()
def MassDensity(): #prints equivalent of number density plot in terms of mass density
    dn=datetime(2022, 8, 22, 18, 0).astimezone(UTC)
    Data=DensitySweepAlt(42.6, -71.2, np.array(np.linspace(60,1000,500)), dn)
    species = ['O', 'O2', 'N2','Ar','O+','O2+','NO+','H+','He+']
    descs = ['O', 'O$_2$', 'N$_2$','Ar','O$^+$','O2$^+$','NO$^+$','H$^+$','He$^+$']
    colors = ['r', 'g', 'b', 'orange','r','g','m','y','hotpink']
    weights=[Cons.m_p*i for i in [16,32,28,39,16,32,30,1,4]]
    linestyles=["--","--","--","--",":",":",":",":",":"]
    # entries=[ [:] for i in species]     
    plt.figure()
    for i in range(len(species)):
        plt.plot(Data[species[i]]*weights[i],Data['alts'],color=colors[i],label=descs[i],linestyle=linestyles[i])
    plt.plot(Data['M_n']+Data['M_i'],Data['alts'],color='k')
    plt.annotate(r'$NO$',xy=(10**-2,620),c='m')
    plt.annotate(r'$NO^+$',xy=(10**-15,125),c='m')
    plt.annotate(r'$O_2$',xy=(10**-19.5,600),c='g')
    plt.annotate(r'$O_2^+$',xy=(10**-19,120),c='g')
    plt.annotate(r'$O$',xy=(10**-10,400),c='r')
    plt.annotate(r'$O^+$',xy=(10**-16,650),c='r')
    plt.annotate(r'$N_2$',xy=(10**-19,850),c='b')
    plt.annotate(r'$N^+$',xy=(10**-2.5,200),c='b')
    plt.annotate(r'$\rho_M$',xy=(10**-14.5,900),c='k')
    plt.annotate(r'$H^+$',xy=(10**-16.7,800),c='y')
    plt.annotate(r'$Ar$',xy=(10**-20,550),c='orange')
    plt.annotate(r'$He^+$',xy=(10**-18.5,900),c='hotpink')
    plt.xscale('log')
    # plt.legend()
    plt.ylabel('alt (km)')
    plt.xlabel(r'$\rho_{m,i}$ $(kg/m^3)$')
    plt.ylim([100,1000])
    plt.xlim([10**(-21),10**(-6)])
    plt.title(dn)
    plt.savefig('MassDensity.png',dpi=300)
    plt.show()
    
    # plt.figure()
    # for i in range(len(species)):
    #     plt.plot(Data[species[i]],Data['alts'],color=colors[i],label=descs[i],linestyle=linestyles[i])

    # plt.plot(Data['e-'],Data['alts'],color='k')
    # plt.annotate(r'$NO$',xy=(10**-2,620),c='m')
    # plt.annotate(r'$NO^+$',xy=(10**3.4,125),c='m')
    # plt.annotate(r'$O_2$',xy=(10**-1.2,700),c='g')
    # plt.annotate(r'$O_2^+$',xy=(10**0,120),c='g')
    # plt.annotate(r'$O$',xy=(10**7.2,500),c='r')
    # plt.annotate(r'$O^+$',xy=(10**1,185),c='r')
    # plt.annotate(r'$N_2$',xy=(10**0.5,750),c='b')
    # plt.annotate(r'$N^+$',xy=(10**-2.5,200),c='b')
    # plt.annotate(r'$e^-$',xy=(10**5.5,250),c='k')
    # plt.annotate(r'$H^+$',xy=(10**2.95,390),c='y')
    # plt.annotate(r'$Ar$',xy=(10**-2,530),c='orange')
    # plt.annotate(r'$He^+$',xy=(10**1.9,600),c='hotpink')
    # plt.xscale('log')
    # # plt.legend()
    # plt.ylabel('alt (km)')
    # plt.xlabel(r'$N_{i}$ $(m^{-3})$')
    # plt.ylim([100,1000])
    # plt.xlim([10**(2),10**(17)])
    # plt.title(dn)
    # plt.savefig('NumberDensity.png',dpi=300)
    # plt.show()
    # return 0
def CheckRange():

    Data1=DensitySweepAlt(15.6, -1.2, np.array(np.linspace(150,1000,100)), datetime(2022, 8, 22, 18, 0).astimezone(UTC))
    Data2=DensitySweepAlt(42.6, -71.2, np.array(np.linspace(150,1000,100)), datetime(2020, 8, 22, 18, 0).astimezone(UTC))
    Data3=DensitySweepAlt(15.6, -1.2, np.array(np.linspace(150,1000,100)), datetime(2016, 8, 22, 18, 0).astimezone(UTC))
    Data4=DensitySweepAlt(42.6, -71.2, np.array(np.linspace(150,1000,100)), datetime(2018, 8, 22, 18, 0).astimezone(UTC))
    Data5=DensitySweepAlt(-24.6, 50.2, np.array(np.linspace(150,1000,100)), datetime(2021, 8, 22, 18, 0).astimezone(UTC))
    Data6=DensitySweepAlt(42.6, 10.2, np.array(np.linspace(150,1000,100)), datetime(2019, 8, 22, 18, 0).astimezone(UTC))
    Data7=DensitySweepAlt(56.1, 70.2, np.array(np.linspace(150,1000,100)), datetime(2015, 8, 22, 18, 0).astimezone(UTC))
    Data8=DensitySweepAlt(42.6, -71.2, np.array(np.linspace(150,1000,100)), datetime(2014, 8, 22, 18, 0).astimezone(UTC))
    plt.figure()
    plt.scatter(Data1['Av_M'],Data1['N_i'])
    plt.scatter(Data2['Av_M'],Data2['N_i'])
    plt.scatter(Data3['Av_M'],Data3['N_i'])
    plt.scatter(Data4['Av_M'],Data4['N_i'])
    plt.scatter(Data5['Av_M'],Data5['N_i'])
    plt.scatter(Data6['Av_M'],Data6['N_i'])
    plt.scatter(Data7['Av_M'],Data7['N_i'])
    plt.scatter(Data8['Av_M'],Data8['N_i'])


    print(np.max([np.max(Data1['Av_M']),np.max(Data2['Av_M']),np.max(Data3['Av_M']),np.max(Data4['Av_M']),np.max(Data5['Av_M']),np.max(Data7['Av_M']),np.max(Data8['Av_M'])]))
    print(np.min([np.max(Data1['Av_M']),np.min(Data2['Av_M']),np.min(Data3['Av_M']),np.min(Data4['Av_M']),np.min(Data5['Av_M']),np.min(Data7['Av_M']),np.min(Data8['Av_M'])]))
    print(np.max([np.max(Data1['N_i']),np.max(Data2['N_i']),np.max(Data3['N_i']),np.max(Data4['N_i']),np.max(Data5['N_i']),np.max(Data7['N_i']),np.max(Data8['N_i'])]))
    print(np.min([np.max(Data1['N_i']),np.min(Data2['N_i']),np.min(Data3['N_i']),np.min(Data4['N_i']),np.min(Data5['N_i']),np.min(Data7['N_i']),np.min(Data8['N_i'])]))
    plt.yscale('log')
    plt.xlabel('Ion Mass (amu)')
    plt.ylabel('Ion density')
    return 0
# CheckRange()
# MassDensity()
# PlotAltSweep2()
# galt=np.array(400)
# CheckPoint(date,glat,glon,galt)