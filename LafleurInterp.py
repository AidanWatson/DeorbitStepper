#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:48:37 2026

@author: faunw
"""

import os 
import sys
module_path = os.path.abspath('/home/faunw/SpyderFiles/PythonPostProcessing/Charged_Aerodynamics_Code/')
if module_path not in sys.path:
    sys.path.append(module_path)
from central_potential import find_drag_force, find_Sheath
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

import pickle
import scipy.constants as Cons
k_B = 1.380649e-23
epsilon_0 = 8.85418782e-12
q_e = 1.60217662e-19
e0=Cons.epsilon_0
q_e=Cons.elementary_charge
m_e = 9.1093837e-31
u = 1.66053906660e-27
N_xi = 1000
root_tol = 1e-8
def TestSingleLafleur():
    w=0.02
    T_e=1997
    Cd=2
    v=7500
    rw=0.005
    rhoQ=4e11
    Mi=16*Cons.m_p
    phiB=20
    Area=w*rw*2
    Fn=rhoQ*Mi*v**2*Cd/2*Area
    F_1 = find_drag_force(N_xi, 2*rw, [-phiB/(k_B*T_e/q_e)], v, rhoQ, Mi, T_e, w,root_tol)
    CC=F_1/Fn*Cd
    print('CC: '+str(CC))
    Fn=rhoQ*Mi*v**2*Cd/2*Area
    F_1 = find_drag_force(N_xi, 2*rw, [-phiB/(k_B*T_e/q_e)], v, 1e16, Cons.m_p*11, T_e, w,root_tol)
    CC=F_1/Fn*Cd
    print('CC: '+str(CC))
    # print(str(10**15))
# TestSingleLafleur()
def CheckLafleurSingle(rhoQEXP,M_i,rw,phi):
    w=0.02
    T_e=1997
    Cd=2
    v=7500
    rhoQ=10**rhoQEXP
    print('rhoQ: '+str(rhoQ))
    Area=rw*w*2
    Fn=rhoQ*M_i*Cons.m_p*v**2*Area*Cd/2
    # F_01 = M_i*n_0*v_0**2*np.pi*R_01**2

    
    F_1 = find_drag_force(N_xi, 2*rw, [-phi/(k_B*T_e/q_e)], v, rhoQ, Cons.m_p*M_i, T_e, w,root_tol)
    CC=F_1/Fn*Cd
    print('CC: '+str(CC))

    return CC

def CheckArray(Interp,AVRes,NiRes):
    AVM=np.linspace(1,30,AVRes)
    NExp=np.linspace(8,12,NiRes)
    RE, MI = np.meshgrid(NExp, AVM, indexing='ij')
    pts = np.stack([RE.ravel(), MI.ravel()], axis=-1)
    # print(pts)
    Z = Interp(pts).reshape(RE.shape)
    plt.figure(figsize=(7, 5))
    plt.pcolormesh(AVM, NExp, Z.T, cmap='viridis', shading='auto')
    plt.colorbar(label='f(x, y)')
    plt.ylabel('Density Exponent') 
    plt.xlabel('Mass (amu)')
    plt.title('Interpolated CC')
    plt.show()

def TestCylinder():
    
    n_0=1e11
    e0=Cons.epsilon_0
    e=Cons.elementary_charge
    # T_e=0.172
    m=Cons.m_p
    # print('m'+str(m))
    v_0=7600
    

    #pt.run_msis()
    w=0.02
    R_01=0.005
    N_xi = 1000
    root_tol = 1e-8
    phi_B=-20
    T_e = 1500
    M_i=Cons.m_p*16
    # Cc=0.5
    # M_i=Average_Mass*n0
    print('Mass: '+str(M_i))
    # Fn=M_i*V**2*Cc/2*Area
    # F_01 = M_i*n0*v**2*np.pi*R_01**2
    F_01 = M_i*n_0*v_0**2*np.pi*R_01**2


    F_1 = find_drag_force(N_xi, R_01, [phi_B/(k_B*T_e/q_e)], v_0, n_0, M_i, T_e,w, root_tol)
    Cc=2*F_1/F_01
    print('CC: '+str(Cc))
    print('Neutral '+str(1e9*F_01)+ ' nN')
    # print('hrmmm'+str(phi_B/(k_B*T_e/q_e)))
    print('Charged '+str(1e9*F_1)+ ' nN')
    return Cc
def CheckLafleurSingle(rhoQEXP,M_i,rw,phi):
    w=0.02
    T_e=1997
    Cd=2
    v=7500
    rhoQ=10**rhoQEXP
    print('rhoQ: '+str(rhoQ))
    Area=rw*w*2
    Fn=rhoQ*M_i*Cons.m_p*v**2*Area*Cd/2
    # F_01 = M_i*n_0*v_0**2*np.pi*R_01**2

    
    F_1 = find_drag_force(N_xi, 2*rw, [-phi/(k_B*T_e/q_e)], v, rhoQ, Cons.m_p*M_i, T_e, w,root_tol)
    CC=F_1/Fn*Cd
    print('CC: '+str(CC))

    return CC
def LoadOrCreateInterpolator(rw, phi):
    filename = f"StepperCSV/lafleur_rw{rw}_phi{phi}.npz"
    
    if os.path.exists(filename):
        print(f"Loading from {filename}")
        data = np.load(filename)
        NExp = data['NExp']
        AVM  = data['AVM']
        Z    = data['Z']
    else:
        print("No cache found, computing...")
        f_vec = np.vectorize(lambda x, y: CheckLafleurSingle(x, y, rw, phi))
        AVM  = np.sort(np.linspace(1, 34, 10))
        NExp = np.sort(np.linspace(7.5, 13, 25))
        RE, MI = np.meshgrid(NExp, AVM, indexing='ij')
        Z = f_vec(RE, MI)
        np.savez(filename, NExp=NExp, AVM=AVM, Z=Z, rw=rw, phi=phi)

    return RegularGridInterpolator((NExp, AVM), Z, method='cubic')
def CreateCSVFixedPhi(rw,phi):
    f_vec = np.vectorize(lambda x, y: CheckLafleurSingle(x, y, rw, phi))

    w=0.02
    v=7500
    
    AVRes=5
    NiRes=5
    AVM=np.linspace(1,30,AVRes)
    NExp=np.linspace(8,13,NiRes)
    AVM=np.sort(AVM)
    NExp=np.sort(NExp)
    RE, MI = np.meshgrid(NExp, AVM, indexing='ij')

    Z=f_vec(RE,MI)
    print("--")
    print(Z)
    NVal=[10**1 for i in NExp]
    # ZI = griddata(points=(NExp.flatten(), AVM.flatten()),values=Z.flatten(), xi=(RE, MI),    method='cubic')   # 'linear' or 'nearest' also available
    filename = f"StepperCSV/lafleur_rw{rw}_phi{phi}.npz"
    np.savez(filename, NExp=NExp, AVM=AVM, Z=Z, rw=rw, phi=phi)
    print(f"Saved to {filename}")
    # IT= RegularGridInterpolator((NExp, AVM), Z, method='cubic')
    IT= RegularGridInterpolator((NExp, AVM), Z, method='cubic')

    print('AAAAAAAAA')
    return IT
def CompareAnswers2(IT,rw,phi):
    rw=0.005
    phi=40

    # AVM = np.sort(np.random.uniform(1, 30.0, size=AVRes))
    # NExp = np.sort(np.random.uniform(8, 12, size=NiRes))
    AVRes=14
    NiRes=14
    AVM = np.sort(np.random.uniform(1, 34.0, size=AVRes))
    NExp = np.sort(np.random.uniform(7.5, 13, size=NiRes))
    # AVM=np.linspace(1,30,AVRes)
    # NExp=np.linspace(8,12,NiRes)

    AVM=np.sort(AVM)
    NExp=np.sort(NExp)
    RE, MI = np.meshgrid(NExp, AVM, indexing='ij')
    f_vec = np.vectorize(lambda x, y: CheckLafleurSingle(x, y, rw, phi))
    
    Z2=f_vec(RE,MI)
    pts = np.stack([RE.ravel(), MI.ravel()], axis=-1)
    Z3 = IT(pts).reshape(RE.shape)


    print(pts)


    Z_pct_diff = np.where(Z2 != 0, (Z2 - Z3) / Z2 * 100, np.nan)
    abs_max = np.nanmax(np.abs(Z_pct_diff))

    # print((Z3-Z2)/Z3)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(NExp, AVM, Z_pct_diff.T, cmap='RdBu', shading='auto',
                       vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax, label='(Z1 - Z2) / Z1 × 100 %')
    plt.scatter(RE,MI,color='black')
    # plt.scatter(RE,RE,color='white')

    ax.set_xlabel(r'$log_{10}(n_i)$')
    ax.set_ylabel('M_i (amu)')
    ax.set_title('% difference')
    plt.tight_layout()
    plt.show()    # fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    print(AVM)
    print(NExp)
    print('_____')
    print(Z2)
    print(Z3)
# IT=CreateCSVFixedPhi(0.005, 40)
# CompareAnswers2(IT)


def TestRelo():
# def TestCylinder():
    nn=np.linspace(8,12,10)
    # n_0=1e11
    e0=Cons.epsilon_0
    e=Cons.elementary_charge
    # T_e=0.172
    m=Cons.m_p
    # print('m'+str(m))
    v_0=7600
    

    #pt.run_msis()
    w=0.02
    R_01=0.005
    N_xi = 1000
    root_tol = 1e-8
    phi_B=-20
    T_e = 1500
    M_i=Cons.m_p*16
    CCArr=[]
    # Cc=0.5
    # M_i=Average_Mass*n0
    print('Mass: '+str(M_i))
    # Fn=M_i*V**2*Cc/2*Area
    # F_01 = M_i*n0*v**2*np.pi*R_01**2
    for n in nn:
        n_0=10**n
        F_01 = M_i*n_0*v_0**2*np.pi*R_01**2


        F_1 = find_drag_force(N_xi, R_01, [phi_B/(k_B*T_e/q_e)], v_0, n_0, M_i, T_e,w, root_tol)
        Cc=2*F_1/F_01
        CCArr.append(Cc)
    plt.figure()
    plt.scatter(nn,CCArr)
    plt.xlabel('Exponent of density')
    plt.ylabel('Cc')
    plt.show()
    print('CC: '+str(Cc))
    print('Neutral '+str(1e9*F_01)+ ' nN')
    # print('hrmmm'+str(phi_B/(k_B*T_e/q_e)))
    print('Charged '+str(1e9*F_1)+ ' nN')
    return Cc
# TestRelo()
    
# TestCylinder()
# TestArray()
# IT=CreateCSVFixedPhi(0.005, 40)
# IT=LoadOrCreateInterpolator(0.005, 40)
# print(IT([])
# print(IT)
# print(float(IT([9,11])[0]))
# print(float(IT([12,12])[0]))

# # CheckArray(IT,15,15)
# CompareAnswers2(IT,0.005,40)

