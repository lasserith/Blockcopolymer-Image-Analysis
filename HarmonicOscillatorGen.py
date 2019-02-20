# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 05:01:01 2018

@author: Moshe
"""
import numpy as np
#%% generate some fake FDisp data
def UnderdampHO(Mass, bfact, k, Amp, Phase, t):
    # remember by rule b^2 < 4*m*k. Insuring bfact is >1 will do this
    if bfact<=1:
        print('b is <=1 you did an oopsy')
    b = np.sqrt(4*Mass*k)/bfact
    Freq = np.sqrt(k/Mass)
    return Amp*np.exp(-b*t/(2*Mass))*np.cos(Freq*t-Phase)

def CritDamp(Mass, k, A, B, t):
    b = np.sqrt(4*Mass*k) # def of critically damped
    return np.exp(-b*t/(2*Mass))*(A+B*t)

def OverDampHO(Mass, bfact, k, A, B, t):
    if bfact<=1:
        print('b is <=1 you did an oopsy')
    b = np.sqrt(4*Mass*k)*bfact
    r1 = (-b+np.sqrt(b**2-4*Mass*k))/(2*Mass)
    r2 = (-b-np.sqrt(b**2-4*Mass*k))/(2*Mass)
    # A and B related to IC. A+B=X0 A*r1+b*r1=V0
    return A*np.exp(r1*t)+B*np.exp(r2*t)

# set some defaults
Amp = 2
Mass = 20
k = 5
bfact = 2
NumMake = 28
TimeStep = 0.5
NumTime = 512
A = 1
B = 1
Rot = True
Phase = np.zeros((1,NumMake))+np.random.rand()*2*np.pi
PhaseRand = np.random.rand(1,NumMake)
FDisp = np.zeros((NumMake,NumTime))
FDispRand = np.zeros((NumMake,NumTime))

for ii in np.arange(NumTime):
    FDisp[:,ii] = UnderdampHO(Mass,bfact,k,Amp,Phase,ii*TimeStep)
    FDispRand[:,ii] = UnderdampHO(Mass,bfact,k,Amp,PhaseRand,ii*TimeStep)
    
#Apply some drift?
if Rot == True:
    MaxOff = Amp # just use amplitude of oscillator as amplitude of drift
    Offset = np.linspace(-MaxOff/2,MaxOff/2,NumTime)
    FDispRot = FDisp-Offset
    FDispRandRot = FDispRand-Offset