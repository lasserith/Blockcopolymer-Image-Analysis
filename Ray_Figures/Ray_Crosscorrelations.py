# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:03:33 2019

@author: Moshe
"""
import os
from PIL import Image
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from glob import glob #glob? GLOB!

#%% Options here

DomPerTrench = int(7)

try:
    os.stat(os.path.join(os.curdir,"output"))
except:
    os.mkdir(os.path.join(os.curdir,"output"))
#%%
    
CrossEF, CrossEAx = plt.subplots(4,3, figsize=(9,12)) # this will make cross temp LER fig

CrossEWPF, CrossEWPAx = plt.subplots(3,3, figsize=(9,9)) # this will make cross temp LER LWR LPR composite fig
#%% loop to import and reduce >
TempList = [150, 210, 240]
for TT in np.arange(len(TempList)):
    # could use iglob but mehhhhhhhhhhhhh
    ELList = glob("./**/Paper*/**/*%i*FitEdgeL.csv" %(TempList[TT]), recursive = True)
    ERList = glob("./**/Paper*/**/*%i*FitEdgeR.csv" %(TempList[TT]), recursive = True)
    FPList = glob("./**/Paper*/**/*%i*FitPeak.csv" %(TempList[TT]), recursive = True)
    FWList = glob("./**/Paper*/**/*%i*FitWidth.csv" %(TempList[TT]), recursive = True)
    for ii in np.arange(len(ELList)): # start per file loop
        # reimport stuff on a file/file basis and rename it as it was so I can copy pasta the figure code from Ray1D here
        FPeak = np.genfromtxt(FPList[ii], delimiter=",", filling_values=float('nan'))
        FPWidth = np.genfromtxt(FWList[ii], delimiter=",", filling_values=float('nan'))
        FEL = np.genfromtxt(ELList[ii], delimiter=",", filling_values=float('nan'))
        FER = np.genfromtxt(ERList[ii], delimiter=",", filling_values=float('nan'))
       

        #% Calc Displacement for peaks to do drift correct
        FDisp = ((FPeak.transpose() - np.nanmean(FPeak,axis=1)).transpose())
        
        #% Do thermal drift correction
        XTD = np.arange(FDisp.shape[1])
        YTD = np.nanmean(FDisp,axis=0)
        TDFit = np.polyfit(XTD,YTD,1)
        TDPlot = np.polyval(TDFit,XTD)
        # now correct the data for drift
        FDispCorrect = (FDisp - TDPlot)
        FELDrift = (FEL - TDPlot)
        FERDrift = (FER - TDPlot)

        #% Calc Displacement for edges/width
        
        FELCorrect = ((FEL.transpose() - np.nanmean(FEL,axis=1)).transpose())
        FERCorrect = ((FER.transpose() - np.nanmean(FER,axis=1)).transpose())
        FPWidthRes = ((FPWidth.transpose() - np.nanmean(FPWidth,axis=1)).transpose())

        #% put edges together
        
        FECorrect = np.zeros((FER.shape[0]*2,FER.shape[1]))
        FECorrect[0::2,:] = FELCorrect
        FECorrect[1::2,:] = FERCorrect




        #%% Cross Corref 
        StackDisp = FDispCorrect.transpose()[:,0:DomPerTrench]
        StackWidth = FPWidthRes.transpose()[:,0:DomPerTrench]
        StackEdge = FECorrect.transpose()[:,0:DomPerTrench*2]
        NumTrench = FDispCorrect.shape[0]/DomPerTrench
        for xx in np.arange(1,NumTrench).astype(int):
            StackDisp=np.concatenate( (StackDisp,FDispCorrect.transpose()[:,xx*DomPerTrench:(xx+1)*DomPerTrench]) )
            StackWidth=np.concatenate((StackWidth,FPWidthRes.transpose()[:,xx*DomPerTrench:(xx+1)*DomPerTrench]))
            StackEdge=np.concatenate((StackEdge,FECorrect.transpose()[:,2*xx*DomPerTrench:2*(xx+1)*DomPerTrench]))
    

        PDStackD = pd.DataFrame(data=StackDisp)
        PDStackW = pd.DataFrame(data=StackWidth)
        PDStackE = pd.DataFrame(data=StackEdge)
        if ii == 0:
            TempStackD = PDStackD.copy()
            TempStackW = PDStackW.copy()
            TempStackE = PDStackE.copy()
        else:
            TempStackD = TempStackD.append(PDStackD)
            TempStackW = TempStackW.append(PDStackW)
            TempStackE = TempStackE.append(PDStackE)
        
        # end per file loop below is per temperature
    print('Temp = %i,n=%f' %(TempList[TT],TempStackD.size))
    
    StackD1O = np.zeros((0,2)) # 1 over correlation
    StackW1O = np.zeros((0,2)) # ditto for width
    StackE1O = np.zeros((0,2)) # ditto for width
    StackD2O = np.zeros((0,2)) # 2 over correlation
    StackW2O = np.zeros((0,2)) # ditto for width
    StackE2O = np.zeros((0,2)) # ditto for width
    StackD3O = np.zeros((0,2)) # 3 over correlation
    StackW3O = np.zeros((0,2)) # ditto for width
    StackE3O = np.zeros((0,2)) # ditto for width
    StackE4O = np.zeros((0,2)) # ditto for width
    StackE5O = np.zeros((0,2)) # ditto for width
    
    
    
    for nn in range(DomPerTrench-1):
        StackD1O = np.append( StackD1O, np.array((TempStackD.values[:,nn],TempStackD.values[:,nn+1])).transpose(),axis = 0 )
        StackW1O = np.append( StackW1O, np.array((TempStackW.values[:,nn],TempStackW.values[:,nn+1])).transpose(),axis = 0 )
        StackE1O = np.append( StackE1O, np.array((TempStackE.values[:,nn],TempStackE.values[:,nn+1])).transpose(),axis = 0 )
        StackE1O = np.append( StackE1O, np.array((TempStackE.values[:,2*nn],TempStackE.values[:,2*nn+1])).transpose(),axis = 0 )
        if nn < DomPerTrench-2:
            StackD2O = np.append( StackD2O, np.array((TempStackD.values[:,nn],TempStackD.values[:,nn+2])).transpose(),axis = 0 )
            StackW2O = np.append( StackW2O, np.array((TempStackW.values[:,nn],TempStackW.values[:,nn+2])).transpose(),axis = 0 )
            StackE2O = np.append( StackE2O, np.array((TempStackE.values[:,nn],TempStackE.values[:,nn+2])).transpose(),axis = 0 )
            StackE2O = np.append( StackE2O, np.array((TempStackE.values[:,2*nn],TempStackE.values[:,2*nn+2])).transpose(),axis = 0 )
        if nn < DomPerTrench-3:
            StackD3O = np.append( StackD3O, np.array((TempStackD.values[:,nn],TempStackD.values[:,nn+3])).transpose(),axis = 0 )
            StackW3O = np.append( StackW3O, np.array((TempStackW.values[:,nn],TempStackW.values[:,nn+3])).transpose(),axis = 0 )
            StackE3O = np.append( StackE3O, np.array((TempStackE.values[:,nn],TempStackE.values[:,nn+3])).transpose(),axis = 0 )
            StackE3O = np.append( StackE3O, np.array((TempStackE.values[:,2*nn],TempStackE.values[:,2*nn+3])).transpose(),axis = 0 )
        if nn < DomPerTrench-4:
            StackE4O = np.append( StackE4O, np.array((TempStackE.values[:,nn],TempStackE.values[:,nn+4])).transpose(),axis = 0 )
            StackE4O = np.append( StackE4O, np.array((TempStackE.values[:,2*nn],TempStackE.values[:,2*nn+4])).transpose(),axis = 0 )
        if nn< DomPerTrench-5:
            StackE5O = np.append( StackE5O, np.array((TempStackE.values[:,nn],TempStackE.values[:,nn+5])).transpose(),axis = 0 )
            StackE5O = np.append( StackE5O, np.array((TempStackE.values[:,2*nn],TempStackE.values[:,2*nn+5])).transpose(),axis = 0 )
    
    
    CCDisp = TempStackD.corr() # calcualte cross correlations
    CCWidth = TempStackW.corr() # calcualte cross correlations
    CCEL = TempStackE.corr()


    #%% Plot LER
    extent = (-5, 5,-5,5)
    LERCrossF , LERCrossAx = plt.subplots(6,1, figsize=(4,12))
    LERCrossAx[0].imshow(CCEL.values[0:14,0:14], cmap="seismic_r",extent=(0,14,0,14),vmin=-1, vmax=1)
    LERCrossAx[0].set_title('Edge')
    LERCrossAx[0].set_xticks([])
    LERCrossAx[0].set_xticks(2*np.arange(0,7)+1)
    LERCrossAx[0].set_xticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    LERCrossAx[0].set_yticks([])
    LERCrossAx[0].set_yticks(2*np.arange(0,7)+1)
    LERCrossAx[0].set_yticklabels(('e7','e6','e5','e4','e3','e2','e1'))
    LERCrossAx[0].set_ylim(0,14)
    LERCrossAx[0].set_xlim(0,14)
    LERCrossAx[1].hexbin(StackE1O[:,0],StackE1O[:,1],gridsize=20,extent=extent)
    LERCrossAx[1].set_aspect('equal')
    LERCrossAx[2].hexbin(StackE2O[:,0],StackE2O[:,1],gridsize=20,extent=extent)
    LERCrossAx[2].set_aspect('equal')
    LERCrossAx[3].hexbin(StackE3O[:,0],StackE3O[:,1],gridsize=20,extent=extent)
    LERCrossAx[3].set_aspect('equal')
    LERCrossAx[4].hexbin(StackE4O[:,0],StackE4O[:,1],gridsize=20,extent=extent)
    LERCrossAx[4].set_aspect('equal')
    LERCrossAx[5].hexbin(StackE5O[:,0],StackE5O[:,1],gridsize=20,extent=extent)
    LERCrossAx[5].set_aspect('equal')
    for ax, color in zip([LERCrossAx[1], LERCrossAx[2],LERCrossAx[3]], ['#00FF00', '#FF00FF', 'k']):
        plt.setp(ax.spines.values(), color=color)
        plt.setp(ax.spines.values(), linewidth=3)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)
    plt.tight_layout(h_pad=1.0)
    LERCrossF.savefig(os.path.join(os.curdir,"output","%iCLER_Cross.png" %(TempList[TT])), dpi=600)
    # w/o guidelines
    # now plot lines to guide eyes
    LERCrossAx[0].plot(np.arange(0,14),13-np.arange(0,14),color='#00FF00',linewidth = 2)
    LERCrossAx[0].plot(np.arange(0,14),12-np.arange(0,14),color='#FF00FF',linewidth = 2)
    LERCrossAx[0].plot(np.arange(0,14),11-np.arange(0,14),color='k',linewidth = 2)
    plt.tight_layout(h_pad=1.0)
    LERCrossF.savefig(os.path.join(os.curdir,"output","%iCLER_GCross.png" %(TempList[TT])), dpi=600)
    #%%
    LERCrossF.clf()
    plt.close(LERCrossF)
    
    #%% now plot LER on composite figs
    CrossEAx[0,TT].imshow(CCEL.values[0:14,0:14], cmap="seismic_r",extent=(0,14,0,14),vmin=-1, vmax=1)
    CrossEAx[0,TT].set_title('%i°C'%(TempList[TT]))
    CrossEAx[0,TT].set_xticks([])
    CrossEAx[0,TT].set_xticks(2*np.arange(0,7)+1)
    CrossEAx[0,TT].set_xticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    CrossEAx[0,TT].set_yticks([])
    CrossEAx[0,TT].set_yticks(2*np.arange(0,7)+1)
    CrossEAx[0,TT].set_yticklabels(('e7','e6','e5','e4','e3','e2','e1'))
    CrossEAx[0,TT].set_ylim(0,14)
    CrossEAx[0,TT].set_xlim(0,14)
    CrossEAx[1,TT].hexbin(StackE1O[:,0],StackE1O[:,1],gridsize=20,extent=extent)
    CrossEAx[1,TT].set_aspect('equal')
    CrossEAx[2,TT].hexbin(StackE2O[:,0],StackE2O[:,1],gridsize=20,extent=extent)
    CrossEAx[2,TT].set_aspect('equal')
    CrossEAx[3,TT].hexbin(StackE3O[:,0],StackE3O[:,1],gridsize=20,extent=extent)
    CrossEAx[3,TT].set_aspect('equal')
    for ax, color in zip([CrossEAx[1,TT], CrossEAx[2,TT],CrossEAx[3,TT]], ['#00FF00', '#FF00FF', 'k']):
        plt.setp(ax.spines.values(), color=color)
        plt.setp(ax.spines.values(), linewidth=3)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)
    # w/ guidelines only
    CrossEAx[0,TT].plot(np.arange(0,14),13-np.arange(0,14),color='#00FF00',linewidth = 2)
    CrossEAx[0,TT].plot(np.arange(0,14),12-np.arange(0,14),color='#FF00FF',linewidth = 2)
    CrossEAx[0,TT].plot(np.arange(0,14),11-np.arange(0,14),color='k',linewidth = 2)
    plt.tight_layout(h_pad=1.0)

    #%% LER is plotted twice. Also on composite with LWR/LPR
    CrossEWPAx[0,TT].imshow(CCEL.values[0:14,0:14], cmap="seismic_r",extent=(0,14,0,14),vmin=-1, vmax=1)
    CrossEWPAx[0,TT].set_title('%i°C'%(TempList[TT]))
    CrossEWPAx[0,TT].set_xticks([])
    CrossEWPAx[0,TT].set_xticks(2*np.arange(0,7)+1)
    CrossEWPAx[0,TT].set_xticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    CrossEWPAx[0,TT].set_yticks([])
    CrossEWPAx[0,TT].set_yticks(2*np.arange(0,7)+1)
    CrossEWPAx[0,TT].set_yticklabels(('e7','e6','e5','e4','e3','e2','e1'))
    CrossEWPAx[0,TT].set_ylim(0,14)
    CrossEWPAx[0,TT].set_xlim(0,14)
    #%% Plot LPR
    LPRCrossF , LPRCrossAx = plt.subplots(4,1, figsize=(4,12))
    LPRCrossF.suptitle('Positional Correlations')
    LPRCrossAx[0].imshow(CCDisp, cmap="seismic_r", vmin=-1, vmax=1)
    LPRCrossAx[0].set_xticks([])
    LPRCrossAx[0].set_xticks(np.arange(0,7))
    LPRCrossAx[0].set_xticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    LPRCrossAx[0].set_yticks([])
    LPRCrossAx[0].set_yticks(np.arange(0,7))
    LPRCrossAx[0].set_yticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    LPRCrossAx[1].hexbin(StackD1O[:,0],StackD1O[:,1],gridsize=20,extent=(-10, 10, -10, 10))
    LPRCrossAx[1].set_aspect('equal')
    LPRCrossAx[2].hexbin(StackD2O[:,0],StackD2O[:,1],gridsize=20,extent=(-10, 10, -10, 10))
    LPRCrossAx[2].set_aspect('equal')
    LPRCrossAx[3].hexbin(StackD3O[:,0],StackD3O[:,1],gridsize=20,extent=(-10, 10, -10, 10))
    LPRCrossAx[3].set_aspect('equal')
    
    LPRCrossF.savefig(os.path.join(os.curdir,"output","%iCLPR_Cross.png" %(TempList[TT])), dpi=600)
    #%%
    LPRCrossF.clf()
    plt.close(LPRCrossF)

    #%% Plot LPR Composite
    CrossEWPAx[1,TT].imshow(CCDisp, cmap="seismic_r", vmin=-1, vmax=1)
    CrossEWPAx[1,TT].set_xticks([])
    CrossEWPAx[1,TT].set_xticks(np.arange(0,7))
    CrossEWPAx[1,TT].set_xticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    CrossEWPAx[1,TT].set_yticks([])
    CrossEWPAx[1,TT].set_yticks(np.arange(0,7))
    CrossEWPAx[1,TT].set_yticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    #%%
    extent = (-5, 5,-5,5)
    LWRCrossF , LWRCrossAx = plt.subplots(4,1, figsize=(4,12))
    LWRCrossF.suptitle('Width Correlations')
    LWRCrossAx[0].imshow(CCWidth, cmap="seismic_r", vmin=-1, vmax=1)
    LWRCrossAx[0].set_xticks([])
    LWRCrossAx[0].set_xticks(np.arange(0,7))
    LWRCrossAx[0].set_xticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    LWRCrossAx[0].set_yticks([])
    LWRCrossAx[0].set_yticks(np.arange(0,7))
    LWRCrossAx[0].set_yticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    LWRCrossAx[1].hexbin(StackW1O[:,0],StackW1O[:,1],gridsize=20,extent=extent)
    LWRCrossAx[1].set_aspect('equal')
    LWRCrossAx[2].hexbin(StackW2O[:,0],StackW2O[:,1],gridsize=20,extent=extent)
    LWRCrossAx[2].set_aspect('equal')
    LWRCrossAx[3].hexbin(StackW3O[:,0],StackW3O[:,1],gridsize=20,extent=extent)
    LWRCrossAx[3].set_aspect('equal')
    
    LWRCrossF.savefig(os.path.join(os.curdir,"output","%iCLWR_Cross.png" %(TempList[TT]) ), dpi=600)
    #%%
    LWRCrossF.clf()
    plt.close(LWRCrossF)

    #%% LWR Composite
    CrossEWPAx[2,TT].imshow(CCWidth, cmap="seismic_r", vmin=-1, vmax=1)
    CrossEWPAx[2,TT].set_xticks([])
    CrossEWPAx[2,TT].set_xticks(np.arange(0,7))
    CrossEWPAx[2,TT].set_xticklabels(('e1','e2','e3','e4','e5','e6','e7'))
    CrossEWPAx[2,TT].set_yticks([])
    CrossEWPAx[2,TT].set_yticks(np.arange(0,7))
    CrossEWPAx[2,TT].set_yticklabels(('e1','e2','e3','e4','e5','e6','e7'))

#%% Save composites
CrossEF.savefig(os.path.join(os.curdir,"output","E_Cross.png"), dpi=600)

CrossEWPAx[0,0].set_ylabel('Edge Cross Correlations')
CrossEWPAx[1,0].set_ylabel('Position Cross Correlations')
CrossEWPAx[2,0].set_ylabel('Width Cross Correlations')
plt.tight_layout(h_pad=1.0)
CrossEWPF.savefig(os.path.join(os.curdir,"output","EWP_Cross.png"), dpi=600)

