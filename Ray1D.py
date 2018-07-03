# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:20:58 2017

AFM test for Raybin
"""
#%% Imports
import tkinter as tk
from tkinter import filedialog, ttk
import os
import csv
import lmfit
from PIL import Image
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy
from skimage import restoration, morphology, filters, feature
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
import matplotlib.animation as manimation
try:
    from igor.binarywave import load as loadibw
except: print('You will be unable to open Asylum data without igor')

#%% Defs
   #%%
def AutoDetect( FileName , Opt ):
    """
    Attempt to autodetect the instrument used to collect the image
    Currently supports the Zeiss Merlin
    V0.2
    
    0.1 - Orig
    0.2 - Asylum AFM added
    """
    if Opt.FExt == ".ibw": # file is igor
        if Opt.Machine!="Asylum AFM":Opt.Machine="Asylum AFM"; # avoid race condition in parallel loop
        RawData= loadibw(FileName)['wave']
        Labels = RawData['labels'][2]
        Labels = [i.decode("utf-8") for i in Labels] # make it strings
        # Need to add a selector here for future height/phase
        AFMIndex=[ i for i, s in enumerate(Labels) if Opt.AFMLayer in s] #they index from 1????
        AFMIndex= AFMIndex[0]-1 # fix that quick
        imarray = RawData['wData'][:,:,AFMIndex]
        #slow scan is column in original data
        # AFM data has to be leveled :(
        TArray=imarray.transpose() # necessary so that slow scan Y and fast scan is X EG afm tip goes along row > < then down to next row etc

        if Opt.AFMLevel == 1: #median leveling
            MeanSlow=imarray.mean(axis=0) # this calculates the mean of each slow scan row
            Mean=imarray.mean() # mean of everything
            MeanOffset=MeanSlow-Mean # determine the offsets
            imfit=imarray-MeanOffset # adjust the image
            
        elif Opt.AFMLevel==2: # median of dif leveling
            DMean=np.diff(TArray,axis=0).mean(axis=1) 
            # calc the 1st order diff from one row to next. Then average these differences 
            DMean=np.insert(DMean,0,0) # the first row we don't want to adjust so pop in a 0
            imfit = imarray-DMean
        elif Opt.AFMLevel==3: # Polynomial leveling
            imfit = np.zeros(imarray.shape)
            FastInd = np.arange(imarray.shape[0]) # this is 0 - N rows
            
            for SlowInd in np.arange(imarray.shape[1]): # for each column eg slowscan axis
                Slow = imarray[:, SlowInd]
                PCoef = np.polyfit(FastInd, Slow, Opt.AFMPDeg)
                POffset = np.polyval(PCoef , FastInd)
                imfit[:, SlowInd] = imarray[:, SlowInd] - POffset
        else: imfit=imarray

        #Brightness/Contrast RESET needed for denoising. Need to figure out how to keep track of this? add an opt?
#        imfit = imfit - imfit.min()
#        imfit = imfit*255/imfit.max()
        if Opt.NmPP!=RawData['wave_header']['sfA'][0]*1e9:Opt.NmPP=RawData['wave_header']['sfA'][0]*1e9 # hopefully avoid issues with parallel
        
        RawData.clear()
        imarray=imfit
        
    else:
        im= Image.open(FileName)
        if im.mode!="P":
            im=im.convert(mode='P')
            print("Image was not in the original format, and has been converted back to grayscale. Consider using the original image.")    
        imarray = np.array(im)
        
        SkimFile = open( FileName ,'rb')
        MetaF=exifread.process_file(SkimFile)
        SkimFile.close()
        try:
            Opt.FInfo=str(MetaF['Image Tag 0x8546'].values);
            Opt.NmPP=float(Opt.FInfo[17:30])*10**9;
            Opt.Machine="Merlin";
        except:
            pass


#    if Opt.NmPP!=0:
#        print("Instrument was autodetected as %s, NmPP is %f \n" % (Opt.Machine ,Opt.NmPP) )
#    else:
#        print("Instrument was not detected, and NmPP was not set. Please set NmPP and rerun")
    return(imarray);

def PeakPara(RawIn, NmPP, CValley, FitWidth, FPeak, FPWidth, ImNum, tt):
    Length = RawIn.shape[1]
    gmodel = lmfit.Model(gaussian)
    Inits = gmodel.make_params()
    
    
    for pp in range(len(CValley)): #loop through peak positions (guesstimates)
        PCur = CValley[pp]
        PLow = int(np.maximum((PCur-FitWidth),0))
        PHigh = int(np.min((PCur+FitWidth+1,Length-1)))
        LocalCurve = abs((RawIn[PLow:PHigh,tt,ImNum]-max(RawIn[PLow:PHigh,tt,ImNum]))) # use with range(PLow,PHigh)
        # set our initial conditions
        #Inits['amp']=lmfit.Parameter(name='amp', value= max(LocalCurve), min=0, max=max(LocalCurve)+5)
        #Inits['wid']=lmfit.Parameter(name='wid', value= FitWidth, min=0, max=100)
        #Inits['cen']=lmfit.Parameter(name='cen', value= PCur, min=PCur-7, max=PCur+7)
        
        Inits['amp']=lmfit.Parameter(name='amp', value= max(LocalCurve))
        Inits['wid']=lmfit.Parameter(name='wid', value= FitWidth)
        Inits['cen']=lmfit.Parameter(name='cen', value= PCur, min=PCur-7, max=PCur+7)
    
        Res = gmodel.fit(LocalCurve, Inits, x=np.arange(PLow,PHigh))
        FPeak[pp,tt] = Res.best_values['cen']
        FPWidth[pp,tt] = abs(np.copy(Res.best_values['wid']*2.35482*NmPP)) # FWHM in NM
        if (abs(Res.best_values['cen'] - PCur) > 5) or (Res.best_values['wid'] > 50) or (Res.best_values['cen']==PCur):
            FPWidth[pp,tt] = np.nan
            FPeak[pp,tt] = np.nan

            
#%%
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

if __name__ == '__main__':
    # OK
    
    Vers = "AAA"
    
    # Will hold options
    class Opt:
        pass
    # WIll hold outputs
    class Output:
        pass
    #%% Default options
    #IndividualLog =1; # Write a log for each sample?
    CombLog = 1 # If One write a combined log, if two clean it out each time(don't append)
    ShowImage = 0 # Show images?
    Opt.NmPP = 0 # Nanometers per pixel scaling (will autodetect)
    Opt.l0 = 50 # nanometer l0
    
    #%% AFM Settings
    Opt.AFMLayer = "Phase" #Matched Phase ZSensor
    Opt.AFMLevel = 3  # 0 = none 1 = Median 2= Median of Dif 3 = polyfit
    Opt.AFMPDeg = 1 # degree of polynomial.
    
    # Autocorrelation max shift
    Opt.ACCutoff = 50
    Opt.ACSize = 200
    
    Opt.AngMP = 5 # Do a midpoint average based on this many points
    # EG AngMP = 5 then 1 2 3 4 5, 3 will be calc off angle 1 - 5
    Opt.Machine = "Unknown"
    
    
    #%% Open
    FOpen=tk.Tk()
    
    currdir = os.getcwd()
    FNFull = tk.filedialog.askopenfilename(parent=FOpen, title='Please select a file', multiple=1)
    FOpen.withdraw()
    #if len(FNFull) > 0:
    #    print("You chose %s" % FNFull)
    
    #%% Do Once
    ImNum = 0
    Opt.Name = FNFull[ImNum] # this hold the full file name
    Opt.FPath, Opt.BName= os.path.split(Opt.Name)  # File Path/ File Name
    (Opt.FName, Opt.FExt) = os.path.splitext(Opt.BName) # File name/File Extension split
    
    firstim = AutoDetect( FNFull[ImNum], Opt)
    Shap0 =firstim.shape[0]
    Shap1 = firstim.shape[1]
    RawIn = np.zeros((Shap0,Shap1,len(FNFull)))
    RawComb = np.zeros((Shap0,Shap1*len(FNFull)))
    
    
    
    #%% Import data ( can't be parallelized as it breaks the importer )
    
    for ii in range(0, len(FNFull)):
        try:
            if Opt.NmPPSet!=0:Opt.NmPP=Opt.NmPPSet # so if we set one then just set it
        except:
            pass
        RawIn[:,:,ii]=AutoDetect( FNFull[ii], Opt) # autodetect the machine, nmpp and return the raw data array
        print('Loading raw data %i/%i' %(ii,len(FNFull)))
        #if ii%2 == 1:
         #   RawIn[:,:,ii] = np.flipud(RawIn[:,:,ii]) # if odd flip upside down
        RawComb[:,ii*Shap1:(ii+1)*Shap1] = RawIn[:,:,ii] # make one array with all data in an order not used do as seperate experiments
    
    #%% Find the markers between arrays
    # per image
    for ImNum in range(0, len(FNFull)):
        
        Opt.Name = FNFull[ImNum] # this hold the full file name
        Opt.FPath, Opt.BName= os.path.split(Opt.Name)  # File Path/ File Name
        (Opt.FName, Opt.FExt) = os.path.splitext(Opt.BName) # File name/File Extension split
            
        # Make output folder if needed
        try:
            os.stat(os.path.join(Opt.FPath,"output"))
        except:
            os.mkdir(os.path.join(Opt.FPath,"output"))
       
        RawComp = RawIn[:,:,ii].sum(axis=1) # sum along the channels to get a good idea where peaks are
        RawTop = RawIn[:,:5,ii].sum(axis=1)
        RawBot = RawIn[:,-5:,ii].sum(axis=1)
        SavFil = scipy.signal.savgol_filter(RawComp,5,2,axis = 0)
        TopFil = scipy.signal.savgol_filter(RawTop,5,2,axis = 0)
        BotFil = scipy.signal.savgol_filter(RawBot,5,2,axis = 0)
        D1SavFil = scipy.signal.savgol_filter(RawComp,5,2,deriv = 1,axis = 0)
        D2SavFil = scipy.signal.savgol_filter(RawComp,5,2, deriv = 2,axis = 0)
        
        FPlot = plt.figure()
        FPlot.suptitle('SavGol Filter and Derivatives in black (Raw is Blue)')
        Xplot = range(0, Shap0)
        
        FPlot1 = FPlot.add_subplot(311)
        FPlot1.plot(Xplot,RawComp,'b',Xplot,SavFil,'k')
        
        FPlot2 = FPlot.add_subplot(312)
        FPlot2.plot(Xplot,D1SavFil,'k',Xplot,D2SavFil,'r')
        
        FPlot3 = FPlot.add_subplot(313)
        FPlot3.plot(Xplot,TopFil,'b',Xplot,BotFil,'k')
        FPlot.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "SVG.png"), dpi=300)
        FPlot.clf()
        plt.close(FPlot)
        # What is the peak corresponding to pattern?
        #%%
        #Following is per image. Find the peaks
        
        #PatPeak = np.zeros((100,RawComp.shape[1]))
        #PolyPeak =  np.zeros((150,RawComp.shape[1]))
        
        PSpace = int(np.floor(Opt.l0/Opt.NmPP*.5)) # peaks must be at least 70% of L0 apart
        
        
        Peak = np.zeros((0,0))
        Valley = np.zeros((0,0))
        PatSep = np.zeros((1,1))#start with zero as a potential separator
        
        
        for xx in range(len(SavFil)-1):
            if D1SavFil[xx]*D1SavFil[xx+1] <= 0: # if 1D changes signs
                if (D2SavFil[xx]+D2SavFil[xx+1])/2 <= 0: # if 2D is neg
                    if (SavFil[xx]+SavFil[xx+1])/2 < SavFil.max()*.6 : # if we're above 3k then it's the pattern
                        Peak = np.append(Peak,xx)
                    else: 
                        PatSep=np.append(PatSep,xx)
                else: 
                    Valley = np.append(Valley,xx)
        #% add the last value as a potential pat separator
        PatSep = np.append(PatSep,len(SavFil)-1)
        
        #%%
        #Dump raws
        
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "Valley.txt"),Valley,fmt='%3u')
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "Peak.txt"),Peak,fmt='%3u')
        #%% Clean up the peaks of the pattern
        #CPeak = np.zeros((0,0))
        #for pp in range(len(Peak)-1):
        #    if (Peak[pp+1]-Peak[pp]) < PSpace: # if peaks too close
        #        if SavFil[int(Peak[pp])] >= SavFil[int(Peak[pp+1])]:
        #            CPeak = np.append(CPeak, Peak[pp]) 
        #        else: # keep the higher value
        #            CPeak = np.append(CPeak, Peak[pp+1])
        #    elif ((Peak[pp]-Peak[pp-1]) >= PSpace) or pp == 0 : 
        #        CPeak = np.append(CPeak, Peak[pp])
        #if (Peak[-1]-Peak[-2]) >= PSpace: # catch the last peak 
        #    CPeak = np.append(CPeak, Peak[-1])
        #CPeak = np.unique(CPeak)
        #%% Clean up the valleys
        #CPeak = np.zeros((0,0))
        #for pp in range(len(Peak)-1):
        #    if (Peak[pp+1]-Peak[pp]) < PSpace: # if peaks too close
        #        if SavFil[int(Peak[pp])] >= SavFil[int(Peak[pp+1])]:
        #            CPeak = np.append(CPeak, Peak[pp]) 
        #        else: # keep the higher value
        #            CPeak = np.append(CPeak, Peak[pp+1])
        #    elif ((Peak[pp]-Peak[pp-1]) >= PSpace) or pp == 0 : 
        #        CPeak = np.append(CPeak, Peak[pp])
        #if (Peak[-1]-Peak[-2]) >= PSpace: # catch the last peak 
        #    CPeak = np.append(CPeak, Peak[-1])
        #CPeak = np.unique(CPeak)
        ##%% clean up the pattern separators
        #CPatSep = np.zeros((0,0))
        #for pp in range(len(PatSep)-1):
        #    if (PatSep[pp+1]-PatSep[pp]) < PSpace*5: # if PatSeps too close
        #        if SavFil[int(PatSep[pp])] >= SavFil[int(PatSep[pp+1])]:
        #            CPatSep = np.append(CPatSep, PatSep[pp]) 
        #        else: # keep the higher value
        #            CPatSep = np.append(CPatSep, PatSep[pp+1])
        #    elif ((PatSep[pp]-PatSep[pp-1]) >= PSpace*5) or pp == 0: 
        #        CPatSep = np.append(CPatSep, PatSep[pp])
        #if (PatSep[-1]-PatSep[-2]) >= PSpace*5: # catch the last PatSep 
        #    CPatSep = np.append(CPatSep, PatSep[-1])
        #
        #CPatSep = np.unique(CPatSep)
        #%%200C Manual
        #CPatSep = np.array([80,150,250,330,420,500])
        #CValley = np.array([92,99,106,114,121,129,136,143,171,178,186,194,204,212,220,228,263,270,278,285,293,300,308,315,349,356,363,371,379,385,393,401,435,442,450,457,465,472,480,488])
        #%% 2018-0611 Manual
        CValley = np.array([37, 44, 52, 60, 68, 76, 83 ,166, 174, 182, 189, 197, 205, 212, 296, 303, 311, 318, 325, 333, 341, 425, 433, 440, 448, 455, 463, 471])
        CPatSep = np.array([30,160,290, 420, 500])
        
        #%% for 150 as is
        # for 175 , -22
        #for 200 - 22
        # for 210  -19
        # for 220 
        #for 240 add 4
        CValley += -22
        CPatSep += -22
        
        
        #%%
        CBins = np.digitize(CValley,CPatSep)
        BinCount = np.histogram(CValley,CPatSep)[0]
        #%%
        FPeak =  np.zeros((len(CValley),RawIn.shape[1]))
        FPWidth = np.zeros((len(CValley),RawIn.shape[1]))
        FitWidth = int(Opt.l0/Opt.NmPP*.5)
        #FitWidth = int(4)
        #% Parallel cus gotta go fast, currently borked though :()
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

        Parallel(n_jobs=8,verbose=5)(delayed(PeakPara)(RawIn, Opt.NmPP, CValley, FitWidth, FPeak, FPWidth, ImNum, tt) # backend='threading' removed
            for tt in range(RawIn.shape[1]))
        print('Done')
        
        
        #%% Save filtered data
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "FitPeak.csv"),FPeak,delimiter=',')
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "FitFWHM.csv"),FPWidth,delimiter=',')      
        
        #%% Calc Displacement
        FDisp = ((FPeak.transpose() - np.nanmean(FPeak,axis=1)).transpose())
        
        #%% Do thermal drift correction
        XTD = np.arange(FDisp.shape[1])
        YTD = np.nanmean(FDisp,axis=0)
        TDFit = np.polyfit(XTD,YTD,1)
        TDPlot = np.polyval(TDFit,XTD)
        
        TDF, TDAx = plt.subplots()
        TDF.suptitle('Thermal Drift y=%f*x+%f' % (TDFit[0],TDFit[1]))
        TDAx.plot(XTD,YTD,XTD,TDPlot)
        TDF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "ThermalDrift.png"), dpi=600)
        TDF.clf()
        plt.close(TDF)
        #%% now correct the data and put in NanoMeters!
        FDispCorrect = (FDisp - TDPlot)*Opt.NmPP
        #%% move on
        
        
        
        PanDisp = pd.DataFrame(data=FDispCorrect.transpose())
        PanWidth = pd.DataFrame(data=FPWidth.transpose())
        
        
        
        #StackDisp 
        #StackWidth
        #%% Cross Corref 
        
        StackDisp = np.concatenate((FDispCorrect.transpose()[:,0:7],
                                    FDispCorrect.transpose()[:,7:14],
                                    FDispCorrect.transpose()[:,14:21],
                                    FDispCorrect.transpose()[:,21:]))
        StackWidth = np.concatenate((FPWidth.transpose()[:,0:7],
                                    FPWidth.transpose()[:,7:14],
                                    FPWidth.transpose()[:,14:21],
                                    FPWidth.transpose()[:,21:]))
        PDStackD = pd.DataFrame(data=StackDisp)
        PDStackW = pd.DataFrame(data=StackWidth)
        
        CCDisp = PDStackD.corr() # calcualte cross correlations
        CCWidth = PDStackW.corr() # calcualte cross correlations

        #%% Plot Pek Cross Corr
        CCF , CCAx = plt.subplots()
        CCF.suptitle('Peak displacement correlations (Pearson)')
        CCIm = CCAx.imshow(CCDisp, cmap="seismic_r", vmin=-1, vmax=1)
        CCF.colorbar(CCIm)
        CCF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "DisplacementCC.png"), dpi=300)
        CCF.clf()
        plt.close(CCF)
        #%%
        CCWidthF , CCWidthAx = plt.subplots()
        CCWidthF.suptitle('FWHM correlations (Pearson)')
        CCWidthIm = CCWidthAx.imshow(CCWidth, cmap="seismic_r", vmin=-1, vmax=1)
        CCWidthF.colorbar(CCWidthIm)
        CCWidthF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "WidthCC.png"), dpi=300)
        CCWidthF.clf()
        plt.close(CCWidthF)
        #%%
        CrossCorF , CrossCorAx = plt.subplots(1,6, figsize=(15,4))
        CrossCorF.suptitle('Line/Line Correlation for each set of lines, X axis is line, Y axis is next line')
        for nn in range(6):
            CrossCorAx[nn].hexbin(PDStackD.values[:,nn],PDStackD.values[:,nn+1],gridsize=20,extent=(-10, 10, -10, 10))
            CrossCorAx[nn].set_aspect('equal')
        CrossCorF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "CrossCor.png"), dpi=600)
        #%% Autocorrelation Opt.AcSize
        ACPeak = pd.DataFrame()
        CCPeak = pd.DataFrame()
        
        ACMSD = pd.DataFrame()
        CCMSD = pd.DataFrame()
        ACWidth = pd.DataFrame() 
        CCWidth = pd.DataFrame()
        
        for lag in range(Opt.ACSize):
            ACPeak = ACPeak.append( PanDisp.corrwith(PanDisp.shift(periods=lag)).rename('lag%i' %lag))
            CCPeak = CCPeak.append( PanDisp.corrwith(PanDisp.shift(periods=lag).shift(1,axis=1)).rename('lag%i' %lag))
            ACMSD = ACMSD.append(((PanDisp.shift(periods=lag)-PanDisp)**2).mean().rename('lag%i' %lag))
            ACWidth = ACWidth.append( PanWidth.corrwith(PanWidth.shift(periods=lag)).rename('lag%i' %lag))
            CCWidth = CCWidth.append( PanWidth.corrwith(PanWidth.shift(periods=lag).shift(1,axis=1)).rename('lag%i' %lag))
            
        #%% Autocorrelation
        ACDispF , ACDispAx = plt.subplots()
        ACDispF.suptitle('Peak displacement Autocorrelation')
        ACDispIm = ACDispAx.imshow(ACPeak, cmap="seismic_r", vmin=-1, vmax=1)
        ACDispF.colorbar(ACDispIm)
        ACDispF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "DisplacementAC.png"), dpi=300)
        ACDispF.clf()
        plt.close(ACDispF)        
        #%% MSD AC
        ACMSDF , ACMSDAx = plt.subplots()
        ACMSDF.suptitle('Peak displacement Autocorrelation')
        ACMSDIm = ACMSDAx.imshow(ACMSD, cmap="seismic_r", vmin=-1, vmax=1)
        ACMSDF.colorbar(ACMSDIm)
        ACMSDF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "MSDAC.png"), dpi=300)
        ACMSDF.clf()
        plt.close(ACMSDF)        
        
        #%%width
        ACWidthF , ACWidthAx = plt.subplots()
        ACWidthF.suptitle('Peak FWHM Autocorrelation')
        ACWidthIm = ACWidthAx.imshow(ACWidth, cmap="seismic_r", vmin=-1, vmax=1)
        ACWidthF.colorbar(ACWidthIm)
        ACWidthF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "WidthAC.png"), dpi=300)
        ACWidthF.clf()
        plt.close(ACWidthF)
        #%% MSD per line
        MSDF , MSDAx = plt.subplots(CBins.max(),BinCount.max(), figsize=(16,16))
        MSDF.suptitle('Mean Squared Displacement per line')
        for nn in range(CBins.max()):
            for ll in range(BinCount.max()):
                ACMSD[nn*CBins.max()+ll].plot(ax=MSDAx[nn,ll])
                MSDAx[nn,ll].set_ylim([0, 2])
                MSDAx[nn,ll].set_xlim([0, 100])
        MSDF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "MSD.png"), dpi=300)
        #MSDF.clf()
        #plt.close(MSDF)
        
        #%% save the kappa/KT
        PSeudoEA = np.nanmean(( FDispCorrect**-2),axis=1)
        PSeudoEA = PSeudoEA.reshape((4,7))
        if ImNum == 0:
            EAOut = PSeudoEA
        else:
            EAOut = np.concatenate((EAOut, PSeudoEA))
    np.savetxt(os.path.join(Opt.FPath,"output","SummedEA.csv"),EAOut,delimiter=',')
    
    #%% 
    
    
    #%%
    #PWidF , PWidAx = plt.subplots(CBins.max(),1, sharex='all', figsize=(16,8))
    #PWidF.suptitle('Peak Width for each set of lines')
    #for nn in range(CBins.max()):
    #    PWidAx[nn].imshow(np.abs(FPWidth[CBins==nn+1,:]), vmin=0, vmax=30, cmap='gray')
    #   
    #PWidF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "WidIm.png"), dpi=600)
    
    #%%
    #PWidBF , PWidBAx = plt.subplots(CBins.max(),1, sharex='all', figsize=(16,8))
    #PWidBF.suptitle('Mean Peak Width for each set of lines')
    #for nn in range(CBins.max()):
    #    PWidBAx[nn].bar(range((CBins==nn+1).sum()),np.nanmean(np.abs(FPWidth[CBins==nn+1,:]),axis=1))
    #    PWidBAx[nn].set_ylim([10,30])
    #PWidF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "WidBar.png"), dpi=600)
    
    ##%%
    #DispF , DispAx = plt.subplots(CBins.max(),BinCount.max(), sharex='col',figsize=(12,12))
    #DispF.suptitle('Displacement for each line (Histogram)')
    #for nn in range(CBins.max()):
    #    for ll in range(BinCount.max()):
    #        try:
    #            DispAx[nn,ll].hist(FDisp[CBins==nn+1,:][ll,:][~np.isnan(FDisp[CBins==nn+1,:][ll,:])],bins = np.arange(-10,10))
    #            DispAx[nn,ll].set_title('StDev '+str(np.round(np.std(FDisp[CBins==nn+1,:][ll,:][~np.isnan(FDisp[CBins==nn+1,:][ll,:])]),2)))
    #        except:
    #            pass
    #        DispAx[nn,ll].get_yaxis().set_visible(False)
    #DispF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "Displacement.png"), dpi=600)
    #
    ###%%
    #CrossCorF , CrossCorAx = plt.subplots(CBins.max(),BinCount.max()-1, figsize=(16,16))
    #CrossCorF.suptitle('Line/Line Correlation for each set of lines, X axis is line, Y axis is next line')
    #for nn in range(CBins.max()):
    #    for ll in range(BinCount.max()):
    #        try:
    #            CrossCorAx[nn,ll].scatter(FDisp[CBins==nn+1,:][ll,:],FDisp[CBins==nn+1,:][ll+1,:])
    #            CrossCorAx[nn,ll].set_xlim([-7, 7])
    #            CrossCorAx[nn,ll].set_ylim([-7, 7])
    #        except:
    #            pass
    #CrossCorF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "CrossCor.png"), dpi=600)
