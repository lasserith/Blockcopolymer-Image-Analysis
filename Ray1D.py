# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:20:58 2017

AFM 1D Data creation script as used for paper DOI:XXX
"""
#%% Imports
import tkinter as tk
from tkinter import filedialog, ttk
import os
import sys
import lmfit
from PIL import Image
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import matplotlib.pyplot as plt
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
            Opt.FInfo=str(MetaF['Image Tag 0x8546'].values)
            Opt.NmPP=float(Opt.FInfo[17:30])*10**9
            Opt.Machine="Merlin"
        except:
            pass


#    if Opt.NmPP!=0:
#        print("Instrument was autodetected as %s, NmPP is %f \n" % (Opt.Machine ,Opt.NmPP) )
#    else:
#        print("Instrument was not detected, and NmPP was not set. Please set NmPP and rerun")
    return(imarray)

def PeakPara(LineIn, NmPP, CValley, SetFWidth):
    Length = LineIn.size
    gmodel = lmfit.Model(gaussian)
    Inits = gmodel.make_params()
    FPeak = np.zeros(CValley.shape)
    FPWidth = np.zeros(CValley.shape)
    GradCurve = np.diff(np.sign((np.gradient(LineIn)))) 
    # this just says where sign of 1D changes
    
    for pp in range(len(CValley)): #loop through peak positions (guesstimates)
        PCur = int(CValley[pp]) # this is our current *peak* guesstimate
        # first goal : Refine the peak guesstimate for this line
        FitWidth = SetFWidth # look at local area only
        PLow = int(np.maximum((PCur-FitWidth),0))
        PHigh = int(np.min((PCur+FitWidth+1,Length-1)))

        try:PCur = int(np.arange(PLow,PHigh)[np.argmax(GradCurve[PLow:PHigh])+1])
        except:pass
        # set peak as the minimum (max 2nd div) 
        # the +1 is to fix the derivative offset from data
        #now expand our range to the next domains
        FitWidth = SetFWidth * 2
        PLow = int(np.maximum((PCur-FitWidth),0))
        PHigh = int(np.min((PCur+FitWidth+1,Length-1)))
        
        # now fix the point to the Right of the valley
        # Remember we are looking for mim of second derivative +1 is to fix diff offset
        PHigh = int(PCur +  np.argmin(GradCurve[PCur:PHigh]) +1) 
        # now fix the point to the left of the valley. 
        #Do the flip so the first point we find is closest to peak
        # do -1 cus we're moving otherway
        PLow = int(PCur - np.argmin(np.flip(GradCurve[PLow:PCur],0)) -1)
        # PLow is now the max peak to the left of the current valley 
        # PHigh is now the max peak to the right of the current valley

        LocalCurve = abs((LineIn[PLow:PHigh]-max(LineIn[PLow:PHigh])))
        
        # this just flips the curve to be right side up with a minimum of 0
        # so we can map it onto the typical gaussian        
        Inits['amp']=lmfit.Parameter(name='amp', value= max(LocalCurve))
        Inits['wid']=lmfit.Parameter(name='wid', value= FitWidth)
        Inits['cen']=lmfit.Parameter(name='cen', value= PCur, min=PCur-7, max=PCur+7)
    
        try:
            Res = gmodel.fit(LocalCurve, Inits, x=np.arange(PLow,PHigh))
            FPeak[pp] = Res.best_values['cen']*NmPP
            FPWidth[pp] = abs(np.copy(Res.best_values['wid']*2.35482*NmPP)) # FWHM in NM
            if (abs(Res.best_values['cen'] - PCur) > 5) or (Res.best_values['wid'] > 50) or (Res.best_values['cen']==PCur):
                FPWidth[pp] = np.nan
                FPeak[pp] = np.nan
        except:
            FPWidth[pp] = np.nan
            FPeak[pp] = np.nan
    return( FPeak, FPWidth)
    
def LERPara(LineIn, NmPP, CValley,LEdgeInd, REdgeInd,  SetFWidth):
    
    Length = LineIn.size
    gmodel = lmfit.Model(gaussian)
    Inits = gmodel.make_params()
    FPeak = np.zeros(CValley.shape)
    FEdgeL = np.zeros(CValley.shape)
    FEdgeR = np.zeros(CValley.shape)
    FPWidth = np.zeros(CValley.shape)
    FirstDer = np.gradient(LineIn)
    GradCurve = np.diff(np.sign((FirstDer))) 
    # this just says where sign of 1D changes
    
    for pp in range(len(CValley)): #loop through peak positions (guesstimates)
        PCur = int(CValley[pp]) # this is our current *peak* guesstimate
        # first goal : Refine the peak guesstimate for this line
        FitWidth = 2 # look at local area only
        
        PLow = int(np.maximum((PCur-FitWidth),0)) #PLow is a fixed distance
        PHigh = int(np.min((PCur+FitWidth+1,Length-1))) #PHigh is fixed +1 still here to fix indexing
        try:PCur = int(np.arange(PLow,PHigh)[np.argmin(LineIn[PLow:PHigh])])
        except:pass
        

        # set peak as the minimum (max 2nd div) 
        # the +1 is to fix the derivative offset from data

        
        #%% Fit Left Edge
        if (pp == LEdgeInd).max() :  # are we on the edge of a trench? if so must fit carefully
            FitWidth = SetFWidth
            PLow = int(np.maximum((PCur-FitWidth),0)) #PLow is a fixed distance
            PHigh = int(np.min((PCur+FitWidth+1+2,Length-1))) # +1 to account for index. +2 for wiggle room
            PHigh = int(PCur +  np.argmax(LineIn[PCur:PHigh]))  # find max in that wiggle room
            LocalCurve = abs((LineIn[PLow:PHigh]-max(LineIn[PLow:PHigh]))) 
            # Set our initial guesses
            
            Inits['amp']=lmfit.Parameter(name='amp', value= max(LocalCurve))
            Inits['wid']=lmfit.Parameter(name='wid', value= LocalCurve.size)
            Inits['cen']=lmfit.Parameter(name='cen', value= PCur, min=PLow, max=PHigh)
            try:
                Res = gmodel.fit(LocalCurve, Inits, x=np.arange(PLow,PHigh))
                FPeak[pp] = Res.best_values['cen']*NmPP
                if (abs(Res.best_values['cen'] <= PLow)) or (abs(Res.best_values['cen'] >= PHigh)):
                    FPeak[pp] = np.nan
            except:
                FPeak[pp] = np.nan
            FEdgeL[pp] = np.nan
        else:

            #now expand our range to the next domains
            FitWidth = SetFWidth * 2
            PLow = int(np.maximum((PCur-FitWidth),0))
            # now fix the point to the left of the valley. 
            #Do the flip so the first point we find is closest to peak
            # do -1 cus we're moving otherway
            PLow = int(PCur - np.argmin(np.flip(GradCurve[PLow:PCur],0)) -1)
            # PLow is now the max peak to the left of the current valley 
            
            
            # now we want to fit edges. so max or mins of derivative. 
            #flip
            LocalCurve = abs((FirstDer[PLow:PCur]-max(FirstDer[PLow:PCur])))
            
            # this just flips the curve to be right side up with a minimum of 0
            # so we can map it onto the typical gaussian        
            Inits['amp']=lmfit.Parameter(name='amp', value= max(LocalCurve))
            Inits['wid']=lmfit.Parameter(name='wid', value= LocalCurve.size)
            Inits['cen']=lmfit.Parameter(name='cen', value= (PLow+PCur)/2, min=PLow, max=PCur)
        
            try:
                Res = gmodel.fit(LocalCurve, Inits, x=np.arange(PLow,PCur))
                FEdgeL[pp] = Res.best_values['cen']*NmPP
                if (abs(Res.best_values['cen'] <= PLow)) or (abs(Res.best_values['cen'] >= PCur)):
                    FEdgeL[pp] = np.nan
            except:
                FEdgeL[pp] = np.nan
            
        #%% Fit Right Edge
        if (pp == REdgeInd).max(): # are we on the edge of a trench? if so must fit carefully
            FitWidth = SetFWidth
            PHigh = int(np.min((PCur+FitWidth+1,Length-1))) #PHigh is fixed +1 still here to fix indexing
            PLow = int(np.maximum((PCur-FitWidth-2),0)) #PLow now needs to be tweaked, minus two for wiggle
            PLow = int(PCur -  np.argmax(np.flip(LineIn[PLow:PCur],0)) -1)  # find max in that wiggle room
            LocalCurve = abs((LineIn[PLow:PHigh]-max(LineIn[PLow:PHigh]))) 
            # Set our initial guesses
            
            Inits['amp']=lmfit.Parameter(name='amp', value= max(LocalCurve))
            Inits['wid']=lmfit.Parameter(name='wid', value= LocalCurve.size)
            Inits['cen']=lmfit.Parameter(name='cen', value= PCur, min=PLow, max=PHigh)
            try:
                Res = gmodel.fit(LocalCurve, Inits, x=np.arange(PLow,PHigh))
                FPeak[pp] = Res.best_values['cen']*NmPP
                if (abs(Res.best_values['cen'] <= PLow)) or (abs(Res.best_values['cen'] >= PHigh)):
                    FPeak[pp] = np.nan
            except:
                FPeak[pp] = np.nan
            FEdgeR[pp] = np.nan
        else:
            #now expand our range to the next domains
            FitWidth = SetFWidth * 2
            PHigh = int(np.min((PCur+FitWidth+1,Length-1)))  
            # now fix the point to the Right of the valley
            # Remember we are looking for mim of second derivative +1 is to fix diff offset
            PHigh = int(PCur +  np.argmin(GradCurve[PCur:PHigh]) +1) 
            LocalCurve = abs((FirstDer[PCur:PHigh]-min(FirstDer[PCur:PHigh])))
            
            # dont flip       
            Inits['amp']=lmfit.Parameter(name='amp', value= max(LocalCurve))
            Inits['wid']=lmfit.Parameter(name='wid', value= LocalCurve.size)
            Inits['cen']=lmfit.Parameter(name='cen', value= (PCur+PHigh)/2, min=PCur, max=PHigh)
        
            try:
                Res = gmodel.fit(LocalCurve, Inits, x=np.arange(PCur,PHigh))
                FEdgeR[pp] = Res.best_values['cen']*NmPP
                if (abs(Res.best_values['cen'] <= PCur)) or (abs(Res.best_values['cen'] >= PHigh)):
                    FEdgeR[pp] = np.nan
            except:
                FEdgeR[pp] = np.nan
        #%% Calc Width and Peak
        if (pp == LEdgeInd).max(): # if we're on left edge
            FPWidth[pp] = (FEdgeR[pp] - FPeak[pp])*2
        elif (pp == REdgeInd).max(): # if we're on right edge
            FPWidth[pp] = (FPeak[pp] - FEdgeL[pp])*2
        else: # calc as usual
            FPWidth[pp]= FEdgeR[pp] - FEdgeL[pp]
            FPeak[pp] = 0.5*(FEdgeL[pp]+FEdgeR[pp])
        
        
    FPWidth[FPWidth < 0] = np.nan # if we have bad data lets throw it out
            
    return( FPeak, FPWidth, FEdgeL, FEdgeR)
            

#%%
def onclick(event):
    global XPeak
    XPeak = np.append(XPeak, event.xdata)
    FPlot1.axvline(x=event.xdata,linewidth=2, color='r')
    plt.draw()

def onkey(event):
    global Block
    global XPeak
    sys.stdout.flush()
    if event.key == "escape":   
        Block = 0
        FPlot.canvas.mpl_disconnect(cid) # disconnect both click    
        FPlot.canvas.mpl_disconnect(kid) # and keypress events
    if event.key == "c":
        XPeak = np.array([])
        FPlot1.cla()
        FPlot1.plot(Xplot,RawComp,'b',Xplot,SavFil,'k')
        FPlot1.legend(['Raw','Filt.'],loc="upper right")
        plt.draw()
    
#%%
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

#%%
def Hooks(x, K, Offset):
    return 0.5*x**2*K+Offset
#%%
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
    #ShowImage = 0 # Show images?
    plt.ioff()# Dont show plots as made, all plots are saved so just use that
    Opt.Boltz = 8.617e-5 # boltzmann, using eV/K here
    
    
    Opt.NmPP = 0 # Nanometers per pixel scaling (will autodetect)
    Opt.l0 = 50 # nanometer l0
    Opt.DomPerTrench = 7 # how many domains are there in a trench?
    Opt.Thread = -1 # how many thread to use. -1 = all possible minus 1, -2 = all possible -2 etc
    #%% AFM Settings
    Opt.AFMLayer = "Phase" #Matched Phase ZSensor
    Opt.AFMLevel = 3  # 0 = none 1 = Median 2= Median of Dif 3 = polyfit
    Opt.AFMPDeg = 1 # degree of polynomial.
    
    # Autocorrelation max shift
    Opt.ACCutoff = 50
    Opt.ACSize = 400
    
    Opt.AngMP = 5 # Do a midpoint average based on this many points
    # EG AngMP = 5 then 1 2 3 4 5, 3 will be calc off angle 1 - 5
    Opt.Machine = "Unknown"
    
    #%% Plot Options
    Opt.TDriftP = 0
    Opt.OptPlots = 0 # this stops all the non paper plots from being made
#    Opt.PColor = "#FFD966"
#    Opt.EColor = "#203864"
#    Opt.WColor = "#C5E0B4"
    
    
    Opt.PColor = "#3c8434"
    Opt.EColor = "#c00000"
    Opt.WColor = "#95a7f5"
    #%% Select Files
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
    XPeak = np.array([])
    Block = 1
    #%%
    print(FNFull[0])
    Opt.TempC = float(input('Temperature in Celsius : '))
    Opt.Temp = Opt.TempC+273.15
    
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
       
        RawComp = RawIn[:,:,ImNum].sum(axis=1) # sum along the channels to get a good idea where peaks are
        RawTop = RawIn[:,:5,ImNum].sum(axis=1)
        RawBot = RawIn[:,-5:,ImNum].sum(axis=1)
        
        #%%
        SavFil = scipy.signal.savgol_filter(RawComp,5,2,axis = 0)
        TopFil = scipy.signal.savgol_filter(RawTop,5,2,axis = 0)
        BotFil = scipy.signal.savgol_filter(RawBot,5,2,axis = 0)
        D1SavFil = scipy.signal.savgol_filter(RawComp,5,2,deriv = 1,axis = 0)
        D2SavFil = scipy.signal.savgol_filter(RawComp,5,2, deriv = 2,axis = 0)
        FPlot = plt.figure(figsize=(8,3))
        FPlot.clf()
        FPlot.suptitle('Click the first domain (valley) for each trench. Then hit ESCAPE \n'
                       +'Hit c to Cancel and restart if you did an oopsy')
        Xplot = range(0, Shap0)
        
        FPlot1 = FPlot.add_subplot(211)
        FPlot1.plot(Xplot,RawComp,'b',Xplot,SavFil,'k')
        FPlot1.legend(['Raw','Filt.'],loc="upper right")
        
        FPlot2 = FPlot.add_subplot(212)
        FPlot2.plot(Xplot,D1SavFil,'r',Xplot,D2SavFil,'k')
        FPlot2.legend(['1st Deriv','2nd Deriv'],loc="upper right")
        
        FPlot.show()
        if Block == 1:
            cid = FPlot.canvas.mpl_connect('button_press_event', onclick)
            kid = FPlot.canvas.mpl_connect('key_press_event', onkey)
            while Block == 1:
                plt.pause(1)
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
        
        #%% use manual sep clicked earlier to do the clean up
        CPatSep = np.zeros_like(XPeak)
        for xx in range(XPeak.size):
            CPatSep[xx] = Valley[np.argmin(abs(Valley-XPeak[xx]))]
        CPatSep = np.unique(CPatSep) # just in case we accidentally double clicked a peak
        CValley = np.zeros(int(CPatSep.size*Opt.DomPerTrench))
        for xx in range(CPatSep.size):
            CPatSepInd = np.nonzero(Valley == CPatSep[xx])[0][0]
            CValley[Opt.DomPerTrench*xx:Opt.DomPerTrench*(xx+1)] = Valley[CPatSepInd:(CPatSepInd+Opt.DomPerTrench)]
        CPatSep -= 5 #scoot our separators off the peak
        CPatSep = np.append(CPatSep, len(SavFil)-1)
        
        #%% Make some indices for later use
        CBins = np.digitize(CValley,CPatSep)
        BinCount = np.histogram(CValley,CPatSep)[0]
        MidInd = np.zeros(CPatSep.size-1,dtype='uint')
        LEdgeInd = np.zeros((CPatSep.size-1),dtype='uint')
        REdgeInd = np.zeros((CPatSep.size-1),dtype='uint')
        EdgeInd = np.zeros((CPatSep.size-1)*2,dtype='uint')
        for bb in range(CPatSep.size-1):
            MidInd[bb] = bb*Opt.DomPerTrench+((Opt.DomPerTrench-1)/2)
            EdgeInd[bb*2] = bb*Opt.DomPerTrench
            LEdgeInd[bb] = bb*Opt.DomPerTrench
            EdgeInd[bb*2+1] = (bb+1)*Opt.DomPerTrench-1
            REdgeInd[bb] = (bb+1)*Opt.DomPerTrench-1
        #%%
        FitWidth = int(Opt.l0/Opt.NmPP*.5)

        #% Parallel cus gotta go fast
        print('\n Image '+Opt.FName+'\n')
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

        POut = Parallel(n_jobs=Opt.Thread,verbose=5)(delayed(LERPara)(RawIn[:,tt,ImNum], Opt.NmPP, CValley, LEdgeInd, REdgeInd, FitWidth) # backend='threading' removed
            for tt in range(RawIn.shape[1]))
        #%%
        FPTuple, FPWTuple, FELTup, FERTup = zip(*POut)
        FPeak = np.array(FPTuple).transpose()
        FPWidth = np.array(FPWTuple).transpose()
        FEL = np.array(FELTup).transpose()
        FER = np.array(FERTup).transpose()
        print('Done')
        #Everything past here is already in Nanometers. PEAK FITTING OUTPUTS IT
        #%% Show Odd EVEN
#       Already verified no odd even effects so skip this
#        OddEveF,OddEveAx = plt.subplots()
#        
#        OddEveAx.plot(Xplot[0::2],FEL[4,0::2],'b.',label='Left Even')
#        OddEveAx.plot(Xplot[0::2],FPeak[4,0::2],'k.',label='Peak Even')
#        OddEveAx.plot(Xplot[0::2],FER[4,0::2],'b.',label='Right Even')
#        OddEveAx.plot(Xplot[1::2],FEL[4,1::2],'b.',label='Left Even')
#        OddEveAx.plot(Xplot[1::2],FPeak[4,1::2],'k.',label='Peak Even')
#        OddEveAx.plot(Xplot[1::2],FER[4,1::2],'b.',label='Right Even')
#        #OddEveF.legend()
#        OddEveAx.set_axis_off()
#        OddEveF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "OddEven.png"), dpi=600)
#        
#            #%%
#        OddEveF.clf()
#        plt.close(OddEveF)
            



        #%% Save filtered data
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "FitPeak.csv"),FPeak,delimiter=',')
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "FitWidth.csv"),FPWidth,delimiter=',')  
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "FitEdgeL.csv"),FEL,delimiter=',')
        np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "FitEdgeR.csv"),FER,delimiter=',')
        
        #%% Calc Displacement for peaks to do drift correct
        FDisp = ((FPeak.transpose() - np.nanmean(FPeak,axis=1)).transpose())

        #%% Do thermal drift correction
        XTD = np.arange(FDisp.shape[1])
        YTD = np.nanmean(FDisp,axis=0)
        TDFit = np.polyfit(XTD,YTD,1)
        TDPlot = np.polyval(TDFit,XTD)
        if Opt.TDriftP == 1:
            TDF, TDAx = plt.subplots()
            TDF.suptitle('Thermal Drift y=%f*x+%f' % (TDFit[0],TDFit[1]))
            TDAx.plot(XTD,YTD,XTD,TDPlot)
            TDF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "ThermalDrift.png"), dpi=600)
            TDF.clf()
            plt.close(TDF)
        #%% now correct the data for drift
        FDispCorrect = (FDisp - TDPlot)
        FELDrift = (FEL - TDPlot)
        FERDrift = (FER - TDPlot)
        
        #%% Calc Displacement for edges/width

        FELCorrect = ((FEL.transpose() - np.nanmean(FEL,axis=1)).transpose())
        FERCorrect = ((FER.transpose() - np.nanmean(FER,axis=1)).transpose())
        FPWidthRes = ((FPWidth.transpose() - np.nanmean(FPWidth,axis=1)).transpose())
        
        #%% put edges together
        
        FECorrect = np.zeros((FER.shape[0]*2,FER.shape[1]))
        FECorrect[0::2,:] = FELCorrect
        FECorrect[1::2,:] = FERCorrect
        
        #%% move on
        
        
        
        PanDisp = pd.DataFrame(data=FDispCorrect.transpose())
        PanWidth = pd.DataFrame(data=FPWidthRes.transpose())
        PanE = pd.DataFrame(data=FECorrect.transpose())

        #%% make demo figs
        if ImNum == 0:
            #%% show an example fit
            FitXLow = FPeak[0,256]-40
            FitXHigh = FPeak[6,256]+40
            FitXPlot = np.linspace(FitXLow,FitXHigh,200)
            FitF,FitAx = plt.subplots()
            FitAx.plot(np.arange(512)*Opt.NmPP,RawIn[:,256,0],'k')
            OffSet1 = min(RawIn[:,256,0])*1.5
            FitAx.set_xlim(FitXLow,FitXHigh)
            FitAx.plot(Xplot*Opt.NmPP,np.gradient(RawIn[:,256,0])+OffSet1,'k')
            OffSet2 = min(np.gradient(RawIn[:,256,0])+OffSet1)*1.5
            for dd in np.arange(Opt.DomPerTrench):
                FitAx.axvline(x=FPeak[dd,256],linestyle = '--',color=Opt.PColor) # plot center
                if dd != 0: 
                    FitAx.axvspan(FEL[dd,256],FPeak[dd,256],color=Opt.WColor ,alpha=0.2)
                    FitAx.axvline(x=FEL[dd,256],linestyle = '--',color= Opt.EColor) # plot Left Edge 
                if dd != Opt.DomPerTrench-1:
                    FitAx.axvspan(FPeak[dd,256],FER[dd,256],color=Opt.WColor ,alpha=0.2)
                    FitAx.axvline(x=FER[dd,256],linestyle = '--',color=Opt.EColor) # plot Right Edge 
            FitAx.get_yaxis().set_visible(False)
            for dd in np.arange(Opt.DomPerTrench):        
                if dd != 0: 
                    FitAx.plot(FitXPlot,gaussian(FitXPlot,-10,FEL[dd,256], FPWidth[dd,256]/4)+OffSet2,'k')
                if dd != Opt.DomPerTrench-1: 
                    FitAx.plot(FitXPlot,gaussian(FitXPlot,10,FER[dd,256], FPWidth[dd,256]/4)+OffSet2,'k')
            FitF.tight_layout()
            FitF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "ExampleFit.png"), dpi=600)
            #%%
            FitF.clf()
            plt.close(FitF)
            #%% show an example LER
            FitF,FitAx = plt.subplots(nrows=3,figsize=(3,6))
            gmodel = lmfit.Model(gaussian) # then the model
            Count , Bin = np.histogram(np.concatenate((FELCorrect[MidInd[0],:],FERCorrect[MidInd[0],:])),bins=200,range=(-5,5),density=True)
            BinX = BinX = Bin[:-1]+(Bin[1]-Bin[0])*0.5 # what is the center of each bin?
            Var = np.sqrt(np.nanvar(np.concatenate((FELCorrect[MidInd[0],:],FERCorrect[MidInd[0],:]))))  # calculate variance exactly
            Inits = gmodel.make_params( amp=Count.max(), cen=0, wid=Var) # make some initial guess
            Inits['wid'] = lmfit.Parameter(name='wid', value=Var, vary=False) # force it to use varianc
            Res = gmodel.fit(Count, Inits, x=BinX) # and fit
            
            FitAx[0].plot(BinX, Count, 'o', color = Opt.EColor, alpha = 0.5)
            FitAx[0].plot(BinX, Res.best_fit, 'black',label='Edge Fit')
            # Show example width
            Count , Bin = np.histogram(FPWidthRes,bins=200,range=(-5,5),density=True)
            BinX = BinX = Bin[:-1]+(Bin[1]-Bin[0])*0.5 # what is the center of each bin?
            Var = np.sqrt(np.nanvar(FPWidthRes))  # calculate variance exactly
            Inits = gmodel.make_params( amp=Count.max(), cen=0, wid=Var) # make some initial guess
            Inits['wid'] = lmfit.Parameter(name='wid', value=Var, vary=False) # force it to use varianc
            Res = gmodel.fit(Count, Inits, x=BinX) # and fit
            FitAx[1].plot(BinX, Count, 'o',color = Opt.WColor, alpha = 0.5)
            FitAx[1].plot(BinX, Res.best_fit, 'black',label='Width Fit')
            # Show example position
            Count , Bin = np.histogram(FDispCorrect,bins=200,range=(-5,5),density=True)
            BinX = BinX = Bin[:-1]+(Bin[1]-Bin[0])*0.5 # what is the center of each bin?
            Var = np.sqrt(np.nanvar(FDispCorrect))  # calculate variance exactly
            Inits = gmodel.make_params( amp=Count.max(), cen=0, wid=Var) # make some initial guess
            Inits['wid'] = lmfit.Parameter(name='wid', value=Var, vary=False) # force it to use varianc
            Res = gmodel.fit(Count, Inits, x=BinX) # and fit
            FitAx[2].plot(BinX, Count, 'o',color = Opt.PColor, alpha = 0.5)
            FitAx[2].plot(BinX, Res.best_fit, 'black',label='Position Fit')
            
            FitF.tight_layout()
            FitF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "ExampleGauss.png"), dpi=600)
            #%%
            FitF.clf()
            plt.close(FitF)
            
        #StackDisp 
        #StackWidth
        #%% Cross Corref 
        StackDisp = FDispCorrect.transpose()[:,0:Opt.DomPerTrench]
        StackWidth = FPWidthRes.transpose()[:,0:Opt.DomPerTrench]
        StackEdge = FECorrect.transpose()[:,0:Opt.DomPerTrench*2]
        for xx in np.arange(1,CPatSep.size-1):
            StackDisp=np.concatenate( (StackDisp,FDispCorrect.transpose()[:,xx*Opt.DomPerTrench:(xx+1)*Opt.DomPerTrench]) )
            StackWidth=np.concatenate((StackWidth,FPWidthRes.transpose()[:,xx*Opt.DomPerTrench:(xx+1)*Opt.DomPerTrench]))
            StackEdge=np.concatenate((StackEdge,FECorrect.transpose()[:,2*xx*Opt.DomPerTrench:2*(xx+1)*Opt.DomPerTrench]))
            

        PDStackD = pd.DataFrame(data=StackDisp)
        PDStackW = pd.DataFrame(data=StackWidth)
        PDStackE = pd.DataFrame(data=StackEdge)

        
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

        
        
        for nn in range(Opt.DomPerTrench-1):
            StackD1O = np.append( StackD1O, np.array((PDStackD.values[:,nn],PDStackD.values[:,nn+1])).transpose(),axis = 0 )
            StackW1O = np.append( StackW1O, np.array((PDStackW.values[:,nn],PDStackW.values[:,nn+1])).transpose(),axis = 0 )
            StackE1O = np.append( StackE1O, np.array((PDStackE.values[:,nn],PDStackE.values[:,nn+1])).transpose(),axis = 0 )
            StackE1O = np.append( StackE1O, np.array((PDStackE.values[:,2*nn],PDStackE.values[:,2*nn+1])).transpose(),axis = 0 )
            if nn < Opt.DomPerTrench-2:
                StackD2O = np.append( StackD2O, np.array((PDStackD.values[:,nn],PDStackD.values[:,nn+2])).transpose(),axis = 0 )
                StackW2O = np.append( StackW2O, np.array((PDStackW.values[:,nn],PDStackW.values[:,nn+2])).transpose(),axis = 0 )
                StackE2O = np.append( StackE2O, np.array((PDStackE.values[:,nn],PDStackE.values[:,nn+2])).transpose(),axis = 0 )
                StackE2O = np.append( StackE2O, np.array((PDStackE.values[:,2*nn],PDStackE.values[:,2*nn+2])).transpose(),axis = 0 )
            if nn < Opt.DomPerTrench-3:
                StackD3O = np.append( StackD3O, np.array((PDStackD.values[:,nn],PDStackD.values[:,nn+3])).transpose(),axis = 0 )
                StackW3O = np.append( StackW3O, np.array((PDStackW.values[:,nn],PDStackW.values[:,nn+3])).transpose(),axis = 0 )
                StackE3O = np.append( StackE3O, np.array((PDStackE.values[:,nn],PDStackE.values[:,nn+3])).transpose(),axis = 0 )
                StackE3O = np.append( StackE3O, np.array((PDStackE.values[:,2*nn],PDStackE.values[:,2*nn+3])).transpose(),axis = 0 )
            if nn < Opt.DomPerTrench-4:
                StackE4O = np.append( StackE4O, np.array((PDStackE.values[:,nn],PDStackE.values[:,nn+4])).transpose(),axis = 0 )
                StackE4O = np.append( StackE4O, np.array((PDStackE.values[:,2*nn],PDStackE.values[:,2*nn+4])).transpose(),axis = 0 )
            if nn< Opt.DomPerTrench-5:
                StackE5O = np.append( StackE5O, np.array((PDStackE.values[:,nn],PDStackE.values[:,nn+5])).transpose(),axis = 0 )
                StackE5O = np.append( StackE5O, np.array((PDStackE.values[:,2*nn],PDStackE.values[:,2*nn+5])).transpose(),axis = 0 )

        
        CCDisp = PDStackD.corr() # calcualte cross correlations
        CCWidth = PDStackW.corr() # calcualte cross correlations
        CCEL = PDStackE.corr()
        




        #%%
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
        
        LPRCrossF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "LPR_Cross.png"), dpi=600)
        #%%
        LPRCrossF.clf()
        plt.close(LPRCrossF)

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
        
        LWRCrossF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "LWR_Cross.png"), dpi=600)
        #%%
        LWRCrossF.clf()
        plt.close(LWRCrossF)
        
                #%%
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
        # now plot lines to guide eyes
        LERCrossAx[0].plot(np.arange(0,14),13-np.arange(0,14),color='#00FF00',linewidth = 2)
        LERCrossAx[0].plot(np.arange(0,14),12-np.arange(0,14),color='#FF00FF',linewidth = 2)
        LERCrossAx[0].plot(np.arange(0,14),11-np.arange(0,14),color='k',linewidth = 2)
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
        LERCrossF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "LER_Cross.png"), dpi=600)
        #%%
        LERCrossF.clf()
        plt.close(LERCrossF)

        

        #%% Power Spectral Density
        
        PSDPeak = np.abs(np.fft.rfft(PanDisp.interpolate(limit_direction='both').values.transpose()))**2
        PSDWidth = np.abs(np.fft.rfft(PanWidth.interpolate(limit_direction='both').values.transpose()))**2
        PSDEdge = np.abs(np.fft.rfft(PanE.interpolate(limit_direction='both').values.transpose()))**2
        PSDFreq = np.fft.rfftfreq(FPeak.shape[1],0.05) # Sampling rate is 20 hz so sample time = .05?
        PSDCk = (4*PSDPeak-PSDWidth)/(4*PSDPeak+PSDWidth)

        #%%
        PSDF , PSDAx = plt.subplots(ncols=3,nrows=3, sharex = True,sharey=True, figsize=(9,9))
        PSDF.suptitle('Power Spectral Density')
        RollAv = int(5)
        
        #% Peak
        PSDMean = np.nanmean(PSDPeak, axis=0)[1:]
        
        if ImNum == 0:
            PSDLPROut = np.vstack( (PSDFreq[1:], PSDMean))
            PSDLPRMid = np.vstack( (PSDFreq[1:], np.nanmean(PSDPeak[MidInd,:], axis=0)[1:]))
            PSDLPREdge = np.vstack( (PSDFreq[1:], np.nanmean(PSDPeak[EdgeInd,:], axis=0)[1:]))
        else:
            PSDLPROut = np.vstack((PSDLPROut, PSDMean))
            PSDLPRMid = np.vstack((PSDLPROut, np.nanmean(PSDPeak[MidInd,:], axis=0)[1:]))
            PSDLPREdge = np.vstack((PSDLPROut, np.nanmean(PSDPeak[EdgeInd,:], axis=0)[1:]))
            
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[0,0].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.PColor,alpha=0.5)
        PSDAx[0,0].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
        PSDAx[0,0].set_title('LPR')
        
        PSDMean = np.nanmean(PSDPeak[MidInd,:], axis=0)[1:]
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[0,1].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.PColor,alpha=0.5)
        PSDAx[0,1].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
        PSDAx[0,1].set_title('Mid')
        
        PSDMean = np.nanmean(PSDPeak[EdgeInd,:], axis=0)[1:]
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[0,2].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.PColor,alpha=0.5)
        PSDAx[0,2].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
        PSDAx[0,2].set_title('Edge')
        
        #% Width
        PSDMean = np.nanmean(PSDWidth, axis=0)[1:]
        
        if ImNum == 0:
            PSDLWROut = np.vstack( (PSDFreq[1:], PSDMean))
            PSDLWRMid = np.vstack( (PSDFreq[1:], np.nanmean(PSDWidth[MidInd,:], axis=0)[1:]))
            PSDLWREdge = np.vstack( (PSDFreq[1:], np.nanmean(PSDWidth[EdgeInd,:], axis=0)[1:]))
        else:
            PSDLWROut = np.vstack((PSDLWROut, PSDMean))
            PSDLWRMid = np.vstack((PSDLWROut, np.nanmean(PSDWidth[MidInd,:], axis=0)[1:]))
            PSDLWREdge = np.vstack((PSDLWROut, np.nanmean(PSDWidth[EdgeInd,:], axis=0)[1:]))
        
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[1,0].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.WColor,alpha=0.5)
        PSDAx[1,0].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
        
        PSDMean = np.nanmean(PSDWidth[MidInd,:], axis=0)[1:]
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[1,1].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.WColor,alpha=0.5)
        PSDAx[1,1].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')

        
        PSDMean = np.nanmean(PSDWidth[EdgeInd,:], axis=0)[1:]
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[1,2].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.WColor,alpha=0.5)
        PSDAx[1,2].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
        PSDAx[1,0].set_title('LWR')
        
        #% Edge
        PSDMean = np.nanmean(PSDEdge, axis=0)[1:]
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        
        if ImNum == 0:
            PSDLEROut = np.vstack( (PSDFreq[1:], PSDMean))
            PSDLERMid = np.vstack( (PSDFreq[1:], np.nanmean(PSDEdge[MidInd,:], axis=0)[1:]))
            PSDLEREdge = np.vstack( (PSDFreq[1:], np.nanmean(PSDEdge[EdgeInd,:], axis=0)[1:]))
        else:
            PSDLEROut = np.vstack((PSDLEROut, PSDMean))
            PSDLERMid = np.vstack((PSDLEROut, np.nanmean(PSDEdge[MidInd,:], axis=0)[1:]))
            PSDLEREdge = np.vstack((PSDLEROut, np.nanmean(PSDEdge[EdgeInd,:], axis=0)[1:]))
        
        PSDAx[2,0].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.EColor,alpha=0.5)
        PSDAx[2,0].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
        
        PSDMean = np.nanmean(PSDEdge[np.sort(np.append(MidInd*2, MidInd*2+1)),:], axis=0)[1:]
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[2,1].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.EColor,alpha=0.5)
        PSDAx[2,1].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')

        
        PSDMean = np.nanmean(PSDEdge[np.sort(np.append(EdgeInd*2, EdgeInd*2+1)),:], axis=0)[1:]
        PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
        PSDAx[2,2].loglog(PSDFreq[1:], PSDMean,'.',color=Opt.EColor,alpha=0.5)
        PSDAx[2,2].loglog(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
        PSDAx[2,0].set_title('LER')

        PSDF.tight_layout(rect=(0,0,1,0.95))
        PSDF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "PSD.png"), dpi=300)
        #%%
        PSDF.clf()
        plt.close(PSDF)
        #%%
        if Opt.OptPlots == 1:
            RollAv = int(5)
            PSDCkF, PSDCkAx = plt.subplots(ncols = 3,sharey = True)
            PSDMean = np.nanmean(PSDCk, axis=0)[1:]
            PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
            PSDCkAx[0].plot(PSDFreq[1:], PSDMean,'k.',alpha=0.2)
            PSDCkAx[0].plot(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
            #mid
            PSDMean = np.nanmean(PSDCk[MidInd,:], axis=0)[1:]
            PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
            PSDCkAx[1].plot(PSDFreq[1:], PSDMean,'k.',alpha=0.2)
            PSDCkAx[1].plot(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
            #Edge
            PSDMean = np.nanmean(PSDCk[EdgeInd,:], axis=0)[1:]
            PSDRMean = np.convolve(PSDMean, np.ones((RollAv,))/RollAv, mode='valid')
            PSDCkAx[2].plot(PSDFreq[1:], PSDMean,'k.',alpha=0.2)
            PSDCkAx[2].plot(PSDFreq[int(np.ceil(RollAv/2)):-int(np.floor(RollAv/2))], PSDRMean,'k')
    
            PSDCkAx[0].set_title('CK')
            PSDCkAx[1].set_title('Mid')
            PSDCkAx[2].set_title('Edge')
            PSDCkAx[1].set_xlabel('Frequency (hz)')
            PSDCkAx[0].set_ylim([0, 1])
            #%%
            PSDCkF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "PSDCk.png"), dpi=300)
            PSDCkF.clf()
            plt.close(PSDCkF)
            #%% Autocorrelation
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
            
            ACDispF , ACDispAx = plt.subplots(nrows=2,figsize=(15,3))
            ACDispF.suptitle('Peak displacement Autocorrelation')
            ACDispIm = ACDispAx[0].imshow(ACPeak.transpose(), cmap="seismic_r", vmin=-1, vmax=1)   
            ACDispF.colorbar(ACDispIm)
            ACDispAx[1].plot(ACPeak.values.mean(axis=1),label='Overall')
            ACDispAx[1].plot(ACPeak.values[:,EdgeInd].mean(axis=1),label='Edge')
            ACDispAx[1].plot(ACPeak.values[:,MidInd].mean(axis=1),label='Center')
            ACDispAx[1].legend()
            ACDispF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "DisplacementAC.png"), dpi=300)
            #%%
            ACDispF.clf()
            plt.close(ACDispF)        
            #%% MSD AC
            ACMSDF , ACMSDAx = plt.subplots(nrows=2,figsize=(15,3))
            ACMSDF.suptitle('MSD Autocorrelation')
            ACMSDIm = ACMSDAx[0].imshow(ACMSD.transpose(), cmap="plasma", vmin=0, vmax=15)
            ACMSDF.colorbar(ACMSDIm)
            ACMSDAx[1].plot(ACMSD.values.mean(axis=1),label='Overall')
            ACMSDAx[1].plot(ACMSD.values[:,EdgeInd].mean(axis=1),label='Edge')
            ACMSDAx[1].plot(ACMSD.values[:,MidInd].mean(axis=1),label='Center')
            ACMSDAx[1].legend()
            ACMSDF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "MSDAC.png"), dpi=300)
            #%%
            ACMSDF.clf()
            plt.close(ACMSDF)        
            
            #%%width
            ACWidthF , ACWidthAx = plt.subplots(nrows=2, figsize=(15,3))
            ACWidthF.suptitle('Peak FWHM Autocorrelation')
            ACWidthIm = ACWidthAx[0].imshow(ACWidth.transpose(), cmap="seismic_r", vmin=-1, vmax=1)
            ACWidthF.colorbar(ACWidthIm)
            ACWidthAx[1].plot(ACWidth.values.mean(axis=1),label='Overall')
            ACWidthAx[1].plot(ACWidth.values[:,EdgeInd].mean(axis=1),label='Edge')
            ACWidthAx[1].plot(ACWidth.values[:,MidInd].mean(axis=1),label='Center')
            ACWidthAx[1].legend()
            ACWidthF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "WidthAC.png"), dpi=300)
            #%%
            ACWidthF.clf()
            plt.close(ACWidthF)
            #%% MSD per line
            MSDF , MSDAx = plt.subplots(CBins.max(),BinCount.max(), figsize=(16,16))
            MSDF.suptitle('Mean Squared Displacement per line')
            for nn in range(CBins.max()):
                for ll in range(BinCount.max()):
                    ACMSD[nn*CBins.max()+ll].plot(ax=MSDAx[nn,ll])
                    MSDAx[nn,ll].set_ylim([0, 10])
                    MSDAx[nn,ll].set_xlim([0, 500])
            MSDF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "MSD.png"), dpi=300)
            MSDF.clf()
            plt.close(MSDF)
        
            #%% save the kappa/KT calculated from disp squared
            PSeudoEA = np.nanmean(( FDispCorrect**-2),axis=1)
            PSeudoEA = PSeudoEA.reshape((CPatSep.size-1,Opt.DomPerTrench))*Opt.Boltz*Opt.Temp
            if ImNum == 0:
                EAOut = PSeudoEA
            else:
                EAOut = np.concatenate((EAOut, PSeudoEA))
                #%% what about from the variance? Change if changing domain counts
            VarEA = np.zeros((1,Opt.DomPerTrench+1))
    
            EAF , EAAx = plt.subplots(2,4, figsize=(16,16))
            # set up the fig first  
            gmodel = lmfit.Model(gaussian) # then the model
            # ok now histogram remember StackDisp comes from FDispCorrect so is already NM
            Count , Bin = np.histogram(StackDisp[np.isfinite(StackDisp)],bins=200,range=(-20,20))
            BinX = BinX = Bin[:-1]+(Bin[1]-Bin[0])*0.5 # what is the center of each bin?
            VarRt = np.sqrt(np.nanvar(StackDisp))  # calculate variance exactly
            Inits = gmodel.make_params( amp=Count.max(), cen=0, wid=VarRt) # make some initial guess
            Inits['wid'] = lmfit.Parameter(name='wid', value=VarRt, vary=False) # force it to use variance
            
            Res = gmodel.fit(Count, Inits, x=BinX) # and fit
            VarEA[0,0] = (Res.best_values['wid'])**-2 * Opt.Boltz*Opt.Temp
    
            EAAx[0,0].plot(BinX, Count, 'bo')
            EAAx[0,0].plot(BinX, Res.best_fit, 'r-')
            EAAx[0,0].set_title('Overall fit Kappa = %f eV/nm^2'%(VarEA[0,0]))
            EAAx[0,0].set_xlim([-10, 10])
            for dd in range(Opt.DomPerTrench):
                Count , Bin = np.histogram(StackDisp[:,dd][np.isfinite(StackDisp[:,dd])],bins=200,range=(-20,20))
                BinX = BinX = Bin[:-1]+(Bin[1]-Bin[0])*0.5 # what is the center of each bin?
                VarRt = np.sqrt(np.nanvar(StackDisp[:,dd]))  # calculate variance exactly
                Inits = gmodel.make_params( amp=Count.max(), cen=0, wid=VarRt) # make some initial guess
                Inits['wid'] = lmfit.Parameter(name='wid', value=VarRt, vary=False) # force it to use variance
    
                Res = gmodel.fit(Count, Inits, x=BinX) # and fit
                VarEA[0,dd+1] = (Res.best_values['wid'])**-2 * Opt.Boltz * Opt.Temp
                # add to the plot
                rc = int((dd+1)/4) # the plot goes on the 0th row if dd <3
                cc = int((dd+1)%4) # plot goes on column related to modulo 4
                EAAx[rc,cc].plot(BinX, Count, 'bo')
                EAAx[rc,cc].plot(BinX, Res.best_fit, 'r-')
                EAAx[rc,cc].set_title('Domain %i fit Kappa = %f eV/nm^2'%(dd+1, VarEA[0,(dd+1)]))
                EAAx[rc,cc].set_xlim([-10, 10])
             #% plot  
            EAF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "VarianceFitting.png"), dpi=300)
            EAF.clf()
            plt.close(EAF)
            if ImNum == 0:
                VarEAOut = np.copy(VarEA)
            else:
                VarEAOut = np.concatenate((VarEAOut, VarEA))
            #%% Delta G plot
            DeltaGF,DeltaGAx = plt.subplots(1,1)
            DeltaGAx.set_ylabel('DeltaG (eV)')
            DeltaGAx.set_xlabel('Displacement (nm)')
            Count , Bin = np.histogram(StackDisp[np.isfinite(StackDisp)],bins=200,range=(-20,20))
            BinX = BinX = Bin[:-1]+(Bin[1]-Bin[0])*0.5 # what is the center of each bin?
            GCount = np.log(Count/Count.sum())*(-Opt.Boltz*Opt.Temp)
            
            Filter = np.isfinite(GCount)
            
            hmodel = lmfit.Model(Hooks)
            Inits = hmodel.make_params(K=0, Offset = 0.1)
            Inits['K'] = lmfit.Parameter(name='K', value=VarEA[0,0], vary=False) # force it to use EA calculated earlier
            Res = hmodel.fit(GCount[Filter], Inits, x=BinX[Filter]) # and fit
            
            GOff = Res.best_values['Offset']
            GTheory = VarEA[0,0]*0.5*BinX**2
            GTheory += GOff
            DeltaGAx.plot(BinX[Filter],GCount[Filter],'k.',BinX[Filter],GTheory[Filter],'r')
            DeltaGF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "HooksLaw"), dpi=300)
            DeltaGF.clf()
            plt.close(DeltaGF)
        
        #%% Sigma Stuff
        SigmaE = np.zeros((1,(Opt.DomPerTrench+1)))
        SigmaW = np.zeros((1,(Opt.DomPerTrench+1)))
        SigmaP = np.zeros((1,(Opt.DomPerTrench+1)))
        
        SigmaE[0,0] = np.abs(3*np.nanstd(FECorrect))
        SigmaW[0,0] = np.abs(3*np.nanstd(FPWidthRes))
        SigmaP[0,0] = np.abs(3*np.nanstd(FDispCorrect))
        for dd in range(Opt.DomPerTrench):
            SigmaE[0,dd+1] = np.abs(3*np.nanstd(StackEdge[:,dd*2:dd*2+2]))
            SigmaW[0,dd+1] = np.abs(3*np.nanstd(StackWidth[:,dd]))
            SigmaP[0,dd+1] = np.abs(3*np.nanstd(StackDisp[:,dd]))
        SigmaC = 1 - SigmaW**2/(2 * SigmaE**2)
        SigmaAll = np.concatenate((SigmaE,SigmaW,SigmaP, SigmaC),axis=1)
        
        if ImNum == 0:
            SigmaOut = np.copy(SigmaAll)
        else:
            SigmaOut = np.concatenate((SigmaOut, SigmaAll))
        #%% plot sigma
        SigmaF, SigmaAx = plt.subplots(nrows=2, sharex=True)
        SigmaLab = ('Overall','1','2','3','4','5','6','7')
        SigmaPos = np.arange(len(SigmaLab))
        SigmaAx[0].plot(SigmaPos,SigmaW[0,:],'k.',label='3 sigma W')
        SigmaAx[0].plot(SigmaPos,SigmaE[0,:],'b.',label='3 sigma E')
        SigmaAx[1].plot(SigmaPos,SigmaP[0,:],'r.',label='3 sigma P')
        SigmaAx[1].plot(SigmaPos,SigmaC[0,:],'k*',label='C')
        SigmaAx[1].axis([-0.5,7.5,0,5])
        SigmaF.legend()
        SigmaAx[1].set_xticks(SigmaPos)
        SigmaAx[1].set_xticklabels(SigmaLab)
        SigmaF.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "Sigma"), dpi=300)
        #%%
        SigmaF.clf()
        plt.close(SigmaF)
        #%%
        
    #outside of imnum loop
    try:np.savetxt(os.path.join(Opt.FPath,"output","SummedEA.csv"),EAOut,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","VarEA.csv"),VarEAOut,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","3Sigma.csv"),SigmaOut,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","LPR.csv"),PSDLPROut,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","LWR.csv"),PSDLWROut,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","LER.csv"),PSDLEROut,delimiter=',')
    except: pass

    try:np.savetxt(os.path.join(Opt.FPath,"output","LPR-MID.csv"),PSDLPRMid,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","LWR-MID.csv"),PSDLWRMid,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","LER-MID.csv"),PSDLERMid,delimiter=',')
    except: pass

    try:np.savetxt(os.path.join(Opt.FPath,"output","LPR-Edge.csv"),PSDLPREdge,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","LWR-Edge.csv"),PSDLWREdge,delimiter=',')
    except: pass
    try:np.savetxt(os.path.join(Opt.FPath,"output","LER-Edge.csv"),PSDLEREdge,delimiter=',')
    except: pass
    
    print('All Done')

