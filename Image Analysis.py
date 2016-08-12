# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:38:16 2016

@author: Moshe Dolejsi MosheDolejsi@uchicago.edu

Block Copolymer Analysis Package by Moshe Dolejsi
Done in Spyder/VStudio2015 Community with Anaconda.
Heavily Uses Numpy, Scipy, Mpltlib, and the rest of the usual

TODO: Classify independent function blocks
Allow the code to run  without inverse
"""
#%%
Vers="0.24"

#%% Imports

# 
import tkinter as tk
from tkinter import filedialog, ttk

import lmfit
from PIL import Image
import os
import csv
import numpy as np
import skimage
from skimage import restoration, morphology, filters, feature

import re #dat regex
import matplotlib.pyplot as plt
import exifread #needed to read tif tags

import scipy

import IAFun
# Will hold options
class Opt:
    pass
# WIll hold outputs
class Output:
    pass
#%% Default options

Opt.AutoDenoise=1;
Opt.AutoThresh=1;

Opt.RSFactor=4;# Implemented: TODO add to GUI
Opt.RSToggle=0; # 
Opt.Inversion=0;
Opt.ACToggle=0; #autocorrelation (currently broken)
Opt.ACCutoff=0; 
Opt.ACSize=50;

Opt.SchCO=5; # Step in from 'Ide' in nm



#IndividualLog =1; # Write a log for each sample?
CombLog = 1; # If One write a combined log, if two clean it out each time(don't append)
ShowImage = 0; # Show images?
# Following is GUI supported
Opt.EDToggle=0; #WIP ED/LER
Opt.FFTToggle=1; #fft
Opt.DenToggle=1; #Denoising ON
Opt.ThreshToggle=1; #thresh
Opt.RSOToggle=1; #remove small objects
Opt.IDEToggle=0; # Mask out the electrodes for YK
Opt.LabelToggle=1; # label domains
Opt.SkeleToggle=1; # Skeleton/Defect analysis
Opt.AngDetToggle=1; # Angle Detection


Opt.Machine="Unknown";
Output.Denoise='NA';

#plt.ioff() # turn off interactive plotting



#%% Gui Cus why not?
class GUI:
    def __init__(self, master):
      
        
        
        self.fftTVAR=tk.IntVar()
        self.DenTVAR=tk.IntVar()
        self.ThreshTVAR=tk.IntVar()
        self.RSOTVAR=tk.IntVar()
        self.SkeleTVAR=tk.IntVar()
        Opt.EDToggle=tk.IntVar()        
        
        #show images?
        Opt.CropSh=tk.IntVar()
        Opt.FFTSh=tk.IntVar()
        Opt.DenSh=tk.IntVar()
        Opt.ThreshSh=tk.IntVar()
        Opt.RSOSh=tk.IntVar()
        Opt.LabelSh=tk.IntVar()
        Opt.SkeleSh=tk.IntVar()
        Opt.EDSh=tk.IntVar()
        #save images?
        Opt.CropSa=tk.IntVar()
        Opt.FFTSa=tk.IntVar()
        Opt.DenSa=tk.IntVar()
        Opt.ThreshSa=tk.IntVar()
        Opt.RSOSa=tk.IntVar()
        Opt.LabelSa=tk.IntVar()
        Opt.SkeleSa=tk.IntVar()
        Opt.EDSa=tk.IntVar()
        
        Note=tk.ttk.Notebook(master)
        
        Note.pack()        
        
        Page1=tk.Frame(Note)
        Page2=tk.Frame(Note)
        Note.add(Page1,text='Options')
        Note.add(Page2,text='Image Options')
        Note.select(Page1)
        
        self.f1 = tk.ttk.Labelframe(Page1)
        self.f1.pack()              
        self.l1 = tk.Label(
            self.f1, text="Nanometers Per Pixel (Merlin Auto)"            
            )
        self.l1.pack(side=tk.LEFT)
        self.e1 = tk.Entry(self.f1)
        self.e1.pack(side=tk.LEFT)
        self.e1.insert(0, "0")  
        
        self.f2 = tk.ttk.Labelframe(Page1)
        self.f2.pack()
        self.l2 = tk.Label(
            self.f2, text="Pixels to crop Top, Left, Right, Bottom"            
            )
        self.l2.pack(side=tk.TOP)
        
        self.e2 = tk.Entry(self.f2)
        self.e2.pack(side=tk.LEFT)
        self.e2.insert(0, "0")  
        
        self.e3 = tk.Entry(self.f2)
        self.e3.pack(side=tk.LEFT)
        self.e3.insert(0, "0")
        
        self.e4 = tk.Entry(self.f2)
        self.e4.pack(side=tk.LEFT)
        self.e4.insert(0, "0")
        
        self.e5 = tk.Entry(self.f2)
        self.e5.pack(side=tk.LEFT)
        self.e5.insert(0, "100")          
        
        self.fftf=tk.ttk.Labelframe(Page1)
        self.fftf.pack()

        self.fftTog=tk.Checkbutton(
            self.fftf,text="Enable FFT",variable=self.fftTVAR)
        self.fftTog.pack(side=tk.LEFT)
        self.fftTog.select()
        self.fftl=tk.Label(
            self.fftf, text="Enter L0 (nm) if not using FFT"
            )
        self.fftl.pack(side=tk.LEFT) 
        self.L0 =tk.Entry(self.fftf)
        self.L0.pack(side=tk.LEFT)
        self.L0.insert(0,"0")
        
        
        self.Denf= tk.ttk.Labelframe(Page1)
        self.Denf.pack()
        self.DenTog=tk.Checkbutton(
            self.Denf,text="Enable Denoising",variable=self.DenTVAR)
        self.DenTog.pack(side=tk.LEFT)
        self.DenTog.select()
        self.l3=tk.Label(
            self.Denf, text="Denoising weight Lower = More Blur"            
            )
        self.l3.pack(side=tk.LEFT)
        self.e6 = tk.Entry(self.Denf)
        self.e6.pack(side=tk.LEFT)
        self.e6.insert(0,"130")
        self
        
        self.Threshf= tk.ttk.Labelframe(Page1)
        self.Threshf.pack()
        self.ThreshTog=tk.Checkbutton(
            self.Threshf,text="Enable Thresholding",variable=self.ThreshTVAR)
        self.ThreshTog.pack(side=tk.LEFT)
        self.ThreshTog.select()
        self.l4=tk.Label(
            self.Threshf, text="Thresholding weight, Lower = Local Thresh, Higher = Global"            
            )
        self.l4.pack(side=tk.LEFT)
        self.e7 = tk.Entry(self.Threshf)
        self.e7.pack(side=tk.LEFT)
        self.e7.insert(0,"2") # normally 2 #2.5 was prev YK
        
        self.RSOf= tk.ttk.Labelframe(Page1)
        self.RSOf.pack()
        self.RSOTog=tk.Checkbutton(
            self.RSOf,text="Remove small features",variable=self.RSOTVAR)
        self.RSOTog.pack(side=tk.LEFT)
        self.RSOTog.select()
        self.l5=tk.Label(
            self.RSOf, text="Remove Clusters < this many Pixels"            
            )
        self.l5.pack(side=tk.LEFT)
        self.e8 = tk.Entry(self.RSOf)
        self.e8.pack(side=tk.LEFT)
        self.e8.insert(0,"10")        
        
        self.Skelef= tk.ttk.Labelframe(Page1)
        self.Skelef.pack()
        self.SkeleTog=tk.Checkbutton(
            self.Skelef,text="Enable Skeleton/Defect Analysis",variable=self.SkeleTVAR)
        self.SkeleTog.pack(side=tk.LEFT)
        self.SkeleTog.select()
        self.l6=tk.Label(
            self.Skelef, text="Defect Analysis Edge Protect (Px)"            
            )
        self.l6.pack(side=tk.LEFT)
        self.e9 = tk.Entry(self.Skelef)
        self.e9.pack(side=tk.LEFT)
        self.e9.insert(0,"10")          
        
        self.EDF=tk.ttk.Labelframe(Page1)
        self.EDF.pack()
        self.EDTog=tk.Checkbutton(self.EDF,text="Enable Edge Detection/LWR",variable=Opt.EDToggle)
        self.EDTog.pack()
        self.EDTog.select()
        
        self.fend=tk.ttk.Labelframe(Page1)
        self.fend.pack()
        self.AcB = tk.Button(self.fend, text="Accept and Select File", command=self.begin)
        self.AcB.pack()
        
        self.ImShow=tk.ttk.Button(Page2,text="Show all Images",command=self.ImShowFun)
        self.ImShow.grid()
        
        self.ImNoShow=tk.ttk.Button(Page2,text="Show no Images",command=self.ImShowNoFun)
        self.ImNoShow.grid(row=0,column=1)
        
        self.ImSave=tk.ttk.Button(Page2,text="Save all Images",command=self.ImSaveFun)
        self.ImSave.grid(row=0,column=2)
        
        self.ImNoSave=tk.ttk.Button(Page2,text="Save no Images",command=self.ImSaveNoFun)
        self.ImNoSave.grid(row=0,column=3)
        #crop
        self.CropShC=tk.Checkbutton(Page2,text='Show Cropped Image',variable=Opt.CropSh)
        self.CropShC.grid(row=5,columnspan=2,column=0)
        self.CropSaC=tk.Checkbutton(Page2,text='Save Cropped Image',variable=Opt.CropSa)
        self.CropSaC.grid(row=5,columnspan=2,column=2)
        #fft
        self.FFTShC=tk.Checkbutton(Page2,text='Show FFT Image',variable=Opt.FFTSh)
        self.FFTShC.grid(row=10,columnspan=2,column=0)
        self.FFTSaC=tk.Checkbutton(Page2,text='Save FFT Image',variable=Opt.FFTSa)
        self.FFTSaC.grid(row=10,columnspan=2,column=2)
        #den
        self.DenShC=tk.Checkbutton(Page2,text='Show Denoised Image',variable=Opt.DenSh)
        self.DenShC.grid(row=15,columnspan=2,column=0)
        self.DenSaC=tk.Checkbutton(Page2,text='Save Denoised Image',variable=Opt.DenSa)
        self.DenSaC.grid(row=15,columnspan=2,column=2)
        #Thresh
        self.ThreshShC=tk.Checkbutton(Page2,text='Show Thresholded Image',variable=Opt.ThreshSh)
        self.ThreshShC.grid(row=20,columnspan=2,column=0)
        self.ThreshSaC=tk.Checkbutton(Page2,text='Save Thresholded Image',variable=Opt.ThreshSa)
        self.ThreshSaC.grid(row=20,columnspan=2,column=2)
        #RSO
        self.RSOShC=tk.Checkbutton(Page2,text='Show RSOd Image',variable=Opt.RSOSh)
        self.RSOShC.grid(row=25,columnspan=2,column=0)
        self.RSOSaC=tk.Checkbutton(Page2,text='Save RSO d Image',variable=Opt.RSOSa)
        self.RSOSaC.grid(row=25,columnspan=2,column=2)
        #label domains
        self.LabelShC=tk.Checkbutton(Page2,text='Show Labeled Image',variable=Opt.LabelSh)
        self.LabelShC.grid(row=26,columnspan=2,column=0)
        self.LabelSaC=tk.Checkbutton(Page2,text='Save Labeled Image',variable=Opt.LabelSa)
        self.LabelSaC.grid(row=26,columnspan=2,column=2)
        #Skele
        self.SkeleShC=tk.Checkbutton(Page2,text='Show Skeletonized Image',variable=Opt.SkeleSh)
        self.SkeleShC.grid(row=30,columnspan=2,column=0)
        self.SkeleSaC=tk.Checkbutton(Page2,text='Save Skeletonized Image',variable=Opt.SkeleSa)
        self.SkeleSaC.grid(row=30,columnspan=2,column=2)
        #EdgeDetect
        self.EDShC=tk.Checkbutton(Page2,text='Show Edge Detection/LER image',variable=Opt.EDSh)
        self.EDShC.grid(row=35,columnspan=2,column=0)
        self.EDSaC=tk.Checkbutton(Page2,text='Save Edge Detection/LER image',variable=Opt.EDSa)
        self.EDSaC.grid(row=35,columnspan=2,column=2)
        
    def ImShowFun(self):
        self.CropShC.select()
        self.FFTShC.select()
        self.DenShC.select()
        self.ThreshShC.select()
        self.RSOShC.select()
        self.LabelShC.select()
        self.SkeleShC.select()
        self.EDShC.select()
        
    def ImShowNoFun(self):
        self.CropShC.deselect()
        self.FFTShC.deselect()
        self.DenShC.deselect()
        self.ThreshShC.deselect()
        self.RSOShC.deselect()
        self.LabelShC.deselect()
        self.SkeleShC.deselect()
        self.EDShC.deselect()
        
    def ImSaveFun(self):
        self.CropSaC.select()
        self.FFTSaC.select()
        self.DenSaC.select()
        self.ThreshSaC.select()
        self.RSOSaC.select()
        self.LabelSaC.select()
        self.SkeleSaC.select()
        self.EDSaC.select()
        
    def ImSaveNoFun(self):
        self.CropSaC.deselect()
        self.FFTSaC.deselect()
        self.DenSaC.deselect()
        self.ThreshSaC.deselect()
        self.RSOSaC.deselect()
        self.LabelSaC.deselect()
        self.SkeleSaC.deselect()
        self.EDSaC.deselect()
        
    def begin(self):
        
        try:
            Opt.NmPP=float(self.e1.get())
        except:
            pass
        try:
            Output.l0=float(self.L0.get())
        except:
            pass
        
        Opt.FFTToggle=int(self.fftTVAR.get())
        Opt.DenToggle=int(self.DenTVAR.get())
        Opt.ThreshToggle=int(self.ThreshTVAR.get())
        Opt.RSOToggle=int(self.RSOTVAR.get())
        Opt.SkeleToggle=int(self.SkeleTVAR.get())
        Opt.EDToggle=int(Opt.EDToggle.get())        
        
        Opt.CropT=float(self.e2.get())
        Opt.CropL=float(self.e3.get())
        Opt.CropR=float(self.e4.get())
        Opt.CropB=float(self.e5.get())
        Opt.DenWeight=float(self.e6.get())
        Opt.ThreshWeight=float(self.e7.get())
        Opt.SPCutoff =float(self.e8.get())
        Opt.DefEdge=float(self.e9.get())
        
        Opt.CropSh=int(Opt.CropSh.get())
        Opt.FFTSh=int(Opt.FFTSh.get())
        Opt.DenSh=int(Opt.DenSh.get())
        Opt.ThreshSh=int(Opt.ThreshSh.get())
        Opt.RSOSh=int(Opt.RSOSh.get())
        Opt.LabelSh=int(Opt.LabelSh.get())
        Opt.SkeleSh=int(Opt.SkeleSh.get())
        Opt.EDSh=int(Opt.EDSh.get())
        #save images?
        Opt.CropSa=int(Opt.CropSa.get())
        Opt.FFTSa=int(Opt.FFTSa.get())
        Opt.DenSa=int(Opt.DenSa.get())
        Opt.ThreshSa=int(Opt.ThreshSa.get())
        Opt.RSOSa=int(Opt.RSOSa.get())
        Opt.LabelSa=int(Opt.LabelSa.get())
        Opt.SkeleSa=int(Opt.SkeleSa.get())
        Opt.EDSa=int(Opt.EDSa.get())
        
        root.destroy()


root = tk.Tk()
root.title("Image Analysis Software by Moshe V"+Vers)
gui=GUI(root)

root.mainloop()

#%% Open
FOpen=tk.Tk()

currdir = os.getcwd()
FNFull = tk.filedialog.askopenfilename(parent=FOpen, title='Please select a file', multiple=1)
FOpen.withdraw()
#if len(FNFull) > 0:
#    print("You chose %s" % FNFull)

for ImNum in range(0, len(FNFull) ):
    
    im= Image.open(FNFull[ImNum])
    FName = os.path.splitext(FNFull[ImNum])[0]
    Opt.FPath, Opt.BName= os.path.split(FName) 
    
    # Make output folder if needed
    try:
        os.stat(os.path.join(Opt.FPath,"output"))
    except:
        os.mkdir(os.path.join(Opt.FPath,"output"))
        
    if im.mode!="P":
        im=im.convert(mode='P')
        print("Image was not in the original format, and has been converted back to grayscale. Consider using the original image.")    
    imarray = np.array(im)
    
    
    
    #%% Autodetect per pixel scaling for merlin, don't have a nanosem image to figure that out
    
    IAFun.AutoDetect( FNFull[ImNum], Opt)
    
    #%% Crop
    (CropArray, Output.CIMH, Output.CIMW)=IAFun.Crop( imarray , Opt )
    imarray=CropArray

    #%% Data Rescaling STUPIDLY INTENSIVE
    
    if Opt.RSToggle==1:
        Output.CIMH*=Opt.RSFactor;
        Output.CIMW*=Opt.RSFactor;
        RSArray=skimage.transform.resize(imarray,(Output.CIMH,Output.CIMW))
        Opt.NmPP*=1/Opt.RSFactor;
        imarray=RSArray
    else:
        Opt.RSFactor=1;

    #$$ Set ArrayIn
    ArrayIn=imarray; 
    
    #%% FFT for period ref 
    # http://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    
    if Opt.FFTToggle==1:   
        Output.Calcl0=IAFun.FFT( ArrayIn, Opt)
        if Output.l0 == 0:
            Output.l0 = Output.Calcl0;
        
    
    #%% Denoise
    if Opt.DenToggle==1:
        (DenArray, Output.Denoise)=IAFun.Denoising(ArrayIn, Opt, Output.l0)
        ArrayIn=DenArray
    #%% Masking
        # Works ok at detecting features in large fields
    if Opt.IDEToggle==1:
        IDEArray=IAFun.YKDetect(DenArray, Opt)
        ArrayIn=IDEArray*ArrayIn
        
    #%% Adaptive Local Thresholding over X pixels, gauss                
    if Opt.ThreshToggle==1:
        (ThreshArray,Output.Thresh)=IAFun.Thresholding(ArrayIn, Opt, Output.l0)
        BinArray=ThreshArray

    #%% Remove Small Objects
    
    if Opt.RSOToggle==1:
        RSOArray=IAFun.RSO(BinArray, Opt)
        BinArray=RSOArray
        


    #%% Feature Finding
    
    
    if Opt.LabelToggle==1:
        (Output.WFrac, Output.BFrac, Output.WDomI, Output.WDomFrac)=IAFun.Label(BinArray,Opt)
        
    
    #%% Skeletonization / Defect could be split but that can be done later
    
    if Opt.SkeleToggle==1:
        (SkelArray, SkelAC, Output.TCount, Output.TCA, Output.JCount, Output.JCA)=IAFun.Skeleton(BinArray,Opt)
    
    #%% Angle Detection
    
    # todo fix so that AngEC is default
    if Opt.AngDetToggle==2:
            AngDetA=IAFun.AngSobel( ArrayIn ) # old method
    if Opt.AngDetToggle==1:
            AngDetA=IAFun.AngEC( BinArray, Opt)          # new method
            
    #%% What to do with angles? 
    (Output.Peak1,Output.Cnt1,Output.Peak2,Output.Cnt2,Output.CntT)=IAFun.AngHist(AngDetA, Opt, MaskArray=BinArray, WeightArray=ArrayIn)
    
    #%% ED
    # Tamar recommended Canny edge so let's try it eh? 
    # We don't use the guassian blur because the denoising/thresholding does this for us
    
    if Opt.EDToggle==1:
        (Output.LERMean,Output.LER3Sig,Output.LERMeanS,Output.LER3SigS)=IAFun.EdgeDetect(BinArray,Opt,SkelArray)
            

    #%% Autocorrel. LETS GO, Currently Not Working
    if Opt.ACToggle==5:
        AutoCor=IAFun.AutoCorrelation(BinArray, Opt, SkelArray)
    
    #%% Find the inverse or 'Dark' Image repeat as above
    if Opt.Inversion==1:
        imarray=255-imarray
        Opt.BName=Opt.BName+"Inv" #change base nameee

    #%% Crop
        (CropArray, Output.CIMH, Output.CIMW)=IAFun.Crop( imarray , Opt )
        ArrayIn=CropArray
    
    #%% Data Rescaling Not Yet Implemented correctly
    
        if Opt.RSToggle==1:
            Output.CIMH*=Opt.RSFactor;
            Output.CIMW*=Opt.RSFactor;
            RSArray=skimage.transform.resize(CropArray,(Output.CIMH,Output.CIMW))
            Opt.NmPP*=1/Opt.RSFactor;
            ArrayIn=RSArray
        else:
                Opt.RSFactor=1; 
    
    #%% Denoise
        if Opt.DenToggle==1:
            (DenArray, Output.Denoise)=IAFun.Denoising(ArrayIn, Opt, Output.l0)
            ArrayIn=DenArray

    #%% Adaptive Local Thresholding over X pixels, gauss                
        if Opt.ThreshToggle==1:
            (ThreshArray,Output.Thresh)=IAFun.Thresholding(ArrayIn, Opt, Output.l0)
            ArrayIn=ThreshArray

    #%% Remove Small Objects
    
        if Opt.RSOToggle==1:
            RSOArray=IAFun.RSO(ArrayIn, Opt)
            ArrayIn=RSOArray
    #%% Feature Finding
    
    
        if Opt.LabelToggle==1:
            (Output.InvWFrac, Output.InvBFrac, Output.InvWDomI, Output.InvWDomFrac)=IAFun.Label(ArrayIn,Opt)
        
    
    #%% Skeletonization / Defect could be split but that can be done later
    
        if Opt.SkeleToggle==1:
            (SkelArray, SkelAC, Output.InvTCount, Output.InvTCA, Output.InvJCount, Output.InvJCA)=IAFun.Skeleton(ArrayIn,Opt)
   
    #%% Logging
    # Check if a log exists, if not, but we want to log: write titles.
    # If we want to wipe the log each time, we also need to write titles
    if (os.path.isfile(os.path.join(Opt.FPath, "output", "output.csv"))==False and CombLog == 1) or CombLog == 2:
        with open(os.path.join(Opt.FPath, "output", "output.csv"), 'w') as Log:
            LogW= csv.writer(Log, dialect='excel', lineterminator='\n')
            LogW.writerow(['Filename',
            'Primary Peak (nm)',
            'Lighter phase',
            'LPhase Area Fraction',
            'DPhase Area Fraction',
            'LDom Index',
            'LDom Fraction',
            'LTerminals',
            'LTerminals/nm^2',
            'LJunctions',
            'Ljunctions/nm^2',
            'LWR Dist nm',
            'LWR 3Sig nm',
            'LWR Dist KDE nm',
            'LWR 3Sig KDE nm',
            'Denoise',
            'Threshold',
            'Denoise Used',
            'Thresh Used',
            'Peak 1 (Degrees)',
            'Count 1',
            'Peak 2 (Degrees)',
            'Count 2',
            'Cumulative Count',
            'Inverse Phase Images',
            'LPhase Area Fraction(FromINV)',
            'LPhase Area Frac AVG',
            'DPhase Area Fraction(FromINV)',
            'DPhase Area Frac AVG',
            'DDom Index',
            'DDom Fraction',
            'DTerminals',
            'DTerminals/nm^2',
            'DJunctions',
            'DJunctions/nm^2'])
    
    if CombLog > 0:
        with open(os.path.join(Opt.FPath, "output", "output.csv"), 'a') as Log:
            LogW= csv.writer(Log, dialect='excel', lineterminator='\n')
            try:
                LogW.writerow([Opt.BName,
                Output.l0,
                '',
                Output.WFrac,
                Output.BFrac,
                Output.WDomI,
                Output.WDomFrac,
                Output.TCount,
                Output.TCA,
                Output.JCount,
                Output.JCA,
                Output.LERMean,
                Output.LER3Sig,
                Output.LERMeanS,
                Output.LER3SigS,
                Opt.DenWeight,
                Opt.ThreshWeight,
                Output.Denoise,
                Output.Thresh,
                Output.Peak1,
                Output.Cnt1,
                Output.Peak2,
                Output.Cnt2,
                Output.CntT])
            except:
                pass;
            try:
                LogW.writerow(['(Names reference original!)',
                Output.InvBFrac,
                (Output.WFrac+Output.InvBFrac)/2,
                Output.InvWFrac,
                (Output.BFrac+Output.InvWFrac)/2,
                Output.InvWDomI,
                Output.InvWDomFrac,
                Output.InvTCount,
                Output.InvTCA,
                Output.InvJCount,
                Output.InvJCA])
            except:
                pass
        
    
    
        
        