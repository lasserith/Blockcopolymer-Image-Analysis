# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:20:58 2017

AFM test for Raybin
"""
#%%
Vers = "AAA"

#%% Imports

#
import tkinter as tk
from tkinter import filedialog, ttk
import os
import csv
import re
import lmfit
from PIL import Image
from joblib import Parallel, delayed
import numpy as np
import scipy
import skimage
from skimage import restoration, morphology, filters, feature
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
import matplotlib.animation as manimation


import IAFun
# Will hold options
class Opt:
    pass
# WIll hold outputs
class Output:
    pass
#%% Default options

Opt.AutoDenoise = 1
Opt.AutoThresh = 1


Opt.Inversion = 0
Opt.ACToggle = 0 #autocorrelation (currently broken)
Opt.ACCutoff = 0
Opt.ACSize = 50

Opt.SchCO = 5 # Step in from 'Ide' in nm



#IndividualLog =1; # Write a log for each sample?
CombLog = 1 # If One write a combined log, if two clean it out each time(don't append)
ShowImage = 0 # Show images?

## TODO : ADD THE BELOW TO GUI
Opt.IDEToggle = 0 # Mask out the electrodes for YK
Opt.LabelToggle = 0 # label domains
Opt.AFMLayer = "Phase" #Matched Phase ZSensor
Opt.AFMLevel = 3  # 0 = none 1 = Median 2= Median of Dif 3 = polyfit
Opt.AFMPDeg = 5 # degree of polynomial.

# Following is GUI supported
#Opt.EDToggle=0; #ED/LER
#Opt.FFTToggle=1; #fft
#Opt.DenToggle=1; #Denoising ON
#Opt.ThreshToggle=1; #thresh
#Opt.RSOToggle=1; #remove small objects

#Opt.SkeleToggle=1; # Skeleton/Defect analysis
#Opt.AngDetToggle=1; # Angle Detection
#Opt.SSFactor=4;#
#Opt.SSToggle=0;#


Opt.Machine = "Unknown"
Output.Denoise = 'NA'

#plt.ioff() # turn off interactive plotting



#%% Gui Cus why not?
class GUI:
    def __init__(self, master):
      

        Opt.SSToggle = tk.IntVar()
        Opt.FFTToggle = tk.IntVar()
        Opt.DenToggle = tk.IntVar()
        Opt.ThreshToggle = tk.IntVar()
        Opt.RSOToggle = tk.IntVar()
        Opt.SkeleToggle=tk.IntVar()
        Opt.EDToggle=tk.IntVar()   
        Opt.AngDetToggle=tk.IntVar()

        #show images?
        Opt.CropSh=tk.IntVar()
        Opt.FFTSh=tk.IntVar()
        Opt.DenSh=tk.IntVar()
        Opt.ThreshSh=tk.IntVar()
        Opt.RSOSh=tk.IntVar()
        Opt.LabelSh=tk.IntVar()
        Opt.SkeleSh=tk.IntVar()
        Opt.EDSh=tk.IntVar()
        Opt.AECSh=tk.IntVar()
        #save images?
        Opt.CropSa=tk.IntVar()
        Opt.FFTSa=tk.IntVar()
        Opt.DenSa=tk.IntVar()
        Opt.ThreshSa=tk.IntVar()
        Opt.RSOSa=tk.IntVar()
        Opt.LabelSa=tk.IntVar()
        Opt.SkeleSa=tk.IntVar()
        Opt.EDSa=tk.IntVar()
        Opt.AECSa=tk.IntVar()
        
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
        
        self.CropT = tk.Entry(self.f2)
        self.CropT.pack(side=tk.LEFT)
        self.CropT.insert(0, "0")  
        
        self.CropL = tk.Entry(self.f2)
        self.CropL.pack(side=tk.LEFT)
        self.CropL.insert(0, "0")
        
        self.CropR = tk.Entry(self.f2)
        self.CropR.pack(side=tk.LEFT)
        self.CropR.insert(0, "0")
        
        self.CropB = tk.Entry(self.f2)
        self.CropB.pack(side=tk.LEFT)
        self.CropB.insert(0, "0")          
        
        self.ssampf=tk.ttk.Labelframe(Page1)
        self.ssampf.pack()
        self.ssTog=tk.Checkbutton(self.ssampf,text="Enable Super Sampling (INTENSIVE)",variable=Opt.SSToggle)
        self.ssTog.pack(side=tk.LEFT)
        self.ssTog.deselect()
        self.ssl=tk.Label(self.ssampf, text="Scaling factor, integer (4=16x total pixels)").pack(side=tk.LEFT)
        self.sse=tk.Entry(self.ssampf)
        self.sse.pack(side=tk.LEFT)
        self.sse.insert(0,"4")
        
        self.fftf=tk.ttk.Labelframe(Page1)
        self.fftf.pack()
        self.fftTog=tk.Checkbutton(self.fftf,text="Enable FFT",variable=Opt.FFTToggle)
        self.fftTog.pack(side=tk.LEFT)
        self.fftTog.select()
        self.fftl=tk.Label(self.fftf, text="Enter L0 (nm) if not using FFT")
        self.fftl.pack(side=tk.LEFT) 
        self.L0 =tk.Entry(self.fftf)
        self.L0.pack(side=tk.LEFT)
        self.L0.insert(0,"50")
        
        
        self.Denf= tk.ttk.Labelframe(Page1)
        self.Denf.pack()
        self.DenTog=tk.Checkbutton(
            self.Denf,text="Enable Denoising",variable=Opt.DenToggle)
        self.DenTog.pack(side=tk.LEFT)
        self.DenTog.select()
        self.l3=tk.Label(
            self.Denf, text="Denoising weight Lower = More Blur"            
            )
        self.l3.pack(side=tk.LEFT)
        self.e6 = tk.Entry(self.Denf)
        self.e6.pack(side=tk.LEFT)
        self.e6.insert(0,"20")

        
        self.Threshf= tk.ttk.Labelframe(Page1)
        self.Threshf.pack()
        self.ThreshTog=tk.Checkbutton(
            self.Threshf,text="Enable Thresholding",variable=Opt.ThreshToggle)
        self.ThreshTog.pack(side=tk.LEFT)
        self.ThreshTog.select()
        self.l4=tk.Label(
            self.Threshf, text="Thresholding weight, Lower = Local Thresh, Higher = Global"            
            )
        self.l4.pack(side=tk.LEFT)
        self.e7 = tk.Entry(self.Threshf)
        self.e7.pack(side=tk.LEFT)
        self.e7.insert(0,"2.5") # normally 2 #2.5 was prev YK
        
        self.RSOf= tk.ttk.Labelframe(Page1)
        self.RSOf.pack()
        self.RSOTog=tk.Checkbutton(
            self.RSOf,text="Remove small features",variable=Opt.RSOToggle)
        self.RSOTog.pack(side=tk.LEFT)
        self.RSOTog.select()
        self.l5=tk.Label(
            self.RSOf, text="Remove Clusters < this many Pixels"            
            )
        self.l5.pack(side=tk.LEFT)
        self.e8 = tk.Entry(self.RSOf)
        self.e8.pack(side=tk.LEFT)
        self.e8.insert(0,"10")  
        
        
        self.Angf= tk.ttk.Labelframe(Page1)
        self.Angf.pack()
        self.Angfl=tk.Label(self.Angf, text="Angle Detection algorithm").pack(side=tk.LEFT)
        self.AngTogN=tk.Radiobutton(self.Angf,text="None",variable=Opt.AngDetToggle, value=0).pack(side=tk.LEFT)
        self.AngTogEC=tk.Radiobutton(self.Angf,text="Edge/Center",variable=Opt.AngDetToggle, value=1).pack(side=tk.LEFT)
        self.AngTogS=tk.Radiobutton(self.Angf,text="Sobel", variable=Opt.AngDetToggle, value=2).pack(side=tk.LEFT)
        Opt.AngDetToggle.set(1) # pick EC as default

        self.Skelef= tk.ttk.Labelframe(Page1)
        self.Skelef.pack()
        self.SkeleTog=tk.Checkbutton(
        self.Skelef,text="Enable Skeleton/Defect Analysis",variable=Opt.SkeleToggle)
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
        self.LabelShC.grid(row=30,columnspan=2,column=0)
        self.LabelSaC=tk.Checkbutton(Page2,text='Save Labeled Image',variable=Opt.LabelSa)
        self.LabelSaC.grid(row=30,columnspan=2,column=2)
        #Skele
        self.SkeleShC=tk.Checkbutton(Page2,text='Show Skeletonized Image',variable=Opt.SkeleSh)
        self.SkeleShC.grid(row=35,columnspan=2,column=0)
        self.SkeleSaC=tk.Checkbutton(Page2,text='Save Skeletonized Image',variable=Opt.SkeleSa)
        self.SkeleSaC.grid(row=35,columnspan=2,column=2)
        #EdgeDetect
        self.EDShC=tk.Checkbutton(Page2,text='Show Edge Detection/LWR image',variable=Opt.EDSh)
        self.EDShC.grid(row=40,columnspan=2,column=0)
        self.EDSaC=tk.Checkbutton(Page2,text='Save Edge Detection/LWR image',variable=Opt.EDSa)
        self.EDSaC.grid(row=40,columnspan=2,column=2)
        #Angle Detection Edge-Center
        self.AECShC=tk.Checkbutton(Page2,text='Show Angle - Edge/Center image',variable=Opt.AECSh)
        self.AECShC.grid(row=45,columnspan=2,column=0)
        self.AECSaC=tk.Checkbutton(Page2,text='Save Angle - Edge/Center image',variable=Opt.AECSa)
        self.AECSaC.grid(row=45,columnspan=2,column=2)
        
        # make buttons to select/deselect all
    def ImShowFun(self):
        self.CropShC.select()
        self.FFTShC.select()
        self.DenShC.select()
        self.ThreshShC.select()
        self.RSOShC.select()
        self.LabelShC.select()
        self.SkeleShC.select()
        self.EDShC.select()
        self.AECShC.select()
        
    def ImShowNoFun(self):
        self.CropShC.deselect()
        self.FFTShC.deselect()
        self.DenShC.deselect()
        self.ThreshShC.deselect()
        self.RSOShC.deselect()
        self.LabelShC.deselect()
        self.SkeleShC.deselect()
        self.EDShC.deselect()
        self.AECShC.deselect()
        
    def ImSaveFun(self):
        self.CropSaC.select()
        self.FFTSaC.select()
        self.DenSaC.select()
        self.ThreshSaC.select()
        self.RSOSaC.select()
        self.LabelSaC.select()
        self.SkeleSaC.select()
        self.EDSaC.select()
        self.AECSaC.select()
        
    def ImSaveNoFun(self):
        self.CropSaC.deselect()
        self.FFTSaC.deselect()
        self.DenSaC.deselect()
        self.ThreshSaC.deselect()
        self.RSOSaC.deselect()
        self.LabelSaC.deselect()
        self.SkeleSaC.deselect()
        self.EDSaC.deselect()
        self.AECSaC.deselect()
        
    def begin(self):
        
        try:
            Opt.NmPPSet=float(self.e1.get()) # this holds it so if we iterate over images we can reset it each time so behavior is consistent
        except:
            pass
        try:
            Output.l0=float(self.L0.get())
        except:
            pass
        
        # convert everything to ints so they can be read outside
        Opt.SSToggle=int(Opt.SSToggle.get())
        Opt.FFTToggle=int(Opt.FFTToggle.get())
        Opt.DenToggle=int(Opt.DenToggle.get())
        Opt.ThreshToggle=int(Opt.ThreshToggle.get())
        Opt.RSOToggle=int(Opt.RSOToggle.get())
        Opt.SkeleToggle=int(Opt.SkeleToggle.get())
        Opt.EDToggle=int(Opt.EDToggle.get())
        Opt.AngDetToggle=int(Opt.AngDetToggle.get())        
        
        Opt.SSFactor=int(self.sse.get())
        Opt.CropT=float(self.CropT.get())
        Opt.CropL=float(self.CropL.get())
        Opt.CropR=float(self.CropR.get())
        Opt.CropB=float(self.CropB.get())
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
        Opt.AECSh=int(Opt.AECSh.get())
        #save images?
        Opt.CropSa=int(Opt.CropSa.get())
        Opt.FFTSa=int(Opt.FFTSa.get())
        Opt.DenSa=int(Opt.DenSa.get())
        Opt.ThreshSa=int(Opt.ThreshSa.get())
        Opt.RSOSa=int(Opt.RSOSa.get())
        Opt.LabelSa=int(Opt.LabelSa.get())
        Opt.SkeleSa=int(Opt.SkeleSa.get())
        Opt.EDSa=int(Opt.EDSa.get())
        Opt.AECSa=int(Opt.AECSa.get())
        
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

#%% Do Once
ImNum = 0
Opt.Name=FNFull[ImNum] # this hold the full file name
Opt.FPath, Opt.BName= os.path.split(Opt.Name)  # File Path/ File Name
(Opt.FName, Opt.FExt) = os.path.splitext(Opt.BName) # File name/File Extension split
    
    
# Make output folder if needed
try:
    os.stat(os.path.join(Opt.FPath,"output"))
except:
    os.mkdir(os.path.join(Opt.FPath,"output"))

firstim = IAFun.AutoDetect( FNFull[ImNum], Opt)
Shap0 =firstim.shape[0]
Shap1 = firstim.shape[1]
imarray = np.zeros((Shap0,Shap1,len(FNFull)))
SkelOut = np.zeros((Shap0,Shap1,len(FNFull))).astype('i1')
FiltOut = np.zeros((Shap0,Shap1,len(FNFull)))
AdOut = np.zeros((Shap0,Shap1,len(FNFull))).astype('i1')
ThreshOut = np.zeros((Shap0,Shap1,len(FNFull))).astype('i1')
CropSkelOut = np.zeros((Shap0,Shap1,len(FNFull))).astype('i1')
FiltCum = np.zeros((Shap0,Shap1))
ThreshCum = np.zeros((Shap0,Shap1)).astype('i2')

#%% Masking
R, C = np.indices(firstim.shape)
Mask = np.ones_like(firstim)
SlTop = (85-70)/C.max() # Slope of top of wedge (Rside y - Lside y * dx)
SlBot = (127-157)/C.max() # slope of bottom of wedge
Mask[ R < C*SlTop+70] = 0 # Lside Y top
Mask[ R > C*SlBot+157 ] = 0 # Lside Y bot
#%% Parallel Loop
IAFun.AFMPara(FNFull,Opt,FiltOut,ThreshOut,AdOut,SkelOut, 0)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    Parallel(n_jobs=8,backend="threading",verbose=50)(delayed(IAFun.AFMPara)(FNFull,Opt,FiltOut,ThreshOut,AdOut,SkelOut, ImNum)
        for ImNum in range(0, len(FNFull)))
    print('Done')
#    


#%%
ThreshCum = ThreshOut.sum(axis=2)
FiltCum = FiltOut.sum(axis=2)    
SkelCum = (SkelOut.sum(axis=2))
SkelX = SkelCum.mean(axis=0) # average across the channel
Term = AdOut == 2
TermSum = Term.sum(axis=2)
Junc = AdOut > 3
JuncSum = Junc.sum(axis=2)
DefMask = skimage.morphology.binary_erosion(Mask)
DefMask = skimage.morphology.binary_erosion(DefMask)

#%% Plotting
MPlot = plt.figure()
MPlot.suptitle('Movement of Domain Skeletons')
MPlot1 = MPlot.add_subplot(211)
MPlot1.axvline(x=42,color='k')
MPlot1.axvline(x=121,color='k')
MPlot1.axvline(x=206,color='k')
MPlot1.axvline(x=283,color='k')
MPlot1.axvline(x=334,color='k')
MPlot1.axvline(x=422,color='k')
MPlot1.axvline(x=478,color='k')
CPlot = MPlot1.imshow(SkelCum, cmap='brg')
#MPlot.colorbar(CPlot)
MPlot1.set_ylim(50,150)

MPlot2 = MPlot.add_subplot(212)
MPlot2.plot(SkelX)
MPlot2.axvline(x=42,color='k')
MPlot2.axvline(x=121,color='k')
MPlot2.axvline(x=206,color='k')
MPlot2.axvline(x=283,color='k')
MPlot2.axvline(x=334,color='k')
MPlot2.axvline(x=422,color='k')
MPlot2.axvline(x=478,color='k')
MPlot2.set_xlim(0, 512)
MPlot2.set_xlabel('Distance along channel (px)')
MPlot2.set_ylabel('Mean Deviation over time')
#%%
TPlot = plt.figure()
TPlot.suptitle('Movement of Terminal Defects')
TPlot1 = TPlot.add_subplot(211)
TPlot1.axvline(x=42,color='k')
TPlot1.axvline(x=121,color='k')
TPlot1.axvline(x=206,color='k')
TPlot1.axvline(x=283,color='k')
TPlot1.axvline(x=334,color='k')
TPlot1.axvline(x=422,color='k')
TPlot1.axvline(x=478,color='k')
CPlot = TPlot1.imshow(TermSum*DefMask, cmap='brg')
#TPlot.colorbar(CPlot)
TPlot1.set_ylim(50,150)

TPlot2 = TPlot.add_subplot(212)
TPlot2.plot((TermSum*DefMask).sum(axis=0))
TPlot2.axvline(x=42,color='k')
TPlot2.axvline(x=121,color='k')
TPlot2.axvline(x=206,color='k')
TPlot2.axvline(x=283,color='k')
TPlot2.axvline(x=334,color='k')
TPlot2.axvline(x=422,color='k')
TPlot2.axvline(x=478,color='k')
TPlot2.set_xlim(0, 512)
TPlot2.set_ylim(0,30)
TPlot2.set_xlabel('Distance along channel (px)')
TPlot2.set_ylabel('Mean Deviation over time')
#%%
JPlot = plt.figure()
JPlot.suptitle('Movement of Junctions')
JPlot1 = JPlot.add_subplot(211)
JPlot1.axvline(x=42,color='k')
JPlot1.axvline(x=121,color='k')
JPlot1.axvline(x=206,color='k')
JPlot1.axvline(x=283,color='k')
JPlot1.axvline(x=334,color='k')
JPlot1.axvline(x=422,color='k')
JPlot1.axvline(x=478,color='k')
CPlot = JPlot1.imshow(JuncSum*DefMask, cmap='brg')
#JPlot.colorbar(CPlot)
JPlot1.set_ylim(50,150)

JPlot2 = JPlot.add_subplot(212)
JPlot2.plot((JuncSum*DefMask).sum(axis=0))
JPlot2.axvline(x=42,color='k')
JPlot2.axvline(x=121,color='k')
JPlot2.axvline(x=206,color='k')
JPlot2.axvline(x=283,color='k')
JPlot2.axvline(x=334,color='k')
JPlot2.axvline(x=422,color='k')
JPlot2.axvline(x=478,color='k')
JPlot2.set_xlim(0, 512)
JPlot2.set_ylim(0,100)
JPlot2.set_xlabel('Distance along channel (px)')
JPlot2.set_ylabel('Mean Deviation over time')

#%%
DefPlot = plt.figure()
DefPlot.suptitle('Movement of Defects')
DefPlot1 = DefPlot.add_subplot(211)
DefPlot1.axvline(x=42,color='k')
DefPlot1.axvline(x=121,color='k')
DefPlot1.axvline(x=206,color='k')
DefPlot1.axvline(x=283,color='k')
DefPlot1.axvline(x=334,color='k')
DefPlot1.axvline(x=422,color='k')
DefPlot1.axvline(x=478,color='k')
CPlot = DefPlot1.imshow((TermSum+JuncSum)*DefMask, cmap='brg')
#DefPlot.colorbar(CPlot)
DefPlot1.set_ylim(50,150)

DefPlot2 = DefPlot.add_subplot(212)
DefPlot2.plot(((TermSum+JuncSum)*DefMask).sum(axis=0))
DefPlot2.axvline(x=42,color='k')
DefPlot2.axvline(x=121,color='k')
DefPlot2.axvline(x=206,color='k')
DefPlot2.axvline(x=283,color='k')
DefPlot2.axvline(x=334,color='k')
DefPlot2.axvline(x=422,color='k')
DefPlot2.axvline(x=478,color='k')
DefPlot2.set_xlim(0, 512)
DefPlot2.set_ylim(0,100)
DefPlot2.set_xlabel('Distance along channel (px)')
DefPlot2.set_ylabel('Mean Deviation over time')
#%%
#%% Animation Setup
MovieFig = plt.figure()
MovieSubplot = MovieFig.add_subplot(111)
ims = []

for ii in range(0, len(FNFull) ):

    Frame = plt.imshow(SkelOut[:,:,ImNum], animated=True)
    ims.append([Frame])

#ims=[]
#for ImNum in range(0, 273 ):
#    Frame=MovieSubplot.imshow(SkelOut[:,:,ImNum], animated=True)
#    ims.append([Frame])
#    print(ImNum)


#ani = manimation.ArtistAnimation(MovieFig, ims,blit=True)
##%% Save animation
#MWriter = manimation.FFMpegWriter(bitrate=10000)
#ani.save('Wedge.mp4' , writer=MWriter)
###%%
#np.save('SkelOutArray',SkelOut)