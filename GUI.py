# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:38:16 2016

@author: Moshe
"""

# -*- coding: utf-8 -*-
"""
Block Copolymer Analysis Package by Moshe Dolejsi
Done in Spyder/VStudio2015 Community with Anaconda.
ToDO: Classify independent function blocks
"""
#%%
Vers="0.21"

#%% Imports
from PIL import Image
# 
import tkinter as tk
from tkinter import *
from tkinter.ttk import *

import os
import csv
import numpy as np
import skimage
from skimage import restoration, morphology, filters, feature

import re #dat regex
import matplotlib.pyplot as plt
import exifread #needed to read tif tags

import scipy

# Will hold options
class Opt:
    pass
# WIll hold outputs
class Output:
    pass
#%% Default options

Opt.AutoDenoise=1;
Opt.AutoThresh=1;

Opt.RSFactor=2;#Not yet implemented
Opt.RSToggle=0; # nyi
# need to make a GUI for this as well
Opt.FFTToggle=1; #fft
Opt.DenToggle=1; #Denoising ON
Opt.ThreshToggle=1; #thresh
Opt.RSOToggle=1; #remove small objects
Opt.LabelToggle=1; # label domains
Opt.SkeleToggle=1; # Skeleton/Defect analysis
Opt.ACToggle=0; #autocorrelation (currently broken)
Opt.ACCutoff=10;
Opt.ACSize=50;

Opt.Machine="Unknown";



#IndividualLog =1; # Write a log for each sample?
CombLog = 1; # If One write a combined log, if two clean it out each time(don't append)
ShowImage = 0; # Show images?

#%% Gui Cus why not?
class GUI:
    def __init__(self, master):
      
        
        
        self.fftTVAR=tk.IntVar()
        self.DenTVAR=tk.IntVar()
        self.ThreshTVAR=tk.IntVar()
        self.RSOTVAR=tk.IntVar()
        self.SkeleTVAR=tk.IntVar()
        #show images?
        Opt.CropSh=tk.IntVar()
        Opt.FFTSh=tk.IntVar()
        Opt.DenSh=tk.IntVar()
        Opt.ThreshSh=tk.IntVar()
        Opt.RSOSh=tk.IntVar()
        Opt.SkeleSh=tk.IntVar()
        #save images?
        Opt.CropSa=tk.IntVar()
        Opt.FFTSa=tk.IntVar()
        Opt.DenSa=tk.IntVar()
        Opt.ThreshSa=tk.IntVar()
        Opt.RSOSa=tk.IntVar()
        Opt.SkeleSa=tk.IntVar()
        
        
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
        self.e5.insert(0, "50")          
        
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
        self.e6.insert(0,"130") #130
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
        self.e7.insert(0,"2")
        
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
        self.DenSaC=tk.Checkbutton(Page2,text='Save Denoises Image',variable=Opt.DenSa)
        self.DenSaC.grid(row=15,columnspan=2,column=2)
        #Thresh
        self.ThreshShC=tk.Checkbutton(Page2,text='Show Thresholded Image',variable=Opt.ThreshSh)
        self.ThreshShC.grid(row=20,columnspan=2,column=0)
        self.ThreshSaC=tk.Checkbutton(Page2,text='Save Thresholded Image',variable=Opt.ThreshSa)
        self.ThreshSaC.grid(row=20,columnspan=2,column=2)
        #RSO
        self.RSOShC=tk.Checkbutton(Page2,text='Show RSOd Image',variable=Opt.RSOSh)
        self.RSOShC.grid(row=25,columnspan=2,column=0)
        self.RSOSaC=tk.Checkbutton(Page2,text='Save RSOd Image',variable=Opt.RSOSa)
        self.RSOSaC.grid(row=25,columnspan=2,column=2)
        #Skele
        self.SkeleShC=tk.Checkbutton(Page2,text='Show Skeletonized Image',variable=Opt.SkeleSh)
        self.SkeleShC.grid(row=30,columnspan=2,column=0)
        self.SkeleSaC=tk.Checkbutton(Page2,text='Save Skeletonized Image',variable=Opt.SkeleSa)
        self.SkeleSaC.grid(row=30,columnspan=2,column=2)
        
        
    def ImShowFun(self):
        self.CropShC.select()
        self.FFTShC.select()
        self.DenShC.select()
        self.ThreshShC.select()
        self.RSOShC.select()
        self.SkeleShC.select()
        
    def ImShowNoFun(self):
        self.CropShC.deselect()
        self.FFTShC.deselect()
        self.DenShC.deselect()
        self.ThreshShC.deselect()
        self.RSOShC.deselect()
        self.SkeleShC.deselect()
        
    def ImSaveFun(self):
        self.CropSaC.select()
        self.FFTSaC.select()
        self.DenSaC.select()
        self.ThreshSaC.select()
        self.RSOSaC.select()
        self.SkeleSaC.select()
        
    def ImSaveNoFun(self):
        self.CropSaC.deselect()
        self.FFTSaC.deselect()
        self.DenSaC.deselect()
        self.ThreshSaC.deselect()
        self.RSOSaC.deselect()
        self.SkeleSaC.deselect()
        
    def begin(self):
        
        try:
            Opt.NmPP=float(self.e1.get())
        except:
            pass
        try:
            Output.l0=float(self.L0.get())
        except:
            pass
        
        Opt.FFTToggle=float(self.fftTVAR.get())
        Opt.DenToggle=float(self.DenTVAR.get())
        Opt.ThreshToggle=float(self.ThreshTVAR.get())
        Opt.SFRToggle=float(self.RSOTVAR.get())
        Opt.SkeleToggle=float(self.SkeleTVAR.get())
                
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
        Opt.SkeleSh=int(Opt.SkeleSh.get())
        #save images?
        Opt.CropSa=int(Opt.CropSa.get())
        Opt.FFTSa=int(Opt.FFTSa.get())
        Opt.DenSa=int(Opt.DenSa.get())
        Opt.ThreshSa=int(Opt.ThreshSa.get())
        Opt.RSOSa=int(Opt.RSOSa.get())
        Opt.SkeleSa=int(Opt.SkeleSa.get())
        
        root.destroy()


root = tk.Tk()
root.title("Image Analysis Software by Moshe V"+Vers)
gui=GUI(root)

root.mainloop()
