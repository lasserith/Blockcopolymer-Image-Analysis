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
Vers="0.22"

#%% Imports
from PIL import Image
# 
import tkinter as tk
from tkinter import filedialog, ttk

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
        Opt.LabelSh=tk.IntVar()
        Opt.SkeleSh=tk.IntVar()
        #save images?
        Opt.CropSa=tk.IntVar()
        Opt.FFTSa=tk.IntVar()
        Opt.DenSa=tk.IntVar()
        Opt.ThreshSa=tk.IntVar()
        Opt.RSOSa=tk.IntVar()
        Opt.LabelSa=tk.IntVar()
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
        #label domains
        self.LabelShC=tk.Checkbutton(Page2,text='Show Labeld Image',variable=Opt.LabelSh)
        self.LabelShC.grid(row=26,columnspan=2,column=0)
        self.LabelSaC=tk.Checkbutton(Page2,text='Save Labeld Image',variable=Opt.LabelSa)
        self.LabelSaC.grid(row=26,columnspan=2,column=2)
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
        self.LabelShC.select()
        self.SkeleShC.select()
        
    def ImShowNoFun(self):
        self.CropShC.deselect()
        self.FFTShC.deselect()
        self.DenShC.deselect()
        self.ThreshShC.deselect()
        self.RSOShC.deselect()
        self.LabelShC.deselect()
        self.SkeleShC.deselect()
        
    def ImSaveFun(self):
        self.CropSaC.select()
        self.FFTSaC.select()
        self.DenSaC.select()
        self.ThreshSaC.select()
        self.RSOSaC.select()
        self.LabelSaC.select()
        self.SkeleSaC.select()
        
    def ImSaveNoFun(self):
        self.CropSaC.deselect()
        self.FFTSaC.deselect()
        self.DenSaC.deselect()
        self.ThreshSaC.deselect()
        self.RSOSaC.deselect()
        self.LabelSaC.deselect()
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
        Opt.LabelSh=int(Opt.LabelSh.get())
        Opt.SkeleSh=int(Opt.SkeleSh.get())
        #save images?
        Opt.CropSa=int(Opt.CropSa.get())
        Opt.FFTSa=int(Opt.FFTSa.get())
        Opt.DenSa=int(Opt.DenSa.get())
        Opt.ThreshSa=int(Opt.ThreshSa.get())
        Opt.RSOSa=int(Opt.RSOSa.get())
        Opt.LabelSa=int(Opt.LabelSa.get())
        Opt.SkeleSa=int(Opt.SkeleSa.get())
        
        root.destroy()


root = tk.Tk()
root.title("Image Analysis Software by Moshe V"+Vers)
gui=GUI(root)

root.mainloop()

#%%
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    http://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

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
    FPath, BName= os.path.split(FName)
    # Make output folder if needed
    try:
        os.stat(os.path.join(FPath,"output"))
    except:
        os.mkdir(os.path.join(FPath,"output"))  
    
    
    
    #%% Autodetect per pixel scaling for merlin, don't have a nanosem image to figure that out
    
    SkimFile = open(FNFull[ImNum],'rb')
    MetaF=exifread.process_file(SkimFile)
    SkimFile.close()
    try:
        Opt.FInfo=str(MetaF['Image Tag 0x8546'].values);
        Opt.NmPP=float(Opt.FInfo[17:30])*10**9;
        Opt.Machine="Merlin";
    except:
        pass
    
    if Opt.NmPP!=0:
        print("Instrument was autodetected as %s, NmPP is %f \n" % (Opt.Machine ,Opt.NmPP) )
    else:
        print("Instrument was not detected, and NmPP was not set. Please set NmPP and rerun")
         
    
    
    
    
    #%% Crop
    if im.mode!="P":
        im=im.convert(mode='P')
        print("Image was not in the original format, and has been converted back to grayscale. Consider using the original image.")
    
    imarray = np.array(im)
    (IMH, IMW) =imarray.shape
    
    # Crop
    
    CropArray=imarray[int(0+Opt.CropT):int(IMH-Opt.CropB),int(Opt.CropL):int(IMW-Opt.CropR)]
    
    (CIMH, CIMW)=CropArray.shape
    ArrayIn=CropArray
    
    
    
    #%% Data Rescaling Not Yet Implemented correctly
    
    if Opt.RSToggle==1:
        CIMH*=Opt.RSFactor;
        CIMW*=Opt.RSFactor;
        RSArray=skimage.transform.resize(CropArray,(CIMH,CIMW))
        Opt.NmPP*=1/Opt.RSFactor;
        ArrayIn=RSArray
    else:
        Opt.RSFactor=1;
    #%% making masks
        
    RImage=Image.new('RGB',(CIMW,CIMH),'Red')
    GImage=Image.new('RGB',(CIMW,CIMH),'Green')
    BImage=Image.new('RGB',(CIMW,CIMH),'Blue')
    
    
    #%% FFT for period ref 
    # http://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    # pyFAI ? AI=pyFAI.load(CropArray)
    #PowerSpec1d=AI.integrate1d(CropArray,100)
    
    if Opt.FFTToggle==1:
    
        
        Opt.FSize=np.min( (CIMH, CIMW) );
        
        FourierArray=np.fft.fft2(ArrayIn, s=(Opt.FSize,Opt.FSize) );
        FreqA=np.fft.fftfreq(Opt.FSize, d=Opt.NmPP);
        SpaceA=1/FreqA;
        F2Array=np.fft.fftshift(FourierArray);
        PowerSpec2d= np.abs( F2Array )**2;
        PowerSpec1d= azimuthalAverage(PowerSpec2d);
        Peak=scipy.signal.find_peaks_cwt(PowerSpec1d[0:int( np.floor(Opt.FSize/2))], np.arange(5,10),);
        
        
        PFreq=np.zeros(np.size(Peak))
        Pspace=np.zeros(np.size(Peak));
        PHeight=np.zeros(np.size(Peak));
        for i in range(0, np.size(Peak)):
            Peak[i]+=1
            PFreq[i]=FreqA[Peak[i]]
            Pspace[i]=1/FreqA[Peak[i]]
            PHeight[i]=PowerSpec1d[Peak[i]]
        if Peak[0] < 10: # if first peak is found at L= infty
            PHeight[0]=0; # dont consider it for characteristic peak
             
        PHMax=PHeight.max()
        PFMax=(PFreq*(PHMax==PHeight)).max()
        PSMax=1/PFMax;
        Output.l0=PSMax 
        # Now save plots
        if Opt.FFTSh==1 or Opt.FFTSa==1:
            Fig=plt.figure()
            PSD1D=Fig.add_subplot(111)
            PSD1D.plot(FreqA[1:int(np.floor(Opt.FSize/2))], PowerSpec1d[1:int( np.floor(Opt.FSize/2))])
            PSD1D.set_yscale('log')
            PSD1D.set_title('1D Power Spectral Density')
            PSD1D.set_xlabel('q (1/nm)')
            PSD1D.set_ylabel('Intensity')
            PSD1D.set_ylim([np.min(PowerSpec1d)*.5, np.max(PowerSpec1d)*10])
            Fig.savefig(os.path.join(FPath,"output",BName + "PowerSpecFreq.png"))
            PSD1D.annotate('Primary Peak at %f' %PFMax, xy=(PFMax, PHMax), xytext=(1.5*PFMax, 1.5*PHMax),
                        arrowprops=dict(facecolor='black', width=2,headwidth=5),
                        )
            Fig.savefig(os.path.join(FPath,"output",BName + "PowerSpecFreqLabel.png"))
            
            
            PS2DImage=Image.fromarray(255/np.max(np.log(PowerSpec2d))*np.log(PowerSpec2d))
            PS2DImage=PS2DImage.convert(mode="RGB")
            if Opt.FFTSh == 1:
                PS2DImage.show()
            if Opt.FFTSa==1:
                PS2DImage.save(os.path.join(FPath,"output",BName + "PowerSpec2d.tif"))
        
    
    
    #%% Denoise
    if Opt.DenToggle==1:
        if Opt.AutoDenoise==1:
            Output.Denoise=( Opt.DenWeight/ (Output.l0/Opt.NmPP )); # 
        else:
            Output.Denoise=Opt.DenWeight
    #
        LDenArray = skimage.restoration.denoise_tv_bregman(ArrayIn,Output.Denoise ) # smaller = more denoise
        LDenArray *= 255
        
        ArrayIn=LDenArray
        
        LDenImage=Image.fromarray(LDenArray)
        LDenImage=LDenImage.convert(mode="RGB")
        
        if Opt.DenSh == 1:
            LDenImage.show()
        if Opt.DenSa == 1:   
            LDenImage.save(os.path.join(FPath,"output",BName + "LDen.tif"))
        
        
    #    CLDenArray = skimage.restoration.denoise_tv_chambolle(ArrayIn, Output.Denoise ) # Larger = more denoise
    #    ArrayIn=CLDenArray    
    #    
    #    CLDenImage=Image.fromarray(CLDenArray)
    #    CLDenImage=CLDenImage.convert(mode="RGB")
    #    
    #    if ShowImage == 1:
    #        CLDenImage.show()
    #        
    #    CLDenImage.save(os.path.join(FPath,"output",BName + "CLDen.tif"))
    
    ##%% Otsu Thresholding (Not used, but can be shown for comp)
    #
    #if Opt.DenToggle==1:
    #    ArrayIn=LDenArray
    #elif Opt.RSToggle==1:
    #    ArrayIn=RSArray
    #else:
    #    ArrayIn=CropArray
    #
    #OtsuV=skimage.filters.threshold_otsu(ArrayIn)
    #OtsuBin= CropArray <= OtsuV
    #
    #OtsuThresh = Image.fromarray(100*np.uint8(OtsuBin))
    #OtsuThresh=OtsuThresh.convert(mode="RGB")
    #OtsuThresh.show()
    #OtsuThresh.save(os.path.join(FPath,"output",BName + "LOtsu.tif") )
    
    #%% Adaptive Local Thresholding over 15 pixels, gauss
    if Opt.ThreshToggle==1:
        if Opt.AutoThresh==1:    
            Output.Thresh=Opt.ThreshWeight*(Output.l0/Opt.NmPP )
            Output.Thresh=np.floor( Output.Thresh )
            Output.Thresh=np.max( (Output.Thresh, 1))
        else:
            Output.Thresh=Opt.ThreshWeight
            
        LAdaptBin=skimage.filters.threshold_adaptive(ArrayIn,Output.Thresh ,'gaussian')
        ArrayIn=LAdaptBin;
        
        LAdaptThresh = Image.fromarray(100*np.uint8(LAdaptBin))
        LAdaptThresh=LAdaptThresh.convert(mode="RGB")
        if Opt.ThreshSh == 1:
            LAdaptThresh.show()
        if Opt.ThreshSa==1:
            LAdaptThresh.save(os.path.join(FPath,"output",BName+"LAThresh.tif"))
    
    
    
    
    
    #%% Small Feature Removal (May not be necessary)
    
    if Opt.SFRToggle==1:
        LAdRSO = skimage.morphology.remove_small_objects(ArrayIn, Opt.SPCutoff)
        ArrayIn=LAdRSO
        LAdRSOI = Image.fromarray(100*np.uint8(LAdRSO)).convert(mode="RGB")
        if Opt.RSOSh == 1:
            LAdRSOI.show()
        if Opt.RSOSa==1:
            LAdRSOI.save(os.path.join(FPath,"output",BName+"LADRSO.tif"))
        
        LDPFrac=(LAdRSO==0).sum()
        LLPFrac=(LAdRSO.size-LDPFrac)
    #%% Feature Finding
    
    
    if Opt.LabelToggle==1:
        LALab, LNumFeat = scipy.ndimage.measurements.label(ArrayIn)
        LDomFrac=(LALab==1).sum()/(LLPFrac)
        LDomI=1
        for i in range(2,LNumFeat):
            TestFrac=(LALab==i).sum()/(LLPFrac)
            if TestFrac > LDomFrac:
                LDomFrac=TestFrac
                LDomI=i
                
        #print("Dominant index %d is %f of total" % (LDomI, LDomFrac))
        LDomMask= ( LALab==LDomI )*255;
        LDomMaskI=Image.fromarray(LDomMask)
        LDomMaskI=LDomMaskI.convert(mode="L")
    
        LALabI=scipy.misc.toimage(LALab).convert(mode="RGB") 
        LADomCI=Image.composite(RImage,Image.fromarray(100*np.uint8(ArrayIn)).convert(mode="RGB"),LDomMaskI)
        LALabDomCI=Image.composite(RImage,LALabI,LDomMaskI)
        if Opt.LabelSh == 1:        
            LALabI.show()
            LADomCI.show()
            LALabDomCI.show()
        if Opt.LabelSa == 1:
            LALabI.save(os.path.join(FPath,"output",BName+"LLab.tif"))
            LADomCI.save(os.path.join(FPath,"output",BName+"LDomC.tif"))
            LALabDomCI.save(os.path.join(FPath,"output",BName+"LLabDomC.tif"))
        
    
    #%% Skeletonization 
    
    if Opt.SkeleToggle==1:
        LASkel = skimage.morphology.skeletonize(ArrayIn)
        
        LASkelI= Image.fromarray(100*LASkel)
        LASkelI=LASkelI.convert(mode="RGB")
        if Opt.SkeleSh == 1:
            LASkelI.show()
        if Opt.SkeleSa==1:
            LASkelI.save(os.path.join(FPath,"output",BName+"LSkel.tif"))
    
    #%% Terminal/Junction finder
    
        LAdCount=scipy.signal.convolve(LASkel, np.ones((3,3)),mode='same')
        # Remove Opt.DefEdge pixels at edge to prevent edge effects. be sure to account for area difference
        
        LAdCount[0:int(Opt.DefEdge-1),:]=0; LAdCount[int(CIMH+1-Opt.DefEdge):int(CIMH),:]=0; 
        LAdCount[:,0:int(Opt.DefEdge-1)]=0; LAdCount[:,int(CIMW+1-Opt.DefEdge):int(CIMW)]=0; 
        DefArea=( CIMW-2*Opt.DefEdge)*( CIMH-2*Opt.DefEdge)*Opt.NmPP*Opt.NmPP; # Area in nm^2
        
        # Terminal
        LTLog = ((LAdCount==2) * (LASkel == 1)) # if next to 1 + on skel
        LTCount = (LTLog==1).sum()
        LTCA=LTCount/DefArea
        LTLog = scipy.signal.convolve(LTLog, np.ones((3,3)),mode='same')
        
        
        LASkelT= Image.fromarray(30*LASkel+100*LTLog)
        if Opt.SkeleSh == 1:
            LASkelT.show()
        if Opt.SkeleSa==1:
            LASkelT.save(os.path.join(FPath,"output",BName+"LASkelTerm.tif"))
        
        # Junctions
        
        LJLog = ((LAdCount > 3) * (LASkel == 1)) # if next to >2 + on skel
        
        LSkelAC = LASkel-LJLog # Pruned Skel to use for autocorrelation
        
        LJCount = (LJLog==1).sum()
        LJCA=LJCount/DefArea
        LJLog = scipy.signal.convolve(LJLog, np.ones((3,3)),mode='same')
        
        
        LASkelJ= Image.fromarray(30*LASkel+100*LJLog)
        if Opt.SkeleSh == 1:
            LASkelJ.show()
        if Opt.SkeleSa==1:
            LASkelJ.save(os.path.join(FPath,"output",BName+"LASkelJunc.tif"))
    
    #%% Autocorrel. LETS GO
    if Opt.ACToggle==1:
    
        class AutoCor:
            pass
        # Struct Tens
    #    LStructTen=skimage.feature.structure_tensor(CropArray, sigma=1) # Take original image and calc derivatives
    #    
    #    LAng=np.arctan2(LStructTen[2],LStructTen[0]) # use arctan dy/dx to find direction of line Rads
    #    LAngS=LAng*LSkelAC #Mask out with Skeleton
        LC1stDy = scipy.ndimage.sobel(LDenArray, axis=0, mode='constant', cval=0)
        LC1stDx = scipy.ndimage.sobel(LDenArray, axis=1, mode='constant', cval=0) 
        
        LAngD=np.float32(np.arctan2(LC1stDy,LC1stDx))
        LAnGDS=LAngD*LASkel
    
        
        
        """
        Note that angles are 0<->pi/2 will use trick later to correct for  
        """
        
        AutoCor.SkI, AutoCor.SkJ=np.nonzero(LASkel); #Get indexes of nonzero try LASkel/LSKelAC
        AutoCor.n=np.zeros(Opt.ACCutoff)
        
        AutoCor.RandoList=np.random.choice(len(AutoCor.SkI), len(AutoCor.SkI), replace=False)
        AutoCor.h=np.zeros(Opt.ACCutoff)
        AutoCor.Indexes=np.array([[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]) # for picking nearby
        AutoCor.IndAngles=np.array([135,90,45,180,0,-135,-90,-45]) #angle of the above in degrees
        AutoCor.IndAngles=AutoCor.IndAngles*np.pi/180 # radians
        AutoCor.Ind=0;
        
        while AutoCor.Ind < Opt.ACSize : # How many points to start at to calc auto correlate
            # The following is the AutoCor Loop
            AutoCor.ntemp=np.zeros(Opt.ACCutoff) # How many times have calculated the n=Index+1 correlation?
            AutoCor.htemp=np.zeros( Opt.ACCutoff ) # what is the current sum value of the correlation(divide by ntemp at end)
            AutoCor.angtemp=np.zeros(Opt.ACCutoff+1) # What is the current angle, 1 prev angle, etc etc
            AutoCor.BBI = 0 # not necessary but helpful to remind us start = BBI 0
            AutoCor.SAD=0;
            #First pick a point, find it's angle
            AutoCor.CCOORD=[AutoCor.SkI[AutoCor.RandoList[AutoCor.Ind]] ,
                            AutoCor.SkJ[AutoCor.RandoList[AutoCor.Ind]] ]
            
            AutoCor.angtemp[0]=LAnGDS[ tuple(AutoCor.CCOORD) ]
            
            AutoCor.BBI=1 #now we at first point... 
            AutoCor.PastN=9 # No previous point to worry about moving back to
                          
            while AutoCor.BBI <= 2*(Opt.ACCutoff): # How far to walk BackBoneIndex total points is 2*Cuttoff+1 (1st point)
                np.roll(AutoCor.angtemp,1) # now 1st angle is index 1 instead of 0 etc
                #what is our next points Cooard?
                print('%F' % AutoCor.BBI)
                AutoCor.WalkDirect=np.random.choice(8,8,replace=False) # pick a spot to move
                for TestNeighbor in np.arange(8):
                    AutoCor.COORD=AutoCor.Indexes[AutoCor.WalkDirect[TestNeighbor]]+AutoCor.CCOORD
                    if np.array( (AutoCor.COORD < LASkel.shape) ).all(): # If we are still in bounds
                        if (LASkel[ tuple(AutoCor.COORD)] == 1 & AutoCor.WalkDirect[TestNeighbor] != 7-AutoCor.PastN): # if we have a valid move
                            if AutoCor.BBI==1: # And its the first move we need to fix 1st angle
                                if AutoCor.angtemp[1] <=0: # if angle is neg
                                    if( np.abs( (AutoCor.angtemp[1]+np.pi)-AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]]) <=
                                np.abs(AutoCor.angtemp[1]-AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]])):
                                        #is angle + pi closer?
                                        AutoCor.angtemp[1]+=np.pi;
                                else: # if angle is postive
                                    if( np.abs( (AutoCor.angtemp[1]-np.pi)-AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]]) <=
                                np.abs(AutoCor.angtemp[1]-AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]])):
                                        #is angle + pi closer?
                                        AutoCor.angtemp[1]+=np.pi;
                            AutoCor.PastN=AutoCor.WalkDirect[TestNeighbor];
                            del TestNeighbor;
                            AutoCor.CCOORD=AutoCor.COORD; # move there
                            AutoCor.angtemp[0]=LAnGDS[tuple(AutoCor.CCOORD)] # set angle to new angle
                            break # break the for loop
                else:
                    # Need to break out of the backbone loop as well...
                    AutoCor.SAD=1; # because
                    del TestNeighbor
                        
                if AutoCor.SAD==1:
                    # Decide if I count this or not...
                    AutoCor.SAD=0;
                    break
                
                # BUT WAIT WE NEED TO FIX THE NEW ANGLE TOO! Keep it within pi of previous
                if AutoCor.angtemp[0] <=0 and ( np.abs( (AutoCor.angtemp[0]+np.pi)-AutoCor.angtemp[1]) <=
                np.abs(AutoCor.angtemp[0]-AutoCor.angtemp[1])):
                        #is angle + pi closer?
                        AutoCor.angtemp[0]+=np.pi;
                elif AutoCor.angtemp[0] > 0 and (np.abs( (AutoCor.angtemp[0]-np.pi)-AutoCor.angtemp[1]) <=
                np.abs(AutoCor.angtemp[0]-AutoCor.angtemp[1])):
                        #is angle + pi closer?
                        AutoCor.angtemp[1]+=np.pi;           
                            
    #            print(np.array_str(AutoCor.CCOORD))        
                for AutoCor.PI in range (0,Opt.ACCutoff): # Persistance Index, 0 = 1 dist etc
                    #Calculating autocorrelation loop
                    if (AutoCor.BBI > 0 & AutoCor.BBI%(AutoCor.PI+1)==0):
                        
                        AutoCor.htemp[AutoCor.PI]+=np.cos(AutoCor.angtemp[0]-AutoCor.angtemp[AutoCor.PI+1]) # dotproduct is cos
                        AutoCor.ntemp[AutoCor.PI]+=1
                
                        
                #FinD next point
                AutoCor.BBI+=1
                
                if AutoCor.BBI==2*(Opt.ACCutoff): # we found all our points!
                    AutoCor.h +=AutoCor.htemp
                    AutoCor.n +=AutoCor.ntemp
                    AutoCor.Ind += 1
            
            
    
    #%% Find the inverse or 'Dark' Image repeat as above
    
    DCropArray=255-CropArray;
    ArrayIn=DCropArray;
    if Opt.RSToggle==1:
        DRSArray=255-RSArray;
        ArrayIn=DRSArray;
        
    #%% Denoise Dark
    
    
    if Opt.DenToggle==1:
            
        DDenArray = skimage.restoration.denoise_tv_bregman(ArrayIn,Output.Denoise)
        DDenArray *= 255
        ArrayIn=DDenArray    
        
        DDenImage=Image.fromarray(DDenArray)
        DDenImage=DDenImage.convert(mode="RGB")
        if Opt.DenSh == 1:
            DDenImage.show()
        if Opt.DenSa==1:
            DDenImage.save(os.path.join(FPath,"output",BName + "DDen.tif"))
    
    
    #%% Adaptive Local Thresholding over 15 pixels, gauss
    if Opt.ThreshToggle==1:
        DAdaptBin=skimage.filters.threshold_adaptive(ArrayIn,Output.Thresh,'gaussian')
        ArrayIn=DAdaptBin;
        DAdaptThresh = Image.fromarray(100*np.uint8(DAdaptBin))
        DAdaptThresh=DAdaptThresh.convert(mode="RGB")
        if Opt.ThreshSh == 1:
            DAdaptThresh.show()
        if Opt.ThreshSa==1:
            DAdaptThresh.save(os.path.join(FPath,"output",BName+"DAThresh.tif"))
    
    #%% Small Feature Removal (May not be necessary)
    if Opt.SFRToggle==1:
        DAdRSO = skimage.morphology.remove_small_objects(ArrayIn, Opt.SPCutoff)
        ArrayIn=DAdRSO;
        DAdRSOI = Image.fromarray(100*np.uint8(DAdRSO))
        DAdRSOI=DAdRSOI.convert(mode="RGB")
        if Opt.RSOSh == 1:
            DAdRSOI.show()
        if Opt.RSOSa==1:
            DAdRSOI.save(os.path.join(FPath,"output",BName+"DADRSO.tif"))
        
        DLPFrac=(DAdRSO==0).sum()
        DDPFrac=(DAdRSO.size-DLPFrac)
    #%% Feature Finding
    if Opt.LabelToggle==1:
        DALab, DNumFeat = scipy.ndimage.measurements.label(ArrayIn)
        
        DDomFrac=(DALab==1).sum()/(DDPFrac)
        DDomI=1;
        for i in range(2,DNumFeat):
            TestFrac=(DALab==i).sum()/(DDPFrac)
            if TestFrac > DDomFrac:
                DDomFrac=TestFrac
                DDomI=i
                
    
                
        #print("Dominant dark index %d is %f of total" % (DDomI, DDomFrac))
        DDomMask= ( DALab==DDomI )*255;
        DDomMaskI=Image.fromarray(DDomMask)
        DDomMaskI=DDomMaskI.convert(mode="L")
        
    
        
        DALabI=scipy.misc.toimage(DALab).convert(mode="RGB")
        DADomCI=Image.composite(RImage,Image.fromarray(100*np.uint8(ArrayIn)).convert(mode="RGB"),DDomMaskI)
        DALabDomCI=Image.composite(RImage,DALabI,DDomMaskI)
        if Opt.LabelSh == 1:
            DALabI.show()
            DADomCI.show()
            DALabDomCI.show()
        if Opt.LabelSa==1:
            DALabI.save(os.path.join(FPath,"output",BName+"DLab.tif"))
            DADomCI.save(os.path.join(FPath,"output",BName+"DDomC.tif"))
            DALabDomCI.save(os.path.join(FPath,"output",BName+"DLabDomC.tif"))
            
    
    #%% Skeletonization 
    if Opt.SkeleToggle==1:
        DASkel = skimage.morphology.skeletonize(DAdaptBin)
        
        DASkelI= Image.fromarray(100*DASkel)
        DASkelI=DASkelI.convert(mode="RGB")
        if Opt.SkeleSh == 1:
            DASkelI.show()
        if Opt.SkeleSa==1:
            DASkelI.save(os.path.join(FPath,"output",BName+"DSkel.tif"))
        
        #%% Terminal/Junction finder
        
        DAdCount=scipy.signal.convolve(DASkel, np.ones((3,3)),mode='same')
        # Remove Opt.DefEdge pixels at edge to prevent edge effects. be sure to account for area difference
        
        DAdCount[0:Opt.DefEdge-1,:]=0; DAdCount[CIMH+1-Opt.DefEdge:CIMH,:]=0; 
        DAdCount[:,0:Opt.DefEdge-1]=0; DAdCount[:,CIMW+1-Opt.DefEdge:CIMW]=0; 
        DefArea=( CIMW-2*Opt.DefEdge)*( CIMH-2*Opt.DefEdge)*Opt.NmPP*Opt.NmPP; # Area in nm^2, don't have to do this again but it makes code readable
        
        # Terminal
        DTLog = ((DAdCount==2) * (DASkel == 1)) # if next to 1 + on skel
        DTCount = (DTLog==1).sum()
        DTCA=DTCount/DefArea
        DTLog = scipy.signal.convolve(DTLog, np.ones((3,3)),mode='same')
        
        
        DASkelT= Image.fromarray(30*DASkel+100*DTLog)
        if Opt.SkeleSh == 1:
            DASkelT.show()
        if Opt.SkeleSa==1:
            DASkelT.save(os.path.join(FPath,"output",BName+"DASkelTerm.tif"))
        
        # Junctions
        
        DJLog = ((DAdCount > 3) * (DASkel == 1)) # if next to >2 + on skel
        DJCount = (DJLog==1).sum()
        DJCA=DJCount/DefArea
        DJLog = scipy.signal.convolve(DJLog, np.ones((3,3)),mode='same')
        
        DASkelJ= Image.fromarray(30*DASkel+100*DJLog)
        if Opt.SkeleSh == 1:
            DASkelJ.show()
        if Opt.SkeleSa==1:
            DASkelJ.save(os.path.join(FPath,"output",BName+"DASkelJunc.tif"))
    
    #%% Logging
    
    #DDomI=0;DDomFrac=0; DTerm=0; DJunct=0;
    #if IndividualLog==1:
    #    with open(FName+"output.csv", 'w') as SampleLog:
    #        SampLogW= csv.writer(SampleLog, dialect='excel', lineterminator='\n')
    #        SampLogW.writerow(['Filename',BName])
    #        
    #        SampLogW.writerow(['Lighter Phase'])
    #        SampLogW.writerow(['LPhase Area Fraction',LLPFrac])
    #        SampLogW.writerow(['DPhase Area Fraction',LDPFrac])
    #        SampLogW.writerow(['LDom Index',LDomI])
    #        SampLogW.writerow(['LDom Fraction',LDomFrac])
    #        SampLogW.writerow(['LTerminals',LTCount])
    #        SampLogW.writerow(['LTerminals/nm^2',LTCA])
    #        SampLogW.writerow(['LJunctions',LJCount])
    #        SampLogW.writerow(['LJunctions/nm^2',LJCA])
    #        SampLogW.writerow(['Inverse Phase Image (Names reference original!)'])
    #        SampLogW.writerow(['DDom Index',DDomI])
    #        SampLogW.writerow(['DDom Fraction',DDomFrac])
    #        SampLogW.writerow(['DTerminals',DTerm])
    #        SampLogW.writerow(['DJunctions',DJunct])
    
    
    # Check if a log exists, if not, but we want to log: write titles.
    # If we want to wipe the log each time, we also need to write titles
    if (os.path.isfile(os.path.join(FPath, "output", "output.csv"))==False and CombLog == 1) or CombLog == 2:
        with open(os.path.join(FPath, "output", "output.csv"), 'w') as Log:
            LogW= csv.writer(Log, dialect='excel', lineterminator='\n')
            LogW.writerow(['Filename',
            'Primary Peak (nm)',
            'Lighter',
            'LPhase Area Fraction',
            'LPhase Area Fraction(FromINV)',
            'LPhase Area Frac AVG',
            'DPhase Area Fraction',
            'DPhase Area Fraction(FromINV)',
            'DPhase Area Frac AVG',
            'LDom Index',
            'LDom Fraction',
            'LTerminals',
            'LTerminals/nm^2',
            'LJunctions',
            'Ljunctions/nm^2',
            'Inverse Phase Images',
            'DDom Index',
            'DDom Fraction',
            'DTerminals',
            'DTerminals/nm^2',
            'DJunctions',
            'DJunctions/nm^2',
            'Denoise',
            'Threshold'])
    
    if CombLog > 0:
        with open(os.path.join(FPath, "output", "output.csv"), 'a') as Log:
            LogW= csv.writer(Log, dialect='excel', lineterminator='\n')
            LogW.writerow([BName,
            Output.l0,
            'Phase',
            LLPFrac/LAdRSO.size,
            DLPFrac/DAdRSO.size,
            (LLPFrac/LAdRSO.size+DLPFrac/DAdRSO.size)/2,
            LDPFrac/LAdRSO.size,
            DDPFrac/DAdRSO.size,
            (LDPFrac/LAdRSO.size+DDPFrac/DAdRSO.size)/2,
            LDomI,
            LDomFrac,
            LTCount,
            LTCA,
            LJCount,
            LJCA,
            '(Names reference original!)',
            DDomI,
            DDomFrac,
            DTCount,
            DTCA,
            DJCount,
            DJCA,
            Output.Denoise,
            Output.Thresh])
        
    
    
        
        