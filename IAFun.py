# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:29:51 2016

@author: Moshe Dolejsi MosheDolejsi@uchicago.edu
"""
#%%
Vers="0.1"

#%%
from PIL import Image
import lmfit
from tkinter import messagebox
import os
import csv
import numpy as np
import pandas as pd
import skimage
from skimage import restoration, morphology, filters, feature
import exifread #needed to read tif tags
try:
    from igor.binarywave import load as loadibw
except: print('You will be unable to open Asylum data without igor')
import re #dat regex
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import scipy


#%%
def AFMPara(RawIn,Opt,FiltOut,ThreshOut,AdOut,SkelOut, EDOut, AngOut, ii):
    """Perform the Denoising thresholding and skeletonization in a parallel compatible manner"""
    try:
        if Opt.NmPPSet!=0:Opt.NmPP=Opt.NmPPSet # so if we set one then just set it
    except:
        pass

#    RawIn[:,:,ii]=AutoDetect( FNFull[ii], Opt) # moved to preloop due to library incompatability
    if Opt.DenToggle == 1:
        FiltOut[:,:,ii] = Denoising(RawIn[:,:,ii], Opt, 50)[0]
    if Opt.BPToggle == 1:
        FiltOut[:,:,ii] = BPFilter(FiltOut[:,:,ii],Opt.NmPP,LW=100,Axes='x') #FFT Filtering
        FiltOut[:,:,ii] = BPFilter(FiltOut[:,:,ii],Opt.NmPP,HW=500,Axes='y') #FFT Filtering
    #ArrayIn = np.multiply(ArrayIn,Mask) #mask in loop

    ThreshOut[:,:,ii] = FiltOut[:,:,ii] > 11

    
    #Thresh = IAFun.Thresholding(ArrayIn, Opt, 50)[0]
    if Opt.SkeleToggle == 1:
        SkelOut[:,:,ii] = skimage.morphology.skeletonize(ThreshOut[:,:,ii])
    if Opt.EDToggle == 1:
        EDOut[:,:,ii] = (ThreshOut[:,:,ii]-skimage.morphology.binary_erosion(ThreshOut[:,:,ii], np.ones((3,3))))
    
    if Opt.AngDetToggle==3:
        AngOut[:,:,ii] = AngMid( ThreshOut[:,:,ii], Opt, SkelArray = SkelOut[:,:,ii])
    if Opt.AngDetToggle==2:
        AngOut[:,:,ii] = AngSobel( FiltOut[:,:,ii] ) # old method
    if Opt.AngDetToggle==1:
        AngOut[:,:,ii] = AngEC(ThreshOut[:,:,ii], Opt, EDArray=EDOut[:,:,ii], SkelArray=SkelOut[:,:,ii])          # new method
    
    
    AdOut[:,:,ii] = scipy.signal.convolve(SkelOut[:,:,ii], np.ones((3,3)),mode='same',method='direct').astype('i1')
    AdOut[:,:,ii] = np.multiply(AdOut[:,:,ii], SkelOut[:,:,ii])
    




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
            StrInt = Opt.FInfo.find('\\n') # find first field
            StrInt = Opt.FInfo.find('\\n', StrInt + 1) #find second
            StrInt = Opt.FInfo.find('\\n', StrInt + 1) #find third (we want NEXT one!)
            StrStart = StrInt + 2
            StrInt = Opt.FInfo.find('\\n', StrInt + 1) # find end of one we want
            StrEnd = StrInt - 2
            Opt.NmPP=float(Opt.FInfo[StrStart:StrEnd])*10**9;
            Opt.Machine="Merlin";
        except:
            pass


#    if Opt.NmPP!=0:
#        print("Instrument was autodetected as %s, NmPP is %f \n" % (Opt.Machine ,Opt.NmPP) )
#    else:
#        print("Instrument was not detected, and NmPP was not set. Please set NmPP and rerun")
    return(imarray);
#%% Croparayy = IAFun.bla( input)
    """
    Crops image
    V0.1
    """

def Crop( imarray , Opt ):

    
    # Crop for zeiss as below
    # 768 : 718 : 50
    # 3072 : 2875 : ~200
    (IMH, IMW) =imarray.shape
    CropArray=imarray[int(0+Opt.CropT):int(IMH-Opt.CropB),int(Opt.CropL):int(IMW-Opt.CropR)]
    (CIMH,CIMW)=CropArray.shape
    return (CropArray, CIMH, CIMW);

#%% Do a time lagged cross correlation based off stackoverflow.com/questions/33171413/
    
def df_shifted(df, target=None, lag=0):
    if not lag and not target:
        return df       
    new = {}
    for c in df.columns:
        if c == target:
            new[c] = df[target]
        else:
            new[c] = df[c].shift(periods=lag)
    return  pd.DataFrame(data=new)

#%% Flood Fill
    """
    We need to pass this function an image array in which 0 is the area not yet filled
    We may need to arbitrarily offset the original data (EG instead of 0-90 we feed in 90-180 and then substract 90 after)
    Might consider adding a mask function, but currently is not in use so it is not important
    V=0.1
    """
def FFill(imarray, size_=(3,3)):
    lcount = 0; # going to keep track of loop number. if it gets too high we need to stop
    while imarray.min() == 0 & lcount < 500:
        lcount += 1
        FillArray=scipy.ndimage.morphology.grey_dilation(imarray, size=size_ )
        imarray+=FillArray*(imarray==0)
    return(imarray)
        
    
        
#%% YKMDetect
def YKDetect(image, Opt):
    """
    Tries to detect IDE areas by looking at sudden changes in derivatives 
    it works ok currently but needs improvement prior to deployment
    V:PreAlpha
    """
    class Ide:
        pass
    
    # Hardcoded cus reasons
    VWidth=np.arange( np.floor(20/Opt.NmPP) , np.floor(30/Opt.NmPP) )  # Vertical lines are narrow (10-15 pixels)
    HWidth=np.arange( np.floor(200/Opt.NmPP) , np.floor(300/Opt.NmPP) ) # horizontal lines are large. (>100 pixels)
    Pdist=20 # Real peaks ought to be > 20 nm apart
    Pdist=np.floor(Pdist/Opt.NmPP) # convert to pixels
   
    
#    Ide.DY=scipy.ndimage.sobel(image, axis=0)
#    Ide.DX=scipy.ndimage.sobel(image, axis=1)
    
    # These values seem optimum for a nice image, but can obviously be adjusted to suit
#    Ide.DYIm=Image.fromarray(np.absolute(Ide.DY)*10/np.absolute(Ide.DY).max())
#    Ide.DXIm=Image.fromarray(np.absolute(Ide.DX)*10/np.absolute(Ide.DX).max())
    
    Ide.SchArray=skimage.filters.scharr(image)
    # Sum along rows
    Ide.SchX=np.average(Ide.SchArray,0) # Each element is average of a column
    Ide.SchY=np.average(Ide.SchArray, 1) # Each element is average of a row
    
    Ide.VEdge=scipy.signal.find_peaks_cwt(Ide.SchX, VWidth) # find peaks
    Ide.HEdge=scipy.signal.find_peaks_cwt(Ide.SchY, HWidth)
    # Remove close peaks multithread soon ^_-
    Rem=np.empty(0)
    for i in range(1, len(Ide.VEdge)-1):
        if Ide.VEdge[i]-Ide.VEdge[i-1] < Pdist: # if points too close
            if Ide.SchX[Ide.VEdge[i]] < Ide.SchX[Ide.VEdge[i-1]]: # remove the point with lower mag
                Rem=np.append(Rem,i)
            else:
                Rem=np.append(Rem,i-1)
    try: #we may not have any extra points to clean up
        Ide.VEdge.remove(Rem)
    except:
        pass
    Rem=np.empty(0)
    for i in range(1, len(Ide.HEdge)-1):
        if Ide.HEdge[i]-Ide.HEdge[i-1] < Pdist*5: # horizontal lines are > 100 nm
            if Ide.SchX[Ide.HEdge[i]] < Ide.SchX[Ide.HEdge[i-1]]:
                Rem=np.append(Rem,i)
            else:
                Rem=np.append(Rem,i-1)
    try: #we may not have any extra points to clean up
        Ide.HEdge.remove(Rem)
    except:
        pass
    # Default behavior is after 1st peak is first zone 
    # so 1-2 = real, 3-4 = real 2-3 = background etc etc
    # future work is to automate this. Dunno how atm
    
    Ide.Mask=np.ones(image.shape)
    
    # this part also crops a bit    
    Ide.Mask[0:Ide.HEdge[0]+2*Pdist,:]=0 # Cut top
    Ide.Mask[Ide.HEdge[1]-2*Pdist:,:]=0 # bottom
    
    
    Ide.Mask[:,Ide.VEdge]=0 # cut lines where the IDE edges are
    if Ide.VEdge[0]< Pdist:
        Ide.Mask[:,0:Ide.VEdge[0]]=0 # Get rid of fluff at start of row region <~ 10 nm
    if image.shape[1]-Ide.VEdge[-1] < Pdist:
        Ide.Mask[:,Ide.VEdge[-1]:]=0 # get rid of last region if width is < 10 nm
    
    Ide.LabMask, Ide.MaskCnt = scipy.ndimage.measurements.label(Ide.Mask)
    Ide.RMask=np.mod(Ide.LabMask,2) # tag alternating rectangles as 1
    Ide.BMask=(1-Ide.RMask)*Ide.Mask # Tag the other alternating rectangles
    # Make images of masks for compositing
    Ide.RMaskI=Image.fromarray(50*Ide.RMask).convert(mode="L") # note the 50 controls the transparency of the color
    Ide.BMaskI=Image.fromarray(50*Ide.BMask).convert(mode="L")

    Ide.RImage=Image.new('RGB',Ide.RMaskI.size,'Red')
    Ide.BImage=Image.new('RGB',Ide.BMaskI.size,'Blue')
    
    Ide.MImage=scipy.misc.toimage(image).convert(mode="RGB") 
    Ide.CImage=Image.composite(Ide.RImage,Image.fromarray(image).convert(mode="RGB"),Ide.RMaskI)
    Ide.CImage=Image.composite(Ide.BImage,Ide.CImage,Ide.BMaskI)
    
    Ide.CImage.show()
    Ide.RTog=messagebox.askyesno(title="IDE Select",message="Is the IDE RED")
    if Ide.RTog==1:
        Ide.Mask=Ide.RMask
    else:
        Ide.Mask=Ide.BMask
    
#    WDomCI=Image.composite(RImage,Image.fromarray(100*np.uint8(image)).convert(mode="RGB"),WDomMaskI)
#    WLabDomCI=Image.composite(RImage,WLabI,WDomMaskI)
    
#    WDomMaskI=Image.fromarray(WDomMask).convert(mode="L")
    
    
#    Ide.SchIm=Image.fromarray( Ide.SchArray*100)
    
    #% Add toggle here for show/save
    #DYIm.show();DXIm.show();LapIm.show();
    
#    DXIm.save(os.path.join(Opt.FPath,"output",Opt.FName+"DX.tif"))
#    DYIm.save(os.path.join(Opt.FPath,"output",Opt.FName+"DY.tif"))
#    SchIm.save(os.path.join(Opt.FPath,"output",Opt.FName+"Sch.tif"))
    return(Ide)

#%% Azimuthal Averaging
def azimuthalAverage(image, center=None, angle=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    http://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    v0.1
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    r = r.astype(np.int)
    
    if not angle:
        Weight = image.ravel()
    else:
        AngMask = np.abs(np.arctan( ( y - center[1] )/(x - center[0] ))) * 180/np.pi
        Weight = (image*(AngMask < angle)).ravel()
    
    tbin = np.bincount(r.ravel(), Weight)
    nr = np.bincount(r.ravel())

    radial_prof = tbin / nr

    return radial_prof

#%%
def FFT( im, Opt):
    """
    Calculate the FFT, and the PSD to find l0
    
    TODO: make peak finding more robust, 
    v0.2
    """
    
    FSize=np.min( im.shape )
    #FSize=500;
    FourierArray=np.fft.fft2( im, s=(FSize,FSize) );
    FreqA=np.fft.fftfreq(FSize, d=Opt.NmPP);
    #
    F2Array=np.fft.fftshift(FourierArray);    
    SpaceA=1/FreqA;
    PowerSpec2d= np.abs( F2Array )**2;
    
    PowerSpec1d= azimuthalAverage(PowerSpec2d);
    
    Peak=scipy.signal.find_peaks_cwt(PowerSpec1d[0:int( np.floor(FSize/2))], np.arange(5,10),);  
    PFreq=np.zeros(np.size(Peak))
    Pspace=np.zeros(np.size(Peak));
    PHeight=np.zeros(np.size(Peak));
    for i in range(0, np.size(Peak)):
        PFreq[i]=FreqA[Peak[i]]
        Pspace[i]=1/FreqA[Peak[i]]
        PHeight[i]=PowerSpec1d[Peak[i]]
        #print("Peak %d found at %f \n" % (i, Pspace[i]))
    if Peak[0] < 4: # if first peak is found at L= infty
        PHeight[0]=0; # dont consider it for characteristic peak

    
    
    PHMax=PHeight.max()
    PFMax=(PFreq*(PHMax==PHeight)).max()
    PSMax=1/PFMax;
    # Now save plots
    if Opt.FFTSh==1 or Opt.FFTSa==1:
        Fig=plt.figure()
        PSD1D=Fig.add_subplot(111)
        PSD1D.plot(FreqA[0:int(np.floor(FSize/2))], PowerSpec1d[0:int( np.floor(FSize/2))])
        PSD1D.set_yscale('log')
        PSD1D.set_title('1D Power Spectral Density')
        PSD1D.set_xlabel('Frequency (1/nm)') # NOTE THIS IS NOT Q
        PSD1D.set_ylabel('Intensity')
        PSD1D.set_ylim([np.min(PowerSpec1d)*.5, np.max(PowerSpec1d)*10])          
        PS2DImage=Image.fromarray(255/np.max(np.log(PowerSpec2d))*np.log(PowerSpec2d))
        PS2DImage=PS2DImage.convert(mode="RGB")
        if Opt.FFTSh == 1:
            PS2DImage.show()
        if Opt.FFTSa==1:
            np.savez(os.path.join(Opt.FPath,"output",Opt.FName + "FFT"),(FreqA[0:int(np.floor(FSize/2))], PowerSpec1d[0:int(np.floor(FSize/2))]))
            Fig.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "PowerSpecFreq.png"))
            PSD1D.annotate('Primary Peak at %f' %PFMax, xy=(PFMax, PHMax), xytext=(1.5*PFMax, 1.5*PHMax),
                    arrowprops=dict(facecolor='black', width=2,headwidth=5),
                    )
            Fig.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "PowerSpecFreqLabel.png"))
            PS2DImage.save(os.path.join(Opt.FPath,"output",Opt.FName + "PowerSpec2d.tif"))
    return(PSMax);
#%%
def FFTAlignment( im, Opt):
    FSize=np.min( im.shape )
    #FSize=500;
    FourierArray=np.fft.fft2( im, s=(FSize,FSize) );
    FreqA=np.fft.fftfreq(FSize, d=Opt.NmPP);
    #
    F2Array=np.fft.fftshift(FourierArray);    
    SpaceA=1/FreqA;
    PowerSpec2d= np.abs( F2Array )**2;
    
    Spec1d = azimuthalAverage(PowerSpec2d)
    SpecAlign1d = azimuthalAverage(PowerSpec2d,angle=Opt.AlignAng)
    PowerSpecAlign = SpecAlign1d/ Spec1d;
    LowCut = (1 - Opt.AlignSize)*Opt.L0
    HighCut = (1 + Opt.AlignSize)*Opt.L0
    Mask = ((SpaceA < HighCut) & (SpaceA > LowCut))[:Spec1d.size]
    PercentAlign = SpecAlign1d[Mask].sum()/Spec1d[Mask].sum()
    PercentZero = SpecAlign1d[Mask].sum()/Spec1d[0]
    if Opt.FFTSh==1 or Opt.FFTSa==1:
        Fig, Ax = plt.subplots(nrows=2)
        Ax[0].semilogy(FreqA[:int(np.floor(FSize/2))],azimuthalAverage(PowerSpec2d)[:int(np.floor(FSize/2))],FreqA[:int(np.floor(FSize/2))],azimuthalAverage(PowerSpec2d,angle=Opt.AlignAng)[:int(np.floor(FSize/2))])
        Ax[0].axvspan(1/HighCut,1/LowCut,alpha=0.2)
        Ax[1].plot(FreqA[:int(np.floor(FSize/2))],PowerSpecAlign[:int(np.floor(FSize/2))])
        Ax[1].axvspan(1/HighCut,1/LowCut,alpha=0.2)
        if Opt.FFTSh == 1:
            Fig.show()
        if Opt.FFTSa==1:
            Fig.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "Alignment.png"))
    
    return(PercentAlign, PercentZero)
#%%
def Denoising(im, Opt, l0):
    """
    Use TV Bregman to denoise the image
    v0.1
    """
    if Opt.AutoDenoise==1:
        DenoiseWeight=( Opt.DenWeight/ (l0/Opt.NmPP )); # 
    else:
        DenoiseWeight=Opt.DenWeight
    
    DenArray = skimage.restoration.denoise_tv_bregman(im,DenoiseWeight ) # smaller = more denoise
    DenArray *= 255

    
    DenImage=Image.fromarray(DenArray)
    DenImage=DenImage.convert(mode="RGB")
    
    if Opt.DenSh == 1:
        DenImage.show()
    if Opt.DenSa == 1:   
        DenImage.save(os.path.join(Opt.FPath,"output",Opt.FName + "Den.tif"))
    
    return( DenArray , DenoiseWeight )
#%%
def Thresholding(im, Opt, l0):
    """
    Adaptive Local thresholding
    returns the thresholded image and the actual weight used
    v0.1
    """
    if Opt.AutoThresh==1:    
        Thresh=Opt.ThreshWeight*(l0/Opt.NmPP )
        Thresh=2*np.floor( Thresh/2 )+1;
        Thresh=np.max( ( Thresh, 1))
    else:
        Thresh=Opt.ThreshWeight
        
    ThreshCut =skimage.filters.threshold_local(im,Thresh ,'gaussian') # thi
    
    AdaptBin = im > ThreshCut
    
    AdaptThresh = Image.fromarray(100*np.uint8(AdaptBin))
    AdaptThresh=AdaptThresh.convert(mode="RGB")
    if Opt.ThreshSh==1:
        AdaptThresh.show()
    if Opt.ThreshSa==1:
        AdaptThresh.save(os.path.join(Opt.FPath,"output",Opt.FName+"AThresh.tif"))
    return(AdaptBin,Thresh)
    
#%%
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))
#%%
def RSO(im, Opt):
    """
    Remove small Objects under size SPCutoff
    v0.1
    """
    RSO = skimage.morphology.remove_small_objects(im, Opt.SPCutoff)
    RSOI = Image.fromarray(100*np.uint8(RSO)).convert(mode="RGB")
    if Opt.RSOSh == 1:
        RSOI.show()
    if Opt.RSOSa==1:
        RSOI.save(os.path.join(Opt.FPath,"output",Opt.FName+"LADRSO.tif"))
    return(RSO);

def MSD(im, Opt):
    Crop = np.array([130, 350])
    PCount = 10 # should have 10 at each location
    
    D1 = np.gradient(im , axis = 0)
    D2 = np.gradient(D1 , axis = 0)
    
    PRough = np.zeros((im.shape[0], len(Crop)*PCount))
    Peaks = np.zeros((im.shape[0], len(Crop)*PCount))
    PWidth = np.zeros((im.shape[0], len(Crop)*PCount))
    Drift = np.zeros((im.shape[0]))
    
    for RI in np.arange(im.shape[0]): # go by row
        Max = np.where( (np.append(np.sign(D1[RI,:-1]) != np.sign(D1[RI,1:]),0))*(D2[RI,:] < 0))[0]
        if len(Max) > PCount*len(Crop):
            for i in np.arange(len(Crop)):
                PDist = np.abs(Max - Crop[i]) # find distance to peaks
                for PC in np.arange(PCount):
    
                    CPeak = np.where(PDist == PDist.min())[0]
                    if len(CPeak!=1): CPeak = CPeak[0] # if two are equidistant just pick 1
                    
                    PeakNum = PC+i*PCount
    
    
                    PRough[RI, PeakNum] = Max[CPeak] # save index of closest peak
                    PDist[CPeak] = PDist.max()       
            PRough[RI, :] = np.sort(PRough[RI, :])
            Drift[RI] = np.mean(PRough[RI]-PRough[0]) # calc drift
    
    DLin = np.polyfit(np.arange(len(Drift)),Drift,deg=1)
    DCorrect = np.polyval(DLin, np.arange(len(Drift)))
    
    for RI in np.arange(im.shape[0]):
        Peaks[RI,:] = PRough[RI,:] - DCorrect[RI]
    
    

    PMean = np.mean(Peaks, axis = 0)
    
                    
    return(Peaks)

def BPFilter(im, NmPP, LW='NA', HW='NA' , Axes='Circ'):
    """
    Bandpass filter with NmPP and Low Wavelength and Highest Wavelength given in nanometers wavelength as opposed to wavenumber
    (Bandpass filter : image, Spacing, LW='NA', HW='NA', Axes = 'Circ' (or x y))
    """
    Wind=np.max(im.shape)
    FT = np.fft.rfft2(im,s=(Wind,Wind)) # take the real fft in 2d
    FT = np.fft.fftshift(FT) # shift so 0 freq is at 0 
    WL=np.fft.rfftfreq(Wind,d=NmPP)
    y, x = np.indices(FT.shape)

    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])   
    x=np.abs(x-center[0]) # x is now dist from center
    y=np.abs(y-center[1]) # y is now dist from center
    r=np.hypot(x,y) # this gives radius in index
    r=r*WL[1] # now in freq space. Could do this before but I think hypot is quicker with int
    x=x*WL[1] # now in frequency space
    y=y*WL[1] # now in frequency space (wavenumber)
 
    LF=r.min()
    HF=r.max()
    if HW != 'NA':LF = 1/HW # lowest frequency allowed (wavenumber)
    if LW != 'NA':HF = 1/LW # Highest frequency allowed (wavenumber)
    if Axes == 'Circ': # symmetric circle filter
        FT[r<LF] = 0
        FT[r>HF] = 0
    elif Axes == 'x': # filter X (ROWS)
        FT[x<LF] = 0
        FT[x>HF] = 0
    elif Axes == 'y': # filter y (Columns)
        FT[y<LF] = 0
        FT[y>HF] = 0

    FT = np.fft.ifftshift(FT)
    Fim = np.fft.irfft2(FT,s=(Wind,Wind))
    Fim = Fim[0:im.shape[0],0:im.shape[1]]
    return(Fim)
#%% Following is Analysis as Opposed to image prep, may split in future
def Label(im, Opt):
    """
    Label connected domains, calculate area fractions of dominant domain
    Also marks domains that are present on both sides with BLUE
    v0.2
    """
    BCount=(im==0).sum()
    WCount=(im.size-BCount)
    BFrac=BCount/(im.size)
    WFrac=WCount/(im.size)
    LabArray, LNumFeat = scipy.ndimage.measurements.label(im)
    WDomFrac=(LabArray==1).sum()/(WCount)
    WDomI=1
    for i in range(2,LNumFeat):
        TestFrac=(LabArray==i).sum()/(WCount)
        if TestFrac > WDomFrac:
            WDomFrac=TestFrac
            WDomI=i
            
    #print("Dominant index %d is %f of total" % (LDomI, LDomFrac))
    WDomMask= ( LabArray==WDomI )*255;
    WDomMaskI=Image.fromarray(WDomMask).convert(mode="L")
    """
    Part 2 doing domains top.bottom
    """
    #Size=int(Opt.NmPP*10) # what is the top zone height? here 10 nm
    Size = 2 # cus lower res IDE 
    TLab=np.unique(LabArray[:Size]);BLab=np.unique(LabArray[-Size:]); # What areas are at top? Which are at bottom?   
    ThroughLab=np.intersect1d(TLab,BLab, assume_unique=True);
    ThroughLab=ThroughLab[ThroughLab!=0]; # remove zero as it is background
    ThroughMask=np.in1d(LabArray, ThroughLab).reshape(LabArray.shape)
    ThroughMaskI=Image.fromarray(ThroughMask*255).convert(mode="L")    
    
    """
    Part 3 making images
    """
    (CIMH,CIMW)=im.shape
    
    RImage=Image.new('RGB',(CIMW,CIMH),'Red')
    BImage=Image.new('RGB',(CIMW,CIMH),'Blue')
    WLabI=scipy.misc.toimage(LabArray).convert(mode="RGB")  # Labeled image
    WDomCI=Image.composite(RImage,Image.fromarray(100*np.uint8(im)).convert(mode="RGB"),WDomMaskI) # Red Dom on Original
    WLabDomCI=Image.composite(RImage,WLabI,WDomMaskI) # red dom on mask image
    WThroughCI=Image.composite(BImage,Image.fromarray(100*np.uint8(im)).convert(mode="RGB"),ThroughMaskI)
    if Opt.LabelSh == 1:        
        WLabI.show()
        WDomCI.show()
        WLabDomCI.show()
        WThroughCI.show()
    if Opt.LabelSa == 1:
        WLabI.save(os.path.join(Opt.FPath,"output",Opt.FName+"Lab.tif"))
        WDomCI.save(os.path.join(Opt.FPath,"output",Opt.FName+"DomC.tif"))
        WDomCI.save(os.path.join(Opt.FPath,"output",Opt.FName+"LabDomC.tif"))
        WThroughCI.save(os.path.join(Opt.FPath,"output",Opt.FName+"ThroughDomC.tif"))
    return(WFrac, BFrac, WDomI, WDomFrac);
    
#%%
def Skeleton(im,Opt):
    """
    Returns a skeleton array for future use, as well as performs defect analysis
    (Diagonals count as connected)
    v0.1
    """
    
    SkelArray = skimage.morphology.skeletonize(im)
    LASkelI = Image.fromarray(100*SkelArray)
    LASkelI = LASkelI.convert(mode="RGB")
    if Opt.SkeleSh == 1:
        LASkelI.show()
    if Opt.SkeleSa == 1:
        LASkelI.save(os.path.join(Opt.FPath, "output", Opt.FName+"Skel.tif"))

    AdCount = scipy.signal.convolve(SkelArray, np.ones((3,3)), mode='same', method='direct')
    # Remove Opt.DefEdge pixels at edge to prevent edge effects. be sure to account for area difference
    (CIMH,CIMW)=im.shape
    AdCount[0:int(Opt.DefEdge-1),:]=0; AdCount[int(CIMH+1-Opt.DefEdge):int(CIMH),:]=0; 
    AdCount[:,0:int(Opt.DefEdge-1)]=0; AdCount[:,int(CIMW+1-Opt.DefEdge):int(CIMW)]=0; 
    DefArea=( CIMW-2*Opt.DefEdge)*( CIMH-2*Opt.DefEdge)*Opt.NmPP*Opt.NmPP; # Area in nm^2
    
    # Terminal
    TLog = ((AdCount==2) * (SkelArray== 1)) # if next to 1 + on skel
    TCount = (TLog==1).sum()
    TCA=TCount/DefArea
    TLog = scipy.signal.convolve(TLog, np.ones((3,3)),mode='same',method='direct')
    
    
    SkelT= Image.fromarray(30*SkelArray+100*TLog)
    if Opt.SkeleSh == 1:
        SkelT.show()
    if Opt.SkeleSa==1:
        SkelT.save(os.path.join(Opt.FPath,"output",Opt.FName+"SkelTerm.tif"))
    
    # Junctions
    
    JLog = ((AdCount > 3) * (SkelArray== 1)) # if next to >2 + on skel
    
    SkelAC = SkelArray-JLog # Pruned Skel to use for autocorrelation
    
    JCount = (JLog==1).sum()
    JCA=JCount/DefArea
    JLog = scipy.signal.convolve(JLog, np.ones((3,3)),mode='same',method='direct')
    
    
    SkelJ= Image.fromarray(30*SkelArray+100*JLog)
    if Opt.SkeleSh == 1:
        SkelJ.show()
    if Opt.SkeleSa==1:
        SkelJ.save(os.path.join(Opt.FPath,"output",Opt.FName+"SkelJunc.tif"))
    
    return(SkelArray, SkelAC, TCount, TCA, JCount, JCA)
#%% Edge Detect
def EdgeDetect(im, Opt, SkelArray = 'none'):
    """
    Detects edges using morphological erosion. Then calculates LWR.
    This often requires super sampling! I may implement super sampling in this process
    v0.2 - Updated Plotting
    """
    
    if SkelArray=='none':
        SkelArray=skimage.morphology.skeletonize(im)
    
    
    EDArray=(im-(skimage.morphology.binary_erosion(im, np.ones((3,3)))).astype(int)) #find my edges (erode image and subtract from orig image)
    EDImage = Image.fromarray(100*np.uint8(EDArray))
    EDImage=EDImage.convert(mode="RGB")
    
    #2nd way. Find distance from each point to center. mask w edge
    lwrA=scipy.ndimage.morphology.distance_transform_edt( (1-SkelArray)) #
    lwrA=lwrA*EDArray #Mask out with edges. Look at distance from edge
    lwrDist, lwrDistCnt = np.unique(lwrA,return_counts=True) # Save the distances, and their counts
    lwrDist=lwrDist[1:];lwrDistCnt=lwrDistCnt[1:] # Ignore the number of zero distance points
    lwrDist=lwrDist*Opt.NmPP # Make the distances nanometers
    
    
    lwrGmod=lmfit.models.GaussianModel()
    lwrGPars=lwrGmod.guess(lwrDistCnt, x=lwrDist)
    lwrGFit=lwrGmod.fit(lwrDistCnt,lwrGPars,x=lwrDist)
    lwr3Sig=3*lwrGFit.params.valuesdict()['sigma'] # extract the sigma
    lwrMean=lwrGFit.params.valuesdict()['center'] # and mean of gauss fit
    
    # smoothing
    lwrDistFlat=lwrA.ravel()[np.flatnonzero(lwrA)] # just collect all the distances into a 1d array, dropping any zero distances
    lwrDistFlat*=Opt.NmPP # convert to nanometers
    lwrDistKDE=scipy.stats.gaussian_kde(lwrDistFlat) # this smooths the data by dropping gaussians
    lwrDistX=np.linspace(lwrDistFlat.min(),lwrDistFlat.max(),20)
    lwrDistY=lwrDistKDE(lwrDistX) #smoothed obviously
    lwrGFitS=lwrGmod.fit(lwrDistY,lwrGPars,x=lwrDistX) #S FOR SMOOTHEDDDD
    lwr3SigS=3*lwrGFitS.params.valuesdict()['sigma'] # extract the sigma
    lwrMeanS=lwrGFitS.params.valuesdict()['center'] # and mean of gauss fit         


    
    # Find feature X and avg them
    Xval, Xcnt = np.unique(im.nonzero()[1],return_counts=True)
    Fend = np.zeros((1,2)) # holds the side of each feature
    Ftot = np.zeros((1,1)) # will hold weighted counts eg Xval*Xnt
    Fcnt = np.zeros((1,1)) # will hold total number of pixels eg sum Xcnt
    Fmid = np.zeros((1,1)) # will hold featuremidpoints 
    Fend[0] = -1
    Fnum = 0
    for i in range(Xval.size):
        Ftot[Fnum]+=Xval[i]*Xcnt[i]
        Fcnt[Fnum]+=Xcnt[i]
        if (i == 0): # are we starting?
            Fend[0,0]=Xval[i] 
        elif (i == Xval.size - 1): # are we done
            Fend[Fnum,1] = Xval[i]
            Fmid[Fnum]=Ftot[Fnum]/Fcnt[Fnum] # calculate midpoint
        elif (i != 0) and (Fend[Fnum,0] == -1): # if Our start of feature is -1. Keep looking
             if (Xval[i] != Xval[i-1]+1): # if we found a start
                 Fend[Fnum,0] = Xval[i] # assign it
        elif (i != 0) and (Fend[Fnum,1] == -1): # if we are on the end of each feature
             if (Xval[i+1] != Xval[i]+1) or (Xcnt[i+1] > Xcnt[i] and Xcnt[i-1] > Xcnt[i]):
                 Fend[Fnum,1] = Xval[i]
                 Fmid[Fnum]=Ftot[Fnum]/Fcnt[Fnum] # calculate midpoint
                 Fend = np.append(Fend, [[-1,-1]],axis=0)
                 Ftot = np.append(Ftot, [0])
                 Fcnt = np.append(Fcnt, [0])
                 Fmid = np.append(Fmid, [0])
                 
                 #print( str(Fend[Fnum])+' mid is at '+str(Fmid[Fnum]))
                 Fnum += 1
                 if (Xcnt[i+1] > Xcnt[i] and Xcnt[i-1] > Xcnt[i]): # if our intersection is blurred add to both
                     Fend[Fnum,0]= Xval[i]
                     Ftot[Fnum]+=Xval[i]*Xcnt[i]
                     Fcnt[Fnum]+=Xcnt[i]
            
    # ok now that we found our midpoints
    EDMidA=np.zeros(im.shape) # make an array of zeros
    EDMidA[:,np.rint(Fmid).astype(int)] = 1 # set the columns where the middle are as 1 so its analogous to skelarray
    
     # now repeat the lwr stuff for ler

    
    lerA = scipy.ndimage.morphology.distance_transform_edt( (1-EDMidA)) #
    lerA = lerA[EDArray==1] #Mask out with edges. Look at distance from edge
    lerDist, lerDistCnt = np.unique(lerA,return_counts=True) # Save the distances, and their counts
    #lerDist=lerDist[1:];lerDistCnt=lerDistCnt[1:] # Ignore the number of zero distance points
    lerDist=lerDist*Opt.NmPP # Make the distances nanometers
    lerGmod=lmfit.models.GaussianModel()
    lerGPars=lerGmod.guess(lerDistCnt, x=lerDist)
    lerGFit=lerGmod.fit(lerDistCnt,lerGPars,x=lerDist)
    ler3Sig=3*lerGFit.params.valuesdict()['sigma'] # extract the sigma
    lerMean=lerGFit.params.valuesdict()['center'] # and mean of gauss fit
    # smoothing
#    lerDistFlat=lerA.ravel()[np.flatnonzero(lerA)] # just collect all the distances into a 1d array, dropping any zero distances
#    lerDistFlat*=Opt.NmPP # convert to nanometers
    lerDistKDE=scipy.stats.gaussian_kde(lerDist) # this smooths the data by dropping gaussians
    lerDistX=np.linspace(lerDist.min(),lerDist.max(),20)
    lerDistY=lerDistKDE(lerDistX) #smoothed obviously
    lerGFitS=lerGmod.fit(lerDistY,lerGPars,x=lerDistX) #S FOR SMOOTHEDDDD
    ler3SigS=3*lerGFitS.params.valuesdict()['sigma'] # extract the sigma
    lerMeanS=lerGFitS.params.valuesdict()['center'] # and mean of gauss fit         
    
    # now repeat the ler stuff for lpr (Line Placement Roughness)
    
    lprA=scipy.ndimage.morphology.distance_transform_edt( (1-EDMidA)) #
    lprA=lprA[SkelArray == 1] #Mask out with Center. Look at distance from edge
    lprD_Raw, lprDC_Raw = np.unique(lprA,return_counts=True) # Save the distances, and their counts
    #lprDist=lprDist[1:];lprDistCnt=lprDistCnt[1:] # Ignore the number of zero distance points
    lprD_Raw=lprD_Raw*Opt.NmPP # Make the distances nanometers
    lprDF = -1*np.flip(lprD_Raw,0) # flip so that we can make symmetric
    lprDCF = 0.5*np.flip(lprDC_Raw,0) # as above but counts (half them)
    lprDC = 0.5*lprDC_Raw # half the counts for right side as well
    lprDC[0]*=2 # except the one for zero cus we won't be adding that back in
    
    lprDist = np.concatenate((lprDF[:-1], lprD_Raw))
    lprDistCnt = np.concatenate((lprDCF[:-1], lprDC))
    
    lprGmod=lmfit.models.GaussianModel()
    lprGPars=lprGmod.guess(lprDistCnt, x=lprDist)
    lprGFit=lprGmod.fit(lprDistCnt,lprGPars,x=lprDist)
    lpr3Sig=3*lprGFit.params.valuesdict()['sigma'] # extract the sigma
    lprMean=lprGFit.params.valuesdict()['center'] # and mean of gauss fit
    
    
    
    # smoothing
#    lprDistFlat=lprA.ravel()[np.flatnonzero(lprA)] # just collect all the distances into a 1d array, dropping any zero distances
#    lprDistFlat*=Opt.NmPP # convert to nanometers
    lprDistKDE=scipy.stats.gaussian_kde(lprDist) # this smooths the data by dropping gaussians
    lprDistX=np.linspace(lprDist.min(),lprDist.max(),20)
    lprDistY=lprDistKDE(lprDistX) #smoothed obviously
    lprGFitS=lprGmod.fit(lprDistY,lprGPars,x=lprDistX) #S FOR SMOOTHEDDDD
    lpr3SigS=3*lprGFitS.params.valuesdict()['sigma'] # extract the sigma
    lprMeanS=lprGFitS.params.valuesdict()['center'] # and mean of gauss fit         
  
    #%% plotting

    EDFig, EDAx =plt.subplots(nrows = 3,figsize=(6,6))
    lwrFig = EDAx[0]
    lwrFig.set_title('1/2th Line Width, \n Mean  = %.2f, LWR 3$\sigma$ = %.2f' %(lwrMean, lwr3Sig))        
    lwrFig.plot(lwrDist, lwrGFit.best_fit/lwrGFit.best_fit.max(), 'r-')
    lwrFig.plot(lwrDistX,lwrDistY/lwrDistY.max(), 'bo')
    lwrFig.set_yticks([])
    
    lerFig = EDAx[1]
    lerFig.set_title('Center - Edge Distance, \n Mean = %.2f, LER 3$\sigma$ = %.2f' %(lerMean, ler3Sig))      
    lerFig.plot(lerDist, lerGFit.best_fit/lerGFit.best_fit.max(), 'r-')
    lerFig.plot(lerDist,lerDistCnt/lerDistCnt.max(), 'bo')
    lerFig.set_yticks([])
    
    lprFig = EDAx[2]
    lprFig.set_title('Position - Mean Position Distance, \n Mean = %.2f, LPR 3$\sigma$ = %.2f' %(lprMean, lpr3Sig))
    lprFig.set_xlabel('Distance (nm)')
    # Can't do smoothed due to proximity to 0
    lprFig.plot(lprDist, lprGFit.best_fit/lprGFit.best_fit.max(), 'r-')
    lprFig.plot(lprDist,lprDistCnt/lprDistCnt.max(), 'bo')
    lprFig.set_yticks([])
    EDFig.tight_layout( pad = 0.5, w_pad=0.5, h_pad=1.0 )
    
#%%
             
            
        
        
        
        
        #todo : now that I have the x values determine cut off with 2nd derivative avg x than use that to find distance and mask with edges then repeat code of above 
                                 
                                      
                                      

    



    if Opt.EDSa==1: #save
        EDFig.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "R.png"))
        EDImage.save(os.path.join(Opt.FPath,"output",Opt.FName+"ED.tif"))
    
    if Opt.EDSh == 1: # show
        EDFig.show()
        EDImage.show()
    else:
        plt.close(EDFig)
    lwr = [lwrMean, lwr3Sig, lwrMeanS,lwr3SigS] # for out
    ler = [lerMean, ler3Sig, lerMeanS, ler3SigS]
    lpr = [lprMean, lpr3Sig, lprMeanS, lpr3SigS]
    return( lwr, ler, lpr);


#%% Angle Detection
def AngSobel(im, Opt):
    """
    Uses sobel derivatives to calculate the maximum gradient direction allowing for 
    rough but noisy orientation detection.
    V0.1
    """
    
    class AngDet:
        pass
    
    AngDet.A1stDY=np.absolute(scipy.ndimage.sobel(im, axis=0))
    AngDet.A1stDX=np.absolute(scipy.ndimage.sobel(im, axis=1))
    AngDet.AngArray=np.float32(np.arctan2(AngDet.A1stDY,AngDet.A1stDX))    
    AngDet.AngArray*=180/np.pi
    if Opt.AECSh == 1 or Opt.AECSa == 1:
        AngPlot=plt.figure()
        AngPlot1=AngPlot.add_subplot(111)
        AngPlot1.imshow(AngDet.AngArray)
    if Opt.AECSh == 1: #REPLACE show

        AngPlot.show() #
    if Opt.AECSa==1: #Replace save
        AngPlot.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "AngEC.png"), dpi=300)
    try:plt.close(AngPlot)
    except:pass
    return(AngDet.AngArray)


def AngEC(im, Opt, EDArray='none', SkelArray='none'):
    """
    Angle detection using the angle from edge of binary image to center of image
    this requires the skeleton to be passed but it will recalculate the edge
    TODO : Try to pass edge array to save time. Probably not worth any time but still.
    V0.1
    """
    if EDArray=='none':
        EDArray=(im-skimage.morphology.binary_erosion(im, np.ones((3,3)))) 
    if SkelArray=='none':
        SkelArray=skimage.morphology.skeletonize(im)
        
    #find my edges (erode image and subtract from orig image)
    EDDistA2=scipy.ndimage.morphology.distance_transform_edt( (1-SkelArray), return_distances=False, return_indices=True)
    
    
    Yind,Xind=np.mgrid[0:im.shape[0],0:im.shape[1]]
    XDist=(EDDistA2[1,:,:]-Xind)
    YDist=(EDDistA2[0,:,:]-Yind)
    AngArray=np.arctan2(YDist,XDist)*180/np.pi
    
    #AngArray is -180- - > +180 lets make it 0<x<=360
    AngArray+=180

    #masking
    EDArray=1.0*EDArray
    EDArray[EDArray==0]=float('nan') # we can mask with NAN if needed
    AngArray*=EDArray 
    # And edge protection, just removes the 10 edge pixels by default to avoid edge effects
    (CIMH,CIMW)=AngArray.shape
    AngArray[0:int(Opt.DefEdge-1),:]=float('nan'); AngArray[int(CIMH+1-Opt.DefEdge):int(CIMH),:]=float('nan'); 
    AngArray[:,0:int(Opt.DefEdge-1)]=float('nan'); AngArray[:,int(CIMW+1-Opt.DefEdge):int(CIMW)]=float('nan'); 
    
    # note that due to the algo the angle is not defined aside from the edges
    # this masking insures the array conveys this fact
    if Opt.AECSh == 1 or Opt.AECSa == 1:
        AngPlot=plt.figure()
        AngPlot1=AngPlot.add_subplot(111)
        AngPlot1.imshow(AngArray)
    if Opt.AECSh == 1: #REPLACE show

        AngPlot.show() #
    if Opt.AECSa==1: #Replace save
        AngPlot.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "AngEC.png"), dpi=300)
    try:plt.close(AngPlot)
    except:pass
    return(AngArray)
    

def AngMid(im, Opt, SkelArray='none'):
    """
    Angle detection using N point averaging
    V0.1
    """
    #
    AngArray = np.zeros(im.shape)
    AngArray[:] = np.nan
    
    # recalc the skel array if we aren't passed it
    if SkelArray=='none':
        SkelArray=skimage.morphology.skeletonize(im).astype('i1')
    # First break at every junction
    # Make a copy of the skeleton array to keep track of where we have been
    TrackArray = np.copy(SkelArray) # zeros in this = non valid moves!
    Adj = scipy.signal.convolve(SkelArray, np.ones((3,3)),mode='same',method='direct').astype('i1')
    TrackArray *= (Adj <= 3) # mark junction as stop points eg zeros
    
    # Now calculate where all the terminals are with the junctions as breaks
    Adj = scipy.signal.convolve(TrackArray, np.ones((3,3)),mode='same',method='direct').astype('i1')
    # if adjacent = 2 and it's on a line it's a terminal
    Term = (( Adj==2)*TrackArray).astype('i1')
    
    TermR, TermC = np.nonzero(Term) # termR is now Rows, TermC : Columns of Terminals
    MoveCoord = np.array([[0, 1], # this is simply the set of moves we allow eg 8 moves
                 [1, 1],
                 [-1, 1],
                 [1, 0],
                 [-1, 0],
                 [-1, 1],
                 [-1, -1],
                 [-1, 0]])
    
    for i in np.arange(len(TermR)):
        # now we need to make a blank Opt.AngMP x 2 array that will carry the x, y coords
        Coord = np.zeros( (Opt.AngMP,2),dtype=int)
        # ok now where are we
        Coord[0,:] = TermR[i],TermC[i] # set coords
        TrackArray[TermR[i],TermC[i]] = 0 # mark we've been here
        Rollin = 1 # now we rollin :)
        
        while Rollin == 1: # yah yah while loops are a trap, lets not fall in eh?
            for TCount in np.arange(8): # we have 8 possible moves
                TestMove = (Coord[0,:] + MoveCoord[TCount, :])
                if (TestMove.min() >= 0) and ((TestMove < TrackArray.shape).all()):
                    if TrackArray[TestMove[0],TestMove[1]] == 1: # if we are on the skeleton
                        Coord = np.roll(Coord,2) # roll back the coordinates
                        Coord[0,:] = np.copy(TestMove) # assign the new coords
                        TrackArray[Coord[0,0],Coord[0,1]] = 0 # mark we've been here
                        
                        if Coord[Opt.AngMP-1,:].sum() != 0: # if we have enough data eg the matrix is full
                            
                            # calculate the distance as current coordinates minus last coords
                            Dist = Coord[0,:] - Coord[-1,:] # note this is [RDist , ColumnDist] aka [DY, DX]
                            

                            AngArray[Coord[int((Opt.AngMP-1)/2),0],Coord[int((Opt.AngMP-1)/2),1]] = (np.arctan2(Dist[1],Dist[0])*180/np.pi) + 180
                        break # go back to while loop and restart for loop
                if TCount == 7: # if we are on the last move and we didn't break we ended our run
                    Rollin = 0 # we ran out of moves and found no valid ones so pick a new terminal
    if Opt.AECSh == 1 or Opt.AECSa == 1:
        AngPlot=plt.figure()
        AngPlot1=AngPlot.add_subplot(111)
        AngPlot1.imshow(AngArray)
    if Opt.AECSh == 1: #REPLACE show
        AngPlot.show() #
    if Opt.AECSa==1: #Replace save
        AngPlot.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "AngMid.png"), dpi=300)
    try:plt.close(AngPlot)
    except:pass
    return(AngArray)
    

    
#%% Angle Hist: Creates histograms of angles
def AngHist(AngArray,Opt, MaskArray=1, WeightArray='N'):
    """
    Simply takes an array and computes the histograms with 1 degree bins from 0-180
    Also weights pixels with weight array if given, and masks with mask array if given
    Currently only dumps to text the unweighted un masked histogram
    Outputs the number of pixels with angles at the highest peak and second highest as well as total pixels counted
    Which can be used to track progress of alignment or percent alignment
    V.2 > Now normalizes so maximum peak is at 90 degrees
    """
    class AngHist:
        pass
    
    Rlow=0 # bottom of angles to look at, now 0 cus we are compressing to one quadrant
    Rhigh=90
    BCount=Rhigh-Rlow+1     
    HPW=10 # half peak width, how many angles to combine
    
    if WeightArray=='N':
        WeightArray=np.ones_like(AngArray)
        
    #angarray=np.absolute(np.pi/4-angarray) #Renormalize to between 0 and pi/4
#    angmask=angarray[maskarray != 0]# Mask out the data note this way flattens
    MaskArray=1.0*MaskArray
    try:
        MaskArray[MaskArray==0]=float('nan')  # replaces 0 with nan in mask array, necessary for histograms to not be overfilled with 0
    except:
        pass
    
    # lets find the mode and set that to 90 to center our results
    # first cast as int and flatten
    

    NANMask=np.logical_not(np.isnan(AngArray))
    
    AngMode=scipy.stats.mode((AngArray[NANMask].flatten()).astype(int))[0]
    
    AngArray+=0-AngMode 
    # make it so max peak is at 90 for clarity
    if AngMode > 0:
        # if mode was over 90 we shifted down, so now we have neg vals
        AngArray[AngArray<= -90] += 180 # so fix it
    elif AngMode < 0: # this should never be the case with algorithms implemented currently for ang det. Leave in for robustness though
        AngArray[AngArray > 90] -= 180
                
    AngArray=np.abs(AngArray)
               
    
    AngMask=AngArray*MaskArray # make the mask array
            
    AngHist.Plot=plt.figure();
    AngHist.Plt1=AngHist.Plot.add_subplot(221) # , range=(0,90)
    hist,bins = np.histogram(AngArray, bins=BCount, range=(Rlow,Rhigh))
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt1.bar(center,hist,align='center',width=width)
    AngHist.Plt1.set_title('OD')
    
    #going to dump this one here TODO dump all as BINS COUNTSA COUNTSB with headers
    np.savetxt(os.path.join(Opt.FPath,"output",Opt.FName + "Hist.csv"),hist,delimiter=',')
    # Also find the max and second highest orientation and dump those to output
    
    CntT=np.sum(hist) # cumulative count
    

    Peak1=hist.argmax()
    PLow=Peak1-HPW;PHigh=Peak1+HPW # ten below, ten above are all combined
    if PLow < -90: 
        PLow+=181; # wrap around if under
        Cnt1=np.sum(hist[:Peak1])+np.sum(hist[PLow:])
        hist[:Peak1]=0;hist[PLow:]=0; # set to zero so we don't find again     
    else:
        Cnt1=np.sum(hist[PLow:Peak1])
        hist[PLow:Peak1]=0;
    if PHigh > 90:
        PHigh-=181; # or over
        Cnt1+=np.sum(hist[Peak1:])+np.sum(hist[:PHigh])
        hist[Peak1:]=0;hist[:PHigh]=0;
    else:
        Cnt1+=np.sum(hist[Peak1:PHigh])
        hist[Peak1:PHigh]=0
    
    Peak2=hist.argmax()
    PLow=Peak2-HPW;PHigh=Peak2+HPW # ten below, ten above are all combined
    if PLow < -90: 
        PLow+=181; # wrap around if under
        Cnt2=np.sum(hist[:Peak2])+np.sum(hist[PLow:])
        hist[:Peak1]=0;hist[PLow:]=0; # set to zero so we don't find again     
    else:
        Cnt2=np.sum(hist[PLow:Peak2])
        hist[PLow:Peak2]=0;
    if PHigh > 90:
        PHigh-=181; # or over
        Cnt2+=np.sum(hist[Peak2:])+np.sum(hist[:PHigh])
        hist[Peak1:]=0;hist[:PHigh]=0;
    else:
        Cnt2+=np.sum(hist[Peak2:PHigh])
        hist[Peak2:PHigh]=0


    
    
    AngHist.Plt2=AngHist.Plot.add_subplot(222)
    hist,bins = np.histogram(AngMask, bins=BCount, range=(Rlow,Rhigh))
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt2.bar(center,hist,align='center',width=width)
    AngHist.Plt2.set_title('OD+Mask') 
    
    
    AngHist.Plt3=AngHist.Plot.add_subplot(223)
    hist,bins = np.histogram(AngArray, bins=BCount, range=(Rlow,Rhigh), weights=WeightArray)
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt3.bar(center,hist,align='center',width=width)
    AngHist.Plt3.set_title('OD+Weight')
    
    
    AngHist.Plt4=AngHist.Plot.add_subplot(224)
    hist,bins = np.histogram(AngMask, bins=BCount, range=(Rlow,Rhigh), weights=WeightArray)
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt4.bar(center,hist,align='center',width=width)
    AngHist.Plt4.set_title('OD+Mask+Weight')    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)    
    
    if 1 == 1: #REPLACE show
        AngHist.Plot.show()
    if 1==1: #Replace save
        AngHist.Plot.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "AngHist.png"))
    plt.close()
    
    return(Peak1,Cnt1,Peak2,Cnt2,CntT)


#%% Autocorrelation T_T
def AutoCorrelation(AngArray, Opt):
    """
    Autocorrelation is a WIP still
    (AutoCorrelation function that is passed the angle array)
    VALPHA
    """
    class AutoCor:
        pass
    
    MaskA = 1-np.isnan(AngArray) # 0 = nan, 1 = valid
    Opt.ACSize = np.min( (Opt.ACSize , MaskA.sum()) )
    
    AutoCor.SkI, AutoCor.SkJ=np.nonzero( MaskA ); #Get indexes of non nan
    PosNeg = np.array([-1, 1])
    
    AutoCor.RandoList=np.random.randint(0,len(AutoCor.SkI),Opt.ACSize)
    AutoCor.uncor=np.zeros((1,2)) # where do we go uncorrelated? how many times.
    AutoCor.n=np.zeros(Opt.ACCutoff)
    AutoCor.h=np.zeros(Opt.ACCutoff)
    AutoCor.Ind = 0
    AutoCor.Indexes=np.array([[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]) # for picking nearby
    AutoCor.IndAngles=np.array([315,270,225,180,135,90,45,0]) #angle of the above in degrees UP is 0 go cw
    AutoCor.HList = np.ones(1)
    AutoCor.DistList = np.ones(1)
    
    AutoCor.IndAngles=AutoCor.IndAngles*np.pi/180 # radians
    AngArray = AngArray * np.pi/180 #radians
    
    
    while AutoCor.Ind < Opt.ACSize : # How many points to start at to calc auto correlate
        # The following is the AutoCor Loop
        AutoCor.ntemp=np.zeros(Opt.ACCutoff) # How many times have calculated the n=Index+1 correlation?
        AutoCor.htemp=np.zeros( Opt.ACCutoff ) # what is the current sum value of the correlation(divide by ntemp at end)
        AutoCor.angtemp=np.ones(Opt.ACCutoff+1)*float('nan') # What is the current angle, 1 prev angle, etc etc
        AutoCor.BBI = 0 # not necessary but helpful to remind us start = BBI 0
        AutoCor.SAD=0;
        ContTemp = np.zeros( Opt.ACCutoff ) # hold the backbone distance
        #First pick a point, find it's angle
        #TODO
        AutoCor.CCOORD=np.append(AutoCor.SkI[AutoCor.RandoList[AutoCor.Ind]],
                         AutoCor.SkJ[AutoCor.RandoList[AutoCor.Ind]])
        
        AutoCor.angtemp[0] = AngArray[AutoCor.CCOORD[0],AutoCor.CCOORD[1]]         
        AutoCor.BBI = 1 #now we at first point... 
        AutoCor.PastN = np.random.randint(8,16) # No previous point to worry about moving back to
         
        while AutoCor.BBI < (2*Opt.ACCutoff+1): # How far to walk BackBoneIndex total points is 
            AutoCor.angtemp = np.roll(AutoCor.angtemp,1) # now 1st angle is index 1 instead of 0 etc
            #what is our next points Coord?
            
            PM1 = AutoCor.PastN + PosNeg[np.random.randint(0,2)] * PosNeg
            PM2 = AutoCor.PastN + PosNeg[np.random.randint(0,2)] * 2 * PosNeg
            PM3 = AutoCor.PastN + PosNeg[np.random.randint(0,2)] * 3 * PosNeg
            
            if AutoCor.PastN%2 == 0: # if even check the cardinals then diagonals
                AutoCor.WalkDirect=np.append(PM1%8, AutoCor.PastN%8)
                AutoCor.WalkDirect=np.concatenate((AutoCor.WalkDirect, PM3%8, PM2%8))
            else: # if odd
                AutoCor.WalkDirect=np.append(AutoCor.PastN%8, PM1%8)
                AutoCor.WalkDirect=np.concatenate((AutoCor.WalkDirect, PM2%8, PM3%8))
                
                
                
            
            for TestNeighbor in np.arange(7): # try moves
                AutoCor.COORD = AutoCor.Indexes[AutoCor.WalkDirect[TestNeighbor]]+AutoCor.CCOORD
                if AutoCor.COORD[0] < AngArray.shape[0] and AutoCor.COORD[1] < AngArray.shape[1]: # if we are in bounds
                    if np.isnan(AngArray[AutoCor.COORD[0],AutoCor.COORD[1]]) == 0: # if we have a good move
#                        if AutoCor.BBI==1: # And its the first move we need to fix 1st angle
#                            if AutoCor.angtemp[1] < AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]] - 90: # if angle is 90 lower
#                                AutoCor.angtemp[1]+=np.pi
#                            elif AutoCor.angtemp[1] > AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]] + 90:
#                                AutoCor.angtemp[1]-=np.pi
                        AutoCor.PastN=AutoCor.WalkDirect[TestNeighbor];
                        
                        Dist = np.hypot(AutoCor.CCOORD[0]-AutoCor.COORD[0],AutoCor.CCOORD[1]-AutoCor.COORD[1])
                        ContTemp = np.roll(ContTemp,1) # roll it so we can accept the 1 unit distance
                        ContTemp[ContTemp!=0] += Dist # Add contour distance of last step to each cell
                        ContTemp[0] = Dist #make sure to set                         
                        
                        AutoCor.CCOORD = AutoCor.COORD; # move there
                        AutoCor.angtemp[0]=AngArray[AutoCor.CCOORD[0],AutoCor.CCOORD[1]] # set angle to new angle
                        
                        for AutoCor.PI in range (0,Opt.ACCutoff): # Persistance Index, 0 = 1 dist etc
                    #Calculating autocorrelation loop
                            if np.isnan(AutoCor.angtemp[AutoCor.PI+1]) == 0:
                                distcalc = ContTemp[AutoCor.PI]
                                hcalc = np.cos(abs(AutoCor.angtemp[0]-AutoCor.angtemp[AutoCor.PI+1]))
                                AutoCor.HList = np.append( AutoCor.HList, hcalc)
                                AutoCor.DistList = np.append( AutoCor.DistList, distcalc)
#                                hcalc = abs(AutoCor.angtemp[0]-AutoCor.angtemp[AutoCor.PI+1])
                                AutoCor.htemp[AutoCor.PI] += hcalc
                                AutoCor.ntemp[AutoCor.PI] += 1

                                
                        break # break the for loop (done finding next point)
                    
                elif TestNeighbor==6: # else if we at the end
                    # Need to break out of the backbone loop as well...
                    AutoCor.SAD=1; # because
            
            if AutoCor.SAD==1: # break out of BB while loop
                # Decide if I count this or not...
                AutoCor.SAD=0;
                break
            AutoCor.BBI+=1
            # BUT WAIT WE NEED TO FIX THE NEW ANGLE TOO!
#            if AutoCor.angtemp[0] < AutoCor.IndAngles[AutoCor.PastN] - 90: # if angle is 90 lower
#                AutoCor.angtemp[0]+=np.pi
#            elif AutoCor.angtemp[0] > AutoCor.IndAngles[AutoCor.PastN] + 90:
#                AutoCor.angtemp[0]-=np.pi         
         # we found all our points done with BBI
        AutoCor.h +=AutoCor.htemp
        AutoCor.n +=AutoCor.ntemp
        AutoCor.Ind += 1
                
    AutoCor.Out = np.divide(AutoCor.h, AutoCor.n)
#    PFit=polyfit(np.arange(1,len(AutoCor.Out)+1)*Opt.NmPP,)

    AutoCor.Bins = np.arange(1,int(AutoCor.DistList.max())+0.5,1)
    ContBin  = np.digitize( AutoCor.HList, AutoCor.Bins ) # which contour length bin is each one in
    AutoCor.MeanCont = np.zeros_like(AutoCor.Bins) # hold mean eelength for each cont length
    for i in np.arange(1,len(AutoCor.Bins)):
        AutoCor.MeanCont[i] = AutoCor.HList[ ContBin == i ].mean()

    
    return(AutoCor)
#%% Autocorrelation T_T
def PersistenceLength(SkelArray, Opt):
    """
    Autocorrelation is a WIP still
    (AutoCorrelation function that is passed the angle array)
    VALPHA
    """
    class PL:
        pass
    
    SkelLocal = SkelArray.copy()
    PL.n=np.zeros(Opt.ACCutoff)
    PL.h=np.zeros(Opt.ACCutoff)
    PL.ContList = np.ones( 1 ) # maybe just keep track of all the points? Will get absurdly big
    PL.EEList = np.ones( 1 ) # dittooooo
    
    PL.Ind = 0
    Indexes=np.array([[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]) # for picking nearby
   
    AdCount=scipy.signal.convolve(SkelLocal, np.ones((3,3)),mode='same', method='direct')
    # Remove Opt.DefEdge pixels at edge to prevent edge effects. be sure to account for area difference
#    (CIMH,CIMW)=SkelArray.shape
#    AdCount[0:int(Opt.DefEdge-1),:]=0; AdCount[int(CIMH+1-Opt.DefEdge):int(CIMH),:]=0; 
#    AdCount[:,0:int(Opt.DefEdge-1)]=0; AdCount[:,int(CIMW+1-Opt.DefEdge):int(CIMW)]=0;
            
    TermArray = np.multiply(SkelLocal, (AdCount==2))
    PointI, PointJ = np.nonzero(TermArray)
    PL.size = np.min((len(PointI), Opt.ACSize))
    print(PL.size,Opt.ACSize,len(PointI))
    RandoList=np.random.randint(0,len(PointI),PL.size)
    
    while PL.Ind < PL.size : # How many points to start at to calc auto correlate
        # The following is the AutoCor Loop
        
#        ntemp = np.zeros(Opt.ACCutoff) # How many times have calculated the n=Index+1 correlation?
#        htemp = np.zeros( Opt.ACCutoff ) # what is the current sum value of the correlation(divide by ntemp at end)


        
        ContTemp = np.zeros( Opt.ACCutoff )
        XYtemp=np.ones((Opt.ACCutoff,2))*float('nan') # What is the current coord prev etc was +1 prev
        BBI = 0 # not necessary but helpful to remind us start = BBI 0
        SAD=0 # used to double loop break
        #First pick a point, find it's angle
        #TODO
        CCOORD=np.array((PointI[RandoList[PL.Ind]],
                         PointJ[RandoList[PL.Ind]]), ndmin=2)
        while SkelLocal[CCOORD[0,0],CCOORD[0,1]] == 0: # if our point isn't good
            PL.Ind +=1 #try the next
            try:CCOORD=np.array((PointI[RandoList[PL.Ind]],PointJ[RandoList[PL.Ind]]),ndmin=2)
            except: # if we are out of points
                BBI = Opt.ACCutoff # skip the big loop
                break # break outta here
        
        XYtemp[0] = CCOORD   
        BBI = 1 #now we at first point... 

         
        while BBI < (Opt.ACCutoff): # How far to walk BackBoneIndex total points is 
            
            #what is our next points Coord?
                           
            
            for TestNeighbor in np.arange(8): # try moves
                COORD = Indexes[TestNeighbor]+CCOORD
                if COORD[0,0] < SkelLocal.shape[0] and COORD[0,1] < SkelLocal.shape[1]: # if we are in bounds
                    if (SkelLocal[COORD[0,0],COORD[0,1]] == 1): # if we have a good move

                        
                        CCOORD = COORD.copy(); # move there
                        RCDist = XYtemp - CCOORD # calculate distance from new points to prev points (END END DIST)
                        Dist = np.hypot(RCDist[:,0], RCDist[:,1]) # find hypot for dist
                        Dist[np.isnan(Dist)] = 0 
                        
                        ContTemp = np.roll(ContTemp,1) # roll it so we can accept the 1 unit distance
                        ContTemp[ContTemp!=0] += Dist[0] # Add contour distance of last step to each cell
                        ContTemp[0] = Dist[0] #make sure to set 
#                        DistRatio = np.divide(Dist,ContTemp)
#                        DistRatio[ np.isnan(DistRatio) ] = 0
                        
                        PL.ContList = np.append( PL.ContList, ContTemp[np.nonzero(Dist)] )
                        PL.EEList = np.append( PL.EEList, Dist[np.nonzero(Dist)] )
                        
#                        htemp += DistRatio # add results to cumulative calc
#                        ntemp += (1-np.isnan(XYtemp[:,0])) # which numbers did we actually accumulate
                        XYtemp = np.roll(XYtemp,2) # now starting point is index 1 instead of 0 etc
                        
                        
                        XYtemp[0] = CCOORD.copy() # set XY to new COORD
                        SkelLocal[CCOORD[0,0],CCOORD[0,1]] = 0 # delete point so we don't hit it again
                                
                        break # break the for loop (done finding next point)
                    
                elif TestNeighbor==7: # else if we at the end
                    # Need to break out of the backbone loop as well...
                    SAD=1; # because
                    
            if SAD==1: # break out of BB while loop
                # Decide if I count this or not...
                SAD=0;
                PL.Ind += 1
                break
            BBI += 1
        
            
#        PL.h += htemp
#        PL.n += ntemp
        PL.Ind += 1
        print(PL.Ind)
        
                
    PL.ContList*=Opt.NmPP
    PL.EEList*=Opt.NmPP

    PL.UNBins, PL.UNBinID =np.unique(np.around(PL.ContList, decimals=0), return_inverse=True)
    PL.UNMeanEE = np.zeros_like(PL.UNBins)
    PL.UNSDEE = np.zeros_like(PL.UNBins)
    PL.UNMeanCL = np.zeros_like(PL.UNBins)
    PL.UNSDCL = np.zeros_like(PL.UNBins)
    PL.UNMeanRat = np.zeros_like(PL.UNBins)
    PL.UNSDRat = np.zeros_like(PL.UNBins)
    
    for i in np.arange(0, len(PL.UNBins)):
        PL.UNMeanEE[i] = PL.EEList[ PL.UNBinID == i].mean()
        PL.UNSDEE[i] = PL.EEList[ PL.UNBinID == i].std() # std dev
        PL.UNMeanCL[i] = PL.ContList[ PL.UNBinID == i].mean()
        PL.UNSDCL[i] = PL.ContList[ PL.UNBinID == i].std()
        PL.UNMeanRat[i] = np.divide(PL.EEList[ PL.UNBinID == i],PL.ContList[ PL.UNBinID == i]).mean()
        PL.UNSDRat[i] = np.divide(PL.EEList[ PL.UNBinID == i],PL.ContList[ PL.UNBinID == i]).std()
        
    
    def PerFunc(cl, P):
        return(np.sqrt( 2*P*cl* (1-P/cl*(1-np.exp(-cl/P) ) )  ))
    PMod = lmfit.Model(PerFunc)
    PL.PRes = PMod.fit(PL.UNMeanEE, cl=PL.UNMeanCL, P=1)
    
    PL.RatFilt = scipy.ndimage.filters.gaussian_filter1d(PL.UNMeanRat,3)
    
    
#    plt.plot(PL.ContList, PL.EEList, 'b.')
    PLPlot=plt.figure()
    PLPlot1=PLPlot.add_subplot(211)
    PLPlot1.plot(PL.UNMeanCL, PL.UNMeanRat,'b.')
    PLPlot1.plot(PL.UNMeanCL, PL.RatFilt,'k-')
    PLPlot1.set_ylim([0,1])
    PLPlot1.set_xlabel('Contour Length')
    PLPlot1.set_ylabel('End to End Distance \n divided by Contour Length')
    PLPlot2=PLPlot.add_subplot(212)
    PLPlot2.errorbar(PL.UNMeanCL, PL.UNMeanEE,  yerr=PL.UNSDEE, xerr=PL.UNSDCL)
    PLPlot2.set_xlabel('Contour Length')
    PLPlot2.set_ylabel('End to End Distance')
#    PLPlot3=PLPlot.add_subplot(313)
#    PLPlot3.plot(PL.UNMeanCL, PL.PRes.best_fit,'b-')
#    PLPlot3.plot(PL.UNMeanCL, PL.UNMeanEE,'k.')
#    PLPlot3.set_xlabel('Contour Length')
#    PLPlot3.set_ylabel('End to End Distance')
#    PLPlot3.set_title('PL Fit : '+str(PL.PRes.params['P'].value))
    
    
    PLPlot.tight_layout( pad = 2, w_pad=2, h_pad=5.0 )
 #    if Opt.AECSh == 1: #REPLACE show   
    PLPlot.show()
    if PL.RatFilt.min() > 0.8:
        PL.Pers = 'Infinite'
        PLPlot.suptitle('Infinite Persistence Length')
    else:
        PL.Pers = PL.UNMeanCL[np.argmax(PL.RatFilt < 0.8)]
        PLPlot.suptitle('Persistence Length (nm) : '+str(PL.Pers.round()))

#    if Opt.AECSa==1: #Replace save

    PLPlot.savefig(os.path.join(Opt.FPath,"output",Opt.FName + "PerLen.png"), dpi=300)
#    plt.close(AngPlot)

    return(PL)
#%% Param Optimizer
def ParamOptimizer(ArrayIn, Opt, l0, Params):
    Opt.DenWeight=Params[0]
    Opt.ThreshWeight=Params[1]
    Array=ArrayIn
    if Opt.DenToggle==1:
        Array=Denoising(Array, Opt, l0)[0]
    if Opt.ThreshToggle==1:
        Array=Thresholding(Array, Opt, l0)[0]
    if Opt.RSOToggle==1:
        Array=RSO(Array, Opt)
    (SkelArray, SkelAC, TCount, TCA, JCount, JCA)=Skeleton(Array, Opt)
#    TCount+=JCount
    print('Denoising weight %.2f Thresholding Weight %.2f, Defect Count %f' %(Params[0], Params[1], TCount))
    return(TCount);