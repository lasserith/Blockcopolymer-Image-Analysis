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
import skimage
from skimage import restoration, morphology, filters, feature
import exifread #needed to read tif tags
try:
    from igor.binarywave import load as loadibw
except: print('You will be unable to open Asylum data without igor')
import re #dat regex
import matplotlib.pyplot as plt
import exifread #needed to read tif tags

import scipy

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
        Opt.Machine="Asylum AFM";
        RawData= loadibw(Opt.Name)['wave']
        Labels = RawData['labels'][2]
        Labels = [i.decode("utf-8") for i in Labels] # make it strings
        # Need to add a selector here for future height/phase
        [AFMIndex]=[ i for i, s in enumerate(Labels) if Opt.AFMLayer in s] #they index from 1????
        AFMIndex-=1 # fix that quick
        imarray = RawData['wData'][:,:,AFMIndex]
        TArray=imarray.transpose() # necessary so that slow scan Y and fast scan is X EG afm tip goes along row > < then down to next row etc
        # AFM data has to be leveled :(
        if Opt.AFMLevel == 1: #median leveling
            MeanRow=TArray.mean(axis=1) # this calculates the mean of each row
            Mean=TArray.mean() # mean of everything
            MeanOffset=MeanRow-Mean # determine the offsets
            imarray=imarray-MeanOffset # adjust the image
            
        elif Opt.AFMLevel==2: # median of dif leveling
            DMean=np.diff(TArray,axis=0).mean(axis=1) 
            # calc the 1st order diff from one row to next. Then average these differences 
            DMean=np.insert(DMean,0,0) # the first row we don't want to adjust so pop in a 0
            
        
        
        #Brightness/Contrast RESET needed for denoising. Need to figure out how to keep track of this? add an opt?
        imarray = imarray/imarray.max()
        Opt.NmPP=RawData['wave_header']['sfA'][0]*1e9
        #RawData.clear()
        
    else:
        im= Image.open(Opt.Name)
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


    if Opt.NmPP!=0:
        print("Instrument was autodetected as %s, NmPP is %f \n" % (Opt.Machine ,Opt.NmPP) )
    else:
        print("Instrument was not detected, and NmPP was not set. Please set NmPP and rerun")
    return(imarray);
#%% Crop
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
    
#    DXIm.save(os.path.join(Opt.FPath,"output",Opt.BName+"DX.tif"))
#    DYIm.save(os.path.join(Opt.FPath,"output",Opt.BName+"DY.tif"))
#    SchIm.save(os.path.join(Opt.FPath,"output",Opt.BName+"Sch.tif"))
    return(Ide)

#%% Azimuthal Averaging
def azimuthalAverage(image, center=None):
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
    F2Array=np.fft.fftshift(FourierArray);    SpaceA=1/FreqA;
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
            np.savez(os.path.join(Opt.FPath,"output",Opt.BName + "FFT"),(FreqA[0:int(np.floor(FSize/2))], PowerSpec1d[0:int(np.floor(FSize/2))]))
            Fig.savefig(os.path.join(Opt.FPath,"output",Opt.BName + "PowerSpecFreq.png"))
            PSD1D.annotate('Primary Peak at %f' %PFMax, xy=(PFMax, PHMax), xytext=(1.5*PFMax, 1.5*PHMax),
                    arrowprops=dict(facecolor='black', width=2,headwidth=5),
                    )
            Fig.savefig(os.path.join(Opt.FPath,"output",Opt.BName + "PowerSpecFreqLabel.png"))
            PS2DImage.save(os.path.join(Opt.FPath,"output",Opt.BName + "PowerSpec2d.tif"))
    return(PSMax);
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
        DenImage.save(os.path.join(Opt.FPath,"output",Opt.BName + "Den.tif"))
    
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
        
    AdaptBin=skimage.filters.threshold_adaptive(im,Thresh ,'gaussian')
    
    
    AdaptThresh = Image.fromarray(100*np.uint8(AdaptBin))
    AdaptThresh=AdaptThresh.convert(mode="RGB")
    if Opt.ThreshSh == 1:
        AdaptThresh.show()
    if Opt.ThreshSa==1:
        AdaptThresh.save(os.path.join(Opt.FPath,"output",Opt.BName+"AThresh.tif"))
    return(AdaptBin,Thresh)
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
        RSOI.save(os.path.join(Opt.FPath,"output",Opt.BName+"LADRSO.tif"))
    return(RSO);
        
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
    Size=int(Opt.NmPP*10) # what is the top zone height? here 10 nm
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
        WLabI.save(os.path.join(Opt.FPath,"output",Opt.BName+"Lab.tif"))
        WDomCI.save(os.path.join(Opt.FPath,"output",Opt.BName+"DomC.tif"))
        WDomCI.save(os.path.join(Opt.FPath,"output",Opt.BName+"LabDomC.tif"))
        WThroughCI.save(os.path.join(Opt.FPath,"output",Opt.BName+"ThroughDomC.tif"))
    return(WFrac, BFrac, WDomI, WDomFrac);
    
#%%
def Skeleton(im,Opt):
    """
    Returns a skeleton array for future use, as well as performs defect analysis
    (Diagonals count as connected)
    v0.1
    """
    
    SkelArray= skimage.morphology.skeletonize(im)
    LASkelI= Image.fromarray(100*SkelArray)
    LASkelI=LASkelI.convert(mode="RGB")
    if Opt.SkeleSh == 1:
        LASkelI.show()
    if Opt.SkeleSa==1:
        LASkelI.save(os.path.join(Opt.FPath,"output",Opt.BName+"Skel.tif"))

    
    AdCount=scipy.signal.convolve(SkelArray, np.ones((3,3)),mode='same')
    # Remove Opt.DefEdge pixels at edge to prevent edge effects. be sure to account for area difference
    (CIMH,CIMW)=im.shape
    AdCount[0:int(Opt.DefEdge-1),:]=0; AdCount[int(CIMH+1-Opt.DefEdge):int(CIMH),:]=0; 
    AdCount[:,0:int(Opt.DefEdge-1)]=0; AdCount[:,int(CIMW+1-Opt.DefEdge):int(CIMW)]=0; 
    DefArea=( CIMW-2*Opt.DefEdge)*( CIMH-2*Opt.DefEdge)*Opt.NmPP*Opt.NmPP; # Area in nm^2
    
    # Terminal
    TLog = ((AdCount==2) * (SkelArray== 1)) # if next to 1 + on skel
    TCount = (TLog==1).sum()
    TCA=TCount/DefArea
    TLog = scipy.signal.convolve(TLog, np.ones((3,3)),mode='same')
    
    
    SkelT= Image.fromarray(30*SkelArray+100*TLog)
    if Opt.SkeleSh == 1:
        SkelT.show()
    if Opt.SkeleSa==1:
        SkelT.save(os.path.join(Opt.FPath,"output",Opt.BName+"SkelTerm.tif"))
    
    # Junctions
    
    JLog = ((AdCount > 3) * (SkelArray== 1)) # if next to >2 + on skel
    
    SkelAC = SkelArray-JLog # Pruned Skel to use for autocorrelation
    
    JCount = (JLog==1).sum()
    JCA=JCount/DefArea
    JLog = scipy.signal.convolve(JLog, np.ones((3,3)),mode='same')
    
    
    SkelJ= Image.fromarray(30*SkelArray+100*JLog)
    if Opt.SkeleSh == 1:
        SkelJ.show()
    if Opt.SkeleSa==1:
        SkelJ.save(os.path.join(Opt.FPath,"output",Opt.BName+"SkelJunc.tif"))
    
    return(SkelArray, SkelAC, TCount, TCA, JCount, JCA)
#%% Edge Detect
def EdgeDetect(im, Opt, SkeleArray):
    """
    Detects edges using morphological erosion. Then calculates LER.
    This often requires super sampling! I may implement super sampling in this process
    v0.1
    """
    
    EDArray=(im-skimage.morphology.binary_erosion(im, np.ones((3,3)))) #find my edges (erode image and subtract from orig image)
    EDImage = Image.fromarray(100*np.uint8(EDArray))
    EDImage=EDImage.convert(mode="RGB")
    
#        # 1st way find distance from each point to edge
#        LDistA=scipy.ndimage.morphology.distance_transform_edt( (1-LEDA)) #
#        LDistA=LDistA*LASkel #Mask out with skeleton so we only look at distances from center
#        LDistHist, LDistBin=np.histogram(LDistA,bins=100)
#        LDistHist=LDistHist[1:] # remove the 0 distance fake data
#        LDistBin=LDistBin[1:]           
    
    #2nd way. Find distance from each point to center. mask w edge
    EDDistA2=scipy.ndimage.morphology.distance_transform_edt( (1-SkeleArray)) #
    EDDistA2=EDDistA2*EDArray #Mask out with edges. Look at distance from edge
    EDDist, EDDistCnt = np.unique(EDDistA2,return_counts=True) # Save the distances, and their counts
    EDDist=EDDist[1:];EDDistCnt=EDDistCnt[1:] # Ignore the number of zero distance points
    EDDist=EDDist*Opt.NmPP # Make the distances nanometers
    EDGmod=lmfit.models.GaussianModel()
    EDGPars=EDGmod.guess(EDDistCnt, x=EDDist)
    
    EDGFit=EDGmod.fit(EDDistCnt,EDGPars,x=EDDist)
    LER3Sig=3*EDGFit.params.valuesdict()['sigma'] # extract the sigma
    LERMean=EDGFit.params.valuesdict()['center'] # and mean of gauss fit
    EDFig=plt.figure()
    # First Figure : Unsmoothed Currently disabled as not relevant
#    EDFigPlt=EDFig.add_subplot(211)
#    EDFigPlt.set_title('Line Width Roughness fitting, \n Mean Line Dist is %.2f, 3$\sigma$ is %.2f' %(LERMean, LER3Sig))
#    EDFigPlt.set_xlabel('Distance (nm)')
#    EDFigPlt.set_ylabel('Counts (arbitrary)')        
#    EDFigPlt.plot(EDDist, EDDistCnt,         'bo')
#    EDFigPlt.plot(EDDist, EDGFit.best_fit, 'r-')
    
    EDDistFlat=EDDistA2.ravel()[np.flatnonzero(EDDistA2)] # just collect all the distances into a 1d array, dropping any zero distances
    EDDistFlat*=Opt.NmPP # convert to nanometers
    EDDistKDE=scipy.stats.gaussian_kde(EDDistFlat) # this smooths the data by dropping gaussians
    EDDistX=np.linspace(0,EDDistFlat.max(),100)
    EDDistY=EDDistKDE(EDDistX) #smoothed obviously
    EDGFitS=EDGmod.fit(EDDistY,EDGPars,x=EDDistX) #S FOR SMOOTHEDDDD
    LER3SigS=3*EDGFitS.params.valuesdict()['sigma'] # extract the sigma
    LERMeanS=EDGFitS.params.valuesdict()['center'] # and mean of gauss fit         
    
    EDFigPlt2=EDFig.add_subplot(212)
    EDFigPlt2.set_title('1/2th Line Width Roughness fitting (Smoothed), \n Mean Line Dist is %.2f, 3$\sigma$ is %.2f' %(LERMeanS, LER3SigS))
    EDFigPlt2.set_xlabel('Distance (nm)')
    EDFigPlt2.set_ylabel('Counts (arbitrary)')        
    EDFigPlt2.plot(EDDistX, EDGFitS.best_fit, 'r-')
    EDFigPlt2.plot(EDDistX,EDDistKDE(EDDistX), 'bo')
    EDFig.tight_layout()
    



    if Opt.EDSa==1: #save
        EDFig.savefig(os.path.join(Opt.FPath,"output",Opt.BName + "LWR.png"))
        EDImage.save(os.path.join(Opt.FPath,"output",Opt.BName+"ED.tif"))
    
    if Opt.EDSh == 1: # show
        EDFig.show()
        EDImage.show()
    else:
        plt.close(EDFig)
    
    return(LERMean,LER3Sig,LERMeanS,LER3SigS);


#%% Angle Detection
def AngSobel(im):
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
    
    #AngArray is -180- - > +180 lets make it 0<x<=180
    AngArray[AngArray<=0]+=180

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
    AngPlot=plt.figure()
    AngPlot1=AngPlot.add_subplot(111)
    AngPlot1.imshow(AngArray)
    if Opt.AECSh == 1: #REPLACE show

        AngPlot.show() #
    if Opt.AECSa==1: #Replace save
        AngPlot.savefig(os.path.join(Opt.FPath,"output",Opt.BName + "AngEC.png"), dpi=300)
    plt.close(AngPlot)
    return(AngArray)
    

    
#%% Angle Hist: Creates histograms of angles
def AngHist(AngArray,Opt, MaskArray=1, WeightArray='none'):
    """
    Simply takes an array and computes the histograms with 1 degree bins from 0-180
    Also weights pixels with weight array if given, and masks with mask array if given
    Currently only dumps to text the unweighted un masked histogram
    Outputs the number of pixels with angles at the highest peak and second highest as well as total pixels counted
    Which can be used to track progress of alignment or percent alignment
    """
    class AngHist:
        pass
        
    if WeightArray=='none':
        WeightArray=np.ones_like(AngArray)
        
    #angarray=np.absolute(np.pi/4-angarray) #Renormalize to between 0 and pi/4
#    angmask=angarray[maskarray != 0]# Mask out the data note this way flattens
    MaskArray=1.0*MaskArray
    try:
        MaskArray[MaskArray==0]=float('nan')  # replaces 0 with nan in mask array, necessary for histograms to not be overfilled with 0
    except:
        pass
    AngMask=AngArray*MaskArray
#    angmask1=scipy.ndimage.binary_erosion(maskarray,structure=np.ones((3,3)))
#    angmask2=scipy.ndimage.binary_erosion(maskarray)  


    AngHist.Plot=plt.figure();
    AngHist.Plt1=AngHist.Plot.add_subplot(221) # , range=(0,90)
    hist,bins = np.histogram(AngArray, bins=181, range=(0,180))
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt1.bar(center,hist,align='center',width=width)
    AngHist.Plt1.set_title('OD')
    
    #going to dump this one here TODO dump all as BINS COUNTSA COUNTSB with headers
    np.savetxt(os.path.join(Opt.FPath,"output",Opt.BName + "Hist.csv"),hist,delimiter=',')
    # Also find the max and second highest orientation and dump those to output
    
    CntT=np.sum(hist) # cumulative count
    
    Peak1=hist.argmax()
    PLow=Peak1-10;PHigh=Peak1+10 # ten below, ten above are all combined
    if PLow < 0: 
        PLow+=181; # wrap around if under
        Cnt1=np.sum(hist[:Peak1])+np.sum(hist[PLow:])
        hist[:Peak1]=0;hist[PLow:]=0; # set to zero so we don't find again     
    else:
        Cnt1=np.sum(hist[PLow:Peak1])
        hist[PLow:Peak1]=0;
    if PHigh > 180:
        PHigh-=181; # or over
        Cnt1+=np.sum(hist[Peak1:])+np.sum(hist[:PHigh])
        hist[Peak1:]=0;hist[:PHigh]=0;
    else:
        Cnt1+=np.sum(hist[Peak1:PHigh])
        hist[Peak1:PHigh]=0
    
    Peak2=hist.argmax()
    PLow=Peak2-10;PHigh=Peak2+10 # ten below, ten above are all combined
    if PLow < 0: 
        PLow+=181; # wrap around if under
        Cnt2=np.sum(hist[:Peak2])+np.sum(hist[PLow:])
        hist[:Peak1]=0;hist[PLow:]=0; # set to zero so we don't find again     
    else:
        Cnt2=np.sum(hist[PLow:Peak2])
        hist[PLow:Peak2]=0;
    if PHigh > 180:
        PHigh-=181; # or over
        Cnt2+=np.sum(hist[Peak2:])+np.sum(hist[:PHigh])
        hist[Peak1:]=0;hist[:PHigh]=0;
    else:
        Cnt2+=np.sum(hist[Peak2:PHigh])
        hist[Peak2:PHigh]=0


    
    
    AngHist.Plt2=AngHist.Plot.add_subplot(222)
    hist,bins = np.histogram(AngMask, bins=181, range=(0,180))
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt2.bar(center,hist,align='center',width=width)
    AngHist.Plt2.set_title('OD+Mask') 
    
    
    AngHist.Plt3=AngHist.Plot.add_subplot(223)
    hist,bins = np.histogram(AngArray, bins=181, range=(0,180), weights=WeightArray)
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt3.bar(center,hist,align='center',width=width)
    AngHist.Plt3.set_title('OD+Weight')
    
    
    AngHist.Plt4=AngHist.Plot.add_subplot(224)
    hist,bins = np.histogram(AngMask, bins=181, range=(0,180), weights=WeightArray)
    width=0.5*(bins[1]-bins[0]);center=(bins[:-1]+bins[1:])/2
    AngHist.Plt4.bar(center,hist,align='center',width=width)
    AngHist.Plt4.set_title('OD+Mask+Weight')    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)    
    
    if 1 == 1: #REPLACE show
        AngHist.Plot.show()
    if 1==1: #Replace save
        AngHist.Plot.savefig(os.path.join(Opt.FPath,"output",Opt.BName + "AngHist.png"))
    plt.close()
    
    return(Peak1,Cnt1,Peak2,Cnt2,CntT)


#%% Autocorrelation T_T
def AutoCorrelation(im,Opt, AngArray, SkelArray):
    """
    Autocorrelation is a WIP still
    VALPHA
    """
    class AutoCor:
        pass
    # Struct Tens
#    LStructTen=skimage.feature.structure_tensor(CropArray, sigma=1) # Take original image and calc derivatives
#    
#    LAng=np.arctan2(LStructTen[2],LStructTen[0]) # use arctan dy/dx to find direction of line Rads
#    LAngS=LAng*LSkelAC #Mask out with Skeleton

    

    AngSkel=AngArray*SkelArray

    
    
    """
    Note that angles are 0<->pi/2 will use trick later to correct for  
    """
    
    AutoCor.SkI, AutoCor.SkJ=np.nonzero(SkelArray); #Get indexes of nonzero try LASkel/LSKelAC
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
                if np.array( (AutoCor.COORD < SkelArray.shape) ).all(): # If we are still in bounds
                    if (SkelArray[ tuple(AutoCor.COORD)] == 1 & AutoCor.WalkDirect[TestNeighbor] != 7-AutoCor.PastN): # if we have a valid move
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
    return(AutoCor)
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