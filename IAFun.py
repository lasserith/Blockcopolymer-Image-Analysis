# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:29:51 2016

@author: Moshe Dolejsi MosheDolejsi@uchicago.edu
"""
#%%
Vers="0.1"

#%%
import lmfit

import os
import csv
import numpy as np
import skimage
from skimage import restoration, morphology, filters, feature

import re #dat regex
import matplotlib.pyplot as plt
import exifread #needed to read tif tags

import scipy

#%%
def AutoDetect( FNFull ):
    """
    Attempt to autodetect the instrument used to collect the image
    Currently supports the Zeiss Merlin
    V0.1
    """
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
