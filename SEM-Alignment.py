# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 17:19:28 2018

@author: Moshe
"""

import tkinter as tk
from tkinter import filedialog, ttk
import os
import IAFun
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

if __name__ == '__main__':
    # OK
    
    Vers = "AAA"
    
    # Will hold options
    class Opt:
        pass

    #%%
    Opt.FFTSh = 0
    Opt.FFTSa = 0
    Opt.CropT = 0
    Opt.CropL = 0
    Opt.CropR = 0
    Opt.CropB = 58
    #Opt.AlignAng = 30
    Opt.AlignSize = 0.1 # +/- 10 %
    #%% 
    
    FOpen=tk.Tk()
    
    currdir = os.getcwd()
    FNFull = tk.filedialog.askopenfilename(parent=FOpen, title='Please select a file', multiple=1)
    FOpen.withdraw()
    # any do once?
    ImNum = 0

    #%%
    for ImNum in range(0, len(FNFull)):
        Opt.AlignAng = 30
        Opt.Name = FNFull[ImNum] # this hold the full file name
        Opt.FPath, Opt.BName= os.path.split(Opt.Name)  # File Path/ File Name
        (Opt.FName, Opt.FExt) = os.path.splitext(Opt.BName) # File name/File Extension split
        RawIn = IAFun.AutoDetect( FNFull[ImNum], Opt)
        CropArray = IAFun.Crop(RawIn, Opt)[0]
        Opt.L0 = IAFun.FFT(CropArray, Opt)
        #%% for nowsies do 30 and 15?
        PercAlign, PercZero = IAFun.FFTAlignment(CropArray,Opt)
        Opt.AlignAng = 15
        PercAlign15, PercZero15 = IAFun.FFTAlignment(CropArray,Opt)
        #%%
        print("%s, %.2f,%.2f"%(Opt.FName,Opt.L0,PercAlign))
        TempOutput = np.vstack( (Opt.FName, Opt.L0, PercAlign, PercZero, PercAlign15, PercZero15 )).transpose()
        if ImNum == 0:
            Output = np.copy(TempOutput)
        else:
            Output = np.vstack((Output,TempOutput))
        
    np.savetxt(os.path.join(Opt.FPath,"PercentAlignment.csv"), Output,fmt='%s',delimiter=',')