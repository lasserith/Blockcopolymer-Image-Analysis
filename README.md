# Blockcopolymer-Image-Analysis

Hopefully I'll get the time to more fully details thing as well as clean up the master branched as I inch onwards to graduation but for now here is the light version.

This software takes SEM (specifically zeiss to read the metadata but FEI should be fine), or AFM images and can perform > Denoising > Thresholding > and then calculates a whole bunch of things such as orientation and defect counting. 

By File: 
IAFun holds all the functions called by 
Image Analysis, which has a cute lil GUI with (most of) the options implemented. 
Param Optimizer will take an image and will bruteforce the optimum denoising and thresholding parameters to faithfully threshold the block copolymer image, which is done by tracking the number of defects identified in the thresholded image. In general poor selection of values results in excess defects. (an alternative would be to accept the incorrect thresholding and then prune the skeletons)
Ray1D is an all in one implementation to be used for an upcoming paper. I'll detail more of what it does when that's submitted (hopefully soon TM)
