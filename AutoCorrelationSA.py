class AutoCor:
    pass

MaskA = 1-np.isnan(AngArray) # 0 = nan, 1 = valid
AutoCor.SkI, AutoCor.SkJ=np.nonzero( MaskA ); #Get indexes of non nan
PosNeg = np.array([-1, 1])

AutoCor.RandoList=np.random.randint(0,len(AutoCor.SkI),Opt.ACSize)
AutoCor.uncor=np.zeros((1,2)) # where do we go uncorrelated? how many times.
AutoCor.n=np.zeros(Opt.ACCutoff)
AutoCor.h=np.zeros(Opt.ACCutoff)
AutoCor.Indexes=np.array([[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]) # for picking nearby
AutoCor.IndAngles=np.array([315,270,225,180,135,90,45,0]) #angle of the above in degrees UP is 0 go cw


AutoCor.IndAngles=AutoCor.IndAngles*np.pi/180 # radians
AngArray = AngDetA * np.pi/180 #radians


for AutoCor.Ind in np.arange(Opt.ACSize) : # How many points to start at to calc auto correlate
    # The following is the AutoCor Loop
    AutoCor.ntemp=np.zeros(Opt.ACCutoff) # How many times have calculated the n=Index+1 correlation?
    AutoCor.htemp=np.zeros( Opt.ACCutoff ) # what is the current sum value of the correlation(divide by ntemp at end)
    AutoCor.angtemp=np.ones(Opt.ACCutoff+1)*float('nan') # What is the current angle, 1 prev angle, etc etc
    AutoCor.BBI = 0 # not necessary but helpful to remind us start = BBI 0
    AutoCor.SAD=0;
    #First pick a point, find it's angle
    #TODO
    AutoCor.CCOORD=np.append(AutoCor.SkI[AutoCor.RandoList[AutoCor.Ind]],
                     AutoCor.SkJ[AutoCor.RandoList[AutoCor.Ind]])
    
    AutoCor.angtemp[0]=AngArray[AutoCor.CCOORD[0],AutoCor.CCOORD[1]]         
    AutoCor.BBI = 1 #now we at first point... 
    AutoCor.PastN = np.random.randint(8,16) # No previous point to worry about moving back to
                  
    while AutoCor.BBI <= 2*(Opt.ACCutoff): # How far to walk BackBoneIndex total points is 2*Cuttoff+1 (1st point)
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
            if AutoCor.COORD > AngArray.shape or AngArray[AutoCor.COORD[0],AutoCor.COORD[1]] != float('nan'): # if we have a good move
#                    if AutoCor.BBI==1: # And its the first move we need to fix 1st angle
#                        if AutoCor.angtemp[1] < AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]] - 90: # if angle is 90 lower
#                            AutoCor.angtemp[1]+=np.pi
#                        elif AutoCor.angtemp[1] > AutoCor.IndAngles[AutoCor.WalkDirect[TestNeighbor]] + 90:
#                            AutoCor.angtemp[1]-=np.pi
                        
                AutoCor.PastN=AutoCor.WalkDirect[TestNeighbor];
                AutoCor.CCOORD=AutoCor.COORD; # move there
                AutoCor.angtemp[0]=AngArray[AutoCor.CCOORD[0],AutoCor.CCOORD[1]] # set angle to new angle
                print(AutoCor.angtemp[0])
                DoOnce = 0 # there has to be a more beautiful way to do this
                for AutoCor.PI in range (0,Opt.ACCutoff): # Persistance Index, 0 = 1 dist etc
            #Calculating autocorrelation loop
                    if AutoCor.angtemp[AutoCor.PI+1] != float('nan'):
                        hcalc = np.cos(AutoCor.angtemp[0]-AutoCor.angtemp[AutoCor.PI+1])
                        AutoCor.htemp[AutoCor.PI] += hcalc
                        AutoCor.ntemp[AutoCor.PI] += 1
                    if hcalc <= 0 & DoOnce == 0:
                        AutoCor.uncor[0,0] += AutoCor.PI
                        AutoCor.uncor[0,1] += 1
                        DoOnce = 1
                        
                break # break the for loop (done finding next point)
                
            elif TestNeighbor==6: # else if we at the end
                # Need to break out of the backbone loop as well...
                AutoCor.SAD=1; # because
                
        if AutoCor.SAD==1: # break out of BB while loop
            # Decide if I count this or not...
            AutoCor.SAD=0;
            break
        
        # BUT WAIT WE NEED TO FIX THE NEW ANGLE TOO!
#            if AutoCor.angtemp[0] < AutoCor.IndAngles[AutoCor.PastN] - 90: # if angle is 90 lower
#                AutoCor.angtemp[0]+=np.pi
#            elif AutoCor.angtemp[0] > AutoCor.IndAngles[AutoCor.PastN] + 90:
#                AutoCor.angtemp[0]-=np.pi         
                          
        
        
                
        #FinD next point
        AutoCor.BBI+=1
        
        if AutoCor.BBI==2*(Opt.ACCutoff): # we found all our points!
            AutoCor.h +=AutoCor.htemp
            AutoCor.n +=AutoCor.ntemp
            AutoCor.Ind += 1