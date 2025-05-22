#!/usr/bin/env python
# coding: utf-8
#Note this is edited outside of the github version on 11/13/2024 to handle entire dataset point simulations
#Rewritten for Python 3.8

#Modified theta and phi for zenith and azimuthal angle (respectively) 11/13/2024
#Zenith is angle from +z axis, azimuthal is angle form +x axis. 

#modified for COMMAS output on 11/12/2024

#Modified for a tree based clound boundary query 7/11/2024

#Modified for a 3d scattering profile 3/28/2024

#Modified for cloud boundary logic 12/15/2025
# 

# ;@knb_mc_sim
# ;WRF Core profile: 8-23-2019, using mean level radii from DSD and conc from dry_air_density*qndrop/ice/etc
# ;Uses 1km average profile. Uncomment the levels in lambda if/then statements to use other averaging values.
# ;+
# ; Author: Kelcy Brunner, kelcy.brunner@ttu.edu
# ;The purpose of this program is to simulate the randomized optical scattering occuring in the 777.4nm wavelength.
# ;-
# 
# ;+
# ; Description:
# ;   This model ingests a number of photons (m), a source position, the shape of the cloud, the output array, and a microphysical profile.
# ;   All of which may be specified in the call line, or in a companion wrapper program title knb_mc_wrapper.pro
# ;
# ; Parameters:
# ;   m: value of the number of photons to be simulated
# ;        Type: int, or long int.
# ;   concentration: number concentration (in count per m^3), same dimensions as model coordinates, in the format [t,z,y,x]
# ;        Typically this is produced in the file knb_calc_distro_MODELTYPE.ipynb
# ;   size: concentration weighted mean particle size (in count per m), same dimensions as model coordinates, in the format [t,z,y,x]
# ;        Typically this is produced in the file knb_calc_distro_MODELTYPE.ipynb
# ;        Type: float 32/64
# ;   profile_in: xarray dataset of the constituents and coordinates of the concentration and size parameters. 
# ;        Typically this is produced in the file knb_calc_distro_MODELTYPE.ipynb
# ;        Type: xarray.core.dataset.Dataset
# ;   tree: KD tree of the xyz locations for each grid space with a mean free path. This is often included in the sim, but
# ;        is kept offline here to avoid repeating the calculation.
# ;        Type: scipy.spatial._kdtree.KDTree
# ;   masktree: KD tree of the xyz locations for each grid space with a mean free path less than the cloud MFP (200m).
# ;        This is often included in the sim, but is kept offline here to avoid repeating the calculation.
# ;        Type: scipy.spatial._kdtree.KDTree
# ;   LAM_arr: 1d array of MFP corresponding to the above concentration and size parameters calculated MFP, same length.
# ;        This is often included in the sim, but is kept offline here to avoid repeating the calculation.
# ;        Type: numpy.ndarray
# ;
# ;
# ; Keywords
# ;   SEEDER: a seed for randomization. The default is systime(/sec).
# ;       Type: a single or double precision value, but a whole number is required by randomu.
# ;
# ;   COORDS: A 6 element array of the cloud coordinates of the faces in the following order: [x+, x-, y+, y-, z+, z-].
# ;   The units must be in the same units as the origin location, and if not specified is chosen as meters (m).
# ;       Type: Single or double precision is sufficient.
# ;
# ;   ORIGIN: A 3 element array [z, y, x] specifying the position of the photons to be simulated. The units must be the same as the COORDS keyword, the default is meters (m).
# ;       Type: Single or double precision is sufficient.
# ;
# ;   TOL: this is a percentage of the MFP that is an allowable range within to consider a 'boundary' - e.g., 25% 
# ;       Type: float, Default: 0.1
# ;   LIMIT: this is the distance (m) from the cloud you may be to still be included in the cloud. In general it is best to be no greater than 1 dx/dy/dz length. 
# ;       Type: float, Default: 100m
# ;
# ; Output/Returns
# ;      [m,9] numpy.ndarray
# ;		 [m,0]: numpy array of final x positions for all photons	 array_out =np.column_stack((x_arr,y_arr,z_arr,xprev_arr, yprev_arr, zprev_arr,k_arr,absorb_flag,dist)) #,zenith_arrout,azimuth_arrout))
# ;		 [m,1]: numpy array of final y positions for all photons
# ;		 [m,2]: numpy array of final z positions for all photons
# ;		 [m,3]: numpy array of second to final x positions for all photons
# ;		 [m,4]: numpy array of second to final y positions for all photons
# ;		 [m,5]: numpy array of second to final z positions for all photons
# ;		 [m,6]: numpy array of the number of times the photon scattered 
# ;		 [m,7]: numpy array of the absorption flag: 0 = not absorbed, 1 = absorbed
# ;		 [m,8]: numpy array of the distance each photon traveled through its path. 
# ;
# ;
# ;
# ; Call example:
# ;   out = knb_mc_sim(m,profile_in['particle_concentration'].values,profile_in['particle_size'].values,profile_in,tree,masktree,LAM_arr,SEEDER  = 10,origin = origin, LIMIT = 500.)
# ;
# ;
# ;-Kelcy
# ;step 1: Photon is emitted in a random direction from the source location.
#   
# ;step 2: Photon travels with an assumed probability that it travels a distance x without an interaction given by P(x) = exp(-x/LAM) where LAM is the mean free path.
# 
# ;step 3: The photon 'encounters' a scatterer, it is either absorbed or scattered
#   
# ;step 4: Check if photon is still within the cloud, if yes then it travels another distance x and step 3 is repeated or it exits the cloud.
#   
# ;step 5: Check if photon is outside the cloud or absorbed, if it is, note the final photon position.
#   
# ;step 6: repeat for another photon.
# 

import numpy as np
from glob import glob
import matplotlib
import xarray as xr
import math
from scipy.io import readsav
from pandas.core.common import flatten
from scipy import spatial, constants
import pandas
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def outside_of_coords(point, coordinates):
    #3d point with format x, y, z
    #Bounding coordinates of the model domain, of the format X+, X-, Y+, Y-, Z+, Z-
    bound = 0
    if point[0] > coordinates[0]:
        bound = 1
    if point[0] < coordinates[1]:
        bound = 1        
    if point[1] > coordinates[2]:
        bound = 1
    if point[1] < coordinates[3]:
        bound = 1
    if point[2] > coordinates[4]:
        bound = 1
    if point[2] < coordinates[5]:
        bound = 1
    return bound
def knb_mc_sim(m,concentration,size,PROFILE_IN,tree,masktree, LAM_arr,SEEDER = 10,
               LIMIT = 100.,
               HOMOGENEOUS = False,TOL = 0.1,
               coords = [40e3,0.,40e3,0.,20e3,0.], 
               origin = [5e3,5e3,5e3]):

    
    #origin order is [z,y,x] because gridcoords is z,y,x
    print(origin)
    previousloc = 0 #0 for out of cloud, 1 for in cloud
    currentloc = 0 #0 for out of cloud, 1 for in cloud
    # print(coords)
    #coords is xmax,xmin, ymax,ymin, zmax,zmin
    #Find the dx, dy, dz
    dx = PROFILE_IN['x'][1].values - PROFILE_IN['x'][0].values
    dy = PROFILE_IN['y'][1].values - PROFILE_IN['y'][0].values
    dz = PROFILE_IN['z'][1].values - PROFILE_IN['z'][0].values
    
    
    #Define your boundaries and arrays to hold when a boundary is crossed, specifically if using a hard lateral or vertical boundary
    s = np.size(coords)
    
    ##########CONSTANTS###########
    #Asymmetry factor for a 10micron water drop in the visible spectral region.
    g = 0.87 #double?
    #single scattering albedo for a water droplet of 10 microns in the visible spectral region.
    w_0 = 0.99998

    #the radius of a typical cloud droplet
    a = 10**-5. #evaluate if e is a double, or long
    N = 10**8.
    LAM = 1./(2.*np.pi*(a**2.)*N)

    
    #####calculate the array of possible scattering angles using the Henyey-Greenstein phase function
    N = 2001#e0
    mu_arr = np.zeros(N)
    mu_arr[0] = -1.#formerly 0D
    for i in np.arange(1,len(mu_arr)-1):
        mu_arr[i] = ((1. + g**2.)-((2.*g)/(N*(1. - g**2.)) + (1. + g**2. - 2.*g*mu_arr[i-1])**(-1./2.))**(-2.))/(2.*g)
        if mu_arr[i] > 1.:
            mu_arr[i] = 1.


    pp = np.arange(len(mu_arr),dtype = int)
    #*********
    
    x_arr = np.zeros(m) #array of xyz positions of final location
    y_arr = np.zeros(m)
    z_arr = np.zeros(m)
    xprev_arr = np.zeros(m) #array of xyz positions of second to last location
    yprev_arr = np.zeros(m)
    zprev_arr = np.zeros(m)
    # zenith_arrout = np.zeros(m) 
    # azimuth_arrout = np.zeros(m)
    absorb_flag = np.zeros(m) #0 for scattered through, 1 for if the photon is absorbed
    dist = np.zeros(m) #distance traveled from emission to point of last scattering
    k_arr = np.zeros(m) #number of segments between emission and final position. If 1, no scattering occurred
 
    seed1 = SEEDER
    seed2 = seed1 + 1
    seed3 = seed2 + 2
    seed4 = seed3 + 3
    seed5 = seed4 + 4
    seed6 = seed5 + 5
    seed7 = seed6 + 6
    seed8 = seed7 + 7
    seed9 = seed8 + 8

    #11/13/2024 Changed phi and theta to azimuth and zenith angles naming convention
    #11/13/2024 Note Phi is the angle from the positive x axis, theta is the angle from the positive z axis. This may be backwards from other conventions
    #changed on on 8/6/2024
    #;initial photon isotropically scattered directions
    rng = np.random.default_rng(seed = seed1)
    u = rng.uniform(0,1,m)#randomu(seed1,m)
    rng = np.random.default_rng(seed = seed2)
    azimuth_0 = rng.uniform(0,2.*math.pi,m)#v = randomu(seed2,m) 
    rng = np.random.default_rng(seed = seed3)
    coszenith = rng.uniform(-1,1,m)
    zenith_0 = np.arccos(coszenith)
    rng = np.random.default_rng(seed = seed4)
    ab_arr_m = rng.uniform(0,1, m)
    rng = np.random.default_rng(seed = seed5)
    r_m = rng.uniform(0,1,m) 
       


    #UNCOMMENT THIS FOR DISTRIBUTED LAMBDA
    LAM = LAM_arr[tree.query(origin)[1]]
    # LAM = 70. #uncomment for homogeneous atmosphere for funsies
    print(LAM)
    x_scat_0 = -LAM*np.log(r_m)
    
    #two cases:
    #1 Emitted in the cloud: 
    #1 emitted out of the cloud
    point = masktree.query(origin)
    if point[0] < LIMIT:
        currentloc = 1
    else: currentloc = 0

    
    
    rng = np.random.default_rng(seed = seed6)
    ind_m = rng.uniform(0,1,m) 
    

    rng = np.random.default_rng(seed = seed7)
    zenith_ind_m_seed = rng.uniform(0,1,m) #used to seed something else
    rng = np.random.default_rng(seed = seed8)
    azimuth_r_m = rng.uniform(0,1,m) #used to seed something else
    rng = np.random.default_rng(seed = seed9)
    r_m_1 = rng.uniform(0,1,m)
        
    for j in np.arange(m): 
        if (j % 10) == 0:
            print(j)
        counter = 0
        bound = 0
        k=0
        rng = np.random.default_rng(seed = (ab_arr_m[j]*100).astype(int))
        ab_arr = rng.uniform(0,1, 2000000)
    
        if not np.logical_and(zenith_0[j], azimuth_0[j]):
            print('began')#  then begin

            mu_x = np.sin(zenith_0[j])*np.cos(azimuth_0[j])
            mu_y = np.sin(zenith_0[j])*np.sin(azimuth_0[j])
            mu_z = np.cos(zenith_0[j])

            #;positions
            x = x + x_scat_0[j]*mu_x
            y = y + x_scat_0[j]*mu_y
            z = z + x_scat_0[j]*mu_z
            endif 

        rng_ind = np.random.default_rng(seed = (ind_m[j]*100.).astype(int))
        ind = rng_ind.uniform(0,1,2000)*1999. + 1.
        zenith_arr = np.arccos((mu_arr[pp[1:]] + mu_arr[0:-1])/2.)
        rng = np.random.default_rng(seed = (zenith_ind_m_seed[j]*100.).astype(int))
        zenith_ind = (rng.uniform(0,1,500000)*2000.).astype(int)
        zenith_p = zenith_arr[zenith_ind]
        rng = np.random.default_rng(seed = (azimuth_r_m[j]*100.).astype(int))
        azimuth_p = rng.uniform(0,2*math.pi,500000)
        rng = np.random.default_rng(seed = (r_m_1[j]*100.).astype(int))
        r = rng.uniform(0,1,500000)
        x_scat = -np.log(r)

    
    #First traveled distance  
       
        mu_x = np.sin(zenith_0[j])*np.cos(azimuth_0[j])
        mu_y = np.sin(zenith_0[j])*np.sin(azimuth_0[j])
        mu_z = np.cos(zenith_0[j])
    
        # ;positions
        xprev = origin[2]
        yprev = origin[1]
        zprev = origin[0]
        
        x = origin[2] + x_scat_0[j]*mu_x
        y = origin[1] + x_scat_0[j]*mu_y
        z = origin[0] + x_scat_0[j]*mu_z
        previousloc = currentloc
        point = masktree.query([z,y,x])
        if point[0]<LIMIT:
            currentloc = 1
        else:
            currentloc = 0
       
 #LOGIC FOR In/Out of cloud options                
        if (previousloc == 0) or (previousloc != currentloc):

            v = [(z - zprev),(y - yprev),(x - xprev)]
            testlam = []
            ptdist = np.sqrt((z-zprev)**2 + (y-yprev)**2 + (x-xprev)**2)

            index = np.arange(500)*(dx/ptdist)
            x1 = xprev + index*v[2]
            y1 = yprev + index*v[1]
            z1 = zprev + index*v[0]
            
            for t in np.arange(500):
                testlam.append((LAM_arr[tree.query([z1[t],y1[t],x1[t]])[1]]))
            newind = np.argmin(np.abs((1+TOL)*LIMIT- np.array(testlam)))   
            if (previousloc == 0) and (currentloc == 0):
                if ~np.any(np.array(testlam) <= (1+TOL)*LIMIT):
                    newind = -1
            min_t = index[newind]
            x = xprev + min_t*v[2]
            y = yprev + min_t*v[1]
            z = zprev + min_t*v[0]
#     #The position has now been updated to the cloud boundary, we can reset and scatter again.
            LAM = testlam[newind]
             

        dist[j] += np.sqrt( (x - xprev)**2 + (y - yprev)**2 + (z - zprev)**2)
        if outside_of_coords([z,y,x], coords):
            # print('ending because bounds')
            x_arr[j] = x
            y_arr[j] = y
            z_arr[j] = z
            xprev_arr[j] = xprev
            yprev_arr[j] = yprev
            zprev_arr[j] = zprev
            k_arr[j]=k
            continue

        while bound < 1: 
            counter +=1
            # ;check to see if absorbed
            if ab_arr[k] > w_0: absorb_flag[j] = 1 # += 1
            if ab_arr[k] > w_0: break


# ;Find the photon's second position
# ;direction cosines
            denom = np.sqrt(1. - mu_z**2.)
            mu_x_new = ((np.sin(zenith_p[k])*(mu_x*mu_z*np.cos(azimuth_p[k]) - mu_y*np.sin(azimuth_p[k])))/denom) + mu_x*np.cos(zenith_p[k])
            mu_y_new = ((np.sin(zenith_p[k])*(mu_y*mu_z*np.cos(azimuth_p[k]) + mu_x*np.sin(azimuth_p[k])))/denom) + mu_y*np.cos(zenith_p[k])
            mu_z_new = -(denom)*np.sin(zenith_p[k])*np.cos(azimuth_p[k]) + mu_z*np.cos(zenith_p[k])

            if not math.isfinite(z):
                print('broken')
                break
            if not math.isfinite(y):
                print('broken')
                break
            if not math.isfinite(x):
                print('broken')
                break
             
            LAM = LAM_arr[tree.query([z,y,x])[1]]

            xprev = x
            yprev = y
            zprev = z
            x = x + LAM*x_scat[k]*mu_x_new
            y = y + LAM*x_scat[k]*mu_y_new
            z = z + LAM*x_scat[k]*mu_z_new
            mu_x = mu_x_new
            mu_y = mu_y_new
            mu_z = mu_z_new
            k += 1
        

            previousloc = currentloc
            point = masktree.query([z,y,x])
            if point[0]<LIMIT:
                currentloc = 1
            else:
                currentloc = 0
       
 #LOGIC FOR In/Out of cloud options                
            if (previousloc == 0) or (previousloc != currentloc):

                v = [(z - zprev),(y - yprev),(x - xprev)]
                testlam = []
                ptdist = np.sqrt((z-zprev)**2 + (y-yprev)**2 + (x-xprev)**2)

                index = np.arange(500)*(dx/ptdist)
                x1 = xprev + index*v[2]
                y1 = yprev + index*v[1]
                z1 = zprev + index*v[0]
            
                for t in np.arange(500):
                    testlam.append((LAM_arr[tree.query([z1[t],y1[t],x1[t]])[1]]))
                newind = np.argmin(np.abs((1+TOL)*LIMIT- np.array(testlam)))   
                if (previousloc == 0) and (currentloc == 0):
                    if ~np.any(np.array(testlam) <= (1+TOL)*LIMIT):
                        newind = -1
                min_t = index[newind]
                x = xprev + min_t*v[2]
                y = yprev + min_t*v[1]
                z = zprev + min_t*v[0]
#     #The position has now been updated to the cloud boundary, we can reset and scatter again.
                LAM = testlam[newind]
             

            dist[j] += np.sqrt( (x - xprev)**2 + (y - yprev)**2 + (z - zprev)**2)

            if outside_of_coords([z,y,x], coords):
                bound = 1
                
        x_arr[j] = x
        y_arr[j] = y
        z_arr[j] = z
        xprev_arr[j] = xprev
        yprev_arr[j] = yprev
        zprev_arr[j] = zprev
        k_arr[j]=k
        # print([z,y,x])
    
    # print(absorb)
    array_out =np.column_stack((x_arr,y_arr,z_arr,xprev_arr, yprev_arr, zprev_arr,k_arr,absorb_flag,dist)) #,zenith_arrout,azimuth_arrout))
    return(array_out)#, pathx,pathy,pathz)        
  

#This is a simple single simulation with a single origin. M is the number of 'photons'. Keep track of seeds for your own use and do not replicate
#seeds in the same simulation. For extended sources repeat this simulation with the updated origin. Origin points are given in meters., Similarly with Coords
#Coords is as legacy input as the cloud is determined by the mean free path. Should you prefer the cloud boundary be a hard boundary set accordingly. 


m = 2500

profile_in = xr.open_dataset('3D_COMMAS_KTAL071015B_30s.002430.nc')

        #make points and kdtree for cloud coordinates
zz,yy,xx = np.meshgrid(profile_in['z'].values, profile_in['y'].values,profile_in['x'].values,indexing='ij')
length = len(profile_in['x'])*len(profile_in['y'])*len(profile_in['z'])
xx = np.reshape(xx,length)
yy = np.reshape(yy,length)
zz = np.reshape(zz,length)
gridcoords = np.stack((zz,yy,xx),axis = 1) 
tree = spatial.KDTree(gridcoords)
    
N_dense = np.reshape(profile_in['particle_concentration'].values,length)
a_arr = np.reshape(profile_in['particle_size'].values,length)
a_arr = np.array([0 if math.isnan(i) else i for i in a_arr])
N_dense = np.array([0 if math.isnan(i) else i for i in N_dense])
    
    # ######Check microphysical profile

LAM_arr = (1./(2.*math.pi*(a_arr**2.)*N_dense)) #make sure a_arr isn't creating a matrix
LAM_arr = np.nan_to_num(LAM_arr,posinf = 999999,neginf = 999999)

    #NOTE, WE USE A MEAN FREE PATH OF 200m here as the cloud boundary. It is on the user to determine if this is appropriate for THEIR CLOUD.
mask = np.where(LAM_arr <= 200)
mask = np.array(list(flatten(mask)))
maskstack = np.stack((zz[mask],yy[mask],xx[mask]),axis=1)
masktree = spatial.KDTree(maskstack)
origin = [4062.5,21e3,15e3]
out = knb_mc_sim(m,profile_in['particle_concentration'].values,profile_in['particle_size'].values,profile_in,
                           tree,masktree,LAM_arr,SEEDER  = 10,origin = origin, LIMIT = 500.)
df = pd.DataFrame(out)
file_name = 'COMMAS30s_'+str(origin[0])+'_'+str(origin[1])+'_'+str(origin[2])+'.csv'


print(file_name)
df.to_csv(file_name, index=False)
