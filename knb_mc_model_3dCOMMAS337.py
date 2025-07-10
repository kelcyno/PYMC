#11/26/2024: This version is for 337nm, 3D commas input. 

#Rewritten for Python 3.8

#Modified theta and phi for zenith and azimuthal angle (respectively) 11/13/2024
#Zenith is angle from +z axis, azimuthal is angle form +x axis. 

#modified for COMMAS output on 11/12/2024

#Modified for a tree based clound boundary query 7/11/2024

#Modified for a 3d scattering profile 3/28/2024

#Modified for cloud boundary logic 12/15/2025

#This simulation considered a cloud boundary, the simulation ends at the edge of the cloud. When the photon exceeds the cloud, the path is traced back to the point along it's trajectory where the cloud boundary was crossed, and that is the 'end' point. If the photon never entered the cloud, then it's final point is considered on the face of the simulation boundary. 6/28/2025


# ;@knb_mc_sim
# ;WRF Core profile: 8-23-2019, using mean level radii from DSD and conc from dry_air_density*qndrop/ice/etc
# ;Uses 1km average profile. Uncomment the levels in lambda if/then statements to use other averaging values.
# ;+
# ; Author: Kelcy Brunner, kelcy.brunner@ttu.edu
# ;The purpose of this program is to simulate the randomized optical scattering occuring in the 337nm wavelength.
# ;-
# 
# ;+
# ;+
# ; Description:
# ;   This model ingests a number of photons (m), a source position, the shape of the cloud, the output array, and a microphysical profile.
# ;   All of which may be specified in the call line, or in a companion wrapper program title knb_mc_wrapper.pro
# ;
# ; Parameters:
# ;   m: value of the number of photons to be simulated
# ;        Type: int, or long int.
# ;   profile_in: xarray dataset of the constituents and coordinates of the concentration and size parameters. 
# ;        Typically this is produced in the file knb_calc_distro_MODELTYPE.ipynb
# ;        Type: xarray.core.dataset.Dataset
# ;   tree: KD tree of the xyz locations for each grid space with a mean free path. This is often included in the sim, but
# ;        is kept offline here to avoid repeating the calculation.
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
#    DISTLIMIT: this is the distance (m) from the cloud you may be to still be included in the cloud. In general it is best to be no greater than 1 dx/dy/dz length. 
#        Type: float, Default: 200m
#    LAMLIMIT: this is MFP limit to be considered a sufficiently different environment to reevaluate the photon position.  
#        Type: float, Default: 200m
# ;
# ; Output/Returns
# ;		 [m,0]: numpy array of final x positions for all photons	
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
# ;   out = knb_mc_sim(m,profile_in,tree,LAM_arr,SEEDER  = 15,origin = origin, DISTLIMIT = 125.,LAMLIMIT=200.,coords = coords)
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
import matplotlib.pyplot as plt
import math
from pandas.core.common import flatten
from scipy import spatial, constants
import pyart
import pandas
import pandas as pd
import re

def outside_of_coords(point, coordinates):
    #3d point with format x, y, z
    #Bounding coordinates of the model domain, of the format X+, X-, Y+, Y-, Z+, Z-
    bound = 0
    if point[0] > coordinates[0]:
        bound = 1
    if point[0] < coordinates[1]:
        bound = 2        
    if point[1] > coordinates[2]:
        bound = 3
    if point[1] < coordinates[3]:
        bound = 4
    if point[2] > coordinates[4]:
        bound = 5
    if point[2] < coordinates[5]:
        bound = 6
    return bound

def find_crossing(prevpoint, point,coords):
    v = [(point[0] - prevpoint[0]),(point[1] - prevpoint[1]),(point[2] - prevpoint[2])]
    test = outside_of_coords(point, coords) - 1
    vi = math.floor(test/2)
    if vi == 2:
        t = (coords[test] - prevpoint[2])/v[vi]
    if vi ==1:
        t = (coords[test] - prevpoint[1])/v[vi]
    if vi == 0:
        t = (coords[test] - prevpoint[0])/v[vi]
    x = prevpoint[0] + t*v[0]
    y = prevpoint[1] + t*v[1]
    z = prevpoint[2] + t*v[2]
    return [x,y,z]
    
    
def knb_mc_sim(m,PROFILE_IN,tree,LAM_arr,SEEDER = 10.,
               DISTLIMIT = 200.,LAMLIMIT = 200.,
               HOMOGENEOUS = False,TOL = 0.1,
               coords = [40e3,0.,40e3,0.,20e3,0.], 
               origin = [5e3,5e3,5e3]):

    concentration = PROFILE_IN['particle_concentration']
    size = PROFILE_IN['particle_size']
    #origin order is [z,y,x] because gridcoords is z,y,x
    # print(origin)

    #coords is xmax,xmin, ymax,ymin, zmax,zmin
    #Find the dx, dy, dz
    dx = PROFILE_IN['x'][1].values - PROFILE_IN['x'][0].values
    dy = PROFILE_IN['y'][1].values - PROFILE_IN['y'][0].values
    dz = PROFILE_IN['z'][1].values - PROFILE_IN['z'][0].values
    
        
    liquid_frac = PROFILE_IN['liquid_concentration'].values/concentration
    frozen_frac = PROFILE_IN['frozen_concentration'].values/concentration
    
    length = len(PROFILE_IN['time'])*len(PROFILE_IN['x'])*len(PROFILE_IN['y'])*len(PROFILE_IN['z'])

    liquid_frac = np.reshape(liquid_frac.data,length)
    frozen_frac= np.reshape(frozen_frac.data, length)
    ind =  np.isnan(liquid_frac)
    liquid_frac[ind] = 0.0 
    ind =  np.isnan(frozen_frac)
    frozen_frac[ind] = 0.0 
    
    #origin order is [z,y,x] because gridcoords is z,y,x
    print(origin)
    print(coords)

    #Define your boundaries and arrays to hold when a boundary is crossed, specifically if using a hard lateral or vertical boundary
    s = np.size(coords)
    
    ##########CONSTANTS###########
    #Asymmetry factor for a 10micron water drop in the visible spectral region.
    g = 0.87 #double?
    g_ice = 0.88
    g_liq = 0.83
    #single scattering albedo for a water droplet of 10 microns in the visible spectral region.
    w_0 = 0.99998
    w_0liq = 0.9999947
    w_0ice = 0.9999975
    liqp = [ 100.,  95.,  90. , 85. , 80. , 75. , 70.  ,65. , 60. , 55., 50., 45. ,40. ,35. ,30. ,25. ,20., 15.,
     10., 5.]
    icep = [0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.]
    #the radius of a typical cloud droplet
    a = 10**-5. #evaluate if e is a double, or long
    N = 10**8.
    LAM = 1./(2.*np.pi*(a**2.)*N)
    # LAM = 70.

    N_dense = np.reshape(concentration.data,length)
    a_arr = np.reshape(size.data,length)
    a_arr = np.array([0 if math.isnan(i) else i for i in a_arr])
    N_dense = np.array([0 if math.isnan(i) else i for i in N_dense])
    
    #     ######Determine mean free path
    LAM_arr = (1./(2.*math.pi*(a_arr**2.)*N_dense)) #make sure a_arr isn't creating a matrix
    LAM_arr = np.nan_to_num(LAM_arr,posinf = 999999,neginf = 999999)
    
    #####calculate the array of possible scattering angles using the Henyey-Greenstein phase function
    N = 2001#e0
    mu_arr = np.zeros(N)
    mu_arr[0] = -1.#formerly 0D
    for i in np.arange(1,len(mu_arr)-1):
        mu_arr[i] = ((1. + g**2.)-((2.*g)/(N*(1. - g**2.)) + (1. + g**2. - 2.*g*mu_arr[i-1])**(-1./2.))**(-2.))/(2.*g)
        if mu_arr[i] > 1.:
            mu_arr[i] = 1.

    pp = np.arange(len(mu_arr),dtype = int)
    
        #####calculate the array of possible scattering angles using the Henyey-Greenstein phase function
    #HG FOR WATER
    N = 2001#e0
    mu_arr_liq = np.zeros(N)
    mu_arr_liq[0] = -1.#formerly 0D
    g = g_liq
    for i in np.arange(1,len(mu_arr_liq)-1):
        mu_arr_liq[i] = ((1. + g**2.)-((2.*g)/(N*(1. - g**2.)) + (1. + g**2. - 2.*g*mu_arr_liq[i-1])**(-1./2.))**(-2.))/(2.*g)
        if mu_arr_liq[i] > 1.:
            mu_arr_liq[i] = 1.
    
    
    #HG FOR ICE
    N = 2001#e0
    mu_arr_ice = np.zeros(N)
    mu_arr_ice[0] = -1.#formerly 0D
    g = g_ice
    for i in np.arange(1,len(mu_arr_ice)-1):
        mu_arr_ice[i] = ((1. + g**2.)-((2.*g)/(N*(1. - g**2.)) + (1. + g**2. - 2.*g*mu_arr_ice[i-1])**(-1./2.))**(-2.))/(2.*g)
        if mu_arr_ice[i] > 1.:
            mu_arr_ice[i] = 1.
    pp_liq = np.arange(len(mu_arr_liq),dtype = int)#;*********
    pp_ice = np.arange(len(mu_arr_ice),dtype = int)#;*********
    
    
    #*********   
    x_arr = np.zeros(m)
    y_arr = np.zeros(m)
    z_arr = np.zeros(m)
    xprev_arr = np.zeros(m)
    yprev_arr = np.zeros(m)
    zprev_arr = np.zeros(m)
    # zenith_arrout = np.zeros(m)
    # azimuth_arrout = np.zeros(m)
    absorb_flag = np.zeros(m) #0 for scattered through, 1 for if the photon is absorbed
    dist = np.zeros(m) #distance traveled from emission to point of last scattering
    k_arr = np.zeros(m) #number of segments between emission and final position. If 1, no scattering occurred
 
    liq = 0
    ice = 0
 
    # zenith_orig = np.zeros(m)
    # azimuth_orig = np.zeros(m)
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
    # LAM = 70.
    x_scat_0 = -LAM*np.log(r_m)

    rng = np.random.default_rng(seed = seed6)
    ind_m = rng.uniform(0,1,m) 
    

    rng = np.random.default_rng(seed = seed7)
    zenith_ind_m_seed = rng.uniform(0,1,m) #used to seed something else
    rng = np.random.default_rng(seed = seed8)
    azimuth_r_m = rng.uniform(0,1,m) #used to seed something else
    rng = np.random.default_rng(seed = seed9)
    r_m_1 = rng.uniform(0,1,m)
     

    for j in np.arange(m): 
        if (j % 100) == 0:
            print(j)
        counter = 0
        bound = 0
        k=0
        rng = np.random.default_rng(seed = (ab_arr_m[j]*100).astype(int))
        ab_arr = rng.uniform(0,1, 200000)
    
        if not np.logical_and(zenith_0[j], azimuth_0[j]):
            print('began the weird case')#  then begin
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
        zenith_arr_liq = np.arccos((mu_arr_liq[pp_liq[1:]] + mu_arr_liq[0:-1])/2.)
        zenith_arr_ice = np.arccos((mu_arr_ice[pp_ice[1:]] + mu_arr_ice[0:-1])/2.)
        rng = np.random.default_rng(seed = (zenith_ind_m_seed[j]*100.).astype(int))
        zenith_ind = (rng.uniform(0,1,20000)*2000.).astype(int)
        zenith_p_liq = zenith_arr_liq[zenith_ind]
        zenith_p_ice = zenith_arr_ice[zenith_ind]
        rng = np.random.default_rng(seed = (azimuth_r_m[j]*100.).astype(int))
        azimuth_p = rng.uniform(0,2*math.pi,20000)
        rng = np.random.default_rng(seed = (r_m_1[j]*100.).astype(int))
        r = rng.uniform(0,1,20000)
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

        
        ptdist = np.sqrt((z-zprev)**2 + (y-yprev)**2 + (x-xprev)**2)
        if LAM_arr[tree.query([z,y,x])[1]] != LAM: 
            #this is only considered if the LAM is not the same as the previous location. 
            #Otherwise it is still travling in the same voxel.
            v = [(x - xprev),(y - yprev),(z - zprev)]
            testlam = []
            index = np.arange(int(math.ceil((ptdist/dx)/10.))*10)*(dx/ptdist)
            index=index[1:]

            x1 = xprev + index*v[0]
            y1 = yprev + index*v[1]
            z1 = zprev + index*v[2]
            
            for t in np.arange(len(index)):
                testlam.append((LAM_arr[tree.query([z1[t],y1[t],x1[t]])[1]]))
            newind = (np.where(np.array(testlam) <= (1+TOL)*LAMLIMIT))

            # if (newind[0].size == 0): #meaning there is nothing in the path below the limit
                    # we assume that if there is nothing in the path, there is also a good chance
                    # This photon has scattered outside of the simulation boundary.
                    # However, if we adjust the location before testing the boundary, then it will never exceed the boundary
            if (len(testlam) > newind[0].size > 0):
                #We assume something in the path, and update the path to the start of that 'something'
                # print('layer or change')
                newind = np.nanmin(newind[0])
                x = x1[newind]
                y = y1[newind]
                z = z1[newind]
                LAM = testlam[newind]
                    
            #note the 3rd instance, where all points along a path between the previous xyx and new xyz are below the 
            #LAM limit, we let the xyz stand. It does not need to be updated for a new environment or boundary. 
        
 
        if outside_of_coords([x,y,z], coords):
            #If the photon exits the boundary of the simulation the end point is drawn
            #by finding the boundary crossed, and what the photon x/y/z position is at the boundary. 
            #The end positions (x,y,z) are updated to that position. 
            x,y,z = find_crossing([xprev,yprev,zprev],[x,y,z],coords)

            dist[j] += np.sqrt( (x - xprev)**2 + (y - yprev)**2 + (z - zprev)**2)
            x_arr[j] = x
            y_arr[j] = y
            z_arr[j] = z
            xprev_arr[j] = xprev
            yprev_arr[j] = yprev
            zprev_arr[j] = zprev
            k_arr[j]=k
            continue
        dist[j] += np.sqrt( (x - xprev)**2 + (y - yprev)**2 + (z - zprev)**2)

        while bound < 1: 
            exceeded_bounds = 0
            counter +=1
            # ;check to see if absorbed
            if ab_arr[k] > w_0: 
                absorb_flag[j] = 1 
                print('absorbed')
                dist[j] += np.sqrt( (x - xprev)**2 + (y - yprev)**2 + (z - zprev)**2)
                x_arr[j] = x
                y_arr[j] = y
                z_arr[j] = z
                xprev_arr[j] = xprev
                yprev_arr[j] = yprev
                zprev_arr[j] = zprev
                break

#Determine if the next interaction is ice or liquid based on height
            liqfrac_here = liquid_frac[tree.query([z,y,x])[1]]
            icefrac_here = frozen_frac[tree.query([z,y,x])[1]]
            if (liqfrac_here == 0.0) and (icefrac_here == 0.0):
                zenith_pk = zenith_p_liq[k]
            if liqfrac_here == 1.0:
                zenith_pk = zenith_p_liq[k]
                liq +=1
            if icefrac_here == 1.0: 
                zenith_pk = zenith_p_ice[k]
                ice += 1
            # print([ab_arr[k],liqfrac_here, icefrac_here])
            if liqfrac_here >0 and liqfrac_here <1.0:
                if (ab_arr[k] >= 0) and (ab_arr[k]<=liqfrac_here):
                    liq +=1 # print('this is liquid')
                    zenith_pk = zenith_p_liq[k]
                if (ab_arr[k] > liqfrac_here) and (ab_arr[k] <= (liqfrac_here + icefrac_here)):
                    ice += 1
                    zenith_pk = zenith_p_ice[k]
                

# ;Find the photon's second position
# ;direction cosines
            denom = np.sqrt(1. - mu_z**2.)
            mu_x_new = ((np.sin(zenith_pk)*(mu_x*mu_z*np.cos(azimuth_p[k]) - mu_y*np.sin(azimuth_p[k])))/denom) + mu_x*np.cos(zenith_pk)
            mu_y_new = ((np.sin(zenith_pk)*(mu_y*mu_z*np.cos(azimuth_p[k]) + mu_x*np.sin(azimuth_p[k])))/denom) + mu_y*np.cos(zenith_pk)
            mu_z_new = -(denom)*np.sin(zenith_pk)*np.cos(azimuth_p[k]) + mu_z*np.cos(zenith_pk)

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
            # LAM = 70.

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

            ptdist = np.sqrt((z-zprev)**2 + (y-yprev)**2 + (x-xprev)**2)
            if LAM_arr[tree.query([z,y,x])[1]] != LAM: 
            #this is only considered if the LAM is not the same as the previous location. 
            #Otherwise it is still travling in the same voxel.
    
                v = [(x - xprev),(y - yprev),(z - zprev)]
                testlam = []
                ind = int(math.ceil((ptdist/dx)/10.))*10
                if ind > 50000:
                    ind = 50000
                index = np.arange(ind)*(dx/ptdist)
                index=index[1:]

                x1 = xprev + index*v[0]
                y1 = yprev + index*v[1]
                z1 = zprev + index*v[2]
            
                for t in np.arange(len(index)):
                    testlam.append((LAM_arr[tree.query([z1[t],y1[t],x1[t]])[1]]))
                newind = (np.where(np.array(testlam) <= (1+TOL)*LAMLIMIT))
                # if (newind[0].size == 0): #meaning there is nothing in the path below the limit
                    # we assume that if there is nothing in the path, there is also a good chance
                    # This photon has scattered outside of the simulation boundary.
                    # However, if we adjust the location before testing the boundary, then it will never exceed the boundary

                if (len(testlam) > newind[0].size > 0):
                    #We assume something in the path, and update the path to the start of that 'something'
                    newind = np.nanmin(newind[0])
                    x = x1[newind]
                    y = y1[newind]
                    z = z1[newind]
                    LAM = testlam[newind]

                    
    #note the 3rd instance, where all points along a path between the previous xyx and new xyz are below the 
    #LAM limit, we let the xyz stand. It does not need to be updated for a new environment or boundary. 

            if outside_of_coords([x,y,z], coords):
                #If the photon exits the boundary of the simulation the end point is drawn
                #by finding the boundary crossed, and what the photon x/y/z position is at the boundary. 
                #The end positions (x,y,z) are updated to that position. 
                x,y,z = find_crossing([xprev,yprev,zprev],[x,y,z],coords)
                #else, just use the originally determined xyz 
                bound = 1

            dist[j] += np.sqrt( (x - xprev)**2 + (y - yprev)**2 + (z - zprev)**2) 
        
                  
        # print(k)    
        x_arr[j] = x
        y_arr[j] = y
        z_arr[j] = z
        xprev_arr[j] = xprev
        yprev_arr[j] = yprev
        zprev_arr[j] = zprev
        k_arr[j]=k

        
        del k,xprev,yprev,zprev,x,y,z,rng,ab_arr, rng_ind, ind,zenith_arr_liq, zenith_arr_ice,r,x_scat,azimuth_p,zenith_ind,zenith_p_liq,zenith_p_ice
        
    # print(absorb)
    array_out =np.column_stack((x_arr,y_arr,z_arr,xprev_arr, yprev_arr, zprev_arr,k_arr,absorb_flag,dist)) #,zenith_arrout,azimuth_arrout))
    return(array_out)  
    
    
#This is a simple single simulation with a single origin. M is the number of 'photons'. Keep track of seeds for your own use and do not replicate
#seeds in the same simulation. For extended sources repeat this simulation with the updated origin. Origin points are given in meters., Similarly with Coords
#Coords is as legacy input as the cloud is determined by the mean free path. Should you prefer the cloud boundary be a hard boundary set accordingly. 
origin = [4062.5,21e3,15e3]
# origin = [5e3,22e3,17e3]
# origin = [5e3,19.5e3,14.5e3]
# origin = [8437.5,19.5e3,14.5e3]
# origin = [4062.5,9000.0,9250.0]

m = 10
profile_in = xr.open_dataset('3D_COMMAS_KTAL071015B_30s.002430.nc')
        #make points and kdtree for cloud coordinates
zz,yy,xx = np.meshgrid(profile_in['z'].values, profile_in['y'].values,profile_in['x'].values,indexing='ij')
length = len(profile_in['time'])*len(profile_in['x'])*len(profile_in['y'])*len(profile_in['z'])
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

coords = [40e3,0.,40e3,0.,20e3,0.]
origin = [4062.5,21e3,15e3]
out = knb_mc_sim(m,profile_in,tree,LAM_arr,SEEDER  = 15,origin = origin, DISTLIMIT = 125.,LAMLIMIT=200.,coords = coords)

df = pd.DataFrame(out)
file_name = 'COMMAS30s_'+str(origin[0])[:-2]+'_'+str(origin[1])[:-2]+'_'+str(origin[2])[:-2]+'_337.csv'
print(file_name)
df.to_csv(file_name, index=False)