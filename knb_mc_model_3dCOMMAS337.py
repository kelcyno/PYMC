#!/usr/bin/env python
# coding: utf-8
#Note this is edited outside of the github version on 11/13/2024 to handle entire dataset point simulations

import argparse

import numpy as np
from glob import glob
import pickle
import matplotlib
import xarray as xr
import matplotlib.pyplot as plt
import math
# import datetime
# import time
from scipy.io import readsav
from pandas.core.common import flatten
from scipy import spatial, constants
import pyart
import pandas
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flashid', metavar='flashid', required=True,
                        dest='FLASHID', action='store',
                        help='the flashid to be processed, e.g., 1, 2,')
    parser.add_argument('--microphysics_file', metavar='microphysics_file', required=True,
                        dest='MICROPHYSICS_FILE', action='store',
                        help='UTC start time, e.g., 20170704_080000')
    return parser



import numpy as np

def knb_mc_sim(m,frozen_frac,liquid_frac,PROFILE_IN,zz,yy,xx,tree,masktree,LAM_arr, SEEDER = 10.,coords = [1e4,0.,1e4,0.,1e4,0.], 
               origin = [5e3,5e3,5e3]):
    
    #Before you start modeling, check that the profile in does or does not have the liquid and frozen
    #size and concentration distributions. If they do, then skip this part. If not, then create the liquid and frozen 
    #size and concentration arrays.
    
    concentration = PROFILE_IN['particle_concentration'].values
    size = PROFILE_IN['particle_size'].values
    
#     test = str(list(PROFILE_IN.keys()))
#     if not re.search('liquid',test):
#         print('Calculating liquid and frozen microphysics')
#         liq_size, liq_count = calculate_liquid(PROFILE_IN)
#         frozen_size, frozen_count = calculate_frozen(PROFILE_IN)
    
#         liquid_frac = liq_count/concentration
#         ind =  np.isnan(liquid_frac)
#         liquid_frac[ind] = 0.0 
        
#         frozen_frac = frozen_count/concentration
#         ind =  np.isnan(frozen_frac)
#         frozen_frac[ind] = 0.0 
        
#     else: 
#         liquid_frac = PROFILE_IN['liquid_count'].values/concentration
#         frozen_frac = PROFILE_IN['frozen_count'].values/concentration
    
#     length = len(PROFILE_IN['x'])*len(PROFILE_IN['y'])*len(PROFILE_IN['z'])

#     liquid_frac = np.reshape(liquid_frac,length)
#     frozen_frac= np.reshape(frozen_frac, length)
    
    #origin order is [z,y,x] because gridcoords is z,y,x
    print(origin)
    # print(coords)
    #coords is xmax,xmin, ymax,ymin, zmax,zmin
    
    #array for distance the particle traveled
    dist_traveled = np.zeros(m)

    #make points and kdtree for cloud coordinates
    zz,yy,xx = np.meshgrid(PROFILE_IN['z'].values, PROFILE_IN['y'].values,PROFILE_IN['x'].values,indexing='ij')
    # length = len(PROFILE_IN['x'])*len(PROFILE_IN['y'])*len(PROFILE_IN['z'])
    xx = np.reshape(xx,length)
    yy = np.reshape(yy,length)
    zz = np.reshape(zz,length)
    gridcoords = np.stack((zz,yy,xx),axis = 1) 
    tree = spatial.KDTree(gridcoords)
    
    #Define your boundaries and arrays to hold when a boundary is crossed, specifically if using a hard lateral or vertical boundary
    s = np.size(coords)
    
    z_min = coords[5] #this is the bottom of the cloud
    z_max = coords[4] #this is the top of the cloud.
    x_min = coords[1] #this is the 1st face, the negative X-face
    x_max = coords[0] # this is the 2nd face, the positive X-face
    y_min = coords[3] #this is the 3rd face, the negative Y-face
    y_max = coords[2] #this is the 4th face, the positive Y-face
    
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

 
    

    # print(LAM)
    N_dense = np.reshape(concentration,length)
    a_arr = np.reshape(size,length)
    a_arr = np.array([0 if math.isnan(i) else i for i in a_arr])
    N_dense = np.array([0 if math.isnan(i) else i for i in N_dense])
    
    # ######Check microphysical profile
    height = PROFILE_IN['z'].values #reform(profile_in['profile'][:,2])
    if np.nanmax(height) <= 100.:
        height = height*1000.
    #     ######Determine mean free path
    #     #####(m) mean free path of photons with a uniform population of drops.
    #     #LAM_ARR = reform(1./(2.*!dpi*(a_arr^2.)*N_dense))
    LAM_arr = (1./(2.*math.pi*(a_arr**2.)*N_dense)) #make sure a_arr isn't creating a matrix
    LAM_arr = np.nan_to_num(LAM_arr,posinf = 999999,neginf = 999999)
    # LAM_arr = np.reshape(LAM_arr,[80,10,10])

    #NOTE, WE USE A MEAN FREE PATH OF 200m here as the cloud boundary. It is on the user to determine if this is appropriate for THEIR CLOUD.
    mask = np.where(LAM_arr <= 200)
    mask = np.array(list(flatten(mask)))
    if len(mask) == 0 :
        print('no clouds here')
        return
    maskstack = np.stack((zz[mask],yy[mask],xx[mask]),axis=1)
    masktree = spatial.KDTree(maskstack)

    
    #******
    point = masktree.query(origin)
    if point[0] >1100.:
        outofCloud = 1
    else:
        outofCloud = 0
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
    zenith_arrout = np.zeros(m)
    azimuth_arrout = np.zeros(m)
    absorb = 0
    liq = 0
    ice = 0
    
    zenith_orig = np.zeros(m)
    azimuth_orig = np.zeros(m)
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
    print(LAM)
    # print(LAM)
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
        counter = 0
        bound = 0
        k=0
        rng = np.random.default_rng(seed = (ab_arr_m[j]*100).astype(int))
        ab_arr = rng.uniform(0,1, 2000000)

    # ;The position of the particle, after being emitted isotropically and scattered a distance x_scat_0 is given by:
        # ;direction cosines
        mu_x = np.sin(zenith_0[j])*np.cos(azimuth_0[j])
        mu_y = np.sin(zenith_0[j])*np.sin(azimuth_0[j])
        mu_z = np.cos(zenith_0[j])

        # ;positions
        x = origin[2] + x_scat_0[j]*mu_x
        y = origin[1] + x_scat_0[j]*mu_y
        z = origin[0] + x_scat_0[j]*mu_z
        #;location = [[location],[x,y,z]]
        # print([x,y,z])
#         if not np.logical_and(zenith_0[j], azimuth_0[j]):
#             print('began the weird case')#  then begin
#         #; This is the 1st scatter calculations - for testing purposes
#         # ;The position of the particle, after being emitted isotropically and scattered a distance x_scat_0 is given by:
#             # ;direction cosines
#             mu_x = np.sin(zenith_0[j])*np.cos(azimuth_0[j])
#             mu_y = np.sin(zenith_0[j])*np.sin(azimuth_0[j])
#             mu_z = np.cos(zenith_0[j])
# 
#             #;positions
#             x = x + x_scat_0[j]*mu_x
#             y = y + x_scat_0[j]*mu_y
#             z = z + x_scat_0[j]*mu_z
#             endif 
        # ;Find angles of first scattering, these are relative to incoming angle from isotropic scatter.
        rng_ind = np.random.default_rng(seed = (ind_m[j]*100.).astype(int))
        ind = rng_ind.uniform(0,1,2000)*1999. + 1.

        #this gives us most angles near 0
        # zenith_arr = np.arccos((mu_arr[pp[1:]] + mu_arr[0:-1])/2.)
        zenith_arr_liq = np.arccos((mu_arr_liq[pp_liq[1:]] + mu_arr_liq[0:-1])/2.)
        zenith_arr_ice = np.arccos((mu_arr_ice[pp_ice[1:]] + mu_arr_ice[0:-1])/2.)
        # # #;*****
        rng = np.random.default_rng(seed = (zenith_ind_m_seed[j]*100.).astype(int))
        zenith_ind = (rng.uniform(0,1,500000)*2000.).astype(int)
        # zenith_p = zenith_arr[zenith_ind]
        zenith_p_liq = zenith_arr_liq[zenith_ind]
        zenith_p_ice = zenith_arr_ice[zenith_ind]
        rng = np.random.default_rng(seed = (azimuth_r_m[j]*100.).astype(int))
        azimuth_p = rng.uniform(0,2*math.pi,500000)
        rng = np.random.default_rng(seed = (r_m_1[j]*100.).astype(int))
        r = rng.uniform(0,1,500000)
        x_scat = -np.log(r)       

        # #;******
        while bound < 1:
            
            counter +=1
# ;check to see if absorbed
            if ab_arr[k] > w_0: absorb += 1
            if ab_arr[k] > w_0: break
#Determine if the next interaction is ice or liquid based on height
            #Determine if the next interaction is ice or liquid based on height
            liqfrac_here = liquid_frac[tree.query([z,y,x])[1]]
            icefrac_here = frozen_frac[tree.query([z,y,x])[1]]
            # print([liqfrac_here, icefrac_here])
            
            if (liqfrac_here == 0.0) and (icefrac_here == 0.0):
                zenith_pk = zenith_p_liq[k]
            if liqfrac_here == 1.0:
                zenith_pk = zenith_p_liq[k]
                liq +=1
            if icefrac_here == 1.0: 
                zenith_pk = zenith_p_ice[k]
                ice += 1
            # print([ab_arr[k],liqfrac_here, icefrac_here])
#             if liqfrac_here >0 and liqfrac_here <1.0:
#                 if (ab_arr[k] >= 0) and (ab_arr[k]<=liqfrac_here):
#                     liq +=1 # print('this is liquid')
#                     zenith_pk = zenith_p_liq[k]
                # if (ab_arr[k] > liqfrac_here) and (ab_arr[k] <= (liqfrac_here + icefrac_here)):
                #     ice += 1
                #     zenith_pk = zenith_p_ice[k]
                
# ;Find the photon's second position
# ;direction cosines
            denom = np.sqrt(1. - mu_z**2.)
            mu_x_new = ((np.sin(zenith_pk)*(mu_x*mu_z*np.cos(azimuth_p[k]) - mu_y*np.sin(azimuth_p[k])))/denom) + mu_x*np.cos(zenith_pk)
            mu_y_new = ((np.sin(zenith_pk)*(mu_y*mu_z*np.cos(azimuth_p[k]) + mu_x*np.sin(azimuth_p[k])))/denom) + mu_y*np.cos(zenith_pk)
            mu_z_new = -(denom)*np.sin(zenith_pk)*np.cos(azimuth_p[k]) + mu_z*np.cos(zenith_pk)

            #UNCOMMENT THIS FOR DISTRIBUTED LAMBDA
            # print([z,y,x])
            # print(LAM)
            if not math.isfinite(z):
                print('broken')
                break
            if not math.isfinite(x):
                print('broken')
                break
            if not math.isfinite(x):
                print('broken')
                break
            
            LAM = LAM_arr[tree.query([z,y,x])[1]]
            #print([z,y,x])
            #print(LAM)
            xprev = x
            yprev = y
            zprev = z
            x = x + LAM*x_scat[k]*mu_x_new
            y = y + LAM*x_scat[k]*mu_y_new
            z = z + LAM*x_scat[k]*mu_z_new
            dist_traveled[j] += LAM*x_scat[k]
            mu_x = mu_x_new
            mu_y = mu_y_new
            mu_z = mu_z_new
            k += 1
            #print(k)
            # ;check boundary conditions
            point = masktree.query([z,y,x])
            if point[0] >1100.:
                if outofCloud == 1:
                    bound = 0
                    outofCloud = 0
                else: bound = 1
                
            
            #These are boundary checks when using a hard boundary. 
            # if origin[2] >= coords[5]:# then begin #if the original point is 'inside the cloud'
            #     if z > z_max: bound = 1
            #     if z <= z_min: bound = 1
            #     if y > y_max: bound = 1
            #     if y <= y_min: bound = 1
            #     if x > x_max: bound = 1
            #     if x <= x_min: bound = 1
            # else:  #if the original point is outside the cloud
            #     if z > z_max: bound = 1
            #     if z <= 0.: bound = 1 #it's below the ground
            #     if y > y_max: bound = 1
            #     if y <= y_min: bound = 1
            #     if x > x_max: bound = 1
            #     if x <= x_min: bound = 1
            # endelse
        #endrep until bound GT 0D 
            #print(k)    
            x_face = 0
            y_face = 0
            z_face = 0


        x_arr[j] = x
        y_arr[j] = y
        z_arr[j] = z
        xprev_arr[j] = xprev
        yprev_arr[j] = yprev
        zprev_arr[j] = zprev
        zenith_arrout[j] = zenith_pk
        azimuth_arrout[j] = azimuth_p[k]

    x_face = 0
    y_face = 0
    z_face = 0
    
    print(absorb)
    array_out =np.column_stack((x_arr,y_arr,z_arr,xprev_arr, yprev_arr, zprev_arr,dist_traveled))#,zenith_arrout,azimuth_arrout))
    return(array_out)  
       
def calculate_liquid(profile_in):
    #ALL LIQUID particles 
    length = np.size(profile_in['droplet_count'].values)
    liquid_concentration = (profile_in['droplet_count'].values + profile_in['rain_count'].values)
    
    #mean size of liquid particles
    #before calculating the average size you need to get rid of nans for the size arrays because 5 + nan = nan, not 5d
    topdrop = (profile_in['droplet_count'].values*profile_in['droplet_size'].values)
    ind = np.isnan(topdrop)
    topdrop[ind] = 0.0
    toprain = (profile_in['rain_count'].values*profile_in['rain_size'].values)
    ind = np.isnan(toprain)
    toprain[ind] = 0.0

    top = topdrop + toprain
    liquid_size = (top/liquid_concentration)
    return liquid_size, liquid_concentration

def calculate_frozen(profile_in):
#ALL frozen particles 
    length = np.size(profile_in['ice_count'].values)
    frozen_concentration = (profile_in['ice_count'].values + profile_in['graupel_count'].values + profile_in['hail_count'].values + profile_in['snow_count'].values)
    
    #mean size of frozen/ice particles
    #before calculating the average size you need to get rid of nans for the size arrays because 5 + nan = nan, not 5d
    topice = (profile_in['ice_count'].values*profile_in['ice_size'].values)
    ind = np.isnan(topice)
    topice[ind] = 0.0
    topgraupel = (profile_in['graupel_count'].values*profile_in['graupel_size'].values)
    ind = np.isnan(topgraupel)
    topgraupel[ind] = 0.0
    tophail = (profile_in['hail_count'].values*profile_in['hail_size'].values)
    ind = np.isnan(tophail)
    tophail[ind] = 0.0
    topsnow = (profile_in['snow_count'].values*profile_in['snow_size'].values)
    ind = np.isnan(topsnow)
    topsnow[ind] = 0.0


    top = topice + topgraupel + tophail + topsnow
    frozen_size = (top/frozen_concentration)
    return frozen_size, frozen_concentration  
  
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
#This is a simple single simulation with a single origin. M is the number of 'photons'. Keep track of seeds for your own use and do not replicate
#seeds in the same simulation. For extended sources repeat this simulation with the updated origin. Origin points are given in meters., Similarly with Coords
#Coords is as legacy input as the cloud is determined by the mean free path. Should you prefer the cloud boundary be a hard boundary set accordingly. 
    flash_file = 'COMMAS30s_flash_points.csv'
    flashes = pd.read_csv(flash_file, names = ['x','y','z','t','ID','area'],header = 0,index_col = False)
# files = sorted(glob('/Users/kelcy/PYTHON/KNB_PYMC/COMMAS_3dmicrophysics_results/3D_COMMAS_KTAL071015B.*.nc'))
# datafiles = sorted(glob('/Volumes/Extreme SSD/DATA/ropesville125m/KTAL071015B.*.nc'))
# end = files[6].split('_')[-1][-8:-3]
# time_arr = []  
    m = 100
    flashid = int(args.FLASHID)
    print(['flashid = ', flashid])
    flash_out = []
    sub = flashes.loc[lambda df: df['ID'] == flashid, :]
    print(len(sub))
#     print(flashid)

    profile_in = xr.open_dataset(str(args.MICROPHYSICS_FILE))
    print(args.MICROPHYSICS_FILE)
    zs = profile_in['z'].values
    xs = profile_in['x'].values
    ys = profile_in['y'].values
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
    # LAM_arr = np.reshape(LAM_arr,[80,10,10])

    #NOTE, WE USE A MEAN FREE PATH OF 200m here as the cloud boundary. It is on the user to determine if this is appropriate for THEIR CLOUD.
    mask = np.where(LAM_arr <= 200)
    mask = np.array(list(flatten(mask)))
    maskstack = np.stack((zz[mask],yy[mask],xx[mask]),axis=1)
    masktree = spatial.KDTree(maskstack)
    liq_size, liq_count = calculate_liquid(profile_in)
    frozen_size, frozen_count = calculate_frozen(profile_in)
    
    liquid_frac = liq_count/profile_in['particle_concentration'].values
    ind =  np.isnan(liquid_frac)
    liquid_frac[ind] = 0.0 
        
    frozen_frac = frozen_count/profile_in['particle_concentration'].values
    ind =  np.isnan(frozen_frac)
    frozen_frac[ind] = 0.0 

    length = len(profile_in['x'])*len(profile_in['y'])*len(profile_in['z'])

    liquid_frac = np.reshape(liquid_frac,length)
    frozen_frac= np.reshape(frozen_frac, length)
    
    
    for i in np.arange(len(sub['x'])-1):
#         print(['looping',str(i)])
        if i % 2 == 0:
            print(i)
            origin = [zs[sub['z'].iloc[i]],ys[sub['y'].iloc[i]],xs[sub['x'].iloc[i]]]
            out = knb_mc_sim(m,profile_in['particle_concentration'].values,profile_in['particle_size'].values,profile_in,
                zz,yy,xx,tree,masktree,LAM_arr,SEEDER  = (i*flashid*8).astype(int),origin = origin)
            if i == 0:
                flash_out = out
            else: flash_out = np.vstack((flash_out, out))
    df = pd.DataFrame(flash_out)
    file_name = 'COMMAS30s_100p_'+str(flashid)+'_3337.csv'
    print(file_name)
    df.to_csv(file_name, index=False)