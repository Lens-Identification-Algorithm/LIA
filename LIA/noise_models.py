# -*- coding: utf-8 -*-
"""
Created on Thu July 28 20:30:11 2018

@author: danielgodinez
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from math import log

def create_noise(median, rms, degree=3):
    """Creates a noise model by fitting a one-dimensional smoothing 
    spline of degree k.

    Parameters
    ----------
    median : array
        Baseline magnitudes.
    rms : array
        Corresponding RMS per baseline. 
    k : int
        Degree of the smoothing spline. Default is a 
        cubic spline of degree 3.

    Returns
    -------
    fn : The kth degree spline fit. 
    """
    f = UnivariateSpline(median, rms, w=None, k=degree)
    return f

def add_noise(mag, fn, zp=24):
    """Adds noise to magnitudes given a noise function. 

    Parameters
    ----------
    mag : array
        Magnitude to add noise to. 
    fn : function
        Spline fit, must be defined using the create_noise function. 
    zp : Zeropoint
        Zeropoint of the instrument, default is 24.
        
    Returns
    -------
    mag : array
        The noise-added magnitudes. 
    magerr : array
        The corresponding magnitude errors.
    """    
    flux = 10**(-(mag-zp)/2.5)
    delta_fobs = flux*fn(mag)*(log(10)/2.5)
    f_obs = np.random.normal(flux, delta_fobs)

    mag_obs = zp - 2.5*np.log10(f_obs)
    magerr = (2.5/log(10))*(delta_fobs/f_obs)
        
    return np.array(mag_obs), np.array(magerr)

def add_gaussian_noise(mag,zp=24):
    """Adds noise to lightcurve given the magnitudes.

    Parameters
    ----------
    mag : array
        Mag array to add noise to. 
    zp : zeropoint
        Zeropoint of the instrument, default is 24.
    convert : boolean, optional 
    
    Returns
    -------
    noisy_mag : array
        The noise-added magnitude. 
    magerr : array
        The corresponding magnitude errors.
    """
    flux = 10**((mag-zp)/-2.5)
    
    noisy_flux= np.random.poisson(flux)#, np.sqrt(flux))
    magerr = 2.5/(log(10)*np.sqrt(noisy_flux))
    
    noisy_mag = zp - 2.5*np.log10(noisy_flux)
    magerr=np.array(magerr)
    mag = np.array(mag)

    return np.array(noisy_mag), np.array(magerr)

'''
    Adds noise to Gaia g magnitudes
    Estimated, using fi
t by  Lukasz Wyrzykowski
'''
def add_gaia_g_noise(mag):
    a1 = 0.2
    b1 = -5.3 #-5.2
    a2=0.2625
    b2= -6.3625 #-6.2625
     
    mag_obs_list = []
    magerr_list = []
    
    for value in mag:
        log_err1 = a1*value + b1
        log_err2 = a2*value + b2
        if (value < 13.5):
            magerr = 10**(a1*13.5+b1)
            magerr_list.append(magerr)
            mag_obs = np.random.normal(value, magerr)
            mag_obs_list.append(mag_obs)
        if value>=13.5 and value<17:
            magerr = 10**log_err1
            magerr_list.append(magerr)
            mag_obs = np.random.normal(value, magerr)
            mag_obs_list.append(mag_obs)            
        if (value>=17): 
            magerr = 10**log_err2
            magerr_list.append(magerr)
            mag_obs = np.random.normal(value, magerr)
            mag_obs_list.append(mag_obs)            
        #this works until 21 mag.
    return np.array(mag_obs_list), np.array(magerr_list)


'''
    Adds noise to ZTF magnitudes according to given distribution of magnitude errors 
'''
def add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max):
    bin_numbers_1 = np.digitize(mag, bin_edges)


    bin_numbers_2 = []
    for j in range(len(mag)):
        if mag[j] <= mag_max:
            bin_numbers_2.append(bin_numbers_1[j])
        else:
            bin_number_new = len(magerr_intervals)-1
            bin_numbers_2.append(bin_number_new)
        
# if mag interval empty
    for k in range(len(mag)):
        for y in range(len(magerr_intervals)): 
            if len(magerr_intervals[bin_numbers_2[k]]) != 0:
                break
            else:
                try:
                    bin_number_up = bin_numbers_2[k]+y
                    if len(magerr_intervals[bin_number_up]) != 0:
                        bin_numbers_2[k] = bin_number_up
                        break
                except(IndexError):
                    bin_numbers_2[k] = bin_numbers_2[k]-y
                    continue


    magerr_random_list_1 = []
    for l in range(len(mag)):
        magerr_random = np.random.choice(a = magerr_intervals[bin_numbers_2[l]], size = 1)
        magerr_random_list_1.append(magerr_random)

    magerr_random_list = []    
    for e in range(len(magerr_random_list_1)):
        magerr_random_list.append(magerr_random_list_1[e][0])
    
    mag_obs_list = []
    for m in range(len(mag)):
        magerr = magerr_random_list[m]
        mag_obs = np.random.normal(mag[m], magerr)
        mag_obs_list.append(mag_obs)
        
    
    return np.array(mag_obs_list), np.array(magerr_random_list) 