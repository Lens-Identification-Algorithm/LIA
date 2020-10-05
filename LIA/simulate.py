#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:19:01 2020

@author: marlen
"""

from __future__ import division
import numpy as np
import os
import cmath
from math import pi
import VBBinaryLensing
#from astropy.io.votable import parse_single_table

def microlensing(timestamps, baseline):
    """Simulates a microlensing event.  
    The microlensing parameter space is determined using data from an 
    analysis of the OGLE III microlensing survey from Y. Tsapras et al (2016).
    See: The OGLE-III planet detection efficiency from six years of microlensing observations (2003 to 2008).
    (https://arxiv.org/abs/1602.02519)

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude at which to simulate the lightcurve.

    Returns
    -------
    mag : array
        Simulated magnitude given the timestamps.
    u_0 : float
        The source minimum impact parameter.
    t_0 : float
        The time of maximum magnification.
    t_E : float
        The timescale of the event in days.
    blend_ratio : float
        The blending coefficient chosen between 0 and 10.     
    """   
 
    mag = constant(timestamps, baseline)
    # Set bounds to ensure enough measurements are available near t_0 
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    
    t_0 = np.random.uniform(lower_bound, upper_bound)        
    u_0 = np.random.uniform(0, 1.5)
    t_e = np.random.normal(50, 10.0)
    blend_ratio = np.random.uniform(0,10)

    u_t = np.sqrt(u_0**2 + ((timestamps - t_0) / t_e)**2)
    magnification = (u_t**2 + 2.) / (u_t * np.sqrt(u_t**2 + 4.))
 
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    
    flux_obs = f_s*magnification + f_b+flux_noise
    microlensing_mag = -2.5*np.log10(flux_obs)
    
    return np.array(microlensing_mag), baseline, u_0, t_0, t_e, blend_ratio, np.array(flux_obs), f_s, f_b
    
def cv(timestamps, baseline):
    """Simulates Cataclysmic Variable event.
    The outburst can be reasonably well represented as three linear phases: a steeply 
    positive gradient in the rising phase, a flat phase at maximum brightness followed by a declining 
    phase of somewhat shallower negative gradient. The period is selected from a normal distribution
    centered about 100 days with a standard deviation of 200 days. The outburtst amplitude ranges from
    0.5 to 5.0 mag, selected from a uniform random function. 

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude at which to simulate the lightcurve.

    Returns
    -------
    mag : array
        Simulated magnitudes given the timestamps and baseline. 
    outburst_start_times : array
        The start time of each outburst.
    outburst_end_times : array
        The end time of each outburst.
    end_rise_times : array
        The end time of each rise (start time of max amplitude).
    end_high_times : array
        The end time of each peak (end time of max amplitude).
    """

    period = abs(np.random.normal(100, 200))
    amplitude = np.random.uniform(0.5, 5.0)
    lc = np.zeros(len(timestamps))
    # First generate the times when outbursts start. Note that the
    # duration of outbursts can vary for a single object, so the t_end_outbursts will be added later.
    start_times = []

    min_start = min(timestamps)
    max_start = min((min(timestamps)+period),max(timestamps))

    first_outburst_time = np.random.uniform(min_start, max_start)

    start_times.append(first_outburst_time)
    t_start = first_outburst_time + period

    for t in np.arange(t_start,max(timestamps),period):
        start_times.append(t)

    outburst_end_times = []
    duration_times = []
    end_rise_times = []
    end_high_times = []    
    
    for t_start_outburst in start_times:
    # Since each outburst can be a different shape,
    # generate the lightcurve morphology parameters for each outburst:
        duration = np.random.uniform(3.0, (period/10.0))
        duration_times.append(duration)
        t_end_outburst = t_start_outburst + duration
        outburst_end_times.append(t_end_outburst)        
        rise_time = np.random.uniform(0.5,1.0)
        high_state_time = np.random.normal(0.4*duration, 0.2*duration)
        drop_time = duration - rise_time - high_state_time
        t_end_rise = t_start_outburst + rise_time
        t_end_high = t_start_outburst + rise_time + high_state_time
        end_rise_times.append(t_end_rise)
        end_high_times.append(t_end_high)  
        # Rise and drop is modeled as a straight lines with differing gradients
        rise_gradient = -1.0 * amplitude / rise_time

        drop_gradient = (amplitude / drop_time)

        for i in range(0,len(timestamps),1):
                if timestamps[i] >= t_start_outburst and timestamps[i] <= t_end_rise:
                        lc[i] = rise_gradient * (timestamps[i] -t_start_outburst)
                elif timestamps[i] >= t_end_rise and timestamps[i] <= t_end_high:
                        lc[i] = -1.0 * amplitude
                elif timestamps[i] > t_end_high and timestamps[i] <= t_end_outburst:
                        lc[i] = -amplitude + ( drop_gradient * (timestamps[i] - t_end_high))

    lc = lc+baseline 
    return np.array(lc), np.array(start_times), np.array(outburst_end_times), np.array(end_rise_times), np.array(end_high_times)


def constant(timestamps, baseline):
    """Simulates a constant source displaying no variability.  

    Parameters
    ----------
    timestamps : array
        Times at which to simulate the lightcurve.
    baseline : float
        Baseline magnitude of the lightcurve.

    Returns
    -------
    mag : array
        Simulated magnitudes given the timestamps.
    """
    mag = [baseline] * len(timestamps)

    return np.array(mag)
     

def parametersRR0():            #McGill et al. (2018): Microlens mass determination for Gaia’s predicted photometric events
    a1=  0.31932222222222223
    ratio12 = 0.4231184105222867 
    ratio13 = 0.3079439089738683 
    ratio14 = 0.19454399944326523
    f1 =  3.9621766666666667
    f2 =  8.201326666666667
    f3 =  6.259693777777778
    return a1, ratio12, ratio13, ratio14, f1, f2, f3

def parametersRR1():            #McGill et al. (2018)
    a1 =  0.24711999999999998
    ratio12 = 0.1740045322110716 
    ratio13 = 0.08066256609474477 
    ratio14 = 0.033964605589727
    f1 =  4.597792666666666
    f2 =  2.881016
    f3 =  1.9828297333333336
    return a1, ratio12, ratio13, ratio14, f1, f2, f3


def uncertainties(time, curve, uncertain_factor):       #optional, add random uncertainties, controlled by the uncertain_factor
    N = len(time)
    uncertainty = np.random.normal(0, uncertain_factor/100, N)
    realcurve = []                                      #curve with uncertainties
    for idx in range(N):
        realcurve.append(curve[idx]+uncertainty[idx])
    return realcurve


def setup_parameters(timestamps, bailey=None):          #setup of random parameters based on physical parameters
    time = np.array(timestamps)   #np.arange(0,300,0.01)    #np.loadtxt('Gbul2000_45.dat')[:,0] 
    if bailey is None:
        bailey = np.random.randint(1,4)
    if bailey < 0 or bailey > 3:
        raise RuntimeError("Bailey out of range, must be between 1 and 3.")
    a1, ratio12, ratio13, ratio14, f1, f2, f3  = parametersRR1()
    if bailey == 1:
        period = np.random.normal(0.6, 0.15)
        a1, ratio12, ratio13, ratio14, f1, f2, f3  = parametersRR0()
    elif bailey == 2:
        period = np.random.normal(0.33, 0.1)
    elif bailey == 3:
        period = np.random.lognormal(0., 0.2)
        period = 10**period
    s = 20
    period=np.abs(period)  
    n1 = np.random.normal(a1, 2*a1/s)
    ampl_k = [n1, np.random.normal(n1*ratio12, n1*ratio12/s), np.random.normal(n1*ratio13, n1*ratio13/s), np.random.normal(n1*ratio14, n1*ratio14/s)]
    phase_k = [0, np.random.normal(f1, f1/s), np.random.normal(f2, f2/s), np.random.normal(f3, f3/s)]
    return time, ampl_k, phase_k, period

def variable(timestamps, baseline, bailey=None):       #theory, McGill et al. (2018)
    time, ampl_k, phase_k, period = setup_parameters(timestamps, bailey)
    lightcurve = np.array(baseline)
    for idx in range(len(ampl_k)):
        lightcurve = lightcurve + ampl_k[idx] * np.cos(((2*pi*(idx+1))/period)*time+phase_k[idx])
    amplitude = np.ptp(lightcurve) / 2.0
    return np.array(lightcurve), amplitude, period 

'''
Simulate LPV - Miras
Miras.dat: OGLE III http://www.astrouw.edu.pl/ogle/ogle3/OIII-CVS/blg/lpv/pap.pdf

mira_table = parse_single_table('Miras_vo.xml')

primary_period = mira_table.array['col4'].data
amplitude_pp = mira_table.array['col5'].data
secondary_period = mira_table.array['col6'].data
amplitude_sp = mira_table.array['col7'].data
tertiary_period = mira_table.array['col8'].data
amplitude_tp = mira_table.array['col9'].data
'''

def random_mira_parameters(primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp):
    len_miras = len(primary_period)
    rand_idx = np.random.randint(0,len_miras,1)
    amplitudes = [amplitude_pp[rand_idx], amplitude_sp[rand_idx], amplitude_tp[rand_idx]]
    periods = [primary_period[rand_idx], secondary_period[rand_idx], tertiary_period[rand_idx]]
    return amplitudes, periods

def simulate_mira_lightcurve(times, baseline, primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp):
    amplitudes, periods = random_mira_parameters(primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp)
    lc = np.array(baseline)
    for idx in range(len(amplitudes)):
        lc = lc + amplitudes[idx]* np.cos((2*np.pi*(idx+1))/periods[idx]*times)
    return np.array(lc)


'''
Simulate Binary Lensing
Using VBB Binary Lensing, complex caustic solver 
Obtain critical curves and caustics from complex polynomials
'''

VBB = VBBinaryLensing.VBBinaryLensing()
VBB.Tol = 0.001
VBB.RelTol = 0.001
VBB.minannuli=2 # stabilizing for rho>>caustics

# Binary lens equation in complex form
def lens_equation_binary(z, m1, m2, z1,z2):
    zco = z.conjugate()
    return z + m1 / (z1.conjugate() - zco) + m2 / (z2.conjugate() - zco)

# Calculate roots of binary system
def direct_roots_binary(m1,z2,varphi):
    
    x0 = cmath.exp(1j*varphi)
    x1 = x0*z2
    x2 = 2*x1
    x3 = 4*m1*x1 - x2
    x4 = z2**2
    x5 = -x0 - 1/2*x4
    x6 = x5**3
    x7 = (m1*x2 - x1)**2
    x8 = m1*x0
    x9 = x4*x8
    x10 = (1/2)*z2
    x11 = 2*z2
    x12 = x11*(x10*x8 - x11*((1/16)*x0 - 1/64*x4))
    x13 = x12 - x9
    x14 = x13*x5
    x15 = (1/3)*x14 - 1/108*x6 - 1/8*x7
    x16 = 2*x15**(1/3)
    x17 = (2/3)*x0 + (1/3)*x4
    x18 = cmath.sqrt(-x16 + x17)
    x19 = x3/x18
    x20 = (4/3)*x0 + (2/3)*x4
    x21 = x16 + x20
    x22 = (1/2)*cmath.sqrt(x19 + x21)
    x23 = (1/2)*x18
    x24 = x10 - x23
    x25 = (1/12)*x5**2
    x26 = (x13 + x25 == 0)
    x27 = -x12 - x25 + x9
    x28 = (-1/6*x14 + (1/216)*x6 + (1/16)*x7 + cmath.sqrt((1/4)*x15**2 + (1/27)*x27**3))**(1/3)
    x29 = 2*x28
    x30 = (2/3)*x27/x28
    x31 = cmath.sqrt(x17 + x29 - x30)
    x32 = x3/x31
    x33 = x20 - x29 + x30
    x34 = (1/2)*cmath.sqrt(x32 + x33)
    x35 = (1/2)*x31
    x36 = x10 - x35
    x37 = (1/2)*cmath.sqrt(-x19 + x21)
    x38 = x10 + x23
    x39 = (1/2)*cmath.sqrt(-x32 + x33)
    x40 = x10 + x35

    c0 = ((x22 + x24) if x26 else (x34 + x36))
    c1 = ((-x22 + x24) if x26 else (-x34 + x36))
    c2 = ((x37 + x38) if x26 else (x39 + x40))
    c3 = ((-x37 + x38) if x26 else (-x39 + x40))

    if x26 :
        return [x22 + x24,-x22 + x24,x37 + x38,-x39 + x40]
    else:
        return [x34 + x36,-x34 + x36,x39 + x40,-x39 + x40]
    return [c3, c2, c1, c0 ]

# Calculate critical curves and caustics
def direct_critpattern(m1,m2,z2,z1,n):
    crx,cry,cax,cay = [],[],[],[]
    for phi in np.arange(0, 2.*np.pi, np.pi / n):
        rf = direct_roots_binary(m1,z2,phi)
        for idx in range(len(rf)):
            b1 = lens_equation_binary(complex(rf[idx].real, rf[idx].imag),m1,m2,z1,z2)
            crx.append(rf[idx].real),cry.append(rf[idx].imag)
            cax.append(b1.real),cay.append(b1.imag)
    return crx, cry, cax, cay

# Obtain two points in source plane to construct a source trajectory
# One point lies on a caustic, one point lies on the x-axis between min and max of the critical curves
def get_two_source_plane_points(m1,m2,z2,z1):
    crx,cry,cax,cay = [],[],[],[]
    for phi in np.random.uniform(0,2.*np.pi,2): 
        rf = direct_roots_binary(m1,z2,phi)
        for idx in range(len(rf)):
            b1 = lens_equation_binary(complex(rf[idx].real, rf[idx].imag),m1,m2,z1,z2)
            crx.append(rf[idx].real),cry.append(rf[idx].imag)
            cax.append(b1.real),cay.append(b1.imag)    
    #min and max value of critical curves on x axis
    index_min = crx.index(min(crx))
    index_max = crx.index(max(crx))
    #pick random xaxis value between min(x) and max(x)
    rand_xaxis_value = np.random.uniform(crx[index_min], crx[index_max])
    #transform this point (rand_xaxis_value, 0) onto source plane
    #remove those values from list of critical curves and caustics
    crx.pop(index_min)  
    index_max_new = crx.index(max(crx))
    crx.pop(index_max_new)

    cry.pop(index_min)
    cry.pop(index_max_new)
    
    cax.pop(index_min)
    cax.pop(index_max_new)
    
    cay.pop(index_min)
    cay.pop(index_max_new)

    #pick random caustic point from shortend list
    rand_idx = np.random.randint(0,len(crx),1)
    
    rand_idx = rand_idx[0]
    caustic_value_x = cax[rand_idx]
    caustic_value_y = cay[rand_idx]
    
    return rand_xaxis_value, caustic_value_x, caustic_value_y

# Calculate source trajectory from the two source plane points            
def get_source_trajectory(rand_xaxis_value, caustic_value_x, caustic_value_y):
    # trajectory through (xaxis,0) and caustic (cax,cay)
    m = caustic_value_y/(caustic_value_x-rand_xaxis_value)
    b = -m*rand_xaxis_value
    return m, b

def source_trajectory(x,m,b):
    y = m*x+b
    return y

# Calculate x-coordinate in the source plane for a given time
def x_from_tE(t, t0, tE, cax0, cay0, m, b):
    xi = (((t-t0)/tE) + cax0 + cay0 - b)/(1+m)
    return xi


def amplification_PSBL(separation, mass_ratio, x_source, y_source):
    """
    The Point Source Binary Lens amplification, based on the work of Valerio Bozza, thanks :)
    "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
    optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
    http://mnras.oxfordjournals.org/content/408/4/2188
    :param array_like separation: the projected normalised angular distance between the two bodies
    :param float mass_ratio: the mass ratio of the two bodies
    :param array_like x_source: the horizontal positions of the source center in the source plane
    :param array_like y_source: the vertical positions of the source center in the source plane
    :return: the PSBL magnification A_PSBL(t)
    :rtype: array_like
    """

    amplification_psbl = []

    for xs, ys, s in zip(x_source, y_source, separation):

        magnification_VBB =VBB.BinaryMag0(s, mass_ratio, xs, ys)

        amplification_psbl.append(magnification_VBB)

    return np.array(amplification_psbl)

# Calculate a binary lightcurve that crosses a caustic at t0
def Binary_caustic_lightcurve(N, timestamps, baseline):
    # N: number of observations
    
    x1l, y1l = 0., 0. #lens 1 at origin
    y2l = 0.0
    log_d = np.random.uniform(-0.7, 0.7)
    x2l = 10**log_d
    z1=x1l+1.j*y1l
    z2=x2l+1.j*y2l
    log_q = np.random.uniform(-2.0,0.0) #-2.0 planetary
    
    # t0 auf Kaustik caustic_value_x, caustic_value_y
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    t0 = np.random.uniform(lower_bound, upper_bound)  
    
    # construct timestamps
    # t0 + 1/2 N, 
    t_start = t0 - 0.5*N
    t_end = t0 + 0.5*N
    times = np.arange(t_start, t_end, 1.0).tolist() # 1 observation/day
    
    tE = np.random.normal(50.0, 10.0)
    q = 10**log_q
    m1 = q/(q+1)
    m2 = 1.-m1
    blend_ratio = np.random.uniform(0,10) 
    # Get source trajectory
    crx, cry, cax, cay = direct_critpattern(m1,m2,z2,z1,1) # vary last parameter, =1 8 solutions
    rand_xaxis_value, caustic_value_x, caustic_value_y = get_two_source_plane_points(m1,m2,z2,z1)
    m,b = get_source_trajectory(rand_xaxis_value, caustic_value_x, caustic_value_y)

    #Get x_values, y_values
    x_values_traj = []
    y_values_traj = []
    
    for t in times:
        x_value = x_from_tE(t, t0, tE, caustic_value_x, caustic_value_y, m, b)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m,b)
        y_values_traj.append(y_value)
    
    
    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)


    separation = np.array(([0]*len(times)))+x2l  #np.array([0]*len(tau))
    magnification = amplification_PSBL(separation, q , x_values_traj, y_values_traj)
    
    # PSBL magnitudes
    mag = constant(times, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    flux_obs = f_s*magnification + f_b+flux_noise
    mu = flux_obs/flux_base
    microlensing_mag = -2.5*np.log10(flux_obs)
    
    return np.array(microlensing_mag),times,mu

# Calculate a planetary lightcurve that crosses a caustic at t0
def Planetary_caustic_lightcurve(N, timestamps, baseline):
    # N: number of observations
    
    x1l, y1l = 0., 0. #lens 1 at origin
    y2l = 0.0
    log_d = np.random.uniform(-0.7, 0.7)
    x2l = 10**log_d
    z1=x1l+1.j*y1l
    z2=x2l+1.j*y2l
    log_q = np.random.uniform(-6.0,-2.0) #-2.0 planetary
    
    # t0 auf Kaustik caustic_value_x, caustic_value_y
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    t0 = np.random.uniform(lower_bound, upper_bound)  
    
    # construct timestamps
    # t0 + 1/2 N, 
    t_start = t0 - 0.5*N
    t_end = t0 + 0.5*N
    times = np.arange(t_start, t_end, 1.0).tolist() # 1 observation/day
    
    tE = np.random.normal(50.0, 10.0)
    q = 10**log_q
    m1 = q/(q+1)
    m2 = 1.-m1
    blend_ratio = np.random.uniform(0,10) 
    # Get source trajectory
    crx, cry, cax, cay = direct_critpattern(m1,m2,z2,z1,1) # vary last parameter, =1 8 solutions
    rand_xaxis_value, caustic_value_x, caustic_value_y = get_two_source_plane_points(m1,m2,z2,z1)
    m,b = get_source_trajectory(rand_xaxis_value, caustic_value_x, caustic_value_y)

    #Get x_values, y_values
    x_values_traj = []
    y_values_traj = []
    
    for t in times:
        x_value = x_from_tE(t, t0, tE, caustic_value_x, caustic_value_y, m, b)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m,b)
        y_values_traj.append(y_value)
    
    
    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)


    separation = np.array(([0]*len(times)))+x2l  #np.array([0]*len(tau))
    magnification = amplification_PSBL(separation, q , x_values_traj, y_values_traj)
    
    # PSBL magnitudes
    mag = constant(times, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    flux_obs = f_s*magnification + f_b+flux_noise
    mu = flux_obs/flux_base
    microlensing_mag = -2.5*np.log10(flux_obs)
    
    return np.array(microlensing_mag),times,mu

def PSBL_caustic_lightcurve(N, timestamps, baseline):
    # N: number of observations
    
    x1l, y1l = 0., 0. #lens 1 at origin
    y2l = 0.0
    log_d = np.random.uniform(-0.7, 0.7)
    x2l = 10**log_d
    z1=x1l+1.j*y1l
    z2=x2l+1.j*y2l
    log_q = np.random.uniform(-6.0,0.0) #-2.0 planetary
    
    # t0 auf Kaustik caustic_value_x, caustic_value_y
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    t0 = np.random.uniform(lower_bound, upper_bound)  
    
    # construct timestamps
    # t0 + 1/2 N, 
    t_start = t0 - 0.5*N
    t_end = t0 + 0.5*N
    times = np.arange(t_start, t_end, 1.0).tolist() # 1 observation/day
    
    tE = np.random.normal(50.0, 10.0)
    q = 10**log_q
    m1 = q/(q+1)
    m2 = 1.-m1
    blend_ratio = np.random.uniform(0,10) 
    # Get source trajectory
    crx, cry, cax, cay = direct_critpattern(m1,m2,z2,z1,1) # vary last parameter, =1 8 solutions
    rand_xaxis_value, caustic_value_x, caustic_value_y = get_two_source_plane_points(m1,m2,z2,z1)
    m,b = get_source_trajectory(rand_xaxis_value, caustic_value_x, caustic_value_y)

    #Get x_values, y_values
    x_values_traj = []
    y_values_traj = []
    
    for t in times:
        x_value = x_from_tE(t, t0, tE, caustic_value_x, caustic_value_y, m, b)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m,b)
        y_values_traj.append(y_value)
    
    
    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)


    separation = np.array(([0]*len(times)))+x2l  #np.array([0]*len(tau))
    magnification = amplification_PSBL(separation, q , x_values_traj, y_values_traj)
    
    # PSBL magnitudes
    mag = constant(times, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    flux_obs = f_s*magnification + f_b+flux_noise
    mu = flux_obs/flux_base
    microlensing_mag = -2.5*np.log10(flux_obs)
    
    return np.array(microlensing_mag),times,mu















