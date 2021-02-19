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
from math import sqrt
from scipy.optimize import brentq
#from astropy.io.votable import parse_single_table


VBB = VBBinaryLensing.VBBinaryLensing()
VBB.Tol = 0.001
VBB.RelTol = 0.001
VBB.minannuli=2 # stabilizing for rho>>caustics

VBB.LoadESPLTable('/home/marlen/LIA/ESPL_1.tbl')
#VBB.LoadESPLTable('/work/lacerta/LIA/ESPL_1.tbl')

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


def impact_parameter(tau, uo):
    """
    The impact parameter U(t).
    "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
    http://adsabs.harvard.edu/abs/1986ApJ...304....1P
    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param array_like uo: the uo define for example in
                              http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :return: the impact parameter U(t)
    :rtype: array_like
    """
    impact_param = (tau ** 2 + uo ** 2) ** 0.5  # u(t)

    return impact_param

def amplification_FSPLarge(tau, uo, rho): 
    """
    The VBB FSPL for large source. Faster than the numba implementations...
    Much slower than Yoo et al. but valid for all rho, all u_o
    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param array_like uo: the uo define for example in
                             http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param float rho: the normalised angular source star radius
    :param float limb_darkening_coefficient: the linear limb-darkening coefficient
    :return: the FSPL magnification A_FSPL(t) for large sources
    :rtype: array_like
    """

    #VBB.LoadESPLTable(os.path.dirname('/home/marlen/LIA/LIA/ESPL.tbl'))
    #VBB.LoadESPLTable('/home/marlen/LIA/ESPL_1.tbl')
    #VBB.LoadESPLTable('/work/lacerta/LIA/ESPL_1.tbl')
    amplification_fspl = []

    impact_param = (tau**2 + uo**2)**0.5

    for ind,u in enumerate(impact_param):

        magnification_VBB = VBB.ESPLMag(u,rho)# VBB.ESPLMagDark(u,rho,limb_darkening_coefficient)

        amplification_fspl.append(magnification_VBB)

    return np.array(amplification_fspl)


def microlensing_ESPL(timestamps, baseline):
    # Set bounds to ensure enough measurements are available near t_0 
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    
    t_0 = np.random.uniform(lower_bound, upper_bound)        
    u_0 = np.random.uniform(0, 1.5)
    logte = np.random.normal(1.5,0.4)
    t_e = 10**logte
    
    blend_ratio = np.random.uniform(0,10)

    log_rho = np.random.normal(-2.4,0.6)
    rho = 10**log_rho

    tau = []
    for t in timestamps:
        tau_i = (t-t_0)/t_e
        tau.append(tau_i)
    tau = np.array(tau)
    
    #u_t = np.sqrt(u_0**2 + ((timestamps - t_0) / t_e)**2)
    #magnification = (u_t**2 + 2.) / (u_t * np.sqrt(u_t**2 + 4.))

    magnification = amplification_FSPLarge(tau, u_0, rho)
    
    #print('tau:',tau)
    #print('magnification:', magnification)
    mag = constant(timestamps, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    
    flux_obs = f_s*magnification + f_b+flux_noise

    microlensing_mag = -2.5*np.log10(abs(flux_obs))
    
    return np.array(microlensing_mag), baseline, u_0, t_0, t_e, blend_ratio, np.array(flux_obs), f_s, f_b,magnification,rho


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
    logte = np.random.normal(1.5,0.4)
    t_e = 10**logte
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
    microlensing_mag = -2.5*np.log10(abs(flux_obs))
    
    return np.array(microlensing_mag), baseline, u_0, t_0, t_e, blend_ratio, np.array(flux_obs), f_s, f_b,magnification
    
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
'''
def lens_equation_binary(z, m1, m2, z1,z2):
    zco = z.conjugate()
    return z + m1 / (z1.conjugate() - zco) + m2 / (z2.conjugate() - zco)


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

def get_two_source_plane_points(m1,m2,z2,z1):
    crx,cry,cax,cay = [],[],[],[]
    for phi in np.random.uniform(0,2.*np.pi,2): #or only 1?
        rf = direct_roots_binary(m1,z2,phi)
        for idx in range(len(rf)):
            b1 = lens_equation_binary(complex(rf[idx].real, rf[idx].imag),m1,m2,z1,z2)
            crx.append(rf[idx].real),cry.append(rf[idx].imag)
            cax.append(b1.real),cay.append(b1.imag)    
    #min and max value of critical curves on x axis
    index_min = crx.index(min(crx))
    index_max = crx.index(max(crx))
    #print(index_min,index_max)
    #pick random xaxis value between min(x) and max(x)
    rand_xaxis_value = np.random.uniform(crx[index_min], crx[index_max])
    #transform this point (rand_xaxis_value, 0) onto source plane, needed??
    #remove those values from list of critical curves and caustics
    #print(crx,cry)
    crx.pop(index_min)  # danach index nicht mehr an gleicher Stelle!!
    index_max_new = crx.index(max(crx))
    crx.pop(index_max_new)
    #print(index_max_new)
    cry.pop(index_min)
    cry.pop(index_max_new)
    
    cax.pop(index_min)
    cax.pop(index_max_new)
    
    cay.pop(index_min)
    cay.pop(index_max_new)

    #print(crx,cry)
    #pick random caustic point from shortend list
    rand_idx = np.random.randint(0,len(crx),1)
    #print(rand_idx, len(cax))
    rand_idx = rand_idx[0]
    caustic_value_x = cax[rand_idx]
    caustic_value_y = cay[rand_idx]
    
    return rand_xaxis_value, caustic_value_x, caustic_value_y,crx, cry, cax, cay

def get_source_trajectory(rand_xaxis_value, caustic_value_x, caustic_value_y):
    # trajectory through (xaxis,0) and caustic (cax,cay)
    m = caustic_value_y/(caustic_value_x-rand_xaxis_value)
    b = -m*rand_xaxis_value
    return m, b

# test what kind of caustic topology
#cf ERDL AND SCHNEIDER 1993, DOMINIK 1999
    
def critc(d,q):
    return (q/(q+1.)**2-(1.0-d**4)**3/(27.*d**8))

def critw(d,q):
    return (d-sqrt((1.+q**(1./3.))**3/(1.+q)))

def limits(q,d):
    if q==0.:
        return 0.,0.
    lc=brentq(critc,0.1,10.,args=(q))
    lw=brentq(critw,0.1,10.,args=(q))
    return lc,lw

# Caustic region of interest
# cf Penny 2014 doi:10.1088/0004-637X/790/2/142
# reference point xc_c

# close topology
def reference_point_close(d,q):
    xc_c = (1/(1+q))*(d-((1-q)/d))
    return xc_c

def close_plan(d,q):
    axis_1 = (3/2)*d**3*sqrt(3)*sqrt(q)
    axis_2 = (2*sqrt(q))/(d*sqrt(1+d**2))
    return axis_1/2, axis_2

# wide topology

def reference_point_wide(d,q):
    xc_w = d - (1/((1+q)*d))
    return xc_w

def wide_plan(d,q):
    axis_1 = (4*sqrt(q))/(d*sqrt(d**2 - 1))
    axis_2 = (4*sqrt(q))/(d*sqrt(d**2 + 1))
    return axis_1/2, axis_2/2

def central_wide_close(d,q):
    cos_phi_c = (3/4)*(d+1/d)*(1-sqrt(1-(32/9)*(d+1/d)**(-2)))
    phi_c = np.arccos(cos_phi_c)
    axis_1 = (4*q)/(d-1/d)**2
    axis_2 = axis_1*((d-1/d)**2 * abs(np.sin(phi_c)**3))/(d+1/d - 2*cos_phi_c)**2
    return axis_1, axis_2/2


# intermediate/resonant topology
def uc_r_range(q):
    uc_r_max = 4.5*q**(1/4)
    return uc_r_max

def point_on_ellipse(gamma,axis_1,axis_2,xc):
    theta = np.random.uniform(0,2*np.pi)
    if axis_1 > axis_2:
        x = gamma*axis_1*np.cos(theta) + xc
        y = gamma*axis_2*np.sin(theta)
    else:
        x = gamma*axis_1*np.sin(theta) + xc
        y = gamma*axis_2*np.cos(theta)
    return x,y

def x_alpha_ellipse(x,y,alpha):
    if np.pi/2 < alpha < 3*np.pi/2:
        x_alpha = x - y/np.tan(alpha)
    else:
        x_alpha = x + y/np.tan(alpha)
    return x_alpha    

def x_alpha_resonant(xr,yr,alpha):
    if np.pi/2 < alpha < 3*np.pi/2:
        x_alpha = xr - yr/np.tan(alpha)
    else:
        x_alpha = xr + yr/np.tan(alpha)
    return x_alpha  

def point_on_circle(uc):
    theta = np.random.uniform(0,2*np.pi)
    x = uc*np.cos(theta)
    y = uc*np.sin(theta)
    return x,y


def get_trajectory_from_two_points(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y1-m*x1
    return m, b

def source_trajectory(x,m,b):
    y = m*x+b
    return y

def x_from_t(t,tc,tE,x_c,y_c,m,b):
    if t > tc:
        x =(-b*m*tE + m*tE*y_c + tE*x_c + sqrt(abs(-b**2*tE**2 - 2*b*m*tE**2*x_c + 2*b*tE**2*y_c + m**2*t**2 - 2*m**2*t*tc - m**2*tE**2*x_c**2 + m**2*tc**2 + 2*m*tE**2*x_c*y_c + t**2 - 2*t*tc - tE**2*y_c**2 + tc**2)))/(tE*(m**2 + 1))
    if t == tc:
        x = x_c
    if t < tc:
        x = (tE*(-b*m + m*y_c + x_c) - sqrt(abs(-b**2*tE**2 - 2*b*m*tE**2*x_c + 2*b*tE**2*y_c + m**2*t**2 - 2*m**2*t*tc - m**2*tE**2*x_c**2 + m**2*tc**2 + 2*m*tE**2*x_c*y_c + t**2 - 2*t*tc - tE**2*y_c**2 + tc**2)))/(tE*(m**2 + 1))
    return x

def u0_binary(m_source,b_source):
    m_u0 = -1/m_source
    x_u0 = -b_source/(m_source-m_u0)
    y_u0 = m_u0*x_u0
    u0 = sqrt(x_u0**2 + y_u0**2)
    return u0

def amplification_ESBL(separation, mass_ratio, x_source, y_source,rho, accuracy):
    """
    The Extended Source Binary Lens amplification, based on the work of Valerio Bozza, thanks :)
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

        magnification_VBB =VBB.BinaryMag(s, mass_ratio, xs, ys,rho,accuracy)

        amplification_psbl.append(magnification_VBB)

    return np.array(amplification_psbl)

def ESBL_binary(timestamps,baseline):
    #draw q and d randomly
    log_d = np.random.normal(0.05,0.2)
    log_q = np.random.uniform(-2.0,0.0)
   
    q = 10**log_q
    d = 10**log_d

    #m1 < m2
    m1 = q/(q+1)
    m2 = 1.-m1

    # lens positions
    x2, y2 = 0., 0. #lens 2 at origin
    x1, y1 = d, 0.  #lens1 at x = d

    #time t_c when u = u_c
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    tc = np.random.uniform(lower_bound, upper_bound)  
    
    logte = np.random.normal(1.5,0.4)
    tE = 10**logte
    blend_ratio = np.random.uniform(0,10)
    log_rho = np.random.normal(-2.4,0.6)
    rho = 10**log_rho
    accuracy_VBB = 0.001
    
    # draw alpha
    alpha = np.random.uniform(0, 2*np.pi,1)

    # check topology
    lc, lw = limits(q,d)
    if d < lc:
        topology = 'close'
        random = np.random.randint(0,2,1)
        if random[0] == 0:
            # planetary
            xc = reference_point_close(d,q)
            axis_1,axis_2 = close_plan(d,q)
            gamma = np.random.uniform(1,10)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)
        else:
            # central
            xc = 0
            axis_1,axis_2 = central_wide_close(d,q)
            gamma = np.random.uniform(1,10)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)  
    if d > lw:
        #wide topology
        topology = 'wide'
        random = np.random.randint(0,2,1)
        if random[0] == 0:
            # planetary
            xc = reference_point_wide(d,q)
            axis_1,axis_2 = wide_plan(d,q)
            gamma = np.random.uniform(1,10)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)
        else:
            # central
            xc = 0
            axis_1,axis_2 = central_wide_close(d,q)
            gamma = np.random.uniform(1,10)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)        
    if lc <= d <= lw:
        topology = 'resonant'
        # resonant topology
        uc_r_max = uc_r_range(q)
        uc_r_max = uc_r_max/2
        uc = np.random.uniform(0,uc_r_max)
        xc = x2
        x_e,y_e = point_on_circle(uc)
        x_alpha = x_alpha_resonant(x_e,y_e,alpha)
        m_source, b_source = get_trajectory_from_two_points(x_e,y_e, x_alpha, 0)
        axis_1,axis_2,gamma = 1,1,1
       #print(uc_r_max)        
                      
    #get u0
    u0 = u0_binary(m_source,b_source)
    
    # get coordinates on source track
    
    x_values_traj = []
    y_values_traj = []
    
    
    for t in timestamps:
        x_value = x_from_t(t,tc,tE,x_e,y_e,m_source,b_source)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m_source,b_source)
        y_values_traj.append(y_value)

    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)
    
    # transform into VBB coordinate system, i.e. origin at center of mass at xs
    xs = m1*d

    x_values_traj_new = []
    for x in x_values_traj:
        x_i = x - abs(xs)
        x_values_traj_new.append(x_i)
 
    x_values_traj_new = np.array(x_values_traj_new)

    separation = np.array(([0]*len(timestamps)))+d  #np.array([0]*len(tau))
    magnification = amplification_ESBL(separation, q , x_values_traj_new, y_values_traj,rho, accuracy_VBB)
    
    # PSBL magnitudes
    mag = constant(timestamps, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    flux_obs = f_s*magnification + f_b+flux_noise

    microlensing_mag = -2.5*np.log10(abs(flux_obs))
    
    return np.array(microlensing_mag), magnification, u0,tE, rho,d, log_q,topology,f_s,alpha


def ESBL_planetary(timestamps,baseline):
    #draw q and d randomly
    log_d = np.random.normal(0.05,0.2)
    log_q = np.random.uniform(-6.0,-2.0)
   
    q = 10**log_q
    d = 10**log_d

    #m1 < m2
    m1 = q/(q+1)
    m2 = 1.-m1

    # lens positions
    x2, y2 = 0., 0. #lens 2 at origin
    x1, y1 = d, 0.  #lens1 at x = d

    #time t_c when u = u_c
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    tc = np.random.uniform(lower_bound, upper_bound)  
    
    logte = np.random.normal(1.5,0.4)
    tE = 10**logte
    blend_ratio = np.random.uniform(0,10)
    log_rho = np.random.normal(-2.4,0.6)
    rho = 10**log_rho
    accuracy_VBB = 0.001
    
    # draw alpha
    alpha = np.random.uniform(0, 2*np.pi,1)

    # check topology
    lc, lw = limits(q,d)
    if d < lc:
        topology = 'close'
        random = np.random.randint(0,2,1)
        if random[0] == 0:
            # planetary
            xc = reference_point_close(d,q)
            axis_1,axis_2 = close_plan(d,q)
            gamma = np.random.uniform(1,50)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)
        else:
            # central
            xc = 0
            axis_1,axis_2 = central_wide_close(d,q)
            gamma = np.random.uniform(1,50)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)  
    if d > lw:
        topology = 'wide'
        #wide topology
        random = np.random.randint(0,2,1)
        if random[0] == 0:
            # planetary
            xc = reference_point_wide(d,q)
            axis_1,axis_2 = wide_plan(d,q)
            gamma = np.random.uniform(1,50)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)
        else:
            # central
            xc = 0
            axis_1,axis_2 = central_wide_close(d,q)
            gamma = np.random.uniform(1,50)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)        
    if lc <= d <= lw:
        topology = 'resonant'
        # resonant topology
        uc_r_max = uc_r_range(q)
        uc_r_max = uc_r_max/2
        uc = np.random.uniform(0,uc_r_max)
        xc = x2
        x_e,y_e = point_on_circle(uc)
        x_alpha = x_alpha_resonant(x_e,y_e,alpha)
        m_source, b_source = get_trajectory_from_two_points(x_e,y_e, x_alpha, 0)
        axis_1,axis_2,gamma = 1,1,1
       #print(uc_r_max)        
                      
    #get u0
    u0 = u0_binary(m_source,b_source)
    
    # get coordinates on source track
    
    x_values_traj = []
    y_values_traj = []
    
    
    for t in timestamps:
        x_value = x_from_t(t,tc,tE,x_e,y_e,m_source,b_source)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m_source,b_source)
        y_values_traj.append(y_value)

        
    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)
    
    # transform into VBB coordinate system, i.e. origin at center of mass at xs
    xs = m1*d

    x_values_traj_new = []
    for x in x_values_traj:
        x_i = x - abs(xs)
        x_values_traj_new.append(x_i)
 
    x_values_traj_new = np.array(x_values_traj_new)

    separation = np.array(([0]*len(timestamps)))+d  #np.array([0]*len(tau))
    magnification = amplification_ESBL(separation, q , x_values_traj_new, y_values_traj,rho, accuracy_VBB)
    
    # ESBL magnitudes
    mag = constant(timestamps, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    flux_obs = f_s*magnification + f_b+flux_noise

    microlensing_mag = -2.5*np.log10(abs(flux_obs))
    
    return np.array(microlensing_mag), magnification, u0,tE, rho,d, log_q,topology,f_s,alpha


def ESBL(timestamps,baseline):
    #draw q and d randomly
    log_d = np.random.normal(0.05,0.2)
    log_q = np.random.uniform(-6.0,0.0)
   
    q = 10**log_q
    d = 10**log_d

    #m1 < m2
    m1 = q/(q+1)
    m2 = 1.-m1

    # lens positions
    x2, y2 = 0., 0. #lens 2 at origin
    x1, y1 = d, 0.  #lens1 at x = d

    #time t_c when u = u_c
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    tc = np.random.uniform(lower_bound, upper_bound)  
    
    logte = np.random.normal(1.5,0.4)
    tE = 10**logte
    blend_ratio = np.random.uniform(0,10)
    log_rho = np.random.normal(-2.4,0.6)
    rho = 10**log_rho
    accuracy_VBB = 0.001
    
    # draw alpha
    alpha = np.random.uniform(0, 2*np.pi,1)

    # check topology
    lc, lw = limits(q,d)
    if d < lc:
        topology = 'close'
        random = np.random.randint(0,2,1)
        if random[0] == 0:
            # planetary
            xc = reference_point_close(d,q)
            axis_1,axis_2 = close_plan(d,q)
            gamma = np.random.uniform(1,25)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)
        else:
            # central
            xc = 0
            axis_1,axis_2 = central_wide_close(d,q)
            gamma = np.random.uniform(1,25)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)  
    if d > lw:
        topology = 'wide'
        #wide topology
        random = np.random.randint(0,2,1)
        if random[0] == 0:
            # planetary
            xc = reference_point_wide(d,q)
            axis_1,axis_2 = wide_plan(d,q)
            gamma = np.random.uniform(1,25)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)
        else:
            # central
            xc = 0
            axis_1,axis_2 = central_wide_close(d,q)
            gamma = np.random.uniform(1,25)
            x_e, y_e = point_on_ellipse(gamma,axis_1,axis_2,xc)
            x_alpha = x_alpha_ellipse(x_e,y_e,alpha)
            m_source, b_source = get_trajectory_from_two_points(x_e, y_e, x_alpha, 0)        
    if lc <= d <= lw:
        topology = 'resonant'
        # resonant topology
        uc_r_max = uc_r_range(q)
        uc_r_max = uc_r_max/2
        uc = np.random.uniform(0,uc_r_max)
        xc = x2
        x_e,y_e = point_on_circle(uc)
        x_alpha = x_alpha_resonant(x_e,y_e,alpha)
        m_source, b_source = get_trajectory_from_two_points(x_e,y_e, x_alpha, 0)
        axis_1,axis_2,gamma = 1,1,1
       #print(uc_r_max)        
                      
    #get u0
    u0 = u0_binary(m_source,b_source)
    
    # get coordinates on source track
    
    x_values_traj = []
    y_values_traj = []
    
    
    for t in timestamps:
        x_value = x_from_t(t,tc,tE,x_e,y_e,m_source,b_source)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m_source,b_source)
        y_values_traj.append(y_value)

    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)
    
    # transform into VBB coordinate system, i.e. origin at center of mass at xs
    xs = m1*d

    x_values_traj_new = []
    for x in x_values_traj:
        x_i = x - abs(xs)
        x_values_traj_new.append(x_i)
 
    x_values_traj_new = np.array(x_values_traj_new)

    separation = np.array(([0]*len(timestamps)))+d  #np.array([0]*len(tau))
    magnification = amplification_ESBL(separation, q , x_values_traj_new, y_values_traj,rho, accuracy_VBB)
    
    # ESBL magnitudes
    mag = constant(timestamps, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    flux_obs = f_s*magnification + f_b+flux_noise

    microlensing_mag = -2.5*np.log10(abs(flux_obs))
    
    return np.array(microlensing_mag), magnification, u0,tE, rho,d, log_q,topology,f_s,alpha


def minimum_distance(baseline):
    if baseline >= 20.0:
        min_distance = 2.0
    if baseline >= 19.0 and baseline < 20.0:
        min_distance = 1.5
    if baseline >= 18.0 and baseline < 19.0:
        min_distance = 1.0
    if baseline < 18.0:
        min_distance = 0.5
    return min_distance


# planetary with caustic criteria
    
def ESBL_plan_caustic(timestamps, baseline):
    log_d = np.random.normal(0.05,0.2)
    d = 10**log_d
        # lens positions
    x1, y1 = 0., 0. #lens 2 at origin
    x2, y2 = d, 0.  #lens1 at x = d

    z1=x1+1.j*y1
    z2=x2+1.j*y2
    log_q = np.random.uniform(-6.0,-2.0) 

    # t0 auf Kaustik caustic_value_x, caustic_value_y
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    t0 = np.random.uniform(lower_bound, upper_bound)
    log_rho = np.random.normal(-2.4,0.6)
    rho = 10**log_rho
    accuracy_VBB = 0.001
    
   
    q = 10**log_q

    m2 = q/(q+1)
    m1 = 1.-m2
    
    lc, lw = limits(q,d)
    if d < lc:
        topology = 'close'    
    if d > lw:
        topology = 'wide'
    if lc <= d <= lw:
        topology = 'resonant'
    #time t_c when u = u_c
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    #tc = np.random.uniform(lower_bound, upper_bound)  
    
    logte = np.random.normal(1.5,0.4)
    tE = 10**logte
    blend_ratio = np.random.uniform(0,10)
    #accuracy_VBB = 0.001
  
    # draw alpha
    #alpha = np.random.uniform(0, 2*np.pi,1)
    
    # construct timestamps
    # t0 + 1/2 N, 
    #t_start = t0 - 0.5*N
    #t_end = t0 + 0.5*N
    #times = np.arange(t_start, t_end, 0.01).tolist() # 1 observation/day
    #print(t0, len(times), times)
    #print(len(times))

    # Get source trajectory
    #crx, cry, cax, cay = direct_critpattern(m1,m2,z2,z1,1) # vary last parameter, =1 8 solutions
    rand_xaxis_value, caustic_value_x, caustic_value_y,crx, cry, cax, cay = get_two_source_plane_points(m1,m2,z2,z1)
    m,b = get_source_trajectory(rand_xaxis_value, caustic_value_x, caustic_value_y)

    #print(tE, t0, m1, m2)
    #crxmin = min(crx)
    #ymin = source_trajectory(crxmin,m,b)
    #crxmax = max(crx)
    #ymax = source_trajectory(crxmax,m,b)

    #len_traj = ((crxmax-crxmin)**2 + (ymax-ymin)**2)**0.5
    
    #Get x_values, y_values
    x_values_traj = []
    y_values_traj = []
    
    for t in timestamps:
        x_value = x_from_t(t, t0, tE, caustic_value_x, caustic_value_y, m, b)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m,b)
        y_values_traj.append(y_value)
    
    
    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)
   # transform into VBB coordinate system, i.e. origin at center of mass at xs
    xs = m2*d

    x_values_traj_new = []
    for x in x_values_traj:
        x_i = x - abs(xs)
        x_values_traj_new.append(x_i)
 
    x_values_traj_new = np.array(x_values_traj_new)
    separation = np.array(([0]*len(timestamps)))+d  #np.array([0]*len(tau))
    magnification = amplification_ESBL(separation, q , x_values_traj_new, y_values_traj,rho, accuracy_VBB)
    #separation = np.array(([0]*len(times)))+x2l  #np.array([0]*len(tau))
    #magnification = amplification_PSBL(separation, q , x_values_traj, y_values_traj)
    #print(len(magnification),len(x_values_traj), len(y_values_traj))
    
    # PSBL magnitudes
    mag = constant(timestamps, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    #print(f_s, len(magnification),f_b,len(flux_noise))
    flux_obs = f_s*magnification + f_b+flux_noise
    #print('mu:', mu)
    microlensing_mag = -2.5*np.log10(flux_obs)
    
    #magnification_pl  = amplification_PSBL(separation, q , x_values_traj_new, y_values_traj)

    #separation = np.array(([0]*len(times)))+x2l  #np.array([0]*len(tau))
    #magnification = amplification_PSBL(separation, q , x_values_traj, y_values_traj)
    #print(len(magnification),len(x_values_traj), len(y_values_traj))
    
    #print(f_s, len(magnification),f_b,len(flux_noise))
    #flux_obs_pl  = f_s*magnification_pl + f_b+flux_noise
    #print('mu:', mu)
    #microlensing_mag_pl  = -2.5*np.log10(flux_obs_pl)
       
    return np.array(microlensing_mag),magnification,tE, rho,d, log_q,f_s, topology

def ESBL_bin_caustic(timestamps, baseline):
    log_d = np.random.normal(0.05,0.2)
    d = 10**log_d
        # lens positions
    x1, y1 = 0., 0. #lens 2 at origin
    x2, y2 = d, 0.  #lens1 at x = d

    z1=x1+1.j*y1
    z2=x2+1.j*y2
    log_q = np.random.uniform(-2.0,0.0) 

    # t0 auf Kaustik caustic_value_x, caustic_value_y
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    t0 = np.random.uniform(lower_bound, upper_bound)
    log_rho = np.random.normal(-2.4,0.6)
    rho = 10**log_rho
    accuracy_VBB = 0.001
    
   
    q = 10**log_q

    m2 = q/(q+1)
    m1 = 1.-m2
    
    lc, lw = limits(q,d)
    if d < lc:
        topology = 'close'    
    if d > lw:
        topology = 'wide'
    if lc <= d <= lw:
        topology = 'resonant'
    #time t_c when u = u_c
    lower_bound = np.percentile(timestamps, 10)
    upper_bound = np.percentile(timestamps, 90)
    #tc = np.random.uniform(lower_bound, upper_bound)  
    
    logte = np.random.normal(1.5,0.4)
    tE = 10**logte
    blend_ratio = np.random.uniform(0,10)
    #accuracy_VBB = 0.001
  
    # draw alpha
    #alpha = np.random.uniform(0, 2*np.pi,1)
    
    # construct timestamps
    # t0 + 1/2 N, 
    #t_start = t0 - 0.5*N
    #t_end = t0 + 0.5*N
    #times = np.arange(t_start, t_end, 0.01).tolist() # 1 observation/day
    #print(t0, len(times), times)
    #print(len(times))

    # Get source trajectory
    #crx, cry, cax, cay = direct_critpattern(m1,m2,z2,z1,1) # vary last parameter, =1 8 solutions
    rand_xaxis_value, caustic_value_x, caustic_value_y,crx, cry, cax, cay = get_two_source_plane_points(m1,m2,z2,z1)
    m,b = get_source_trajectory(rand_xaxis_value, caustic_value_x, caustic_value_y)

    #print(tE, t0, m1, m2)
    #crxmin = min(crx)
    #ymin = source_trajectory(crxmin,m,b)
    #crxmax = max(crx)
    #ymax = source_trajectory(crxmax,m,b)

    #len_traj = ((crxmax-crxmin)**2 + (ymax-ymin)**2)**0.5
    
    #Get x_values, y_values
    x_values_traj = []
    y_values_traj = []
    
    for t in timestamps:
        x_value = x_from_t(t, t0, tE, caustic_value_x, caustic_value_y, m, b)
        x_values_traj.append(x_value)
        y_value = source_trajectory(x_value,m,b)
        y_values_traj.append(y_value)
    
    
    x_values_traj = np.array(x_values_traj)
    y_values_traj = np.array(y_values_traj)
   # transform into VBB coordinate system, i.e. origin at center of mass at xs
    xs = m2*d

    x_values_traj_new = []
    for x in x_values_traj:
        x_i = x - abs(xs)
        x_values_traj_new.append(x_i)
 
    x_values_traj_new = np.array(x_values_traj_new)
    separation = np.array(([0]*len(timestamps)))+d  #np.array([0]*len(tau))
    magnification = amplification_ESBL(separation, q , x_values_traj_new, y_values_traj,rho, accuracy_VBB)
    #separation = np.array(([0]*len(times)))+x2l  #np.array([0]*len(tau))
    #magnification = amplification_PSBL(separation, q , x_values_traj, y_values_traj)
    #print(len(magnification),len(x_values_traj), len(y_values_traj))
    
    # PSBL magnitudes
    mag = constant(timestamps, baseline)
    flux = 10**((mag) / -2.5)
    baseline = np.median(mag)
    flux_base = np.median(flux)
    flux_noise = flux-flux_base
    f_s = flux_base / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    #print(f_s, len(magnification),f_b,len(flux_noise))
    flux_obs = f_s*magnification + f_b+flux_noise
    #print('mu:', mu)
    microlensing_mag = -2.5*np.log10(flux_obs)
    
    #magnification_pl  = amplification_PSBL(separation, q , x_values_traj_new, y_values_traj)

    #separation = np.array(([0]*len(times)))+x2l  #np.array([0]*len(tau))
    #magnification = amplification_PSBL(separation, q , x_values_traj, y_values_traj)
    #print(len(magnification),len(x_values_traj), len(y_values_traj))
    
    #print(f_s, len(magnification),f_b,len(flux_noise))
    #flux_obs_pl  = f_s*magnification_pl + f_b+flux_noise
    #print('mu:', mu)
    #microlensing_mag_pl  = -2.5*np.log10(flux_obs_pl)
       
    return np.array(microlensing_mag),magnification,tE, rho,d, log_q,f_s, topology













