#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:33:07 2020

@author: marlen
"""

import numpy as np
import random
from astropy.io import fits
from sklearn import decomposition
import os
from math import log
from math import sqrt
from scipy import linalg
from scipy.stats import mstats
import contextlib
from astropy.table import Table
from scipy.signal import find_peaks
from astropy.io.votable import parse_single_table

from LIA import simulate
from LIA import noise_models
from LIA import quality_check
from LIA import extract_features
from LIA import PSPL_Fisher_matrix


def create(all_oids, all_mag, all_magerr, all_mjd, noise=None, Planetary_events = 'Yes', PSBL_condition='Strong', n_class=500, ml_n1=7, cv_n1=7, cv_n2=1):
    """Creates a training dataset using adaptive cadence.
    Simulates each class n_class times, adding errors from
    a noise model either defined using the create_noise
    function, or Gaussian by default

    Parameters
    __________
    all_oids: 
        oids of dataset 
    all_mag:
        magnitudes of dataset
    all_mjd:
        times of dataset 
        
    noise : function, optional 
        Noise model, can be created using the create_noise function.
        If None it defaults to adding Gaussian noise. 
    Planetary_events: 'Yes' - Simulate two classes: Binary and planetary ML events
        Binary: logq [-2,0]     Planetary: logq [-6,-2]
        'No': Simulate only one PSBL ML class with logq [-6,0]
        
    PSBL_cond: 'Strong', condition for a Binary Lightcurve: len(magnification >= 3.0)>= 4 and number of peaks >=2 
                Else: Only len(magnification >= 3.0)>= 4 condition for lightcurve
    n_class : int, optional
        The amount of lightcurve (per class) to simulate.
        Defaults to 500. 
    ml_n1 : int, optional
        The mininum number of measurements that should be within the 
        microlensing signal when simulating the lightcurves. 
    cv_n1 : int, optional
        The mininum number of measurements that should be within 
        at least one CV outburst when simulating the lightcurves.
    cv_n2 : int, optional
        The mininum number of measurements that should be within the 
        rise or drop of at least one CV outburst when simulating the lightcurves.

    Outputs
    _______
    dataset : FITS
        All simulated lightcurves in a FITS file, sorted by class and ID
    all_features : txt file
        A txt file containing all the features plus class label.
    pca_stats : txt file
        A txt file containing all PCA features plus class label. 
    """
    if n_class < 12:
        raise ValueError("Parameter n_class must be at least 12 for principal components to be computed.")    

    times_list=[]
    mag_list=[]
    magerr_list=[]
    id_list = []
    source_class_list=[]
    stats_list = []

    # prepare timestamps and baselines depending on noise model
    # Gaia
    if noise == 'Gaia':
        time_baseline_pairs = []
        seen = set()
        uniq = [x for x in all_oids if x not in seen and not seen.add(x)]

        for idx in uniq:
            msk = all_oids == idx
            lcmag = all_mag[msk]
            #lcmagerr = all_magerr[msk]
            mjd = all_mjd[msk]
            winsorized_values = mstats.winsorize(np.array(lcmag), limits=[0.1, 0.1])
            baseline = sum(winsorized_values)/len(winsorized_values)
            pair = [mjd, baseline]
            time_baseline_pairs.append(pair)


    # ZTF
    if noise == 'ZTF':
        time_baseline_pairs = []

        seen = set()
        uniq = [x for x in all_oids if x not in seen and not seen.add(x)]

        for idx in uniq:
            times = []
            msk = all_oids == idx
            lcmag = all_mag[msk]
            #lcmagerr = all_magerr[msk]
            mjd = all_mjd[msk]
            for k in range(len(mjd)-1):
                delta_t = mjd[k+1]-mjd[k]
                if delta_t >= 0.3:
                    times.append(mjd[k])
                else:
                    continue
            times.append(mjd[-1])
            winsorized_values = mstats.winsorize(np.array(lcmag), limits=[0.1, 0.1])
            baseline = sum(winsorized_values)/len(winsorized_values)
            pair = [times, baseline]
            time_baseline_pairs.append(pair)

        #mag_data = table.array['mag'].data
        #magerr_data = table.array['magerr'].data
        print('Length pairs', len(time_baseline_pairs))

        tuples = list(zip(all_mag, all_magerr))
        tuples_sorted = sorted(tuples)

        mag_data_sorted = []
        magerr_data_sorted = []

        for i in range(len(tuples_sorted)):
            mag_data_sorted.append(tuples_sorted[i][0])
            magerr_data_sorted.append(tuples_sorted[i][1])

        mag_min = min(all_mag)
        mag_max = max(all_mag)

        hist, bin_edges = np.histogram(mag_data_sorted, bins = 'auto')

        interval_list = []

        for i in range(len(bin_edges)):
            interval = []
            for mag_magerr_tuple in tuples_sorted:
                try:
                    if bin_edges[i] <= mag_magerr_tuple[0] < bin_edges[i+1]:
                        interval.append(mag_magerr_tuple)
#                       interval.append(mag_magerr_tuple[1])
                except(IndexError):
                    pass
            interval_list.append(interval)            

        magerr_intervals = []
        for interval_i in interval_list:
            magerr_int = []
            for j in range(len(interval_i)):
                magerr_int.append(interval_i[j][1])
            magerr_intervals.append(magerr_int)

    while True:
        try:
            x=len(time_baseline_pairs[0][0])
            break
        except TypeError:
            raise ValueError("Incorrect format -- append the timestamps to a list and try again.")
        

    print("Now simulating variables...")
    for k in range(1,n_class+1):
        choosen_pair = random.choice(time_baseline_pairs)
        time = choosen_pair[0]
        time = np.array(time)
        baseline = choosen_pair[1]
        mag, amplitude, period = simulate.variable(time,baseline)
           
        if noise == 'Gaia':
            mag, magerr = noise_models.add_gaia_g_noise(mag)
        if noise == 'ZTF':
            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
        if noise is None:
            mag, magerr = noise_models.add_gaussian_noise(mag) # zp=max_mag+3
           

        source_class = ['VARIABLE']*len(time)
        source_class_list.append(source_class)

        id_num = [k]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_features.extract_all(mag,magerr,convert=True)
        stats = [i for i in stats]
        stats = ['VARIABLE'] + [k] + stats
        stats_list.append(stats)
        
    print("Variables successfully simulated")
    
    mira_table = parse_single_table('Miras_vo.xml')

    primary_period = mira_table.array['col4'].data
    amplitude_pp = mira_table.array['col5'].data
    secondary_period = mira_table.array['col6'].data
    amplitude_sp = mira_table.array['col7'].data
    tertiary_period = mira_table.array['col8'].data
    amplitude_tp = mira_table.array['col9'].data
    
    
    print("Now simulating LPVs (Miras)...")
    for k in range(1,n_class+1):
        choosen_pair = random.choice(time_baseline_pairs)
        time = choosen_pair[0]
        time = np.array(time)
        baseline = choosen_pair[1]
        mag = simulate.simulate_mira_lightcurve(time, baseline, primary_period, amplitude_pp, secondary_period, amplitude_sp, tertiary_period, amplitude_tp)
           
        if noise == 'Gaia':
            mag, magerr = noise_models.add_gaia_g_noise(mag)
        if noise == 'ZTF':
            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
        if noise is None:
            mag, magerr = noise_models.add_gaussian_noise(mag) # zp=max_mag+3
           

        source_class = ['LPV']*len(time)
        source_class_list.append(source_class)

        id_num = [1*n_class+k]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_features.extract_all(mag,magerr,convert=True)
        stats = [i for i in stats]
        stats = ['LPV'] + [1*n_class+k] + stats
        stats_list.append(stats)
        
    print("LPVs successfully simulated")
    
    
    print("Now simulating constants...")
    for k in range(1,n_class+1):
        choosen_pair = random.choice(time_baseline_pairs)
        time = choosen_pair[0]
        time = np.array(time)
        baseline = choosen_pair[1]
        mag = simulate.constant(time, baseline)
        
        if noise == 'Gaia':
             mag, magerr = noise_models.add_gaia_g_noise(mag)
        if noise == 'ZTF':
            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
        #if noise is not None:
        #    mag, magerr = noise_models.add_gaia_g_noise(mag)
#            mag, magerr = noise_models.add_noise(mag, noise)
        if noise is None:
            mag, magerr = noise_models.add_gaussian_noise(mag) #zp=max_mag+3
           
        source_class = ['CONSTANT']*len(time)
        source_class_list.append(source_class)

#        id_num = [2*n_class+k]*len(time)
        id_num = [2*n_class+k]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_features.extract_all(mag,magerr,convert=True)
        stats = [i for i in stats]
#        stats = ['CONSTANT'] + [2*n_class+k] + stats
        stats = ['CONSTANT'] + [2*n_class+k] + stats
        stats_list.append(stats)
        
    print("Constants successfully simulated")
    print("Now simulating CV...")
    for k in range(1,n_class+1):
        for j in range(10000):
            choosen_pair = random.choice(time_baseline_pairs)
            time = choosen_pair[0]
            time = np.array(time)
            baseline = choosen_pair[1]
            mag, burst_start_times, burst_end_times, end_rise_times, end_high_times = simulate.cv(time, baseline)
            
            quality = quality_check.test_cv(time, burst_start_times, burst_end_times, end_rise_times, end_high_times, n1=cv_n1, n2=cv_n2)
            if quality is True:
                try:
                    if noise == 'Gaia':
                        mag, magerr = noise_models.add_gaia_g_noise(mag)
                    if noise == 'ZTF':
                        mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                    if noise is None:
                        mag, magerr = noise_models.add_gaussian_noise(mag) #zp=max_mag+3
                except ValueError:
                    continue
                
                source_class = ['CV']*len(time)
                source_class_list.append(source_class)
        #        id_num = [3*n_class+k]*len(time)
                id_num = [3*n_class+k]*len(time)        
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_features.extract_all(mag,magerr,convert=True)
                stats = [i for i in stats]
#                stats = ['CV'] + [3*n_class+k] + stats
                stats = ['CV'] + [3*n_class+k] + stats
                stats_list.append(stats)
                break
            if j == 9999:
                raise RuntimeError('Unable to simulate proper CV in 10k tries with current cadence -- inspect cadence and try again.')
    
    print("CVs successfully simulated")
    print ("Now simulating microlensing PSPL...")
    for k in range(1,n_class+1):
        for j in range(100000):
            choosen_pair = random.choice(time_baseline_pairs)
            time = choosen_pair[0]
            time = np.array(time)
            baseline = choosen_pair[1]
            mag, baseline, u_0, t_0, t_e, blend_ratio, flux_obs, f_s, f_b = simulate.microlensing(time, baseline)
            try:
                if noise == 'Gaia':
                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                if noise == 'ZTF':
                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                if noise is None:
                    mag, magerr= noise_models.add_gaussian_noise(mag)
            except ValueError:
                continue

            F = np.zeros(25).reshape((5,5))
            for t in range(len(time)):
                fluxerr = (log(10)/2.5)*flux_obs[t]*magerr[t]
                F11, F12, F13, F14, F15, F22, F23, F24, F25, F33, F34, F35, F44, F45, F55 = PSPL_Fisher_matrix.fisher_matrix_contribution_single_measurement(time[t],t_e,u_0,t_0,fluxerr, f_s, f_b)
                F21 = F12
                F31 = F13
                F32 = F23
                F41 = F14
                F42 = F24
                F43 = F34
                F51 = F15
                F52 = F25
                F53 = F35
                F54 = F45
                F_t = [[F11, F12, F13, F14, F15], [F21, F22, F23, F24, F25], [F31, F32, F33, F34, F35], [F41, F42, F43, F44, F45], [F51, F52, F53, F54, F55]]
                F = np.add(F, F_t)
            
            C = linalg.inv(F)
            sigma_t_e_rel = sqrt(abs(C[0][0]))/t_e
            sigma_u_0_rel = sqrt(abs(C[1][1]))/u_0
            
            quality = quality_check.test_microlensing(time, mag, magerr, baseline, u_0, t_0, t_e, blend_ratio, n=ml_n1)
            if quality is True and sigma_t_e_rel <= 0.1 and sigma_u_0_rel <= 0.1:
                
                source_class = ['ML']*len(time)
                source_class_list.append(source_class)
#                id_num = [4*n_class+k]*len(time)
                id_num = [4*n_class+k]*len(time)
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_features.extract_all(mag,magerr, convert=True)
                stats = [i for i in stats]
#                stats = ['ML'] + [4*n_class+k] + stats
                stats = ['ML'] + [4*n_class+k] + stats
                stats_list.append(stats)
                break
            if j == 99999:
                raise RuntimeError('Unable to simulate proper ML in 100k tries with current cadence -- inspect cadence and/or noise model and try again.')
                    
    print("Microlensing events successfully simulated")

    if Planetary_events == 'Yes':
        print ("Now simulating microlensing PSBL Binary events...")
        '''
         I.e. simulate two classes of binary events for two different mass ranges:
         binary: log q = [-2.0,0], planetary: log q = [-6.0,2.0]
         Strong condition: Simulate binary lightcurves that cross caustics and fullfill two conditions:
         magnification >= 3.0 for at least 4 observations and find_peaks functions identifies at least 2 peaks
        '''
        if PSBL_condition == 'Strong':
            for k in range(1,n_class+1):
                for j in range(100000):
                    choosen_pair = random.choice(time_baseline_pairs)
                    time = choosen_pair[0]
                    time = np.array(time)
                    baseline = choosen_pair[1]
                    N = len(time) #N = number of observations 
                    
                    # Simulate PSBL event
                    I = 10000
                    for i in range(I):
                        try:
                            mu_value = 3.0 
                            mag, new_timestamps , mu = simulate.Binary_caustic_lightcurve(N, time, baseline) 
                            count = len([h for h in mu if h >= mu_value])
                            if count >= 4: 
                                break
                            else:
                                continue 
                            if i == 9999:
                                raise RuntimeError('Unable to simulate PSBL Binary in 10k tries.')     
                        except ZeroDivisionError:
                            continue

                    # noise
                    try:
                        if noise == 'Gaia':
                            mag, magerr = noise_models.add_gaia_g_noise(mag)
                        if noise == 'ZTF':
                            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                        if noise is None:
                            mag, magerr= noise_models.add_gaussian_noise(mag)
                    except ValueError:
                        continue                    
            
                    peaks, properties = find_peaks(mag, prominence=1)
    
                    if len(peaks) >= 2:
                        source_class = ['PSBL_Binary']*len(new_timestamps)
                        source_class_list.append(source_class)
                        id_num = [5*n_class+k]*len(new_timestamps)
                        id_list.append(id_num)
            
                        times_list.append(new_timestamps)
                        mag_list.append(mag)
                        magerr_list.append(magerr)
                
                        stats = extract_features.extract_all(mag,magerr, convert=True)
                        stats = [i for i in stats]
                        stats = ['PSBL_Binary'] + [5*n_class+k] + stats
                        stats_list.append(stats)
                    
                        break 
                    if j == 99999:
                        raise RuntimeError('Unable to simulate proper ML in 100k tries with current cadence.')
    
            print("PSBL Binary events  with strong condition successfully simulated")
            
        # Weak condition: Simulate binary lightcurves that cross caustics and fullfill one condition:
        # magnification >= 3.0 for at least 4 observations
        if PSBL_condition == 'Weak':
            for k in range(1,n_class+1):
                choosen_pair = random.choice(time_baseline_pairs)
                time = choosen_pair[0]
                time = np.array(time)
                baseline = choosen_pair[1]
                N = len(time) #N = number of observations 
                    
                # Simulate PSBL event
                I = 10000
                for i in range(I):
                    try:
                        mu_value = 3.0 
                        mag, new_timestamps , mu = simulate.Binary_caustic_lightcurve(N, time, baseline) 
                        count = len([h for h in mu if h >= mu_value])
                        if count >= 4:
                            break
                        else:
                            continue
                        if i == 9999:
                            raise RuntimeError('Unable to simulate PSBL Binary in 10k tries.')     
                    except ZeroDivisionError:
                        continue

                    # noise
                try:
                    if noise == 'Gaia':
                        mag, magerr = noise_models.add_gaia_g_noise(mag)
                    if noise == 'ZTF':
                        mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                    if noise is None:
                        mag, magerr= noise_models.add_gaussian_noise(mag)
                except ValueError:
                    continue                    
            
                source_class = ['PSBL_Binary']*len(new_timestamps)
                source_class_list.append(source_class)
                id_num = [5*n_class+k]*len(new_timestamps)
                id_list.append(id_num)

                times_list.append(new_timestamps)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_features.extract_all(mag,magerr, convert=True)
                stats = [i for i in stats]
                stats = ['PSBL_Binary'] + [5*n_class+k] + stats
                stats_list.append(stats)
                        
            print("PSBL Binary events with weak condition successfully simulated")        

        print ("Now simulating microlensing PSBL Planetary events...")    
        # Strong
        if PSBL_condition == 'Strong':
            for k in range(1,n_class+1):
                for j in range(100000):
                    choosen_pair = random.choice(time_baseline_pairs)
                    time = choosen_pair[0]
                    time = np.array(time)
                    baseline = choosen_pair[1]
                    N = len(time) #N = number of observations 
                    
                    # Simulate PSBL event
                    I = 10000
                    for i in range(I):
                        try:
                            mu_value = 3.0 
                            mag, new_timestamps , mu = simulate.Planetary_caustic_lightcurve(N, time, baseline) 
                            count = len([h for h in mu if h >= mu_value])
                            if count >= 4:
                                break
                            else:
                                continue
                            if i == 9999:
                                raise RuntimeError('Unable to simulate PSBL Planetary in 10k tries.')     
                        except ZeroDivisionError:
                            continue

                    # noise
                    try:
                        if noise == 'Gaia':
                            mag, magerr = noise_models.add_gaia_g_noise(mag)
                        if noise == 'ZTF':
                            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                        if noise is None:
                            mag, magerr= noise_models.add_gaussian_noise(mag)
                    except ValueError:
                        continue                    
            
                    peaks, properties = find_peaks(mag, prominence=1)
    
                    if len(peaks) >= 2:
                        source_class = ['PSBL_Planetary']*len(new_timestamps)
                        source_class_list.append(source_class)
                        id_num = [6*n_class+k]*len(new_timestamps)
                        id_list.append(id_num)
            
                        times_list.append(new_timestamps)
                        mag_list.append(mag)
                        magerr_list.append(magerr)
                
                        stats = extract_features.extract_all(mag,magerr, convert=True)
                        stats = [i for i in stats]
                        stats = ['PSBL_Planetary'] + [6*n_class+k] + stats
                        stats_list.append(stats)
                    
                        break 
                    if j == 99999:
                        raise RuntimeError('Unable to simulate proper planetary ML in 100k tries with current cadence.')
    
            print("PSBL Planetary events  with strong condition successfully simulated")
            
        # Weak
        if PSBL_condition == 'Weak':
            for k in range(1,n_class+1):
                choosen_pair = random.choice(time_baseline_pairs)
                time = choosen_pair[0]
                time = np.array(time)
                baseline = choosen_pair[1]
                N = len(time) #N = number of observations 
                    
                # Simulate PSBL event
                I = 10000
                for i in range(I):
                    try:
                        mu_value = 3.0 
                        mag, new_timestamps , mu = simulate.Planetary_caustic_lightcurve(N, time, baseline) 
                        count = len([h for h in mu if h >= mu_value])
                        if count >= 4:
                            break
                        else:
                            continue
                        if i == 9999:
                            raise RuntimeError('Unable to simulate PSBL Planetary in 10k tries.')     
                    except ZeroDivisionError:
                        continue

                # noise
                try:
                    if noise == 'Gaia':
                        mag, magerr = noise_models.add_gaia_g_noise(mag)
                    if noise == 'ZTF':
                        mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                    if noise is None:
                        mag, magerr= noise_models.add_gaussian_noise(mag)
                except ValueError:
                    continue                    
            
                source_class = ['PSBL_Planetary']*len(new_timestamps)
                source_class_list.append(source_class)
                id_num = [6*n_class+k]*len(new_timestamps)
                id_list.append(id_num)
                
                times_list.append(new_timestamps)
                mag_list.append(mag)
                magerr_list.append(magerr)
            
                stats = extract_features.extract_all(mag,magerr, convert=True)
                stats = [i for i in stats]
                stats = ['PSBL_Planetary'] + [6*n_class+k] + stats
                stats_list.append(stats)
                    
    
            print("PSBL planetary events with weak condition successfully simulated")    
    
    if Planetary_events == 'No':
        print ("Now simulating microlensing PSBL Binary events...")
        # Simulate one class of binary lightcurves for mass ranges: logq = [-6.0,0]
        # Strong
        if PSBL_condition == 'Strong':
            for k in range(1,n_class+1):
                for j in range(100000):
                    choosen_pair = random.choice(time_baseline_pairs)
                    time = choosen_pair[0]
                    time = np.array(time)
                    baseline = choosen_pair[1]
                    N = len(time) #N = number of observations 
                    
                    # Simulate PSBL event
                    I = 10000
                    for i in range(I):
                        try:
                            mu_value = 3.0 
                            mag, new_timestamps , mu = simulate.PSBL_caustic_lightcurve(N, time, baseline) 
                            count = len([h for h in mu if h >= mu_value])
                            if count >= 4:
                                break
                            else:
                                continue
                            if i == 9999:
                                raise RuntimeError('Unable to simulate PSBL in 10k tries.')     
                        except ZeroDivisionError:
                            continue

                    # noise
                    try:
                        if noise == 'Gaia':
                            mag, magerr = noise_models.add_gaia_g_noise(mag)
                        if noise == 'ZTF':
                            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                        if noise is None:
                            mag, magerr= noise_models.add_gaussian_noise(mag)
                    except ValueError:
                        continue                    
            
                    peaks, properties = find_peaks(mag, prominence=1)
    
                    if len(peaks) >= 2:
                        source_class = ['PSBL_ML']*len(new_timestamps)
                        source_class_list.append(source_class)
                        id_num = [5*n_class+k]*len(new_timestamps)
                        id_list.append(id_num)
            
                        times_list.append(new_timestamps)
                        mag_list.append(mag)
                        magerr_list.append(magerr)
                
                        stats = extract_features.extract_all(mag,magerr, convert=True)
                        stats = [i for i in stats]
                        stats = ['PSBL_ML'] + [5*n_class+k] + stats
                        stats_list.append(stats)
                    
                        break 
                    if j == 99999:
                        raise RuntimeError('Unable to simulate proper PSBL ML in 100k tries with current cadence.')
    
            print("PSBL ML events with strong condition successfully simulated")
            
        # Weak
        if PSBL_condition == 'Weak':
            for k in range(1,n_class+1):
                choosen_pair = random.choice(time_baseline_pairs)
                time = choosen_pair[0]
                time = np.array(time)
                baseline = choosen_pair[1]
                N = len(time) #N = number of observations 
                    
                # Simulate PSBL event
                I = 10000
                for i in range(I):
                    try:
                        mu_value = 3.0 
                        mag, new_timestamps , mu = simulate.PSBL_caustic_lightcurve(N, time, baseline) 
                        count = len([h for h in mu if h >= mu_value])
                        if count >= 4:
                            break
                        else:
                            continue
                        if i == 9999:
                            raise RuntimeError('Unable to simulate PSBL Binary in 10k tries.')     
                    except ZeroDivisionError:
                        continue

                # noise
                try:
                    if noise == 'Gaia':
                        mag, magerr = noise_models.add_gaia_g_noise(mag)
                    if noise == 'ZTF':
                        mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                    if noise is None:
                        mag, magerr= noise_models.add_gaussian_noise(mag)
                except ValueError:
                    continue                    
            
                source_class = ['PSBL_ML']*len(new_timestamps)
                source_class_list.append(source_class)
                id_num = [5*n_class+k]*len(new_timestamps)
                id_list.append(id_num)

                times_list.append(new_timestamps)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_features.extract_all(mag,magerr, convert=True)
                stats = [i for i in stats]
                stats = ['PSBL_ML'] + [5*n_class+k] + stats
                stats_list.append(stats)
                    
    
            print("PSBL ML events with weak condition successfully simulated")       
    
    print("Writing files...")
    col0 = fits.Column(name='Class', format='20A', array=np.hstack(source_class_list))
    col1 = fits.Column(name='ID', format='E', array=np.hstack(id_list))
    col2 = fits.Column(name='time', format='D', array=np.hstack(times_list))
    col3 = fits.Column(name='mag', format='E', array=np.hstack(mag_list))
    col4 = fits.Column(name='magerr', format='E', array=np.hstack(magerr_list))
    cols = fits.ColDefs([col0, col1, col2, col3, col4])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto('lightcurves.fits',overwrite=True)

    print("Saving features...")
    np.savetxt('feats.txt',np.array(stats_list).astype(str),fmt='%s')

    #output_file = open('feats.txt','w')
    #for line in stats_list:
    #    print >>output_file, line
    #output_file.close()
    
    with open(r'feats.txt', 'r') as infile, open(r'all_features.txt', 'w') as outfile:
         
         data = infile.read()
         data = data.replace("'", "")
         data = data.replace(",", "")
         data = data.replace("[", "")
         data = data.replace("]", "")
         outfile.write(data)
         
    os.remove('feats.txt')
    print("Computing principal components...")
    coeffs = np.loadtxt('all_features.txt',usecols=np.arange(2,49))
    pca = decomposition.PCA(n_components=47, whiten=True, svd_solver='auto')
    pca.fit(coeffs)
    #feat_strengths = pca.explained_variance_ratio_
    X_pca = pca.transform(coeffs) 
    
    
    if Planetary_events == 'Yes':
        classes = ["VARIABLE"]*n_class+['LPV']*n_class+["CONSTANT"]*n_class+["CV"]*n_class+["ML"]*n_class+['PSBL_Binary']*n_class+['PSBL_Planetary']*n_class
        np.savetxt('pca_features.txt',np.c_[classes,np.arange(1,n_class*7+1),X_pca[:,:47]],fmt='%s') 
    else:
        classes = ["VARIABLE"]*n_class+['LPV']*n_class+["CONSTANT"]*n_class+["CV"]*n_class+["ML"]*n_class+['PSBL_ML']*n_class
        np.savetxt('pca_features.txt',np.c_[classes,np.arange(1,n_class*6+1),X_pca[:,:47]],fmt='%s')
    print("Complete!")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    