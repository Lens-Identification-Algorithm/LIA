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
from astropy.io.votable import parse_single_table
from itertools import groupby
from scipy.signal import find_peaks

from LIA import simulate
from LIA import noise_models
from LIA import quality_check
from LIA import extract_features
from LIA import PSPL_Fisher_matrix


def create(mira_table,ztf_table, noise=None, PL_criteria = 'Strong', Binary_events = 'Yes',BL_criteria='Strong',BL_classes='Yes', n_class=500, ml_n1=7, cv_n1=7, cv_n2=1):
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
    log_10 = np.log(10)
    # prepare timestamps and baselines depending on noise model
    # Gaia
    if noise == 'Gaia':
        time_baseline_pairs = ztf_table['time_base']

# bin_edges, magerr_intervals, mag_max, mag_intervals
            
    if noise == 'ZTF':
        time_baseline_pairs = ztf_table['time_base']
        bin_edges = ztf_table['bin_edges']
        magerr_intervals = ztf_table['magerr_intervals']
        mag_max = ztf_table['mag_max']
        mag_intervals = ztf_table['mag_intervals']

# noise

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
            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
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
            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
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
            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
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
                        mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
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
    
        
    print ("Now simulating PL microlensing...")
    
    N134 = []
    u0 = []
    tE = []
    rho_par = []
    median_mag_list = []
    ids_ml = []

    col_names = ['Source class','id','N134', 'u0','tE','rho','median mag','d','log_q','topology', 'alpha']
    data_parameters = []
    
    if PL_criteria == 'Strong':
        for k in range(1,n_class+1):
            for j in range(100000):
                choosen_pair = random.choice(time_baseline_pairs)
                time = choosen_pair[0]
                time = np.array(time)
                baseline = choosen_pair[1]
                if Binary_events == 'Yes':
                    mag, baseline, u_0, t_0, t_e, blend_ratio, flux_obs, f_s, f_b, magnification,rho = simulate.microlensing_ESPL(time, baseline)
                if Binary_events == 'No':
                    mag, baseline, u_0, t_0, t_e, blend_ratio, flux_obs, f_s, f_b,magnification = simulate.microlensing(time, baseline)                
                mag_value = 1.34 
                count = len([h for h in magnification if h >= mag_value])
                if count >= 20:   # count geringer?       
                    try:
                        if noise == 'Gaia':
                            mag, magerr = noise_models.add_gaia_g_noise(mag)
                        if noise == 'ZTF':
                            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
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
                    try:
                        C = linalg.inv(F)
                    except (linalg.LinAlgError, ValueError):
                        continue
            
            
                    sigma_t_e_rel = sqrt(abs(C[0][0]))/t_e
                    sigma_u_0_rel = sqrt(abs(C[1][1]))/u_0

                    quality = quality_check.test_microlensing(time, mag, magerr, baseline, u_0, t_0, t_e, blend_ratio, n=ml_n1)
                
                    if quality is True and sigma_t_e_rel <= 0.1 and sigma_u_0_rel <= 0.1:                
                        source_class = ['ML']*len(time)
                        source_class_list.append(source_class)
#                id_num = [4*n_class+k]*len(time)
                        id_num = [4*n_class+k]*len(time)
                        id_list.append(id_num)
                
                        ids_ml.append(id_num)
                        count = len([h for h in magnification if h >= 1.34])
                        N134.append(count)
                        u0.append(u_0)
                        tE.append(t_e)
                        if Binary_events == 'Yes':
                            rho_par.append(rho)
                        if Binary_events == 'No':
                            rho_par.append(0)
                        median_mag_list.append(np.median(mag))
            
                        times_list.append(time)
                        mag_list.append(mag)
                        magerr_list.append(magerr)
                    
                        stats = extract_features.extract_all(mag,magerr, convert=True)
                        stats = [i for i in stats]
#                        stats = ['ML'] + [4*n_class+k] + stats
                        stats = ['ML'] + [4*n_class+k] + stats
                        stats_list.append(stats)
                        break
                if j == 99999:
                    raise RuntimeError('Unable to simulate proper ML in 100k tries with current cadence -- inspect cadence and/or noise model and try again.')
    

    if PL_criteria == 'Weak':
        for k in range(1,n_class+1):
            for j in range(100000):
                choosen_pair = random.choice(time_baseline_pairs)
                time = choosen_pair[0]
                time = np.array(time)
                baseline = choosen_pair[1]
                if Binary_events == 'Yes':
                    mag, baseline, u_0, t_0, t_e, blend_ratio, flux_obs, f_s, f_b, magnification,rho = simulate.microlensing_ESPL(time, baseline)
                if Binary_events == 'No':
                    mag, baseline, u_0, t_0, t_e, blend_ratio, flux_obs, f_s, f_b,magnification = simulate.microlensing(time, baseline)                
                mag_value = 1.34 
                count = len([h for h in magnification if h >= mag_value])
                if count >= 20:   # count geringer?       
                    try:
                        if noise == 'Gaia':
                            mag, magerr = noise_models.add_gaia_g_noise(mag)
                        if noise == 'ZTF':
                            mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                        if noise is None:
                            mag, magerr= noise_models.add_gaussian_noise(mag)
                    except ValueError:
                        continue
            
                    sigma_mu_134 = []
                    for j in range(len(magnification)):
                        if magnification[j] >= mag_value:
                            sigma_F_k = magerr[j]*(0.4*log_10)/(10**(0.4*mag[j]))
                            sigma_mu_k = sigma_F_k*f_s**(-1)
                            d_k = magnification[j] - 1
                            #print(magnification[k],sigma_mu_k,d_k)
                            if d_k >= sigma_mu_k:
                                sigma_mu_134.append(1)
                            else:
                                sigma_mu_134.append(0)
                        else:
                            sigma_mu_134.append(0)
                    count_cons_1 = [len(list(g[1])) for g in groupby(sigma_mu_134) if g[0]==1]
                    try:
                        max_counts = max(count_cons_1)
                    except ValueError:
                        continue
                    
                    min_distance = simulate.minimum_distance(np.median(mag))
                    diff_mag = max(mag)-min(mag)                

                    quality = quality_check.test_microlensing(time, mag, magerr, baseline, u_0, t_0, t_e, blend_ratio, n=ml_n1)
                
                    if quality is True and max_counts >=6 and diff_mag >= min_distance:                
                        source_class = ['ML']*len(time)
                        source_class_list.append(source_class)
#                id_num = [4*n_class+k]*len(time)
                        id_num = [4*n_class+k]*len(time)
                        id_list.append(id_num)
                
                        ids_ml.append(id_num)
                        count = len([h for h in magnification if h >= 1.34])
                        N134.append(count)
                        u0.append(u_0)
                        tE.append(t_e)
                        if Binary_events == 'Yes':
                            rho_par.append(rho)
                        if Binary_events == 'No':
                            rho_par.append(0)
                        median_mag_list.append(np.median(mag))
            
                        times_list.append(time)
                        mag_list.append(mag)
                        magerr_list.append(magerr)
                    
                        stats = extract_features.extract_all(mag,magerr, convert=True)
                        stats = [i for i in stats]
#                        stats = ['ML'] + [4*n_class+k] + stats
                        stats = ['ML'] + [4*n_class+k] + stats
                        stats_list.append(stats)
                        break
                if j == 99999:
                    raise RuntimeError('Unable to simulate proper ML in 100k tries with current cadence -- inspect cadence and/or noise model and try again.')
    
    # Save relevant parameters
    if Binary_events == 'Yes':
        ML_type = 'ESPL'
    if Binary_events == 'No':
        ML_type = 'PSPL'
    for i in range(len(N134)):
        rowdata = [ML_type,ids_ml[i][0], N134[i],u0[i],tE[i],rho_par[i],median_mag_list[i],0,0,0,0]
        data_parameters.append(rowdata)
    '''
    # N134 table
    print('Writing N134 ESPL table')
    col_names_ESPL = ['ids', 'N134']
    data_ESPL =  []
    for i in range(len(N134)):
        rowdata_ESPL = [ids_ml[i],N134[i]]
        data_ESPL.append(rowdata_ESPL)
    
    t_ESPL = Table(rows=data_ESPL, names=col_names_ESPL)
    t_ESPL.write('Parameters_ESPL.xml', format = 'votable')  
    '''             
    print("Microlensing events successfully simulated")

    if Binary_events == 'Yes':
        
        if BL_classes == 'Yes':
            print ("Now simulating microlensing ESBL Binary events...")
            N134_bin = []
            u0_bin = []
            tE_bin = []
            rho_par_bin = []
            median_mag_bin_list = []
            ids_bin = []
            d_bin = []
            logq_bin = []
            topology_bin = []
            alpha_bin = []
            
            if BL_criteria == 'Weak':
                for k in range(1,n_class+1):
                    for l in range(10000):
                        choosen_pair = random.choice(time_baseline_pairs)
                        time = choosen_pair[0]
                        time = np.array(time)
                        baseline = choosen_pair[1]
                        try:
                            mag,magnification,u_0,t_e, rho,d, log_q,topology,f_s, alpha = simulate.ESBL_binary(time,baseline)                 
                        except ZeroDivisionError:
                            continue  
                        mag_value = 1.34 
                        count = len([h for h in magnification if h >= mag_value])
                        if count >= 20:
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue     

                    
                            sigma_mu_134 = []
                            for j in range(len(magnification)):
                                if magnification[j] >= mag_value:
                                    sigma_F_k = magerr[j]*(0.4*log_10)/(10**(0.4*mag[j]))
                                    sigma_mu_k = sigma_F_k*f_s**(-1)
                                    d_k = magnification[j] - 1
                                    #print(magnification[k],sigma_mu_k,d_k)
                                    if d_k >= sigma_mu_k:
                                        sigma_mu_134.append(1)
                                    else:
                                        sigma_mu_134.append(0)
                                else:
                                    sigma_mu_134.append(0)
                            count_cons_1 = [len(list(g[1])) for g in groupby(sigma_mu_134) if g[0]==1]
                            try:
                                max_counts = max(count_cons_1)
                            except ValueError:
                                continue
                    
                            min_distance = simulate.minimum_distance(np.median(mag))
                            diff_mag = max(mag)-min(mag)
                            if max_counts >=6 and diff_mag >= min_distance:
                                source_class = ['ESBL_Binary']*len(time)
                                source_class_list.append(source_class)
                                id_num = [5*n_class+k]*len(time)
                                id_list.append(id_num)
                                
                                ids_bin.append(id_num)
                                count = len([h for h in magnification if h >= 1.34])
                                N134_bin.append(count)
                                u0_bin.append(u_0)
                                tE_bin.append(t_e)
                                rho_par_bin.append(rho)
                                median_mag_bin_list.append(np.median(mag))
                                d_bin.append(d)
                                logq_bin.append(log_q)
                                topology_bin.append(topology)
                                alpha_bin.append(alpha)
                                
                                times_list.append(time)
                                mag_list.append(mag)
                                magerr_list.append(magerr)
                                
                                stats = extract_features.extract_all(mag,magerr, convert=True)
                                stats = [i for i in stats]
                                stats = ['ESBL_Binary'] + [5*n_class+k] + stats
                                stats_list.append(stats)
                                break  
                        if l == 9999:
                            raise RuntimeError('Unable to simulate ESBL Binary lightcurve in 10k tries.')                      
            
            if BL_criteria == 'Strong':
                for k in range(1,n_class+1):
                    for l in range(1000000):
                        choosen_pair = random.choice(time_baseline_pairs)
                        time = choosen_pair[0]
                        time = np.array(time)
                        baseline = choosen_pair[1]
                        try:
                            mag,magnification,t_e, rho,d, log_q,f_s,topology = simulate.ESBL_bin_caustic(time,baseline)                 
                        except ZeroDivisionError:
                            continue  
                        mag_value = 1.34 
                        count = len([h for h in magnification if h >= mag_value])
                        if count >= 20:
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue   
                            peaks, properties = find_peaks(mag, prominence=1)
                            if len(peaks) >= 2:
                                source_class = ['ESBL_Binary']*len(time)
                                source_class_list.append(source_class)
                                id_num = [5*n_class+k]*len(time)
                                id_list.append(id_num)
                                #print(id_num)
                                ids_bin.append(id_num)
                                count = len([h for h in magnification if h >= 1.34])
                                N134_bin.append(count)
                                u0_bin.append(0)
                                tE_bin.append(t_e)
                                rho_par_bin.append(rho)
                                median_mag_bin_list.append(np.median(mag))
                                d_bin.append(d)
                                logq_bin.append(log_q)
                                topology_bin.append(topology)
                                alpha_bin.append(0)
                                
                                times_list.append(time)
                                mag_list.append(mag)
                                magerr_list.append(magerr)
                                
                                stats = extract_features.extract_all(mag,magerr, convert=True)
                                stats = [i for i in stats]
                                stats = ['ESBL_Binary'] + [5*n_class+k] + stats
                                stats_list.append(stats)
                                break  
                        if l == 999999:
                            raise RuntimeError('Unable to simulate ESBL Binary lightcurve in 1m tries.')                      
    
            # Save relevant parameters 
            for i in range(len(N134_bin)):
                rowdata = ['ESBL_bin',ids_bin[i][0], N134_bin[i],u0_bin[i],tE_bin[i],rho_par_bin[i],median_mag_bin_list[i],d_bin[i],logq_bin[i],topology_bin[i],alpha_bin[i]]
                data_parameters.append(rowdata)
        
            print("ESBL binary events successfully simulated")

            print ("Now simulating microlensing ESBL planetary events...")
            N134_plan = []
            u0_plan = []
            tE_plan = []
            rho_par_plan = []
            median_mag_plan_list = []
            ids_plan = []
            d_plan = []
            logq_plan = []
            topology_plan = []
            alpha_plan = []

            if BL_criteria == 'Strong':        
                for k in range(1,n_class+1):
                    for l in range(1000000):
                        choosen_pair = random.choice(time_baseline_pairs)
                        time = choosen_pair[0]
                        time = np.array(time)
                        baseline = choosen_pair[1]
                        try:
                            mag,magnification,t_e, rho,d, log_q,f_s,topology = simulate.ESBL_plan_caustic(time,baseline)                                         
                        except ZeroDivisionError:
                            continue  
                        mag_value = 1.34 
                        count = len([h for h in magnification if h >= mag_value])
                        if count >= 20:
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue  

                            peaks, properties = find_peaks(mag, prominence=1)
                            if len(peaks) >= 2:                    
                                source_class = ['ESBL_Plan']*len(time)
                                source_class_list.append(source_class)
                                id_num = [6*n_class+k]*len(time)
                                id_list.append(id_num)
                                #print(id_num)
                                ids_plan.append(id_num)
                                count = len([h for h in magnification if h >= 1.34])
                                N134_plan.append(count)
                                u0_plan.append(0)
                                tE_plan.append(t_e)
                                rho_par_plan.append(rho)
                                median_mag_plan_list.append(np.median(mag))
                                d_plan.append(d)
                                logq_plan.append(log_q)
                                topology_plan.append(topology)
                                alpha_plan.append(0)
                        
                                times_list.append(time)
                                mag_list.append(mag)
                                magerr_list.append(magerr)
                                
                                stats = extract_features.extract_all(mag,magerr, convert=True)
                                stats = [i for i in stats]
                                stats = ['ESBL_Plan'] + [6*n_class+k] + stats
                                stats_list.append(stats)
                                break  
                        if l == 999999:
                            raise RuntimeError('Unable to simulate ESBL Binary lightcurve in 1m tries.')                      
        
            if BL_criteria == 'Weak':        
                for k in range(1,n_class+1):
                    for l in range(10000):
                        choosen_pair = random.choice(time_baseline_pairs)
                        time = choosen_pair[0]
                        time = np.array(time)
                        baseline = choosen_pair[1]
                        try:
                            mag,magnification,u_0,t_e, rho,d, log_q,topology,f_s, alpha = simulate.ESBL_planetary(time,baseline)                 
                        except ZeroDivisionError:
                            continue  
                        mag_value = 1.34 
                        count = len([h for h in magnification if h >= mag_value])
                        if count >= 20:
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue  
                    
                            sigma_mu_134 = []
                            for j in range(len(magnification)):
                                if magnification[j] >= mag_value:
                                    sigma_F_k = magerr[j]*(0.4*log_10)/(10**(0.4*mag[j]))
                                    sigma_mu_k = sigma_F_k*f_s**(-1)
                                    d_k = magnification[j] - 1
                                    #print(magnification[k],sigma_mu_k,d_k)
                                    if d_k >= sigma_mu_k:
                                        sigma_mu_134.append(1)
                                    else:
                                        sigma_mu_134.append(0)
                                else:
                                    sigma_mu_134.append(0)
                            count_cons_1 = [len(list(g[1])) for g in groupby(sigma_mu_134) if g[0]==1]
                            try:
                                max_counts = max(count_cons_1)
                            except ValueError:
                                continue
                    
                            min_distance = simulate.minimum_distance(np.median(mag))
                            diff_mag = max(mag)-min(mag)
                            if max_counts >=6 and diff_mag >= min_distance:
                                source_class = ['ESBL_Plan']*len(time)
                                source_class_list.append(source_class)
                                id_num = [6*n_class+k]*len(time)
                                id_list.append(id_num)
                                
                                ids_plan.append(id_num)
                                count = len([h for h in magnification if h >= 1.34])
                                N134_plan.append(count)
                                u0_plan.append(u_0)
                                tE_plan.append(t_e)
                                rho_par_plan.append(rho)
                                median_mag_plan_list.append(np.median(mag))
                                d_plan.append(d)
                                logq_plan.append(log_q)
                                topology_plan.append(topology)
                                alpha_plan.append(alpha)
                        
                                times_list.append(time)
                                mag_list.append(mag)
                                magerr_list.append(magerr)
                                
                                stats = extract_features.extract_all(mag,magerr, convert=True)
                                stats = [i for i in stats]
                                stats = ['ESBL_Plan'] + [6*n_class+k] + stats
                                stats_list.append(stats)
                                break  
                        if l == 9999:
                            raise RuntimeError('Unable to simulate ESBL Binary lightcurve in 10k tries.')                      
                                    
            # Save relevant parameters 
            for i in range(len(N134_bin)):
                rowdata = ['ESBL_plan',ids_plan[i][0], N134_plan[i],u0_plan[i],tE_plan[i],rho_par_plan[i],median_mag_plan_list[i],d_plan[i],logq_plan[i],topology_plan[i],alpha_plan[i]]
                data_parameters.append(rowdata)

            print("ESBL planetary events successfully simulated")

        if BL_classes == 'No':
            print ("Now simulating microlensing ESBL Binary events...")
            N134_bin = []
            u0_bin = []
            tE_bin = []
            rho_par_bin = []
            median_mag_bin_list = []
            ids_bin = []
            d_bin = []
            logq_bin = []
            topology_bin = []
            alpha_bin = []
            
            if BL_criteria == 'Weak':
                for k in range(1,n_class+1):
                    for l in range(10000):
                        choosen_pair = random.choice(time_baseline_pairs)
                        time = choosen_pair[0]
                        time = np.array(time)
                        baseline = choosen_pair[1]
                        try:
                            mag,magnification,u_0,t_e, rho,d, log_q,topology,f_s, alpha = simulate.ESBL(time,baseline)                 
                        except ZeroDivisionError:
                            continue  
                        mag_value = 1.34 
                        count = len([h for h in magnification if h >= mag_value])
                        if count >= 20:
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue     

                    
                            sigma_mu_134 = []
                            for j in range(len(magnification)):
                                if magnification[j] >= mag_value:
                                    sigma_F_k = magerr[j]*(0.4*log_10)/(10**(0.4*mag[j]))
                                    sigma_mu_k = sigma_F_k*f_s**(-1)
                                    d_k = magnification[j] - 1
                                    #print(magnification[k],sigma_mu_k,d_k)
                                    if d_k >= sigma_mu_k:
                                        sigma_mu_134.append(1)
                                    else:
                                        sigma_mu_134.append(0)
                                else:
                                    sigma_mu_134.append(0)
                            count_cons_1 = [len(list(g[1])) for g in groupby(sigma_mu_134) if g[0]==1]
                            try:
                                max_counts = max(count_cons_1)
                            except ValueError:
                                continue
                    
                            min_distance = simulate.minimum_distance(np.median(mag))
                            diff_mag = max(mag)-min(mag)
                            if max_counts >=6 and diff_mag >= min_distance:
                                source_class = ['ESBL_Binary']*len(time)
                                source_class_list.append(source_class)
                                id_num = [5*n_class+k]*len(time)
                                id_list.append(id_num)
                                
                                ids_bin.append(id_num)
                                count = len([h for h in magnification if h >= 1.34])
                                N134_bin.append(count)
                                u0_bin.append(u_0)
                                tE_bin.append(t_e)
                                rho_par_bin.append(rho)
                                median_mag_bin_list.append(np.median(mag))
                                d_bin.append(d)
                                logq_bin.append(log_q)
                                topology_bin.append(topology)
                                alpha_bin.append(alpha)
                                
                                times_list.append(time)
                                mag_list.append(mag)
                                magerr_list.append(magerr)
                                
                                stats = extract_features.extract_all(mag,magerr, convert=True)
                                stats = [i for i in stats]
                                stats = ['ESBL_Binary'] + [5*n_class+k] + stats
                                stats_list.append(stats)
                                break  
                        if l == 9999:
                            raise RuntimeError('Unable to simulate ESBL Binary lightcurve in 10k tries.')     


            if BL_criteria == 'Strong':
                num_ml = n_class/2
                for k in range(1,int(num_ml)+1):
                    for l in range(1000000):
                        choosen_pair = random.choice(time_baseline_pairs)
                        time = choosen_pair[0]
                        time = np.array(time)
                        baseline = choosen_pair[1]
                        try:
                            mag,magnification,t_e, rho,d, log_q,f_s,topology = simulate.ESBL_bin_caustic(time,baseline)                 
                        except ZeroDivisionError:
                            continue  
                        mag_value = 1.34 
                        count = len([h for h in magnification if h >= mag_value])
                        if count >= 20:
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue   
                            peaks, properties = find_peaks(mag, prominence=1)
                            if len(peaks) >= 2:
                                source_class = ['ESBL_Binary']*len(time)
                                source_class_list.append(source_class)
                                id_num = [5*n_class+k]*len(time)
                                id_list.append(id_num)
                                
                                ids_bin.append(id_num)
                                count = len([h for h in magnification if h >= 1.34])
                                N134_bin.append(count)
                                u0_bin.append(0)
                                tE_bin.append(t_e)
                                rho_par_bin.append(rho)
                                median_mag_bin_list.append(np.median(mag))
                                d_bin.append(d)
                                logq_bin.append(log_q)
                                topology_bin.append(topology)
                                alpha_bin.append(0)
                                
                                times_list.append(time)
                                mag_list.append(mag)
                                magerr_list.append(magerr)
                                
                                stats = extract_features.extract_all(mag,magerr, convert=True)
                                stats = [i for i in stats]
                                stats = ['ESBL_Binary'] + [5*n_class+k] + stats
                                stats_list.append(stats)
                                break  
                        if l == 999999:
                            raise RuntimeError('Unable to simulate ESBL Binary lightcurve in 1m tries.')   
                #1/2 planetary
                for k in range(1,int(num_ml)+1):
                    for l in range(1000000):
                        choosen_pair = random.choice(time_baseline_pairs)
                        time = choosen_pair[0]
                        time = np.array(time)
                        baseline = choosen_pair[1]
                        try:
                            mag,magnification,t_e, rho,d, log_q,f_s,topology = simulate.ESBL_plan_caustic(time,baseline)                                         
                        except ZeroDivisionError:
                            continue  
                        mag_value = 1.34 
                        count = len([h for h in magnification if h >= mag_value])
                        if count >= 20:
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max, mag_intervals)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue  

                            peaks, properties = find_peaks(mag, prominence=1)
                            if len(peaks) >= 2:                    
                                source_class = ['ESBL_Binary']*len(time)
                                source_class_list.append(source_class)
                                id_num = [5*n_class+k+num_ml]*len(time)
                                id_list.append(id_num)
                                
                                ids_bin.append(id_num)
                                count = len([h for h in magnification if h >= 1.34])
                                N134_bin.append(count)
                                u0_bin.append(0)
                                tE_bin.append(t_e)
                                rho_par_bin.append(rho)
                                median_mag_bin_list.append(np.median(mag))
                                d_bin.append(d)
                                logq_bin.append(log_q)
                                topology_bin.append(topology)
                                alpha_bin.append(0)
                        
                                times_list.append(time)
                                mag_list.append(mag)
                                magerr_list.append(magerr)
                                
                                stats = extract_features.extract_all(mag,magerr, convert=True)
                                stats = [i for i in stats]
                                stats = ['ESBL_Binary'] + [5*n_class+k+num_ml] + stats
                                stats_list.append(stats)
                                break  
                        if l == 999999:
                            raise RuntimeError('Unable to simulate ESBL Binary lightcurve in 1m tries.')

            for i in range(len(N134_bin)):
                rowdata = ['ESBL_bin',ids_bin[i][0], N134_bin[i],u0_bin[i],tE_bin[i],rho_par_bin[i],median_mag_bin_list[i],d_bin[i],logq_bin[i],topology_bin[i],alpha_bin[i]]
                data_parameters.append(rowdata)
    print('Writing parameter table')    
    
    t = Table(rows=data_parameters, names=col_names)
    t.write('Parameters_ML.xml', format = 'votable')  
    
    
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
    
    
    if Binary_events == 'Yes':
        if BL_classes == 'Yes':
            classes = ["VARIABLE"]*n_class+['LPV']*n_class+["CONSTANT"]*n_class+["CV"]*n_class+["ML"]*n_class+['ESBL_Binary']*n_class+['ESBL_Plan']*n_class
            np.savetxt('pca_features.txt',np.c_[classes,np.arange(1,n_class*7+1),X_pca[:,:47]],fmt='%s') 
        else:
            classes = ["VARIABLE"]*n_class+['LPV']*n_class+["CONSTANT"]*n_class+["CV"]*n_class+["ML"]*n_class+['ESBL_Binary']*n_class
            np.savetxt('pca_features.txt',np.c_[classes,np.arange(1,n_class*6+1),X_pca[:,:47]],fmt='%s')             
    else:
        classes = ["VARIABLE"]*n_class+['LPV']*n_class+["CONSTANT"]*n_class+["CV"]*n_class+["ML"]*n_class
        np.savetxt('pca_features.txt',np.c_[classes,np.arange(1,n_class*5+1),X_pca[:,:47]],fmt='%s')
    print("Complete!")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    