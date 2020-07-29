# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
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


from LIA import simulate
from LIA import noise_models
from LIA import quality_check
from LIA import extract_features
from LIA import PSPL_Fisher_matrix
    
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlmodels
from pyLIMA import microloutputs


def create(all_oids, all_mag, all_magerr, all_mjd, noise=None, PSBL_class = 'yes', PSBL_method='Peak', n_class=500, ml_n1=7, cv_n1=7, cv_n2=1):
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
    PSBL_class: 'yes' - include a class for PSBL events
        
    PSBL_method: 'Fit' uses a pyLIMA fit to check whether a simulated PSBL lightcurve is characterizable, 
                'Peak' checks whether a simulated PSBL lightcurve has at least two maxima
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
        id_num = [1*n_class+k]*len(time)
        id_list.append(id_num)

        times_list.append(time)
        mag_list.append(mag)
        magerr_list.append(magerr)
        
        stats = extract_features.extract_all(mag,magerr,convert=True)
        stats = [i for i in stats]
#        stats = ['CONSTANT'] + [2*n_class+k] + stats
        stats = ['CONSTANT'] + [1*n_class+k] + stats
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
                id_num = [2*n_class+k]*len(time)        
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_features.extract_all(mag,magerr,convert=True)
                stats = [i for i in stats]
#                stats = ['CV'] + [3*n_class+k] + stats
                stats = ['CV'] + [2*n_class+k] + stats
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
                id_num = [3*n_class+k]*len(time)
                id_list.append(id_num)
            
                times_list.append(time)
                mag_list.append(mag)
                magerr_list.append(magerr)
                
                stats = extract_features.extract_all(mag,magerr, convert=True)
                stats = [i for i in stats]
#                stats = ['ML'] + [4*n_class+k] + stats
                stats = ['ML'] + [3*n_class+k] + stats
                stats_list.append(stats)
                break
            if j == 99999:
                raise RuntimeError('Unable to simulate proper ML in 100k tries with current cadence -- inspect cadence and/or noise model and try again.')
                    
    print("Microlensing events successfully simulated")
    
    if PSBL_class == 'yes':
        print ("Now simulating microlensing PSBL...")
        # FIT METHOD 
    
        if PSBL_method == 'Fit':
            col_names_PSBL = ['id', 't0', 'u0', 'tE', 'log_d', 'log_q', 'alpha']
            data_rows_PSBL = []
            for k in range(1,n_class+1):
                for j in range(100000):
                    choosen_pair = random.choice(time_baseline_pairs)
                    time = choosen_pair[0]
                    time = np.array(time)
                    baseline = choosen_pair[1]
                    # Simulate PSBL event
                    N = 10000
                    for i in range(N):
                        mu_value = 3.0
                        mag, base, u_0, t_0, t_e, blend_ratio, flux_obs, f_s, f_b, log_q, log_d, alpha, mu = simulate.PSBL_microlensing(time, baseline)
                        count = len([h for h in mu if h >= mu_value])
                        if count >= 4:
                            # Noise model
                            try:
                                if noise == 'Gaia':
                                    mag, magerr = noise_models.add_gaia_g_noise(mag)
                                if noise == 'ZTF':
                                    mag, magerr = noise_models.add_ztf_noise(mag, bin_edges, magerr_intervals, mag_max)
                                if noise is None:
                                    mag, magerr= noise_models.add_gaussian_noise(mag)
                            except ValueError:
                                continue 
                            
                            # Peaks
                            peaks, properties = find_peaks(mag, prominence=1)
                            if len(peaks) >= 2:
                                break
                       
              
           
            
            # quality check, pyLIMA
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        your_event = event.Event()
                        your_event.name = 'Simulated PSBL'
                        data = np.column_stack((time, mag, magerr))  
                        telescope1 = telescopes.Telescope(name='ZTF', camera_filter='r', light_curve_magnitude= data)
                        your_event.telescopes.append(telescope1)
                        model_1 = microlmodels.create_model('PSBL', your_event)
                        # parameter guess [t0, u0, tE, logs, logq, alpha]
                        model_1.parameters_guess = [t_0, u_0, t_e, log_d, log_q, alpha]

                        your_event.fit(model_1,'TRF') #LM
                        #your_event.fits[0].produce_outputs()
                        parameters = your_event.fits[-1].fit_results
                        #errors = your_event.fits[-1].outputs.fit_errors
                        # covariance matrix
                        covariance_matrix = your_event.fits[-1].fit_covariance
                    errors = abs(covariance_matrix.diagonal()) ** 0.5
                    # relative errors
                    sigma_te_rel = abs(errors[2]/parameters[2])
                    sigma_u0_rel = abs(errors[1]/parameters[1])
                    sigma_log10q = abs(errors[4])
                    q = 10**log_q
                    sigma_q = np.log(10)*q*sigma_log10q
                    sigma_q_rel = sigma_q/q

                    if sigma_te_rel <= 0.02 and sigma_u0_rel <= 0.1 and sigma_q_rel <= 0.08:
                        #print('PSBL LC ok', sigma_te_rel, sigma_u0_rel, sigma_logq_rel)
                        source_class = ['PSBL ML']*len(time)
                        source_class_list.append(source_class)
#                       id_num = [4*n_class+k]*len(time)
                        id_num = [4*n_class+k]*len(time)
                        id_list.append(id_num)
            
                        times_list.append(time)
                        mag_list.append(mag)
                        magerr_list.append(magerr)
                
                        stats = extract_features.extract_all(mag,magerr, convert=True)
                        stats = [i for i in stats]
#                       stats = ['ML'] + [4*n_class+k] + stats
                        stats = ['PSBL ML'] + [4*n_class+k] + stats
                        stats_list.append(stats)
                        rowdata_PSBL = [id_num[0], t_0, u_0, t_e, log_d, log_q, alpha]
                        data_rows_PSBL.append(rowdata_PSBL)
    
                        break
                    if j == 99999:
                        raise RuntimeError('Unable to simulate proper ML in 100k tries with current cadence -- inspect cadence and/or noise model and try again.')
    
            t = Table(rows=data_rows_PSBL, names=col_names_PSBL)
            print('Write parameter table PSBL')
            t.write('Simulated_parameters_PSBL.xml', format = 'votable')                
            print("PSBL Microlensing events successfully simulated")                
            #print('Fit parameters:',parameters, 'errors:', errors)
            #print('parameter guess:', t_0, u_0, t_e, log_d, log_q, alpha)
            #print('sigma tE:', sigma_te_rel, 'sigma u0:', sigma_u0_rel, 'sigma logq:', sigma_logq_rel)
    
    
        if PSBL_method == 'Peak':
            for k in range(1,n_class+1):
                for j in range(100000):
                    choosen_pair = random.choice(time_baseline_pairs)
                    time = choosen_pair[0]
                    time = np.array(time)
                    baseline = choosen_pair[1]
                    # mag PSBL
                    N = 10000
                    for i in range(N):
                        mu_value = 3.0
                        mag, base, u_0, t_0, t_e, blend_ratio, flux_obs, f_s, f_b, log_q, log_d, alpha, mu = simulate.PSBL_microlensing(time, baseline)
                        count = len([h for h in mu if h >= mu_value])
                        if count >= 4:
                            break
                        else:
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
                        source_class = ['PSBL ML']*len(time)
                        source_class_list.append(source_class)
                        id_num = [4*n_class+k]*len(time)
                        id_list.append(id_num)
            
                        times_list.append(time)
                        mag_list.append(mag)
                        magerr_list.append(magerr)
                
                        stats = extract_features.extract_all(mag,magerr, convert=True)
                        stats = [i for i in stats]
                        stats = ['PSBL ML'] + [4*n_class+k] + stats
                        stats_list.append(stats)
                    
                        break 
                    if j == 99999:
                        raise RuntimeError('Unable to simulate proper ML in 100k tries with current cadence.')
    
            print("PSBL Microlensing events successfully simulated")   
        
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
    
    
    if PSBL_class == 'yes':
        classes = ["VARIABLE"]*n_class+["CONSTANT"]*n_class+["CV"]*n_class+["ML"]*n_class+['PSBL ML']*n_class
        np.savetxt('pca_features.txt',np.c_[classes,np.arange(1,n_class*5+1),X_pca[:,:47]],fmt='%s') 
    else:
        classes = ["VARIABLE"]*n_class+["CONSTANT"]*n_class+["CV"]*n_class+["ML"]*n_class
        np.savetxt('pca_features.txt',np.c_[classes,np.arange(1,n_class*4+1),X_pca[:,:47]],fmt='%s')
    print("Complete!")
'''
    np.savetxt('pca_features.txt',np.column_stack(
        ((classes), (np.array(X_pca[:,0])), (np.array(X_pca[:,1])), (np.array(X_pca[:,2])),
         (np.array(X_pca[:,3])), (np.array(X_pca[:,4])), (np.array(X_pca[:,5])), (np.array(X_pca[:,6])), 
         (np.array(X_pca[:,7])),(np.array(X_pca[:,8])), (np.array(X_pca[:,9])), (np.array(X_pca[:,10])),
         (np.array(X_pca[:,11])),(np.array(X_pca[:,12])),(np.array(X_pca[:,13])),(np.array(X_pca[:,14])),
         (np.array(X_pca[:,15])),(np.array(X_pca[:,16])),(np.array(X_pca[:,17])),(np.array(X_pca[:,18])),
         (np.array(X_pca[:,19])),(np.array(X_pca[:,20])),(np.array(X_pca[:,21])),(np.array(X_pca[:,22])),
         (np.array(X_pca[:,23])), (np.array(X_pca[:,24])),(np.array(X_pca[:,25])),(np.array(X_pca[:,26])),
         (np.array(X_pca[:,27])),(np.array(X_pca[:,28])),(np.array(X_pca[:,29])),(np.array(X_pca[:,30])),
         (np.array(X_pca[:,31])),(np.array(X_pca[:,32])),(np.array(X_pca[:,33])),(np.array(X_pca[:,34])),
         (np.array(X_pca[:,35])),(np.array(X_pca[:,36])),(np.array(X_pca[:,37])),(np.array(X_pca[:,38])),
         (np.array(X_pca[:,39])),(np.array(X_pca[:,40])),(np.array(X_pca[:,41])),(np.array(X_pca[:,42])),
         (np.array(X_pca[:,43])),(np.array(X_pca[:,44])),(np.array(X_pca[:,45])),(np.array(X_pca[:,46])))), fmt='%5s')
'''  
'''       
    # For unknown reasons np.savetxt does not always entirely print the final lines, this iteration 
    # is to circumnavigate this bug.
    for i in range(100):
        try:
            np.loadtxt('pca_features.txt',dtype=str)
            break
        except ValueError:
            np.savetxt('pca_features.txt',np.column_stack(
                ((classes), (np.array(X_pca[:,0])), (np.array(X_pca[:,1])), (np.array(X_pca[:,2])),
                 (np.array(X_pca[:,3])), (np.array(X_pca[:,4])), (np.array(X_pca[:,5])), (np.array(X_pca[:,6])), 
                 (np.array(X_pca[:,7])),(np.array(X_pca[:,8])), (np.array(X_pca[:,9])), (np.array(X_pca[:,10])),
                 (np.array(X_pca[:,11])),(np.array(X_pca[:,12])),(np.array(X_pca[:,13])),(np.array(X_pca[:,14])),
                 (np.array(X_pca[:,15])),(np.array(X_pca[:,16])),(np.array(X_pca[:,17])),(np.array(X_pca[:,18])),
                 (np.array(X_pca[:,19])),(np.array(X_pca[:,20])),(np.array(X_pca[:,21])),(np.array(X_pca[:,22])),
                 (np.array(X_pca[:,23])), (np.array(X_pca[:,24])),(np.array(X_pca[:,25])),(np.array(X_pca[:,26])),
                 (np.array(X_pca[:,27])),(np.array(X_pca[:,28])),(np.array(X_pca[:,29])),(np.array(X_pca[:,30])),
                 (np.array(X_pca[:,31])),(np.array(X_pca[:,32])),(np.array(X_pca[:,33])),(np.array(X_pca[:,34])),
                 (np.array(X_pca[:,35])),(np.array(X_pca[:,36])),(np.array(X_pca[:,37])),(np.array(X_pca[:,38])),
                 (np.array(X_pca[:,39])),(np.array(X_pca[:,40])),(np.array(X_pca[:,41])),(np.array(X_pca[:,42])),
                 (np.array(X_pca[:,43])),(np.array(X_pca[:,44])),(np.array(X_pca[:,45])),(np.array(X_pca[:,46])))), fmt='%5s')
'''
    
