# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017
    
    @author: danielgodinez
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from warnings import warn
from LIA import extract_features

def predict(mag, magerr, rf_model, pca_model):
    """This function uses machine learning to classify any given lightcurve as either
        a cataclysmic variable (CV), a variable source, microlensing, or a constant star 
        displaying no variability.
        
    Parameters
    ----------
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    rf_model: fn
        The random forest model created using the function in the
        create_models module.
    pca_model: fn
        The PCA transformation devised using the function in the
        create_models module. 

    Returns
    -------
    prediction : str
        Predicted class.
    ml_pred : float
        Probability source is microlensing
    cons_pred : float
        Probability source is constant
    cv_pred : float
        Probability source is CV
    var_pred : float
        Probability source is variable
    """
    if len(mag) < 30:
        warn('The number of data points is low -- results may be unstable')
        

    #classes = ['CONSTANT', 'CV', 'ML', 'VARIABLE', 'LPV'] 
    classes = ['CONSTANT', 'CV', 'LPV', 'ML', 'VARIABLE']
    array=[]
    stat_array = array.append(extract_features.extract_all(mag, magerr, convert=True))
    array=np.array([i for i in array])
    stat_array = pca_model.transform(array)

    #prediction =rf_model.predict(stat_array)
    pred = rf_model.predict_proba(stat_array)
    #cons_pred, cv_pred, ml_pred, var_pred, lpv_pred = pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4]
    cons_pred, cv_pred, lpv_pred, ml_pred, var_pred = pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4]
    prediction = classes[np.argmax(pred)]
    predic = rf_model.predict(stat_array)
    print(predic)
    print(rf_model.classes_)
    print(prediction, pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4])
    return prediction, ml_pred, cons_pred, cv_pred, var_pred, lpv_pred

def predict_binary_2class(mag, magerr, rf_model, pca_model):
    """This function uses machine learning to classify any given lightcurve as either
        a cataclysmic variable (CV), a variable source, microlensing, or a constant star 
        displaying no variability.
        
    Parameters
    ----------
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    rf_model: fn
        The random forest model created using the function in the
        create_models module.
    pca_model: fn
        The PCA transformation devised using the function in the
        create_models module. 

    Returns
    -------
    prediction : str
        Predicted class.
    ml_pred : float
        Probability source is microlensing
    cons_pred : float
        Probability source is constant
    cv_pred : float
        Probability source is CV
    var_pred : float
        Probability source is variable
    """
    if len(mag) < 30:
        warn('The number of data points is low -- results may be unstable')
        

    classes = ['CONSTANT', 'CV', 'ESBL_Binary', 'ESBL_Plan','LPV','ML', 'VARIABLE'] 
    array=[]
    stat_array = array.append(extract_features.extract_all(mag, magerr, convert=True))
    array=np.array([i for i in array])
    stat_array = pca_model.transform(array)

    #prediction =rf_model.predict(stat_array)
    pred = rf_model.predict_proba(stat_array)
    cons_pred, cv_pred, esbl_bin_pred, esbl_plan_pred, lpv_pred, ml_pred, var_pred = pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4],pred[:,5],pred[:,6]
    prediction = classes[np.argmax(pred)]
    #predic = rf_model.predict(stat_array)
    #print(predic)
    #print(rf_model.classes_)
    #print(prediction, pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4],pred[:,5],pred[:,6])
    return prediction, ml_pred, cons_pred, cv_pred, var_pred, lpv_pred, esbl_bin_pred, esbl_plan_pred

def predict_binary_1class(mag, magerr, rf_model, pca_model):
    """This function uses machine learning to classify any given lightcurve as either
        a cataclysmic variable (CV), a variable source, microlensing, or a constant star 
        displaying no variability.
        
    Parameters
    ----------
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    rf_model: fn
        The random forest model created using the function in the
        create_models module.
    pca_model: fn
        The PCA transformation devised using the function in the
        create_models module. 

    Returns
    -------
    prediction : str
        Predicted class.
    ml_pred : float
        Probability source is microlensing
    cons_pred : float
        Probability source is constant
    cv_pred : float
        Probability source is CV
    var_pred : float
        Probability source is variable
    """
    if len(mag) < 30:
        warn('The number of data points is low -- results may be unstable')
        

    classes = ['CONSTANT', 'CV','ESBL_Binary', 'LPV', 'ML', 'VARIABLE'] 
    array=[]
    stat_array = array.append(extract_features.extract_all(mag, magerr, convert=True))
    array=np.array([i for i in array])
    stat_array = pca_model.transform(array)

    #prediction =rf_model.predict(stat_array)
    pred = rf_model.predict_proba(stat_array)
    cons_pred, cv_pred,esbl_bin_pred, lpv_pred, ml_pred, var_pred = pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4],pred[:,5]
    prediction = classes[np.argmax(pred)]
    #predic = rf_model.predict(stat_array)
    #print(predic)
    #print(rf_model.classes_)
    #print(prediction, pred[:,0],pred[:,1],pred[:,2],pred[:,3],pred[:,4],pred[:,5])
    return prediction, ml_pred, cons_pred, cv_pred, var_pred, lpv_pred, esbl_bin_pred



def predict_OV(mag, magerr, rf_model, pca_model):
    """This function uses machine learning to classify any given lightcurve as either
        a cataclysmic variable (CV), a variable source, microlensing, or a constant star 
        displaying no variability.
        
    Parameters
    ----------
    mag : array
        Magnitude array.
    magerr : array
        Corresponing photometric errors.  
    rf_model: fn
        The random forest model created using the function in the
        create_models module.
    pca_model: fn
        The PCA transformation devised using the function in the
        create_models module. 

    Returns
    -------
    prediction : str
        Predicted class.
    ml_pred : float
        Probability source is microlensing
    cons_pred : float
        Probability source is constant
    cv_pred : float
        Probability source is CV
    var_pred : float
        Probability source is variable
    """
    if len(mag) < 30:
        warn('The number of data points is low -- results may be unstable')
        

    classes = ['CONSTANT', 'CV', 'ML', 'VARIABLE'] 
    array=[]
    stat_array = array.append(extract_features.extract_all(mag, magerr, convert=True))
    array=np.array([i for i in array])
    stat_array = pca_model.transform(array)

    #prediction =rf_model.predict(stat_array)
    pred = rf_model.predict_proba(stat_array)
    cons_pred, cv_pred, ml_pred, var_pred = pred[:,0],pred[:,1],pred[:,2],pred[:,3]
    prediction = classes[np.argmax(pred)]
    predic = rf_model.predict(stat_array)
    #print(predic)
    #print(rf_model.classes_)
    #print(prediction, pred[:,0],pred[:,1],pred[:,2],pred[:,3])
    return prediction, ml_pred, cons_pred, cv_pred, var_pred

