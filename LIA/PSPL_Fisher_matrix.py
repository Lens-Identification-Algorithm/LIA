#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:57:45 2020

@author: marlen
"""

import math
import numpy as np
import cmath


'''
Function that calculates the entries of the Fisher matrix for the PSPL model.
Parameters: PSPL model parameters
Returns: Fisher matrix entries
'''

def fisher_matrix_contribution_single_measurement(t,te,u0,t0,sigma,fs,fb):
    x0 = sigma**(-2)
    x1 = te**(-2)
    x2 = (t - t0)**2
    x3 = u0**2 + x1*x2
    x4 = 1/math.sqrt(x3)
    x5 = x3 + 4
    x6 = 1/math.sqrt(x5)
    x7 = x4*x6
    x8 = 2*x7
    x9 = fs*x2/te**3
    x10 = x6/x3**(3/2)
    x11 = x3 + 2
    x12 = x11*x9
    x13 = x4/x5**(3/2)
    x14 = x10*x12 + x12*x13 - x8*x9
    x15 = fs*u0
    x16 = x11*x15
    x17 = -x10*x16 - x13*x16 + x15*x8
    x18 = x0*x14
    x19 = fs*x1*(-2*t + 2*t0)
    x20 = (1/2)*x11*x19
    x21 = -x10*x20 - x13*x20 + x19*x7
    x22 = x11*x7
    x23 = x0*x17
    x24 = x0*x21

    c0 = x0*x14**2
    c1 = x17*x18
    c2 = x18*x21
    c3 = x18*x22
    c4 = x18
    c5 = x0*x17**2
    c6 = x21*x23
    c7 = x22*x23
    c8 = x23
    c9 = x0*x21**2
    c10 = x22*x24
    c11 = x24
    c12 = x0*x11**2/(x3*x5)
    c13 = x0*x22
    c14 = x0

    return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 ]

'''
Only F11, no correlations
def fisher_matrix_contribution_single_measurement(t,te,u0,t0,sigma, fs):
    x0 = (t - t0)**2
    x1 = u0**2 + x0/te**2
    x2 = 1/np.sqrt(x1)
    x3 = te**(-3)
    x4 = x1 + 4
    x5 = fs*x0*x3/np.sqrt(x4)
    x6 = x1 + 2
    return (fs*x0*x2*x3*x6/x4**(3/2) - 2*x2*x5 + x5*x6/x1**(3/2))**2/sigma**2

'''



