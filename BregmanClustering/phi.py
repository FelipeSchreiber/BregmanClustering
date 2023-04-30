#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 28 15:43:00 2023

@author: Felipe Schreiber
"""
import numpy as np

def phi_kl( a ):
    return np.log(1+np.exp(a))

def phi_euclidean( a ):
    return np.power(a,2)
