#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pyscript.py
#  
#  Copyright 2019 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#  
#  version 1.0

import numpy as np
import os, sys

import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

import hdf5storage as hdf
import path

def totalBout(dpath, stimuli):

    tmp = hdf.loadmat(dpath / 'data_72.mat')
    data_1 = tmp['freq_1_raw']
    data_2 = tmp['freq_2_raw']

    return 0


def correctness(dpath, stimuli):
       
    
    return 0
    

def performance(dpath, stimuli):


    return 0

if __name__ == '__main__':

    experiment = ''
    stimuli = 8

    dpath = path.Path() / '..' / experiment
    
    totalBout(dpath, stimuli)
    correctness(dpath, stimuli)
    performance(dpath, stimuli)

    sys.exit()
