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

import pandas as pd

def makeDf(data_1, data_2, stimuli):

    if stimuli % 2 == 0:

        half = stimuli // 2

        data_1 = (data_1[:,:half] + data_1[:,half:]) / 2.0
        data_2 = (data_2[:,:half] + data_2[:,half:]) / 2.0
        
    else:

        half = (stimuli - 1) // 2

        data_1 = (data_1[:,:half] + data_1[:,half:-1]) / 2.0
        data_2 = (data_2[:,:half] + data_2[:,half:-1]) / 2.0

    
    stacked = np.concatenate((data_1, data_2), axis=0)

    df_data = pd.DataFrame(stacked, columns=[str(i) for i in range(stacked.shape[1])])
    df_data['group'] = np.concatenate(([1]*data_1.shape[0], [2]*data_2.shape[0]))

    return df_data

def totalBout(dpath, stimuli):

    tmp = hdf.loadmat(dpath / 'data_72.mat')
    data_1 = tmp['freq_1_raw']
    data_2 = tmp['freq_2_raw']

    df_data = makeDf(data_1, data_2, stimuli)
    
    return 0


def correctness(dpath, stimuli):

    tmp = hdf.loadmat(dpath / 'data_correctness.mat')
    data_1 = tmp['correct_1_raw']
    data_2 = tmp['correct_2_raw']

    df_data = makeDf(data_1, data_2, stimuli)
       
    return 0
    

def performance(dpath, stimuli):


    return 0

if __name__ == '__main__':

    experiment = 'd7_07_01_2021'
    stimuli = 8

    dpath = path.Path() / '..' / experiment
    
    totalBout(dpath, stimuli)
    correctness(dpath, stimuli)
    performance(dpath, stimuli)

    sys.exit()
