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
import pingouin as pg

def makeDf(data_1, data_2, stimuli, rename):

    if stimuli % 2 == 0:

        half = stimuli // 2

        data_1 = (data_1[:,:half] + data_1[:,half:]) / 2.0
        data_2 = (data_2[:,:half] + data_2[:,half:]) / 2.0
        
    else:

        half = (stimuli - 1) // 2

        data_1 = (data_1[:,:half] + data_1[:,half:-1]) / 2.0
        data_2 = (data_2[:,:half] + data_2[:,half:-1]) / 2.0

    
    stacked = np.concatenate((data_1, data_2), axis=0)
    col_names = [str(i) for i in range(stacked.shape[1])]

    df_temp = pd.DataFrame(stacked, columns=col_names)
    df_temp['Group'] = np.concatenate(([1]*data_1.shape[0], [2]*data_2.shape[0]))
    df_temp['ID'] = np.concatenate((np.arange(data_1.shape[0]), np.arange(data_2.shape[0])))
    
    df_data = df_temp.melt(id_vars=['Group', 'ID'], value_vars=col_names, \
        var_name='Stimulus', value_name=rename)

    return df_data

def totalBout(dpath, stimuli):

    tmp = hdf.loadmat(dpath / 'data_72.mat')
    data_1 = tmp['freq_1_raw']
    data_2 = tmp['freq_2_raw']

    df_data = makeDf(data_1, data_2, stimuli, 'Frequency')

    #sns.boxplot(x='Stimulus', y='Frequency', hue='Group', data=df_data, palette='Set3')

    results = pg.rm_anova(dv='Frequency', within=['Stimulus','Group'], subject='ID', data=df_data, detailed=True)
    print(results)
    
    return 0

def correctness(dpath, stimuli):

    tmp = hdf.loadmat(dpath / 'data_correctness_72.mat')
    data_1 = tmp['correct_1_raw']
    data_2 = tmp['correct_2_raw']

    df_data = makeDf(data_1, data_2, stimuli, 'Correctness')

    #sns.boxplot(x='Stimulus', y='Correctness', hue='Group', data=df_data, palette='Set3')

    results = pg.rm_anova(dv='Correctness', within=['Stimulus','Group'], subject='ID', data=df_data, detailed=True)
    print(results)
       
    return 0
    

def performance(dpath, stimuli):


    return 0

if __name__ == '__main__':

    experiment = 'd7_07_01_2021'
    stimuli = 8

    dpath = path.Path() / '..' / experiment

    totalBout(dpath, stimuli)
    correctness(dpath, stimuli)
    #performance(dpath, stimuli)

    sys.exit()
