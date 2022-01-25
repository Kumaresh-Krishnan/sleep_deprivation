#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pyscript.py
#  
#  Copyright 2020 Kumaresh <kumaresh_krishnan@g.harvard.edu>
#
#  version 1.0
#  

import os, sys
import numpy as np

import hdf5storage as hdf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

import path
import pickle

from matplotlib import cm
from colorspacious import cspace_converter

def plotHistogram(experiment, num_bins):

    stimuli = 8

    data_path = path.Path() / '..' / experiment / f'data_correctness_{num_bins}'
    tmp = hdf.loadmat(data_path)

    save_dir = path.Path() / '..' / experiment
    
    data_1 = tmp['correct_1']
    data_2 = tmp['correct_2']
    sem_1 = tmp['sem_correct_1']
    sem_2 = tmp['sem_correct_2']

    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')

    if stimuli % 2 == 0:

        half = stimuli // 2
        
        data_1 = (data_1[:half] + data_1[half:]) / 2.0
        data_2 = (data_2[:half] + data_2[half:]) / 2.0

        sem_1 = np.sqrt((sem_1[:half]**2 + sem_1[half:]**2) / 2.0)
        sem_2 = np.sqrt((sem_2[:half]**2 + sem_2[half:]**2) / 2.0)

    else:

        half = (stimuli - 1) // 2
        
        data_1 = (data_1[:half] + data_1[half:-1]) / 2.0
        data_2 = (data_2[:half] + data_2[half:-1]) / 2.0

        sem_1 = np.sqrt((sem_1[:half]**2 + sem_1[half:-1]**2) / 2.0)
        sem_2 = np.sqrt((sem_2[:half]**2 + sem_2[half:-1]**2) / 2.0)

    data_sub = data_2 - data_1 # sleep deprived - control
    sem_sub = np.sqrt(sem_1**2 + sem_2**2) # Must add variances
    
    x_range = range(half)
    
    sns.set()
    sns.set_style('white')
    sns.set_style('ticks')

    f, ax = plt.subplots()

        
    ax.errorbar(x_range, data_sub[:half], yerr=sem_sub, label='expt - ctrl', marker='o', markersize=2.0, color = 'xkcd:greyish blue')
    
    ax.set_xlabel(f'Coherence')
    ax.set_ylabel('Difference from control')
    
    ax.set_title(f'Alignment accuracy')
    ax.set_xticks(x_range)
    text = [str(x) for x in [0,25,50,100]]
    ax.set_xticklabels(text)
    ax.legend()
    ax.grid(False)
    
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fig_correctness_subtracted.pdf')
    plt.close(f)

    return 0

if __name__ == '__main__':

    experiment = 'd7_07_01_2021'
    num_bins = 72

    plotHistogram(experiment, num_bins)

    sys.exit(0)
