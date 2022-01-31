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

def boutRate(raw_data, stimulus, num_bins):

    start = 'bouts_start_stimulus_%03d'%(stimulus)
    end = 'bouts_end_stimulus_%03d'%(stimulus)

    # Find bout timestamps and fish location at start
    timestamps = raw_data[start]['timestamp']
    pos_x = raw_data[start]['fish_position_x']
    pos_y = raw_data[start]['fish_position_y']

    if timestamps.size == 0:
        return np.array([np.nan]*num_bins)
    
    freq = np.full(num_bins, np.nan)
    
    for period in range(num_bins):
        
        lim1 = period*60; lim2 = lim1 + 60.0
        locs = (timestamps>lim1) & (timestamps<lim2) & (pos_x**2 + pos_y**2 < 1.0)
        normalizer = (lim2-lim1)

        freq[period] = locs.sum()

    return freq

def extractData(experiment, root):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    #num_bins = info['minutes']
    num_bins = 30
    
    total_fish = np.sum(fish)
    
    fish_ctr = 0

    data = np.full((total_fish, trials, num_bins), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):

                folder = root / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)
                    
                rate = boutRate(raw_data, 0, num_bins)
                data[fish_ctr, t] = rate

                tmp.close()

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')

    # Return a reshaped array with trials concatenated since it is 24 hr experiment
    
    return data.reshape(data.shape[0], -1)

def processData(experiment, data):

    info_path = path.Path() / '..' / experiment
    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    group_1 = info['control']
    group_2 = info['sleep']

    freq_data_1 = np.nanmean(data[group_1], axis=0)
    freq_data_2 = np.nanmean(data[group_2], axis=0)

    sem_data_1 = sem(data[group_1], axis=0, nan_policy='omit')
    sem_data_2 = sem(data[group_2], axis=0, nan_policy='omit')

    to_save = {}

    to_save['freq_1'] = freq_data_1
    to_save['freq_sem_1'] = sem_data_1.data if np.ma.isMaskedArray(sem_data_1) else sem_data_1
    to_save['freq_2'] = freq_data_2
    to_save['freq_sem_2'] = sem_data_2.data if np.ma.isMaskedArray(sem_data_2) else sem_data_2
    to_save['freq_1_raw'] = data[group_1]
    to_save['freq_2_raw'] = data[group_2]
    
    return to_save

def main(experiment):

    #root = path.Path() / '..' / '..' / '..' / 'data_hanna_test_06_16_2021' # directory for data
    root = path.Path() / '..' / experiment
    
    data = extractData(experiment, root,)

    to_save = processData(experiment, data)

    save_dir = path.Path() / '..' / experiment / f'data_rate'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotHistogram(experiment, prob=False):

    data_path = path.Path() / '..' / experiment / f'data_rate'
    tmp = hdf.loadmat(data_path)

    data_1 = tmp['freq_1']
    data_2 = tmp['freq_2']
    sem_1 = tmp['freq_sem_1']
    sem_2 = tmp['freq_sem_2']
    save_dir = path.Path() / '..' / experiment

    times = np.linspace(0, data_1.shape[0], data_1.shape[0])
    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')

    sns.set()    

    f, ax = plt.subplots()

    ax.plot(times, data_1, label='control', color = 'xkcd:greyish blue')
    ax.plot(times, data_2, label='sleep deprived', color = 'xkcd:aquamarine')
    
    plt.fill_between(times, \
        data_1-sem_1, \
        data_1+sem_1, \
        color='gray', alpha=0.5)

    plt.fill_between(times, \
        data_2-sem_2, \
        data_2+sem_2, \
        color='gray', alpha=0.5)
    
    ax.set_xlabel(f'Time (min)')
    ax.set_ylabel(f'Boute rate (bouts/min)')
    ax.set_title(f'Bout rate through experiment')
    ax.legend()
    sns.color_palette("PuRd", as_cmap=True)
    ax.grid(False)
    sns.set_style('white')
    sns.set_style('ticks')
    sns.despine(top=True, right=True)

    f.savefig(save_dir / f'fig_activity_grey.pdf')
    plt.close(f)

    return 0

if __name__ == '__main__':

    experiment = '24_test'

    main(experiment)
    plotHistogram(experiment)

    sys.exit(0)
