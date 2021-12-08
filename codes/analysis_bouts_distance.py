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

def boutInfo(raw_data, stimulus, num_bins):

    start = 'bouts_start_stimulus_%03d'%(stimulus)
    end = 'bouts_end_stimulus_%03d'%(stimulus)
    
    try:
        # Compute differences (convention uses start-end)
        b = raw_data[end]['timestamp'] - raw_data[start]['timestamp']
        pos_x = raw_data[end]['fish_position_x'] - raw_data[start]['fish_position_x']
        pos_y = raw_data[end]['fish_position_y'] - raw_data[start]['fish_position_y']
        bdist = np.sqrt(pos_x**2 + pos_y**2)

        if b.size == 0:
            return np.array([np.nan]*num_bins), np.array([np.nan]*num_bins)

        ib = raw_data[start]['timestamp'][1:] - raw_data[end]['timestamp'][:-1]
        pos_x = raw_data[start]['fish_position_x'][1:] - raw_data[end]['fish_position_x'][:-1]
        pos_y = raw_data[start]['fish_position_y'][1:] - raw_data[end]['fish_position_y'][:-1]
        ibdist = np.sqrt(pos_x**2 + pos_y**2)
        
        if ib.size == 0:
            return np.array([np.nan]*num_bins), np.array([np.nan]*num_bins)

        freq_ib, _ = np.histogram(ib, bins=num_bins, range=(0,10))
        freq_bdist, _ = np.histogram(bdist, bins=num_bins, range=(0,0.15))

        return freq_ib, freq_bdist
    
    except:
        return np.array([np.nan]*num_bins), np.array([np.nan]*num_bins)

def extractAngles(experiment,root, num_bins):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    total_fish = np.sum(fish)

    fish_ctr = 0
    stimuli = 8

    data_ib = np.full((total_fish, trials, stimuli, num_bins), np.nan)
    data_bdist = np.full((total_fish, trials, stimuli, num_bins), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):

                folder = root / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):
                    
                    ib, bdist = boutInfo(raw_data, stimulus, num_bins)
                    data_ib[fish_ctr, t, stimulus] = ib
                    data_bdist[fish_ctr, t, stimulus] = bdist

                tmp.close()

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data_ib, data_bdist

def processAngles(experiment, data_ib, data_bdist, num_bins):

    info_path = path.Path() / '..' / experiment
    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()

    group_1 = info['control']
    group_2 = info['sleep']

    data_ib_1 = np.nanmean(data_ib[group_1], axis=(0,1))
    data_ib_2 = np.nanmean(data_ib[group_2], axis=(0,1))

    data_bdist_1 = np.nanmean(data_bdist[group_1], axis=(0,1))
    data_bdist_2 = np.nanmean(data_bdist[group_2], axis=(0,1))

    sem_tmp_1 = np.nanmean(data_ib[group_1], axis=1)
    sem_tmp_2 = np.nanmean(data_ib[group_2], axis=1)

    sem_ib_1 = sem(sem_tmp_1, axis=0, nan_policy='omit')
    sem_ib_2 = sem(sem_tmp_2, axis=0, nan_policy='omit')

    sem_tmp_1 = np.nanmean(data_bdist[group_1], axis=1)
    sem_tmp_2 = np.nanmean(data_bdist[group_2], axis=1)

    sem_bdist_1 = sem(sem_tmp_1, axis=0, nan_policy='omit')
    sem_bdist_2 = sem(sem_tmp_2, axis=0, nan_policy='omit')

    to_save = {}

    to_save['ib_1'] = data_ib_1
    to_save['ib_2'] = data_ib_2
    to_save['sem_ib_1'] = sem_ib_1
    to_save['sem_ib_2'] = sem_ib_2
    to_save['bdist_1'] = data_bdist_1
    to_save['bdist_2'] = data_bdist_2
    to_save['sem_bdist_1'] = sem_bdist_1
    to_save['sem_bdist_2'] = sem_bdist_2
    
    return to_save

def main(experiment, num_bins):

    #root = path.Path() / '..' / '..' / '..' / 'data_hanna_test_06_16_2021' # directory for data
    root = path.Path() / '..' / experiment
    
    data_ib, data_bdist = extractAngles(experiment, root, num_bins)

    to_save = processAngles(experiment, data_ib, data_bdist, num_bins)

    save_dir = path.Path() / '..' / experiment / f'bout_data_{num_bins}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotHistogram(experiment, num_bins, prob=False):

    stimuli = 8

    data_path = path.Path() / '..' / experiment / f'bout_data_{num_bins}'
    tmp = hdf.loadmat(data_path)

    data_ib_1 = tmp['ib_1']
    data_ib_2 = tmp['ib_2']
    sem_ib_1 = tmp['sem_ib_1']
    sem_ib_2 = tmp['sem_ib_2']

    data_bdist_1 = tmp['bdist_1']
    data_bdist_2 = tmp['bdist_2']
    sem_bdist_1 = tmp['sem_bdist_1']
    sem_bdist_2 = tmp['sem_bdist_2']
    
    save_dir = path.Path() / '..' / experiment / f'bouts_distance_histograms_{num_bins}'

    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')

    if stimuli % 2 == 0:

        half = stimuli // 2
        
        data_ib_1 = (data_ib_1[:half] + data_ib_1[half:]) / 2.0
        data_ib_2 = (data_ib_2[:half] + data_ib_2[half:]) / 2.0
        data_bdist_1 = (data_bdist_1[:half] + data_bdist_1[half:]) / 2.0
        data_bdist_2 = (data_bdist_2[:half] + data_bdist_2[half:]) / 2.0

        sem_ib_1 = np.sqrt((sem_ib_1[:half]**2 + sem_ib_1[half:]**2) / 2.0)
        sem_ib_2 = np.sqrt((sem_ib_2[:half]**2 + sem_ib_2[half:]**2) / 2.0)
        sem_bdist_1 = np.sqrt((sem_bdist_1[:half]**2 + sem_bdist_1[half:]**2) / 2.0)
        sem_bdist_2 = np.sqrt((sem_bdist_2[:half]**2 + sem_bdist_2[half:]**2) / 2.0)

    else:

        half = (stimuli - 1) // 2
        
        data_ib_1 = (data_ib_1[:half] + data_ib_1[half:-1]) / 2.0
        data_ib_2 = (data_ib_2[:half] + data_ib_2[half:-1]) / 2.0
        data_bdist_1 = (data_bdist_1[:half] + data_bdist_1[half:-1]) / 2.0
        data_bdist_2 = (data_bdist_2[:half] + data_bdist_2[half:-1]) / 2.0

        sem_ib_1 = np.sqrt((sem_ib_1[:half]**2 + sem_ib_1[half:-1]**2) / 2.0)
        sem_ib_2 = np.sqrt((sem_ib_2[:half]**2 + sem_ib_2[half:-1]**2) / 2.0)
        sem_bdist_1 = np.sqrt((sem_bdist_1[:half]**2 + sem_bdist_1[half:-1]**2) / 2.0)
        sem_bdist_2 = np.sqrt((sem_bdist_2[:half]**2 + sem_bdist_2[half:-1]**2) / 2.0)

    if prob:
        norm = data_ib_1.sum(axis=1).reshape(-1,1)
        data_ib_1 = data_ib_1 / norm; sem_ib_1 = sem_ib_1 / norm
        norm = data_ib_2.sum(axis=1).reshape(-1,1)
        data_ib_2 = data_ib_2 / norm; sem_ib_2 = sem_ib_2 / norm
        norm = data_bdist_1.sum(axis=1).reshape(-1,1)
        data_bdist_1 = data_bdist_1 / norm; sem_bdist_1 = sem_bdist_1 / norm
        norm = data_bdist_2.sum(axis=1).reshape(-1,1)
        data_bdist_2 = data_bdist_2 / norm; sem_bdist_2 = sem_bdist_2 / norm

        save_dir = path.Path() / '..' / experiment / f'bouts_distance_histograms_prob_{num_bins}'

    os.makedirs(save_dir, exist_ok=True)

    sns.set_style('white')
    sns.set_style('ticks')
    
    for stimulus in range(half):

        f, (ax1, ax2) = plt.subplots(2, figsize=(10,12))

        ax1.plot(np.linspace(0,10,num_bins), data_ib_1[stimulus], label='control', color = 'xkcd:greyish blue')
        ax1.plot(np.linspace(0,10,num_bins), data_ib_2[stimulus], label='sleep deprived', color = 'xkcd:aquamarine' )

        ax2.plot(np.linspace(0,0.15,num_bins), data_bdist_1[stimulus], label='control', color = 'xkcd:greyish blue')
        ax2.plot(np.linspace(0,0.15,num_bins), data_bdist_2[stimulus], label='sleep deprived', color = 'xkcd:aquamarine' )

        ax1.fill_between(np.linspace(0,10,num_bins), \
            data_ib_1[stimulus]-sem_ib_1[stimulus], \
            data_ib_1[stimulus]+sem_ib_1[stimulus], \
            color='gray', alpha=0.5)

        ax1.fill_between(np.linspace(0,10,num_bins), \
            data_ib_2[stimulus]-sem_ib_2[stimulus], \
            data_ib_2[stimulus]+sem_ib_2[stimulus], \
            color='gray', alpha=0.5)

        ax2.fill_between(np.linspace(0,0.15,num_bins), \
            data_bdist_1[stimulus]-sem_bdist_1[stimulus], \
            data_bdist_1[stimulus]+sem_bdist_1[stimulus], \
            color='gray', alpha=0.5)

        ax2.fill_between(np.linspace(0,0.15,num_bins), \
            data_bdist_2[stimulus]-sem_bdist_2[stimulus], \
            data_bdist_2[stimulus]+sem_bdist_2[stimulus], \
            color='gray', alpha=0.5)

        
        #ax1.bar(np.linspace(0,2,num_bins), data_ib_1[stimulus], label='control', color = 'xkcd:greyish blue')
        #ax1.bar(np.linspace(0,2,num_bins), data_ib_2[stimulus], alpha=0.7, label='sleep deprived', color = 'xkcd:aquamarine' )

        #ax2.bar(np.linspace(0,0.3,num_bins), data_bdist_1[stimulus], label='control', color = 'xkcd:greyish blue')
        #ax2.bar(np.linspace(0,0.3,num_bins), data_bdist_2[stimulus], alpha=0.7, label='sleep deprived', color = 'xkcd:aquamarine' )
        
        ax1.set_xlabel(f'Time (s)'); ax2.set_xlabel('Distance (Normalized radius 1.0)')
        ax1.set_ylabel(f'Counts'); ax2.set_ylabel(f'Counts')
        
        ax1.set_title(f'{id_map[str(half+stimulus)][0]} Stimulus {id_map[str(half+stimulus)][1]} % - Interbout Intervals')
        ax2.set_title(f'{id_map[str(half+stimulus)][0]} Stimulus {id_map[str(half+stimulus)][1]} % - Bout distance')
        ax1.legend(); ax2.legend()
        ax1.grid(False); ax2.grid(False)
        sns.despine(top=True, right=True)

        f.savefig(save_dir / f'fig_{id_map[str(half+stimulus)][0]}_{id_map[str(half+stimulus)][1]}_grey.pdf')
        plt.close(f)

    return 0

if __name__ == '__main__':

    experiment = 'd8_07_14_2021'
    num_bins = 50

    main(experiment, num_bins)
    plotHistogram(experiment, num_bins)
    plotHistogram(experiment, num_bins, True)

    sys.exit(0)