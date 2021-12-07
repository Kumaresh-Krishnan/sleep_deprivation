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

def headingAngle(raw_data, stimulus, num_bins):

    start = 'bouts_start_stimulus_%03d'%(stimulus)
    end = 'bouts_end_stimulus_%03d'%(stimulus)
    
    try:
        # Compute differences (convention uses start-end)
        angles = raw_data[start]['fish_accumulated_orientation'] - \
                           raw_data[end]['fish_accumulated_orientation']

        if angles.size == 0:
            return np.nan

        # Find bout timestamps and fish location at start
        timestamps = raw_data[start]['timestamp']
        pos_x = raw_data[start]['fish_position_x']
        pos_y = raw_data[start]['fish_position_y']

        # Filter angles based on stimulus time and distance from edge of dish
        # Normalize by exposure to stimulus and suitable scaling for numerical value
        scale = 1000

        lim1, lim2 = 5.0, 15.00
        locs = np.where((timestamps>lim1) & (timestamps<lim2) & (pos_x**2 + pos_y**2 < 0.81))
        normalizer = (lim2-lim1)/scale

        angles = angles[locs]

        # Restrict range to -180,180 and compute frequencies with specified num_bins
        angles[angles > 180] -= 360

        if stimulus < 4:
            prob = (np.sum(angles < 0) + 0.5*np.sum(angles == 0)) / angles.shape[0]
        else:
            prob = (np.sum(angles > 0) + 0.5*np.sum(angles == 0)) / angles.shape[0]

        return prob
    
    except:
        return np.nan

def extractAngles(experiment,root, num_bins):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    total_fish = np.sum(fish)

    fish_ctr = 0
    stimuli = 8

    data = np.full((total_fish, trials, stimuli), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):

                folder = root / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):
                    
                    correct = headingAngle(raw_data, stimulus, num_bins)
                    data[fish_ctr, t, stimulus] = correct

                tmp.close()

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data

def processAngles(experiment, data, num_bins):

    info_path = path.Path() / '..' / experiment
    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    group_1 = info['control']
    group_2 = info['sleep']

    avg_data_1 = np.nanmean(data[group_1], axis=1)
    avg_data_2 = np.nanmean(data[group_2], axis=1)

    correct_1 = np.nanmean(avg_data_1, axis=0)
    correct_2 = np.nanmean(avg_data_2, axis=0)

    sem_correct_1 = sem(avg_data_1, axis=0, nan_policy='omit')
    sem_correct_2 = sem(avg_data_2, axis=0, nan_policy='omit')

    to_save = {}
    to_save['correct_1'] = correct_1
    to_save['correct_2'] = correct_2
    to_save['sem_correct_1'] = sem_correct_1
    to_save['sem_correct_2'] = sem_correct_2
    
    return to_save

def main(experiment, num_bins):

    #root = path.Path() / '..' / '..' / '..' / 'data_hanna_test_06_16_2021' # directory for data
    root = path.Path() / '..' / experiment
    
    data = extractAngles(experiment, root, num_bins)

    to_save = processAngles(experiment, data, num_bins)

    save_dir = path.Path() / '..' / experiment / f'data_correctness_{num_bins}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

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
    
    x_range = range(4)
    
    sns.set()
    sns.set_style('white')
    sns.set_style('ticks')

    f, ax = plt.subplots()

        
    ax.errorbar(x_range, data_1[:4], yerr=sem_1, label='control', marker='o', markersize=2.0, color = 'xkcd:greyish blue')
    ax.errorbar(x_range, data_2[:4], yerr=sem_2, label='sleep deprived', marker='o', markersize=2.0, color = 'xkcd:aquamarine' )
    
    ax.set_xlabel(f'Coherence')
    ax.set_ylabel('Probability Correct')
    
    ax.set_title(f'Alignment accuracy')
    ax.set_xticks(x_range)
    text = [str(x) for x in [0,25,50,100]]
    ax.set_xticklabels(text)
    ax.legend()
    ax.grid(False)
    
    sns.despine(top=True, right=True)


    f.savefig(save_dir / f'fig_correctness_grey.pdf')
    plt.close(f)

    return 0

if __name__ == '__main__':

    experiment = 'd7_07_01_2021'
    num_bins = 72

    main(experiment, num_bins)
    plotHistogram(experiment, num_bins)

    sys.exit(0)
