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

def headingAngle(raw_data, stimulus, experiment, num_bins):

    start = 'bouts_start_stimulus_%03d'%(stimulus)
    end = 'bouts_end_stimulus_%03d'%(stimulus)

    first = np.nan; first_correct = np.nan

    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')
    
    # Find bout timestamps and fish location at start
    timestamps = raw_data[start]['timestamp']
    pos_x = raw_data[start]['fish_position_x']
    pos_y = raw_data[start]['fish_position_y']

    angles = raw_data[start]['fish_accumulated_orientation'] - \
                       raw_data[end]['fish_accumulated_orientation']

    if angles.size == 0:
        return first, first_correct

    # Find time to first bout

    lim1, lim2 = 5.0, 15.00
    first_loc = np.where((timestamps>lim1) & (timestamps<lim2))[0][0]

    # First bout
    if (pos_x[first_loc]**2 + pos_y[first_loc]**2) < 0.81:
        first = timestamps[first_loc] - 5.0

    direction = id_map[str(stimulus)][0]

    if direction == -1:
        correct_loc = np.where((angles < 0) & (timestamps>lim1) & (timestamps<lim2))[0][0]
        if (pos_x[correct_loc]**2 + pos_y[correct_loc]**2) < 0.81:
            first_correct = timestamps[correct_loc] - 5.0
    elif direction == 1:
        correct_loc = np.where((angles > 0) & (timestamps>lim1) & (timestamps<lim2))[0][0]
        if (pos_x[correct_loc]**2 + pos_y[correct_loc]**2) < 0.81:
            first_correct = timestamps[correct_loc] - 5.0
    else:
        first_correct = first

    first_hist, _ = np.histogram(first, bins=num_bins, range=(0,3.0))
    first_correct_hist, _ = np.histogram(first_correct, bins=num_bins, range=(0,3.0))

    return first_hist, first_correct_hist

def extractAngles(experiment,root, num_bins):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    total_fish = np.sum(fish)

    fish_ctr = 0
    stimuli = 8

    data_first = np.full((total_fish, trials, stimuli, num_bins), np.nan)
    data_first_correct = np.full((total_fish, trials, stimuli, num_bins), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):

                folder = root / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):
                    
                    f_hist, fc_hist = headingAngle(raw_data, stimulus, experiment, num_bins)
                    data_first[fish_ctr, t, stimulus] = f_hist
                    data_first_correct[fish_ctr, t, stimulus] = fc_hist

                tmp.close()

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data_first, data_first_correct

def processAngles(experiment, data_first, data_first_correct):

    info_path = path.Path() / '..' / experiment
    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    group_1 = info['control']
    group_2 = info['sleep']
    
    first_1 = data_first[group_1]
    first_2 = data_first[group_2]

    tot_first_1 = np.nansum(first_1, axis=1)
    tot_first_2 = np.nansum(first_2, axis=1)

    avg_first_1 = np.nanmean(tot_first_1, axis=0)
    avg_first_2 = np.nanmean(tot_first_2, axis=0)

    sem_first_1 = sem(tot_first_1, axis=0, nan_policy='omit')
    sem_first_2 = sem(tot_first_2, axis=0, nan_policy='omit')

    first_correct_1 = data_first_correct[group_1]
    first_correct_2 = data_first_correct[group_2]

    tot_first_correct_1 = np.nansum(first_correct_1, axis=1)
    tot_first_correct_2 = np.nansum(first_correct_2, axis=1)

    avg_first_correct_1 = np.nanmean(tot_first_correct_1, axis=0)
    avg_first_correct_2 = np.nanmean(tot_first_correct_2, axis=0)

    sem_first_correct_1 = sem(tot_first_correct_1, axis=0, nan_policy='omit')
    sem_first_correct_2 = sem(tot_first_correct_2, axis=0, nan_policy='omit')

    to_save = {}
    to_save['avg_first_1'] = avg_first_1
    to_save['sem_first_1'] = sem_first_1
    to_save['avg_first_correct_1'] = avg_first_correct_1
    to_save['sem_first_correct_1'] = sem_first_correct_1
    to_save['avg_first_2'] = avg_first_2
    to_save['sem_first_2'] = sem_first_2
    to_save['avg_first_correct_2'] = avg_first_correct_2
    to_save['sem_first_correct_2'] = sem_first_correct_2
    
    return to_save

def main(experiment, num_bins):

    #root = path.Path() / '..' / '..' / '..' / 'data_hanna_test_06_16_2021' # directory for data
    root = path.Path() / '..' / experiment

    data_first, data_first_correct = extractAngles(experiment, root, num_bins)

    to_save = processAngles(experiment, data_first, data_first_correct)

    save_dir = path.Path() / '..' / experiment / f'data_time'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotHistogram(experiment, num_bins, prob=False):

    data_path = path.Path() / '..' / experiment / f'data_time'
    tmp = hdf.loadmat(data_path)

    first_1 = tmp['avg_first_1']
    first_2 = tmp['avg_first_2']

    sem_first_1 = tmp['sem_first_1']
    sem_first_2 = tmp['sem_first_2']

    first_correct_1 = tmp['avg_first_correct_1']
    first_correct_2 = tmp['avg_first_correct_2']

    sem_first_correct_1 = tmp['sem_first_correct_1']
    sem_first_correct_2 = tmp['sem_first_correct_2']

    stimuli = first_1.shape[0]

    save_dir = path.Path() / '..' / experiment / f'time_figures'
    save_dir_db = path.Path() / '..' / experiment / f'doubled_time_figures'

    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_db, exist_ok=True)

    sns.set()

    x_vals = np.linspace(0,3.0,num_bins)

    for stimulus in range(stimuli):

        f, ax = plt.subplots()
        g, ax2 = plt.subplots()

        ax.bar(x_vals, first_1[stimulus], width=0.1, yerr= sem_first_1[stimulus], label='control')
        ax.bar(x_vals, first_2[stimulus], width=0.1, yerr= sem_first_2[stimulus], alpha=0.7, label='sleep deprived')

        ax2.bar(x_vals, first_correct_1[stimulus], width=0.1,  yerr= sem_first_correct_1[stimulus], label='control')
        ax2.bar(x_vals, first_correct_2[stimulus], width=0.1, yerr= sem_first_correct_2[stimulus], alpha=0.7, label='sleep deprived')
        
        ax.set_xlabel(f'Time'); ax2.set_xlabel(f'Time')
        ax.set_ylabel(f'Count'); ax2.set_ylabel(f'Count')
        ax.set_title(f'{id_map[str(stimulus)][0]} Stimulus - Time to first bout')
        ax2.set_title(f'{id_map[str(stimulus)][0]} Stimulus - Time to first correct bout')
        ax.legend()
        ax2.legend()

        f.savefig(save_dir / f'fig_{stimulus}_{id_map[str(stimulus)][0]}_{id_map[str(stimulus)][1]}_first.png')
        g.savefig(save_dir / f'fig_{stimulus}_{id_map[str(stimulus)][0]}_{id_map[str(stimulus)][1]}_first_correct.png')

        plt.close(f)
        plt.close(g)

    if stimuli % 2 == 0:

        half = stimuli // 2
        
        first_1 = (np.fliplr(first_1[:half]) + first_1[half:]) / 2.0
        first_2 = (np.fliplr(first_2[:half]) + first_2[half:]) / 2.0

        sem_first_1 = np.sqrt((np.fliplr(sem_first_1[:half])**2 + sem_first_1[half:]**2) / 2.0)
        sem_first_2 = np.sqrt((np.fliplr(sem_first_2[:half])**2 + sem_first_2[half:]**2) / 2.0)

        first_correct_1 = (np.fliplr(first_correct_1[:half]) + first_correct_1[half:]) / 2.0
        first_correct_2 = (np.fliplr(first_correct_2[:half]) + first_correct_2[half:]) / 2.0

        sem_first_correct_1 = np.sqrt((np.fliplr(sem_first_correct_1[:half])**2 + sem_first_correct_1[half:]**2) / 2.0)
        sem_first_correct_2 = np.sqrt((np.fliplr(sem_first_correct_2[:half])**2 + sem_first_correct_2[half:]**2) / 2.0)

    else:

        half = (stimuli - 1) // 2
        
        first_1 = (np.fliplr(first_1[:half]) + first_1[half:-1]) / 2.0
        first_2 = (np.fliplr(first_2[:half]) + first_2[half:-1]) / 2.0

        sem_first_1 = np.sqrt((np.fliplr(sem_first_1[:half])**2 + sem_first_1[half:-1]**2) / 2.0)
        sem_first_2 = np.sqrt((np.fliplr(sem_first_2[:half])**2 + sem_first_2[half:-1]**2) / 2.0)

        first_correct_1 = (np.fliplr(first_correct_1[:half]) + first_correct_1[half:-1]) / 2.0
        first_correct_2 = (np.fliplr(first_correct_2[:half]) + first_correct_2[half:-1]) / 2.0

        sem_first_correct_1 = np.sqrt((np.fliplr(sem_first_correct_1[:half])**2 + sem_first_correct_1[half:-1]**2) / 2.0)
        sem_first_correct_2 = np.sqrt((np.fliplr(sem_first_correct_2[:half])**2 + sem_first_correct_2[half:-1]**2) / 2.0)

    for stimulus in range(half):

        f, ax = plt.subplots()
        g, ax2 = plt.subplots()

        ax.bar(x_vals, first_1[stimulus], width=0.1, yerr= sem_first_1[stimulus], label='control')
        ax.bar(x_vals, first_2[stimulus], width=0.1, yerr= sem_first_2[stimulus], alpha=0.7, label='sleep deprived')

        ax2.bar(x_vals, first_correct_1[stimulus], width=0.1,  yerr= sem_first_correct_1[stimulus], label='control')
        ax2.bar(x_vals, first_correct_2[stimulus], width=0.1, yerr= sem_first_correct_2[stimulus], alpha=0.7, label='sleep deprived')
        
        ax.set_xlabel(f'Time'); ax2.set_xlabel(f'Time')
        ax.set_ylabel(f'Count'); ax2.set_ylabel(f'Count')
        ax.set_title(f'{id_map[str(stimulus)][0]} Stimulus - Time to first bout')
        ax2.set_title(f'{id_map[str(stimulus)][0]} Stimulus - Time to first correct bout')
        ax.legend()
        ax2.legend()

        f.savefig(save_dir_db / f'fig_{stimulus}_{id_map[str(stimulus)][0]}_{id_map[str(stimulus)][1]}_first.png')
        g.savefig(save_dir_db / f'fig_{stimulus}_{id_map[str(stimulus)][0]}_{id_map[str(stimulus)][1]}_first_correct.png')

        plt.close(f)
        plt.close(g)
    
    return 0

if __name__ == '__main__':

    experiment = 'd8_07_15_2021'
    num_bins = 30

    main(experiment, num_bins)
    plotHistogram(experiment, num_bins)
    plotHistogram(experiment, num_bins, True)

    sys.exit(0)
