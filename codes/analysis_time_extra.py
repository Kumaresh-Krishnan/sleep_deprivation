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

def headingAngle(raw_data, stimulus, experiment):

    start = 'bouts_start_stimulus_%03d'%(stimulus)
    end = 'bouts_end_stimulus_%03d'%(stimulus)

    first = np.nan; first_correct = np.nan

    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')
    
    try:
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

        return first, first_correct
    
    except:
        return np.nan, np.nan

def extractAngles(experiment,root):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    total_fish = np.sum(fish)

    fish_ctr = 0
    stimuli = 8

    data_first = np.full((total_fish, trials, stimuli), np.nan)
    data_first_correct = np.full((total_fish, trials, stimuli), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):

            for t in range(trials):

                folder = root / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):
                    
                    f_time, fc_time = headingAngle(raw_data, stimulus, experiment)
                    data_first[fish_ctr, t, stimulus] = f_time
                    data_first_correct[fish_ctr, t, stimulus] = fc_time

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

    first_correct_1 = data_first_correct[group_1]
    first_correct_2 = data_first_correct[group_2]

    to_save = {}
    to_save['first_1'] = first_1
    to_save['first_2'] = first_2
    to_save['first_correct_1'] = first_correct_1
    to_save['first_correct_2'] = first_correct_2
    
    return to_save

def main(experiment):

    #root = path.Path() / '..' / '..' / '..' / 'data_hanna_test_06_16_2021' # directory for data
    root = path.Path() / '..' / experiment

    data_first, data_first_correct = extractAngles(experiment, root)

    to_save = processAngles(experiment, data_first, data_first_correct)

    save_dir = path.Path() / '..' / experiment / f'data_time'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def removeNan(arr):

    arr = arr[~np.isnan(arr)]

    return np.ravel(arr)

def plotHistogram(experiment, prob=False):

    stimuli = 8

    data_path = path.Path() / '..' / experiment / f'data_time'
    tmp = hdf.loadmat(data_path)

    first_1 = tmp['first_1']
    first_2 = tmp['first_2']

    first_correct_1 = tmp['first_correct_1']
    first_correct_2 = tmp['first_correct_2']

    save_dir = path.Path() / '..' / experiment / f'time_figures'

    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')

    os.makedirs(save_dir, exist_ok=True)

    sns.set()    

    for stimulus in range(stimuli):

        f, ax = plt.subplots()
        g, ax2 = plt.subplots()

        ax.hist(removeNan(first_1[:,:,stimulus]), label='freq_ctrl', range=(0,1.0))
        ax.hist(removeNan(first_2[:,:,stimulus]), alpha=0.7, label='freq_sleep', range=(0,1.0))

        ax2.hist(removeNan(first_correct_1[:,:,stimulus]), label='freq_ctrl', range=(0,1.0))
        ax2.hist(removeNan(first_correct_2[:,:,stimulus]), alpha=0.7, label='freq_sleep', range=(0,1.0))
        
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

    return 0

if __name__ == '__main__':

    experiment = 'd8_07_15_2021'

    main(experiment)
    plotHistogram(experiment)
    plotHistogram(experiment, True)

    sys.exit(0)
