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

    # Compute differences (convention uses start-end)
    angles = raw_data[start]['fish_accumulated_orientation'] - \
                       raw_data[end]['fish_accumulated_orientation']

    if angles.size == 0:
        return np.array([np.nan]*num_bins)

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
    freq, _ = np.histogram(angles, bins=num_bins, range=(-180,180))

    return freq/normalizer

def extractAngles(experiment,root, num_bins):

    info_path = path.Path() / '..' / experiment

    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    days = info['days']
    fish = info['fish']
    trials = info['trials']
    total_fish = np.sum(fish)

    fish_ctr = 0
    stimuli = 8

    data = np.full((total_fish, trials, stimuli, num_bins), np.nan)

    for day_idx, day in enumerate(days):

        for f in range(fish[day_idx]):
            
            for t in range(trials):

                folder = root / f'{day}_fish{f+1:03d}' / 'raw_data' / f'trial{t:03d}.dat'
                tmp = open(folder, 'rb')
                raw_data = pickle.load(tmp)

                for stimulus in range(stimuli):
                    
                    angles = headingAngle(raw_data, stimulus, num_bins)
                    data[fish_ctr, t, stimulus] = angles
                    
                tmp.close()

            fish_ctr += 1
                    
        print(day, fish_ctr, 'fish done')
        
    return data

def processAngles(experiment, data, num_bins):

    info_path = path.Path() / '..' / experiment
    info = np.load(info_path / 'expt_info.npy', allow_pickle=True).item()
    group_1 = info['control']
    group_2 = info['sleep']

    avg_data_1 = np.nanmean(data[group_1], axis=(0,1))
    avg_data_2 = np.nanmean(data[group_2], axis=(0,1))

    sem_tmp_1 = np.nanmean(data[group_1], axis=1)
    sem_tmp_2 = np.nanmean(data[group_2], axis=1)

    sem_data_1 = sem(sem_tmp_1, axis=0, nan_policy='omit')
    sem_data_2 = sem(sem_tmp_2, axis=0, nan_policy='omit')

    norm_data_1 = avg_data_1.sum(axis=1).reshape(-1,1)
    norm_data_2 = avg_data_2.sum(axis=1).reshape(-1,1)

    prob_data_1 = avg_data_1 / norm_data_1
    prob_data_2 = avg_data_2 / norm_data_2

    prob_sem_1 = sem_data_1 / norm_data_1
    prob_sem_2 = sem_data_2 / norm_data_2

    tmp = np.nansum(data[group_1], axis=3)
    freq_tmp_1 = np.nanmean(tmp, axis=1)
    freq_data_1 = np.nanmean(freq_tmp_1, axis=0)
    freq_sem_1 = sem(freq_tmp_1, axis=0, nan_policy='omit')
    freq_std_1 = np.nanstd(freq_tmp_1, axis=0)

    tmp = np.nansum(data[group_2], axis=3)
    freq_tmp_2 = np.nanmean(tmp, axis=1)
    freq_data_2 = np.nanmean(freq_tmp_2, axis=0)
    freq_sem_2 = sem(freq_tmp_2, axis=0, nan_policy='omit')
    freq_std_2 = np.nanstd(freq_tmp_2, axis=0)

    to_save = {}
    to_save['mean_1'] = avg_data_1
    to_save['sem_1'] = sem_data_1.data if np.ma.isMaskedArray(sem_data_1) else sem_data_1
    to_save['prob_1'] = prob_data_1
    to_save['prob_sem_1'] = prob_sem_1.data if np.ma.isMaskedArray(prob_sem_1) else prob_sem_1
    to_save['mean_2'] = avg_data_2
    to_save['sem_2'] = sem_data_2.data if np.ma.isMaskedArray(sem_data_2) else sem_data_2
    to_save['prob_2'] = prob_data_2
    to_save['prob_sem_2'] = prob_sem_2.data if np.ma.isMaskedArray(prob_sem_2) else prob_sem_2
    #added for SEM
    to_save['freq_1_raw'] = freq_tmp_1
    to_save['freq_1'] = freq_data_1
    to_save['freq_sem_1'] = freq_sem_1.data if np.ma.isMaskedArray(freq_sem_1) else freq_sem_1
    to_save['freq_std_1'] = freq_std_1
    to_save['freq_2_raw'] = freq_tmp_2
    to_save['freq_2'] = freq_data_2
    to_save['freq_sem_2'] = freq_sem_2.data if np.ma.isMaskedArray(freq_sem_2) else freq_sem_2
    to_save['freq_std_2'] = freq_std_2

    return to_save

def main(experiment, num_bins):

    #root = path.Path() / '..' / '..' / '..' / 'data_hanna_test_06_16_2021' # directory for data
    root = path.Path() / '..' / experiment
    
    data = extractAngles(experiment, root, num_bins)

    to_save = processAngles(experiment, data, num_bins)

    save_dir = path.Path() / '..' / experiment / f'data_{num_bins}'
    hdf.savemat(save_dir, to_save, format='7.3', oned_as='column', store_python_metadata=True)

    return 0

def plotHistogram(experiment, num_bins, prob=False):

    data_path = path.Path() / '..' / experiment / f'data_{num_bins}'
    tmp = hdf.loadmat(data_path)

    if prob:
        data_1 = tmp['prob_1']
        data_2 = tmp['prob_2']
        sem_1 = tmp['prob_sem_1']
        sem_2 = tmp['prob_sem_2']
        save_dir = path.Path() / '..' / experiment / f'stimulus_histograms_{num_bins}_prob'
        save_dir_db = path.Path() / '..' / experiment / f'doubled_stimulus_histograms_{num_bins}_prob'
        
    else:
        data_1 = tmp['mean_1']
        data_2 = tmp['mean_2']
        sem_1 = tmp['sem_1']
        sem_2 = tmp['sem_2']
        save_dir = path.Path() / '..' / experiment / f'stimulus_histograms_{num_bins}'
        save_dir_db = path.Path() / '..' / experiment / f'doubled_stimulus_histograms_{num_bins}'

    stimuli = data_1.shape[0]
    
    angles = np.linspace(-180,180, num_bins)
    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_db, exist_ok=True)
    sns.set()    

    for stimulus in range(stimuli):

        f, ax = plt.subplots()

        ax.plot(angles, data_1[stimulus], label='control', color = 'xkcd:greyish blue')
        ax.plot(angles, data_2[stimulus], label='sleep deprived', color = 'xkcd:aquamarine')
        
        plt.fill_between(angles, \
            data_1[stimulus]-sem_1[stimulus], \
            data_1[stimulus]+sem_1[stimulus], \
            color='gray', alpha=0.5)

        plt.fill_between(angles, \
            data_2[stimulus]-sem_2[stimulus], \
            data_2[stimulus]+sem_2[stimulus], \
            color='gray', alpha=0.5)
        
        ax.set_xlabel(f'$\\Delta$ Angle (°)')
        if prob == 1:
            ax.set_ylabel(f'Probability')
        else:
            ax.set_ylabel(f'Frequency (mHz)')
        ax.set_title(f'{id_map[str(stimulus)][0]} Stimulus {id_map[str(stimulus)][1]} %, (Bin: 5$^\circ$)')
        ax.legend()
        #sns.color_palette("PuRd", as_cmap=True)
        ax.grid(False)
        sns.set_style('white')
        sns.set_style('ticks')
        sns.despine(top=True, right=True)

        f.savefig(save_dir / f'fig_{stimulus}_{id_map[str(stimulus)][0]}_{id_map[str(stimulus)][1]}_grey.pdf')
        plt.close(f)

    if stimuli % 2 == 0:

        half = stimuli // 2
        
        data_1 = (np.fliplr(data_1[:half]) + data_1[half:]) / 2.0
        data_2 = (np.fliplr(data_2[:half]) + data_2[half:]) / 2.0

        sem_1 = np.sqrt((np.fliplr(sem_1[:half])**2 + sem_1[half:]**2) / 2.0)
        sem_2 = np.sqrt((np.fliplr(sem_2[:half])**2 + sem_2[half:]**2) / 2.0)

    else:

        half = (stimuli - 1) // 2
        
        data_1 = (np.fliplr(data_1[:half]) + data_1[half:-1]) / 2.0
        data_2 = (np.fliplr(data_2[:half]) + data_2[half:-1]) / 2.0

        sem_1 = np.sqrt((np.fliplr(sem_1[:half])**2 + sem_1[half:-1]**2) / 2.0)
        sem_2 = np.sqrt((np.fliplr(sem_2[:half])**2 + sem_2[half:-1]**2) / 2.0)

    for stimulus in range(half):

        f, ax = plt.subplots()
        
        ax.plot(angles, data_1[stimulus], label='control', color = 'xkcd:greyish blue')
        ax.plot(angles, data_2[stimulus], label='sleep deprived', color = 'xkcd:aquamarine' )
        
        plt.fill_between(angles, \
            data_1[stimulus]-sem_1[stimulus], \
            data_1[stimulus]+sem_1[stimulus], \
            color='gray', alpha=0.5)

        plt.fill_between(angles, \
            data_2[stimulus]-sem_2[stimulus], \
            data_2[stimulus]+sem_2[stimulus], \
            color='gray', alpha=0.5)
        
        ax.set_xlabel(f'$\\Delta$ Angle (°)')
        if prob == 1:
            ax.set_ylabel(f'Probability')
        else:
            ax.set_ylabel(f'Frequency (mHz)')
        ax.set_title(f'{id_map[str(half+stimulus)][0]} Stimulus {id_map[str(half+stimulus)][1]} %, (Bin: 5$^\circ$)')
        ax.legend()
        ax.grid(False)
        sns.set_style('white')
        sns.set_style('ticks')
        sns.despine(top=True, right=True)


        f.savefig(save_dir_db / f'fig_{id_map[str(half+stimulus)][0]}_{id_map[str(half+stimulus)][1]}_grey.pdf')
        plt.close(f)

    return 0

def boutFrequency(experiment, num_bins):

    data_path = path.Path() / '..' / experiment / f'data_{num_bins}'
    tmp = hdf.loadmat(data_path)

    id_map = hdf.loadmat(path.Path() / '..' / experiment / 'ID_map.mat')

    save_dir = path.Path() / '..' / experiment

    freq_1 = tmp['freq_1']
    freq_2 = tmp['freq_2']
    sem_1 = tmp['freq_sem_1']
    sem_2 = tmp['freq_sem_2']
    raw_1 = tmp['freq_1_raw']
    raw_2 = tmp['freq_2_raw']
    
    x_range = range(freq_1.shape[0])

    sns.set_style('white')
    sns.set_style('ticks')

    f, ax = plt.subplots()

    ax.bar([e + 1. for e in list(x_range)], freq_1, yerr=sem_1, capsize=5.0, label='control', alpha=0.5, width=0.4, color = 'xkcd:greyish blue')
    ax.bar([e + 1.4 for e in list(x_range)], freq_2, yerr=sem_2, ecolor='grey', capsize=3.0, alpha=0.7, label='sleep deprived', width=0.4, color = 'xkcd:aquamarine')
    
    for i in x_range:
        x_1 = [i+1] * raw_1.shape[0]
        x_2 = [i+1.4] * raw_2.shape[0]

        ax.scatter(x_1, raw_1[:,i], color = 'grey')
        ax.scatter(x_2, raw_2[:,i], color = 'grey')
    
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Total bout rate')
    ax.set_title('Total response to stimulus')
    ax.set_xticks(x_range)
    text = [str(x) for x in x_range]
    ax.set_xticklabels(text)
    ax.legend()
    ax.grid(False)
    
    sns.despine(top=True, right=True)


    f.savefig(save_dir / f'fig_total_response_grey.pdf')
    plt.close(f)

    stimuli = freq_1.shape[0]
    
    if stimuli % 2 == 0:

        half = stimuli // 2

        raw_1 = (raw_1[:,:half] + raw_1[:,half:]) / 2.0
        raw_2 = (raw_2[:,:half] + raw_2[:,half:]) / 2.0
        '''
        freq_1 = (freq_1[:half] + freq_1[half:]) / 2.0
        freq_2 = (freq_2[:half] + freq_2[half:]) / 2.0
        
        sem_1 = np.sqrt((sem_1[:half]**2 + sem_1[half:]**2) / 2.0)
        sem_2 = np.sqrt((sem_2[:half]**2 + sem_2[half:]**2) / 2.0)
        '''
        
    else:

        half = (stimuli - 1) // 2

        raw_1 = (raw_1[:,:half] + raw_1[:,half:-1]) / 2.0
        raw_2 = (raw_2[:,:half] + raw_2[:,half:-1]) / 2.0
        '''
        freq_1 = (freq_1[:half] + freq_1[half:-1]) / 2.0
        freq_2 = (freq_2[:half] + freq_2[half:-1]) / 2.0
        
        sem_1 = np.sqrt((sem_1[:half]**2 + sem_1[half:-1]**2) / 2.0)
        sem_2 = np.sqrt((sem_2[:half]**2 + sem_2[half:-1]**2) / 2.0)
        '''

    freq_1 = np.nanmean(raw_1, axis=0)
    sem_1 = sem(raw_1, axis=0, nan_policy='omit')
    freq_2 = np.nanmean(raw_2, axis=0)
    sem_2 = sem(raw_2, axis=0, nan_policy='omit')

    x_range = range(half)
    
    f, ax = plt.subplots()

    ax.bar([e + 1. for e in list(x_range)], freq_1, yerr=sem_1, capsize=5.0, label='control', alpha=0.5, width=0.4, color = 'xkcd:greyish blue')
    ax.bar([e + 1.4 for e in list(x_range)], freq_2, yerr=sem_2, ecolor='grey', capsize=3.0, alpha=0.7, label='sleep deprived', width=0.4, color = 'xkcd:aquamarine')

    for i in x_range:
        x_1 = [i+1] * raw_1.shape[0]
        x_2 = [i+1.4] * raw_2.shape[0]

        ax.scatter(x_1, raw_1[:,i], color = 'grey')
        ax.scatter(x_2, raw_2[:,i], color = 'grey')
    
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Total bout rate')
    ax.set_title('Total response to stimulus')
    ax.set_xticks(x_range)
    text = [str(x) for x in x_range]
    ax.set_xticklabels(text)
    ax.legend()
    ax.grid(False)
    sns.set_style('white')
    sns.set_style('ticks')
    sns.despine(top=True, right=True)


    f.savefig(save_dir / f'fig_total_response_grey_doubled.pdf')
    plt.close(f)

if __name__ == '__main__':

    experiment = 'd7_07_01_2021'
    num_bins = 72

    main(experiment, num_bins)
    plotHistogram(experiment, num_bins)
    plotHistogram(experiment, num_bins, True)
    boutFrequency(experiment, num_bins)

    sys.exit(0)
