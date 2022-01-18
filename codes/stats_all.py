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
import csv

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

    bout_loc = df_data.loc[df_data['Stimulus'] =='0']
    sleep = bout_loc.loc[bout_loc['Group']== 2]
    control = bout_loc.loc[bout_loc['Group']==1]
    stats_bout = ss.ttest_ind(sleep.Frequency, control.Frequency, equal_var=False)

    sns.boxplot(x='Stimulus', y='Frequency', hue='Group', data=df_data, palette='Set3')

    results = pg.rm_anova(dv='Frequency', within=['Stimulus','Group'], subject='ID', data=df_data, detailed=True)
    save_dir = path.Path() / '..' / experiment
    doc_name = save_dir / 'bouts_stats.xlsx'
    results.to_excel(doc_name, index=False)
    
    return 0

def correctness(dpath, stimuli):

    tmp = hdf.loadmat(dpath / 'data_correctness_72.mat')
    data_1 = tmp['correct_1_raw']
    data_2 = tmp['correct_2_raw']

    df_data = makeDf(data_1, data_2, stimuli, 'Correctness')
    save_dir = path.Path() / '..' / experiment

    sns.boxplot(x='Stimulus', y='Correctness', hue='Group', data=df_data, palette='Set3')

    results = pg.rm_anova(dv='Correctness', within=['Stimulus','Group'], subject='ID', data=df_data, detailed=True)

    doc_name = save_dir / 'correct_stats.xlsx'
    results.to_excel(doc_name, index=False)

    return 0

def rate24(dpath):

    tmp = hdf.loadmat(dpath / 'data_rate.mat')
    data_1 = tmp['freq_1_raw']
    data_2 = tmp['freq_2_raw']

    g1_d1 = data_1[:,0:14*30].mean(axis=1) # test significance between 3 parts
    g1_night = data_1[:,14*30:34*30].mean(axis=1) # These have dimension fish x time series
    g1_d2 = data_1[:,35*30:48*30].mean(axis=1) # 2 dimension because trials are concatenated

    g2_d1 = data_2[:,0:14*30].mean(axis=1) # test significance between 3 parts
    g2_night = data_2[:,14*30:34*30].mean(axis=1) # These have dimension fish x time series
    g2_d2 = data_2[:,35*30:48*30].mean(axis=1) # 2 dimension because trials are concatenated

    stat_d1 = ss.ttest_ind(g1_d1, g2_d1, equal_var=False)
    stat_night = ss.ttest_ind(g1_night, g2_night, equal_var=False)
    stat_d2 = ss.ttest_ind(g1_d2, g2_d2, equal_var=False)

    print(stats_d1); print(stats_d1); print(stats_d1)
    # How do we want to save this? Excel? Will look into it and add
    
    return 0

def performance(dpath, stimuli):

    tmp = hdf.loadmat(dpath / 'data_72.mat')
    data_1 = tmp['mean_1_raw']
    data_2 = tmp['mean_2_raw']

    angles = np.linspace(-180,180,72) # Num bins has not changed in years

    sum_1 = data_1.sum(axis=2)
    sum_1 = np.repeat(sum_1[:,:,np.newaxis], data_1.shape[2], axis=2)
    sum_2 = data_2.sum(axis=2)
    sum_2 = np.repeat(sum_2[:,:,np.newaxis], data_2.shape[2], axis=2)

    prob_1 = data_1 / sum_1
    prob_2 = data_2 / sum_2

    score_1 = np.sum(prob_1 * angles, axis=2)
    score_2 = np.sum(prob_2 * angles, axis=2)

    if stimuli % 2 == 0:

        half = stimuli // 2

        score_1 = (score_1[:,:half] + 1. - score_1[:,half:]) / 2.0
        score_2 = (score_2[:,:half] + 1. -  score_2[:,half:]) / 2.0
        
    else:

        half = (stimuli - 1) // 2

        score_1 = (score_1[:,:half] + 1. - score_1[:,half:-1]) / 2.0
        score_2 = (score_2[:,:half] + 1. - score_2[:,half:-1]) / 2.0

    score_avg_1 = np.mean(score_1, axis=0)
    score_avg_2 = np.mean(score_2, axis=0)

    score_sem_1 = sem(score_1, axis=0, nan_policy='omit')
    score_sem_2 = sem(score_2, axis=0, nan_policy='omit')

    x_range = range(half)
    
    f, ax = plt.subplots()

    ax.bar([e + 1. for e in list(x_range)], score_avg_1, yerr=score_sem_1, capsize=5.0, label='control', alpha=0.5, width=0.4, color = 'xkcd:greyish blue')
    ax.bar([e + 1.4 for e in list(x_range)], score_avg_2, yerr=score_sem_2, ecolor='grey', capsize=3.0, alpha=0.7, label='sleep deprived', width=0.4, color = 'xkcd:aquamarine')

    for i in x_range:
        x_1 = [i+1] * score_1.shape[0]
        x_2 = [i+1.4] * score_2.shape[0]

        ax.scatter(x_1, score_1[:,i], color = 'grey')
        ax.scatter(x_2, score_2[:,i], color = 'grey')
    
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Performance index')
    ax.set_title('Performance across stimuli')
    ax.set_xticks(x_range)
    text = [str(x) for x in x_range]
    ax.set_xticklabels(text)
    ax.legend()
    ax.grid(False)
    sns.set_style('white')
    sns.set_style('ticks')
    sns.despine(top=True, right=True)


    f.savefig(dpath / f'fig_performance_index.pdf')
    plt.close(f)

    return 0

if __name__ == '__main__':

    experiment = 'd8_07_08_2021'
    stimuli = 8

    dpath = path.Path() / '..' / experiment

    totalBout(dpath, stimuli)
    correctness(dpath, stimuli)
    #performance(dpath, stimuli)

    sys.exit()
