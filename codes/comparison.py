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
import hdf5storage as hdf
import matplotlib.pyplot as plt
import seaborn as sns
import path

def compare(expt_1, expt_2, num_bins, prob=False):
    
    tmp1 = hdf.loadmat(path.Path() / '..' / expt_1 / f'data_{num_bins}.mat')
    tmp2 = hdf.loadmat(path.Path() / '..' / expt_2 / f'data_{num_bins}.mat')
    id_map = hdf.loadmat(path.Path() / '..' / expt_1 / 'ID_map.mat')

    stimuli = 8

    g1 = 'control'; g2 = 'sleep'
    e1 = 'd7'; e2 = 'd8'

    if prob:

        g1_e1 = tmp1['prob_1']; s1_e1 = tmp1['prob_sem_1']
        g2_e1 = tmp1['prob_2']; s2_e1 = tmp1['prob_sem_2']
        g1_e2 = tmp2['prob_1']; s1_e2 = tmp2['prob_sem_1']
        g2_e2 = tmp2['prob_2']; s2_e2 = tmp2['prob_sem_2']

        save_dir = path.Path() / '..' / expt_2 / 'comparison_prob'

    else:
        
        g1_e1 = tmp1['mean_1']; s1_e1 = tmp1['sem_1']
        g2_e1 = tmp1['mean_2']; s2_e1 = tmp1['sem_2']
        g1_e2 = tmp2['mean_1']; s1_e2 = tmp2['sem_1']
        g2_e2 = tmp2['mean_2']; s2_e2 = tmp2['sem_2']

        save_dir = path.Path() / '..' / expt_2 / 'comparison'

    os.makedirs(save_dir, exist_ok=True)
    angles = np.linspace(-180,180, num_bins)

    sns.set()
    
    for stimulus in range(stimuli):
        f, ax = plt.subplots()
        g, ax2 = plt.subplots()

        ax.plot(angles, g1_e1[stimulus], label=e1, color = 'xkcd:greyish blue')
        ax.plot(angles, g1_e2[stimulus], label=e2, color = 'xkcd:aquamarine')

        ax2.plot(angles, g2_e1[stimulus], label=e1, color = 'xkcd:greyish blue')
        ax2.plot(angles, g2_e2[stimulus], label=e2, color = 'xkcd:aquamarine')

        ax.fill_between(angles, \
            g1_e1[stimulus]-s1_e1[stimulus], \
            g1_e1[stimulus]+s1_e1[stimulus], \
            color='gray', alpha=0.5)

        ax.fill_between(angles, \
            g1_e2[stimulus]-s1_e2[stimulus], \
            g1_e2[stimulus]+s1_e2[stimulus], \
            color='gray', alpha=0.5)

        ax2.fill_between(angles, \
            g2_e1[stimulus]-s2_e1[stimulus], \
            g2_e1[stimulus]+s2_e1[stimulus], \
            color='gray', alpha=0.5)

        ax2.fill_between(angles, \
            g2_e2[stimulus]-s2_e2[stimulus], \
            g2_e2[stimulus]+s2_e2[stimulus], \
            color='gray', alpha=0.5)

        ax.set_xlabel(f'$\\theta$'); ax2.set_xlabel(f'$\\Delta$ Angel (°)')
        ax.set_ylabel(f'Frequency (mHz)'); ax2.set_ylabel(f'Frequency (mHz)')
        ax.set_title(f'Comparison of group {g1} for {e1} and {e2}')
        ax2.set_title(f'Comparison of group {g2} for {e1} and {e2}')
        ax.legend(); ax2.legend()
        ax.grid(False)
        sns.set_style('white')
        sns.set_style('ticks')
        sns.despine(top=True, right=True)

        f.savefig(save_dir / f'fig_{g1}_{stimulus}_{id_map[str(stimulus)][0]}_{id_map[str(stimulus)][1]}.pdf')
        g.savefig(save_dir / f'fig_{g2}_{stimulus}_{id_map[str(stimulus)][0]}_{id_map[str(stimulus)][1]}.pdf')

        plt.close(f); plt.close(g)

    # Frequencies for each group/experiment combination
    if not prob:
        f1_e1 = tmp1['freq_1']; s1_e1 = tmp1['freq_sem_1']
        f2_e1 = tmp1['freq_2']; s2_e1 = tmp1['freq_sem_2']
        f1_e2 = tmp2['freq_1']; s1_e2 = tmp1['freq_sem_1']
        f2_e2 = tmp2['freq_2']; s2_e2 = tmp1['freq_sem_2']

        x_range = range(f1_e1.shape[0])
        
        f, ax = plt.subplots()
        g, ax2 = plt.subplots()

        ax.bar(x_range, f1_e1, yerr=s1_e1, capsize=5.0, width=0.4, label=e1,color= 'xkcd:greyish blue')
        ax.bar(x_range, f1_e2, yerr=s1_e2, capsize=3.0, width=0.4, alpha =0.7, label=e2,color='xkcd:greyish blue')
        ax2.bar(x_range, f2_e1, yerr=s2_e1, capsize=5.0, width=0.4,, label=e1,color='xkcd:greyish blue')
        ax2.bar(x_range, f2_e2, yerr=s2_e2, capsize=3.0, width=0.4, alpha=0.7, label=e2,color='xkcd:greyish blue')

        ax.set_xlabel('Stimulus'); ax2.set_xlabel('Stimulus')
        ax.set_ylabel('Count'); ax2.set_ylabel('Count')
        ax.set_title(f'{g1} - Total response to stimulus')
        ax2.set_title(f'{g2} - Total response to stimulus')
        ax.set_xticks(x_range); ax2.set_xticks(x_range)
        text = [str(x) for x in x_range]
        ax.set_xticklabels(text); ax2.set_xticklabels(text)
        ax.legend(); ax2.legend()
        ax.grid(False)
        sns.set_style('white')
        sns.set_style('ticks')
        sns.despine(top=True, right=True)


        f.savefig(save_dir / f'fig_{g1}_total_response.pdf')
        g.savefig(save_dir / f'fig_{g2}_total_response.pdf')

        plt.close(f); plt.close(g)
    
    return 0

if __name__ == '__main__':
    
    expt_1 = 'd7_07_01_2021'
    expt_2 = 'd8_07_02_2021'
    
    num_bins = 72

    compare(expt_1, expt_2, num_bins, False)
    compare(expt_1, expt_2, num_bins, True)
    
    sys.exit()
