# -*- coding: utf-8 -*-
"""
This code is for seizure detection task on TUSZ dataset. Version is 1.5.2.
https://isip.piconepress.com/projects/tuh_eeg/ 

@author: Yuanda Zhu, PhD student at Georgia Institute of Technology, Atlanta, GA, USA
Year: 2023
"""

import os
from data_reader_new import *

# Seizure types
seizure_types = ['fnsz', 'gnsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'bckg']

data_root = os.path.join('/datadrive', 'TUSZ', 'edf', 'train', '01_tcp_ar')
session_paths, all_patients = get_all_TUH_EEG_session_paths(data_root)

seizure_duration = [0 for i in range(len(seizure_types))]
seizure_sessions = [0 for i in range(len(seizure_types))]

for data_path in session_paths:
    
    ### Read tse labels
    labels = []
    tseFile = data_path[:-4] + '.tse'
    with open(tseFile,'r') as tseReader:
        rawText = tseReader.readlines()[2:]
        seizPeriods = []
        for item in rawText:
            labels.append([int(item.split()[0].split('.')[0]),int(item.split()[1].split('.')[0]),item.split()[2]])
            
    for label in labels:
        if label[2] in seizure_types:            
            index = seizure_types.index(label[2])
            seizure_duration[index] = seizure_duration[index] + label[1] - label[0]
            seizure_sessions[index] += 1
            
    # break

print('Seizure types are: ', seizure_types)
print('Seizure durations are: ', seizure_duration)
print('Seizure sessions are: ', seizure_sessions)