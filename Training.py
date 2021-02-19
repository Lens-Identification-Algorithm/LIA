#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:23:08 2020

@author: marlen
"""

from LIA import training_set
from astropy.io.votable import parse_single_table
import time
import json

start_time = time.time()

#table = parse_single_table('SPP_LCs_Len_r_Cut_150.xml')
#table = parse_single_table('SPP_LCs_Len_r_Cut_100.xml')
#table = parse_single_table('Gaia_LCs_35.xml')



mira_table = parse_single_table('Miras_vo.xml')
# /home/marlen/SPP_r_100_01_noise.json
#data_Gaia40_diff01_pairs.json
# data_Gaia60_pairs.json
#data_Gaia80_diff01_pairs
#
#SPP_g_100_pairs_noise.json  data_Gaia40_diff01_pairs SPP_r_40_01_noise
with open('/home/marlen/SPP_r_100_pairs_noise.json') as json_file:
    ztf_table = json.load(json_file)


training_set.create(mira_table,ztf_table,noise='ZTF',PL_criteria = 'Weak', Binary_events = 'Yes',BL_criteria='Strong',BL_classes='No', n_class=1000)

print("--- %s seconds ---" % (time.time() - start_time))




   