#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Sample script demonstrating the use of the Estrangement library to plot temporal communities. """

__author__ = """\n""".join(['Vikas Kawadia (vkawadia@bbn.com)',
                            'Sameet Sreenivasan <sreens@rpi.edu>',
                            'Stephen Dabideen <dabideen@bbn.com>'])

#   Copyright (C) 2012 by 
#   Vikas Kawadia <vkawadia@bbn.com>
#   Sameet Sreenivasan <sreens@rpi.edu>
#   Stephen Dabideen <dabideen@bbn.com>
#   All rights reserved. 
#   BSD license. 


import sys
import os
sys.path.append(os.getcwd() + "/..")
sys.path.append(os.getcwd() + "../..")
import estrangement_parser
import estrangement_plots
import estrangement

# use argparse to parse command-line arguments using optionsadder.py
opt = estrangement_parser.parse_args()

# set the values of delta for which to create plots
deltas = [0.01,0.025,0.05,1.0]

# dictionary to pass the simulation output to the plot function
matched_labels_dict = {}

# Prompt the user for a name of the experiment. A folder is created in 
# the current working directory and all the files from the experiment will
# be placed in this file. 
exp_name = raw_input("Please enter experiment name:")
if(not os.path.exists(exp_name)):
    os.mkdir(exp_name)
os.chdir(exp_name)

# Prompt the user for the path to the dataset
path_to_data=raw_input("Please enter path to data:")
if(not os.path.isdir(path_to_data)):
     sys.exit("data folder %s does not exist"%path_to_data)

for d in deltas:
	# check if the matched_labels.log file file exists, and prompt the user if it does 
        if(os.path.isfile("task_delta_" + str(d) + "/matched_labels.log")):
	    use_log = raw_input("Do you wish to use the existing log files for delta=%s? [Y/n]"%d)
	    if(use_log != 'n'):
		with open("task_delta_" + str(d) + "/matched_labels.log", 'r') as ml:
                	matched_labels = ml.read()
			matched_labels_dict[str(d)] = eval(matched_labels)
			continue

	# run the simulation if the matched_labels.log file does not exist or the user specifies this is desired
	matched_labels = estrangement.ERA(dataset_dir=path_to_data,delta=d,increpeats=opt.increpeats,minrepeats=opt.minrepeats)
        matched_labels_dict[str(d)] = matched_labels
	matched_label_file = open("task_delta_" + str(d) +"/matched_labels.log", 'w')
        matched_label_file.write(str(matched_labels))
	matched_label_file.close()

# plot the temporal communities 
estrangement_plots.plot_temporal_communities(matched_labels_dict)
os.chdir("..")


# to plot other parameters, set write_stats=True in estrangement.ERA() 
# and use estrangement_plots.plot_function(). For example,
# estrangement_plots.plot_function(['Estrangement'])
# estrangement_plots.plot_function(['ierr','feasible'])

