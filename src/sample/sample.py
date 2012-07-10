#!/usr/bin/python

import sys
import os
sys.path.append(os.getcwd() + "/..")
sys.path.append(os.getcwd() + "../..")
import utils
import agglomerate
import optionsadder
import postpro
import lpa
import string
import estrangement


opt = optionsadder.parse_args()
deltas = [0.01,0.025,0.05,1.0]

matched_labels_dict = {}

exp_name = raw_input("Please enter experiment name:")
if(not os.path.exists(exp_name)):
    os.mkdir(exp_name)
os.chdir(exp_name)

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
#			print(matched_labels)
			continue
	# run the simulation if the matched_labels.log file does not exist or the user specifies this is desired
	matched_labels = estrangement.ERA(dataset_dir=path_to_data,delta=d,increpeats=opt.increpeats,minrepeats=opt.minrepeats)
        matched_labels_dict[str(d)] = matched_labels
	matched_label_file = open("task_delta_" + str(d) +"/matched_labels.log", 'w')
        matched_label_file.write(str(matched_labels))
#	print(str(matched_labels))
	matched_label_file.close()

 
postpro.plot_temporal_communities(matched_labels_dict)
os.chdir("..")



