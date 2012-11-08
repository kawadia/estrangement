#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script demonstrating the use of the estrangement library to detect and
visualize temporal communities. 
"""

__author__ = """\n""".join(['Vikas Kawadia (vkawadia@bbn.com)',
                            'Sameet Sreenivasan <sreens@rpi.edu>',
                            'Stephen Dabideen <dabideen@bbn.com>'])

#   Copyright (C) 2012 by 
#   Vikas Kawadia <vkawadia@bbn.com>
#   Sameet Sreenivasan <sreens@rpi.edu>
#   Stephen Dabideen <dabideen@bbn.com>
#   All rights reserved. 


import sys
import os
from Estrangement import estrangement
from Estrangement import plots
from Estrangement import options_parser
import multiprocessing



def detect_and_plot_temporal_communities():
    """ Function to run simulations, based on a specified dataset, and created 
    tiled plots of the temporal communities. 
    
    Parameters can be specified at the command line, when calling this script.
    Alternatively, a config file specifed at the command line can be used to set
    the parameter. At the very minimum, a path to the data set must be specified.

    Each experiment requires a name, which is used to create a folder to store the
    results of the simulation. If the results already exist in the folder specified
    by the experiment name, plots are created using these existing results and the 
    simulation is not run on subsequent calls to EstrangementDemo.py. 
    To run the simulation again, delete the experiment folder before running this script,
    or use a different experiment name. 

    Examples
    --------
    >>> # To see all configuarable parameters use the -h option 
    >>> EstrangementDemo.py -h
    >>> # Configurable parameters can be specified at the command line
    >>> EstrangementDemo.py --dataset_dir ./data --display_on True --exp_name my_experiment
    >>> # A config file can be used, but it must be preceeded by an '@'
    >>> # Three config files are provided as examples, check that that path to the dataset is valid.
    >>> EstrangementDemo.py @senate.conf
    >>> EstrangementDemo.py @markovian.conf
    >>> EstrangementDemo.py @realitymining.conf 
    """

    # use argparse to parse command-line arguments using optionsadder.py
    opt = options_parser.parse_args()

    # A dir is created, specified by the --exp_name argument in 
    # the current working directory to place all output from the experiment
    if(not os.path.exists(opt.exp_name)):
        os.mkdir(opt.exp_name)
    else:
        print("""Output dir for exp %s already exists, will use partial results. \\
            Will redo visualization only, if results for all delta exist.\\
            Delete %s to redo everything""" % (opt.exp_name, opt.exp_name))

    # set the values of delta to find communities for
    deltas = opt.delta

    # dictionary to pass the simulation output to the plot function
    matched_labels_dict = {}
    
    if(not os.path.isdir(opt.dataset_dir)):
        sys.exit("ERROR: 'dataset_dir' %s invalid, please specify using --dataset_dir at the command line or in config file." 
            % opt.dataset_dir)

    # @todo run multiple processes in parallel, each for a different value of delta
    # multiprocessing  module is not very portable so leave it out from this # demo
    # q = multiprocessing.Queue()

    for d in deltas:
        results_file = os.path.join(opt.exp_name, "task_delta_" + str(d) , "matched_labels.log")
        if not os.path.exists(results_file):
            print("Detecting temporal communities for delta=%s"%d)
            result = estrangement.ECA(dataset_dir = opt.dataset_dir, delta = d,
                        minrepeats = opt.minrepeats, increpeats = opt.increpeats)
            # result is a dictionary of the form: {time : {node : label}}
            with open(results_file, 'w') as fw:
                fw.write(str(result))
        else:
            print("result file %s found, so using it and not running ECA for this delta value" % results_file)
            with open(results_file, 'r') as fr:
                result = eval(fr.read())
        # combine the results into a single dictionary for plotting
        matched_labels_dict[d] = result

    
    # plot the temporal communities 
    plots.plot_temporal_communities(matched_labels_dict)

    # to plot other parameters, set write_stats=True in estrangement.ECA() 
    # and use plots.plot_function(). For example,
    # estrangement.plots.plot_function(['Estrangement'])

if __name__ == "__main__":
    detect_and_plot_temporal_communities()




#p = multiprocessing.Process(target=estrangement.ECA,
#        args=(opt.dataset_dir,opt.tolerance,opt.convergence_tolerance,
#            d,opt.minrepeats,opt.increpeats,500,False,q))
