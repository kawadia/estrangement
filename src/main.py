#!/usr/bin/env python 

import networkx as nx
import random
import sys
import math
import os
import configparse
import datetime
import inspect
import logging
import cProfile
import pstats
import pdb
#local modules
import estrangement
import agglomerate
import optionsadder


def parse_args(reader_functions={}):
    """@brief parse cmd line and conf file options 
    @retval (opt, args) as returned by configparse.OptionParser.parse_args"""
    # read in options from cmdline and conffile
    usage="""usage: %prog [options] (--help for help)\n"""

    parser = configparse.OptionParser(description="Estrangement Confinement Algorithm",
         usage=usage)


    optionsadder.add_options(parser, reader_functions=reader_functions)
    (opt, args) = parser.parse_args(files=['./simulation.conf'])
    #check_options(opt, parser)
    return (opt, args)



def summary(g1):
    logging.info(nx.info(g1))
    logging.info("Number of components: %d " % nx.number_connected_components(g1))


def maxQ(g1):
    """ a convenenience function to do plain Q maximization"""
    dictPartition = {}
    dictQ = {}
    # set params to do pure Q maximization
    lambduh = 0.0
    Zgraph = nx.Graph()
    # do 100 runs and pick the best
    for r in range(10*opt.minrepeats):
        # best_partition calls agglomerative lpa
        dictPartition[r] = agglomerate.best_partition(g1, opt, lambduh, Zgraph)
        dictQ[r] = agglomerate.modularity(dictPartition[r], g1)
    logging.info("dictQ = %s", str(dictQ))
    bestr = max(dictQ, key=dictQ.get)
    return dictPartition[bestr]



def read_general(datadir):
    """ generator function to read many datasets including mit and random_with_stable_core"""
    # passed in datadir can be one of these
    repo_datadir = "../../../../datasets/"
    #datadir = "random_with_stable_core"
    datadir = os.path.join(repo_datadir, datadir)
    print "datadir: ", datadir
    
    if os.path.exists(os.path.join(datadir, "network.merged")):
        g1 = nx.read_edgelist(os.path.join(datadir, "network.merged"), nodetype=int, data=(('weight',float),))
        if not os.path.exists(os.path.join(datadir, "merged_label_dict")):
            merged_label_dict = maxQ(g1)
            with open(os.path.join(datadir, 'merged_label_dict.txt'), 'w') as lf:
                lf.write(repr(merged_label_dict))
    
    raw_file_list = os.listdir(datadir)
    timestamps = sorted([int(f.rstrip(".ncol")) for f in raw_file_list if f.endswith(".ncol")])

    initial_label_dict_filename = os.path.join(datadir, 'initial_label_dict.txt')

    beginning = True
    for t in timestamps:
        f = str(t) + ".ncol"
        
        fpath = os.path.join(datadir,f)

        # skip empty files but increment timestamp
        if os.path.getsize(fpath) == 0:
            continue
        
        g1 = nx.read_edgelist(fpath, nodetype=int, data=(('weight',float),))

        summary(g1)

        if beginning is True:
            # when called for the first time just return initial_label_dict
            if not os.path.exists(initial_label_dict_filename):
                initial_label_dict = maxQ(g1)
                with open(initial_label_dict_filename, 'w') as lf:
                    lf.write(repr(initial_label_dict))
            
            with open(initial_label_dict_filename, 'r') as lf:
                initial_label_dict = eval(lf.read())
            yield (t, g1, initial_label_dict)
            beginning = False
        else:
            yield (t, g1, None)



if __name__ == '__main__':

    # get names of all functions in this module
    all_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    reader_functions = dict([ t for t in all_functions if t[0].startswith("read") ])

    (opt, args) = parse_args(reader_functions)
    print(opt)

    # Write all option settings to the log file
    #opt_dict = eval(str(opt))
    with open("options.log", 'w') as optf:
       optf.write(str(opt))
    
    logging.basicConfig(level=getattr(logging, opt.loglevel.upper(), None))


    if opt.profiler_on is True:
        # run main through the profiler
        cProfile.run('estrangement.ERA(reader_functions[opt.graph_reader_fn], opt)',
            "simulation.prof")
        # print 40 most expensive fuctions sorted by tot_time, after we are done
        st = pstats.Stats("simulation.prof")
        st.sort_stats('time')
        st.print_stats(40)
    else:
        # run without profiler
        estrangement.ERA(reader_functions[opt.graph_reader_fn], opt)
 
    print "#################### DONE ##############################"
