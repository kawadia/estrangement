#!/usr/bin/env python

import matplotlib
#matplotlib.use('SVG')
matplotlib.use('Agg')
from matplotlib import pyplot
import networkx as nx
import pylab
import sys
import configparse
import os
import random
import numpy
import color_list
import collections
import itertools
import logging
import pprint
import csv
import utils
import visualoptions
import postpro
import math


def parse_args():
    """@brief parse cmd line and conf file options 
    @retval (opt, args) as returned by configparse.OptionParser.parse_args"""
    # read in options from cmdline and conffile
    usage="""usage: %prog [options] (--help for help)\n"""

    parser = configparse.OptionParser(description="Analysis and visualization of Impression Propagation Algorithm",
         usage=usage)
    visualoptions.add_options(parser)
    (opt, args) = parser.parse_args(files=['./analysis.conf'])
    #check_options(opt, parser)
    return (opt, args)



def layout():
    """ produce layouts from .gexfs saved while simulation"""

    for t in summary_dict['snapshots_list']:
        filename = "%s.gexf"%str(t)
        print "reading " , filename
        with open(filename, 'r') as f:
            glay = nx.read_gexf(f)
        nodecolors = [ndata[1]['comlabel'] for ndata in glay.nodes(data=True)]    
        edgecolors = []
        for e in glay.edges(data=True):    
            if e[2]['estranged'] is 1:
                edgecolors.append('r')
            else:    
                edgecolors.append('b')
        pyplot.figure(figsize=(8,8))    
        nx.draw_networkx(glay, node_color=nodecolors, edge_color=edgecolors,)
        #    cmap=pylab.cm.get_cmap(opt.label_cmap))
        pyplot.axis('off')
        pyplot.savefig('layout_t%s.%s'%(str(t),opt.image_extension))


def confidence_interval(nums):
    """Return (half) the 95% confidence interval around the mean for nums:
    1.96 * std_deviation / sqrt(len(nums)).
    """
    return 1.96 * numpy.std(nums) / math.sqrt(len(nums))


def plot_with_lambdas():
    """ plot F with lambdas for various snapshots """

    with open("Fdetails.log", 'r') as Fdetails_file:
        Fdetails_dict = eval(Fdetails_file.read())

    # Fdetails_dict is {time: {lambda: {run_number: F}}}

    with open("Qdetails.log", 'r') as Qdetails_file:
        Qdetails_dict = eval(Qdetails_file.read())


    with open("Edetails.log", 'r') as Edetails_file:
        Edetails_dict = eval(Edetails_file.read())

    
    with open("lambdaopt.log", 'r') as f:
      lambdaopt_dict = eval(f.read())  # {time: lambdaopt}

    with open("best_feasible_lambda.log", 'r') as f:
      best_feasible_lambda_dict = eval(f.read())  # {time: best_feasible_lambda}

    with open("Q.log", 'r') as f:
      Q_dict = eval(f.read())  # {time: lambdaopt}

    with open("F.log", 'r') as f:
      F_dict = eval(f.read())  # {time: lambdaopt}

    for t in sorted(Fdetails_dict.keys()):
        Flam = Fdetails_dict[t]
        Qlam = Qdetails_dict[t]
        Elam = Edetails_dict[t]

        dictX = collections.defaultdict(list)
        dictY = collections.defaultdict(list)
        dictErr = collections.defaultdict(list)
        for l in sorted(Flam.keys()):
            
            dictX['Q'].append(l)
            dictY['Q'].append(max(Qlam[l].values()))
            dictErr['Q'].append( confidence_interval(Qlam[l].values()) )
            
            #dictX['E'].append(l)
            #dictY['E'].append(max(Elam[l].values()))
            #dictErr['E'].append( confidence_interval(Elam[l].values()) )
            
            dictX['F'].append(l)
            dictY['F'].append(max(Flam[l].values()))
            dictErr['F'].append( confidence_interval(Flam[l].values()) )

        ax2 = postpro.plot_by_param(dictX, dictY, opt,
            listLinestyles=['b-', 'g-', 'r-',], 
            xlabel="$\lambda$", ylabel="Dual function", title="Dual function at t=%s"%(str(t)), 
            dictErr=dictErr)

        ax2.axvline(x=lambdaopt_dict[t], color='m', linewidth=opt.linewidth,
            linestyle='--', label="$\lambda_{opt}$")
        
        ax2.axvline(x=best_feasible_lambda_dict[t], color='k', linewidth=opt.linewidth,
            linestyle='--', label="best feasible $\lambda$")

        
        ax2.axhline(F_dict[t], color='b', linewidth=opt.linewidth,
            linestyle='--', label="best feasible F")

        ax2.axhline(Q_dict[t], color='g', linewidth=opt.linewidth,
            linestyle='--', label="best feasible Q")

        pyplot.legend()

        pyplot.savefig('with_lambda_at_t%s.%s'%(str(t), opt.image_extension))



if __name__ == '__main__':

    (opt, args) = parse_args()

    print opt
    
    logging.basicConfig(level=getattr(logging, opt.loglevel.upper(), None))

    summary_file = open("summary.log", 'r')
    summary_dict = eval(summary_file.read())
    print "summary_dict: ", summary_dict

    #plot_with_lambdas()
    #layout()
