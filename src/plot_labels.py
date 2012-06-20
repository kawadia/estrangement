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



def COW_country_codes():
    """ read COW country codes """
    cow_dict = {}
    #codefilename = "/home/vkawadia/datasets/COW_IGO/COWStatelist.csv"
    codefilename = "../../../../datasets/COW_IGO/COWStatelist.csv"
    with open(codefilename, 'rU') as f:
        reader = csv.reader(f)
        #skip header
        reader.next()
        for row in reader:
            cow_dict[int(row[1])] = row[2]

    return cow_dict


def rca_country_codes():
    """ read rca country codes """
    cow_dict = {}
    #codefilename = "/home/vkawadia/datasets/COW_IGO/COWStatelist.csv"
    codefilename = "../../../../../datasets/rca_country_list.txt"
    with open(codefilename, 'rU') as f:
        for line in f:
            words = line.split()
            cow_dict[int(words[0])] = words[1]

    return cow_dict


def create_node_index_dict():
    # assign indices to nodes to try to cluster communities together
    node_index_dict  = {}
    # also assign indices to labels as they may be tuples in the case of
    # overlapping communities derived from linegraphs
    label_index_dict  = {} # key = label, val = the index to be used for plotting
    index = 0

    # key = node, val = list of labels the node gets over time
    label_time_series_dict = collections.defaultdict(list)

    # track all labels seen to that we can assign them indices
    all_labels_set = set([])

    label_file = open("simplified_matched_labels.log", 'r')
    for l in label_file:
        # reading dict with key=time t and val = label_dict (key=node_name, val=label
        # at time t
        line_dict = eval(l)
        time = line_dict.keys()[0]
        label_dict = line_dict[time] # key= node, val = label at time t
          
        for n,l in label_dict.items():
            label_time_series_dict[n].append(l)

            
        if opt.node_indexing == "lexicographical":
            ### just a lexicographical order
            for n,l in sorted(label_dict.items()):
                if not node_index_dict.has_key(n):
                    node_index_dict[n] = index
                    index += 1

        for n,l in label_dict.items():
            all_labels_set.add(l)

    #if opt.node_indexing == "lexicographical":
    #    return node_index_dict

    label_count_dict = {} # key = node, val = tuple of labels, ordered by freq
    for n , label_list in label_time_series_dict.items():
        label_freq = collections.defaultdict(int) # key = label, val = freq
        for l in label_list:
            label_freq[l] += 1
        label_count_dict[n] = sorted(label_freq.keys(), key=label_freq.get, reverse=True)

    #print("label_count_dict : ",  str(label_count_dict))        
    
    ordered_nodes = sorted(label_count_dict.keys(), key=label_count_dict.get)



    #print("ordered_nodes : ",  str(ordered_nodes))        

    node_index_dict =  dict(zip(ordered_nodes, range(len(label_count_dict.keys()))))
    reverse_node_index_dict = dict(zip(range(len(label_count_dict.keys())),ordered_nodes))

    label_index_dict = dict(zip(sorted(all_labels_set), range(len(all_labels_set))))

    print("node_index_dict : ")        
    pprint.pprint(node_index_dict)        

    label_file.close()

    return node_index_dict, reverse_node_index_dict, label_index_dict


def plot_temporal_communities():
    fig1 = pylab.figure(figsize=(14,24))
    #ax1 = fig1.add_subplot(111, frameon=False)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Nodename")
    ax1.set_title("Dynamic communities (%s)" % os.path.basename(os.getcwd()).lstrip('task-'),
        fontsize=opt.fontsize)
    #ax1.set_title(opt.title, fontsize=opt.fontsize)
    

    pylab.hold(True)

    # assign indices to labels and nodename for plotting purposes

    label_file = open("simplified_matched_labels.log", 'r')
    
    x = numpy.array(summary_dict['snapshots_list'])
    y = numpy.array(sorted(node_index_dict.values()))
    Labels = numpy.empty((len(y), len(x)), int)
    Labels.fill(-1)

    
    # iterate over the lines in the logfile and plot data for each snapshot
    for l in label_file:
        # reading dict with key=time t and val = label_dict (key=node_name, val=label
        # at time t
        line_dict = eval(l)
        time = line_dict.keys()[0]
        label_dict = line_dict[time]
        #print "label_dict", label_dict
        #print "unique labels", numpy.unique1d(label_dict.values())

        for nodename in label_dict.keys():
            if opt.use_separate_label_indices is True:
                Labels[node_index_dict[nodename], t_index_dict[time]] = label_index_dict[label_dict[nodename]]
            else:    
                Labels[node_index_dict[nodename], t_index_dict[time]] = node_index_dict[label_dict[nodename]]

        numpy.set_printoptions(threshold=400)
        #print("time: %d, Labels: %s" %(time, str(Labels)))
    # mask the nodes not seenin this snapshot
    Labels_masked = numpy.ma.masked_equal(Labels, -1)

    #pylab.pcolor(Labels_masked, cmap=pylab.cm.get_cmap(opt.label_cmap, 250),
    pylab.pcolor(Labels_masked, cmap=pylab.cm.get_cmap(opt.label_cmap),
        alpha=opt.alpha, edgecolors='none')
    # @todo apply the right labels using set_xlabel and python unzip
    # specifying x and y leads to last row and col of Labels beting omitted from plotting
    #pylab.pcolor(x, y, Labels_masked, cmap=pylab.cm.get_cmap(opt.label_cmap), alpha=opt.alpha)
    levels = numpy.unique1d(Labels_masked)
    #print("Levels are: %s", levels)
    if opt.colorbar is True:
        if opt.label_with_country_names is True:
            levels_ccodes = [ COW_ccode_dict[reverse_node_index_dict[l]] for l in levels.compressed() ]
            cb = pylab.colorbar(ticks=levels)
            cb.ax.set_yticklabels(levels_ccodes)
        else:
            pylab.colorbar(ticks=levels)


    if opt.label_with_country_names is True:
        ylocs = sorted(node_index_dict.values(), key=int)
        ylabs = sorted(node_index_dict.keys(), key=lambda x:int(node_index_dict[x]))
        country_names = [COW_ccode_dict[c] for c in ylabs]
        pylab.yticks(ylocs, country_names, fontsize=11)

    xlocs = sorted(t_index_dict.values(), key=int)
    xlabs = sorted(t_index_dict.keys(), key=lambda x:int(t_index_dict[x]))
    pylab.xticks(xlocs, xlabs, fontsize=14, rotation=75)


    label_file.close()
    pylab.savefig('dyncon.%s'%opt.image_extension)
    if opt.display_on is True:
        pylab.show()


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


def plot_node_appearance_histogram():
    """ plot a histogram of the frequency of node appearances over all snapshots """
    fig2 = pyplot.figure(figsize=(opt.xfigsize,opt.yfigsize))
    ax2 = fig2.add_subplot(111)
    xlabel="Number of appearances"
    ylabel="Number of nodes"
    title="Histogram of number of appearances of a node"
    ax2.set_title(title, fontsize=opt.fontsize)
    ax2.set_xlabel(xlabel, fontsize=opt.fontsize)
    ax2.set_ylabel(ylabel, fontsize=opt.fontsize)
    
    xticklabels = pyplot.getp(pyplot.gca(), 'xticklabels')
    pyplot.setp(xticklabels, fontsize=opt.label_fontsize)

    yticklabels = pyplot.getp(pyplot.gca(), 'yticklabels')
    pyplot.setp(yticklabels, fontsize=opt.label_fontsize)
    pyplot.hold(True)

    with open("node_appearances.log", 'r') as freq_file:
        node_appearances_dict = dict(eval(freq_file.read()))

    pyplot.hist(sorted(node_appearances_dict.values()), bins=30)    
    pyplot.savefig('node_appearance_histogram.%s'%opt.image_extension)
    if opt.display_on is True:
        pylab.show()



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


def match_labels():
    """ read labels.log and create simplified_matched_labels.log using max
    weight matching"""
    
    prev_matched_label_dict = {}
    label_file = open("labels.log", 'r')
    matched_label_file = open("simplified_matched_labels.log", 'w')
    for l in label_file:
        # reading dict with key=time t and val = label_dict (key=node_name, val=label
        # at time t
        line_dict = eval(l)
        time = line_dict.keys()[0]
        label_dict = line_dict[time] # key= node, val = label at time t

        matched_label_dict = utils.simplified_match_labels(label_dict,
            prev_matched_label_dict)


        matched_label_file.write("{%d:%s}\n" % (time,str(matched_label_dict)))
        prev_matched_label_dict = matched_label_dict

    label_file.close()
    matched_label_file.close()


if __name__ == '__main__':

    (opt, args) = parse_args()

    print opt
    
    
    logging.basicConfig(level=getattr(logging, opt.loglevel.upper(), None))

    match_labels()

    summary_file = open("summary.log", 'r')
    summary_dict = eval(summary_file.read())
    print "summary_dict: ", summary_dict


    ## dict which assigns the nodes a number for plotting purposes
    # key is the node name and value is an integer
    node_index_dict, reverse_node_index_dict, label_index_dict = create_node_index_dict()

    t_index_dict = dict(zip(summary_dict['snapshots_list'],
        range(len(summary_dict['snapshots_list']))))

    ## dict which assigns the labels a color for plotting purposes
    # key is the node name and value is an integer
    #colorfile = open("colorfile.dat")
    #for l in colorfile:
    #    label_color_dict = eval(l)

    figsize = (opt.xfigsize, opt.yfigsize)
    #print "markersize, markerheight = ", markersize, markerheight

    verts = [
        (0., 0.), # left, bottom
        (0., opt.markerheight), # left, top
        (1., opt.markerheight), # right, top
        (1., 0.), # right, bottom
        (0., 0.), # ignored
    ]

    #COW_ccode_dict  = COW_country_codes()
    #COW_ccode_dict  = rca_country_codes()
    plot_temporal_communities()
    #plot_sigmas()
    #plot_node_appearance_histogram()
    #plot_with_lambdas()
    #layout()
