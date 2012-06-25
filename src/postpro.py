#!/usr/bin/env python

import matplotlib
#matplotlib.use('SVG')
matplotlib.use('WXAgg')
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
import pylab
import sys
import configparse
import os
import numpy
import color_list
import collections
import itertools
import random
import operator
import cProfile
import pstats
import logging
import numpy
import pprint
import csv
import math
import re

import utils
import visualoptions


markers = [
  'o'	,
  'v'	,
  '^'	,
  '<'	,
  '>'	,
  '1'	,
  '2'	,
  '3'	,
  '4'	,
  's'	,
  'p'	,
  '*'	,
  'h'	,
  'H'	,
  '+'	,
  'x'	,
  'D'	,
  'd'	,
  '|'	,
  '_'	,
]



def parse_args():
    """@brief parse cmd line and conf file options 
    @retval (opt, args) as returned by configparse.OptionParser.parse_args"""
    # read in options from cmdline and conffile
    usage="""usage: %prog [options] (--help for help)\n"""

    parser = configparse.OptionParser(description="Analysis and visualization of Impression Propagation Algorithm",
         usage=usage)
    visualoptions.add_options(parser)
    (opt, args) = parser.parse_args(files=['./postpro.conf'])
    #check_options(opt, parser)
    return (opt, args)


def parseOptions():
    # read the options dict for each run
    dictOptions = {} # key = dirname, val = options dict for that set of repeated runs
    for dirname in os.listdir(os.getcwd()):
        if not os.path.isdir(dirname):
            continue
        if not dirname.startswith("task"):
            continue
        infile = open(os.path.join(dirname, "options.log"), 'r')
        for l in infile:
            dictOptions[dirname] = eval(l)
            # append dirname to the dict for easy reference later
            dictOptions[dirname].update({'dirname': dirname})

    logging.debug("dictOptions: %s ", str(dictOptions))
    return dictOptions



def plot_by_param(dictX, dictY, opt, fname=None, listLinestyles=None, xlabel="",
    ylabel="", title="", xscale='linear', yscale='linear', dictErr=None,
    display_on=False):
    """
    Given dicts, dictX with key=label, val = iterable of X values, 
    dictY with key=label, val = iterable of Y values, 
    plots lines for all the labels on the same plot.  """
    pyplot.clf()
    fig2 = pyplot.figure(figsize=(opt.xfigsize,opt.yfigsize))
    ax2 = fig2.add_subplot(111)
    ax2.set_title(title, fontsize=opt.fontsize)
    ax2.set_xlabel(xlabel, fontsize=opt.fontsize)
    ax2.set_ylabel(ylabel, fontsize=opt.fontsize)

    ax2.set_xscale(xscale)
    ax2.set_yscale(yscale)

    xticklabels = pyplot.getp(pyplot.gca(), 'xticklabels')
    pyplot.setp(xticklabels, fontsize=opt.label_fontsize)

    yticklabels = pyplot.getp(pyplot.gca(), 'yticklabels')
    pyplot.setp(yticklabels, fontsize=opt.label_fontsize)
    pyplot.hold(True)
    
    line_dict = {} # key = label, val = pyplot line object

    logging.debug( "dictX: %s", str(dictX))
    logging.debug( "dictY: %s", str(dictY))
    logging.debug( "dictErr: %s", str(dictErr))

    i=0
    for label in sorted(dictX.keys()):
        arrayX = dictX[label]
        arrayY = dictY[label]

        if listLinestyles is not None:
            fmt = listLinestyles[i]
            i += 1
        else:
            fmt = random.choice(markers)

        if dictErr is not None:
            arrayErr = dictErr[label]
            # plot with errorbars
            line_dict[label] = pyplot.errorbar(
                arrayX, arrayY, yerr=[arrayErr, numpy.zeros(len(arrayErr))],
                fmt=fmt,
                label="%s"%str(label), linewidth=opt.linewidth,
                elinewidth=opt.linewidth / 2.0,
                markersize=opt.markersize)
        else:
            line_dict[label] = pyplot.plot(
                arrayX, arrayY, fmt,
                label="%s"%str(label), linewidth=opt.linewidth, 
                markersize=opt.markersize)

    # for the choosing delta plot only          
    #pyplot.ylim(ymin=-0.005)        
    #pyplot.xlim((-0.05, 1.05))        

    # for the atypical senator plot only
    #pyplot.xlim((1885, 2015))        
    #pyplot.ylim((-1, 25))        
    pyplot.legend()

    # magic function to adjust the various spacings in the plot
    # not availabe until matplotlib 1.1.0 which needs dist upgrade to oneric ;(
    # pyplot.tight_layout()

    # But do not grief. Save as .svg and edit the final plots in inkscape. Remember to ungroup.
    
    if fname is not None:
        pyplot.savefig('%s'%fname)
    if display_on is True or opt.display_on is True:
        pyplot.show()

    return ax2

def plot_function(listNames):
    """ name is any attribute of SnapshotStatistics """
    
    runsdict = collections.defaultdict(list) # key = task name without run
                                        #number, val = list of run numbers
                                        
    # averaging over runs is unnecessary now, since we do repeats internally,
    # but leave it here for now                                    
    for dirname in os.listdir(os.getcwd()):
        if not os.path.isdir(dirname):
            continue
        if not dirname.startswith("task"):
            continue
        print(dirname)
        pieces = dirname.split('-run_')
        if len(pieces) > 2:
            taskey = pieces[0]+pieces[1][pieces[1].find('-'):]
        else:
            taskey = pieces[0]
        runsdict[taskey].append(dirname)

    print "runsdict: ", runsdict
    dictX = collections.defaultdict(list)
    dictY = collections.defaultdict(list)

    concat_datadict = {}
    avg_datadict = {}
    for task in runsdict.keys():
        for name in listNames:
            if task != 'task-single':
              label = name + ':' + task[5:] 
            else:
                label = name
            concat_datadict[label] = collections.defaultdict(list)
            avg_datadict[label] = collections.defaultdict(float)
            for dirname in runsdict[task]:
                numruns = len(runsdict[task])
                with open(os.path.join(dirname,"%s.log"%name), 'r') as infile:
                    data_dict = eval(infile.read())
                for t in data_dict.keys():
                    concat_datadict[label][t].append(data_dict[t])
            
            for t in concat_datadict[label].keys():
                avg_datadict[label][t] = numpy.mean(concat_datadict[label][t])

            for k in sorted(avg_datadict[label].keys(),key=int):
                dictX[label].append(int(k))
                dictY[label].append(avg_datadict[label][k])

      
    plot_by_param(dictX, dictY, opt, fname='%s.%s'%('-'.join(listNames), opt.image_extension),
        listLinestyles=['bo-', 'ro-', 'go-', 'mo-', 'ko-', 'yo-', 'co-',
                  'bs-', 'rs-', 'gs-', 'ms-', 'ks-', 'ys-', 'cs-',
                  'b*-', 'r*-', 'g*-', 'm*-', 'k*-', 'y*-', 'c*-',],
        xlabel="Time", ylabel=name, title="%s evolution"% ', '.join(listNames))



def ChoosingDelta():
    """ plot avg Q*-Q vs delta to get insights into the best delta """
    listOptionsDict = sorted(dictOptions.values(), key=operator.itemgetter('delta')) 
    logging.debug( "listOptionsDict: %s" , listOptionsDict )
    

    dictX = collections.defaultdict(list)
    dictY = collections.defaultdict(list)

    Qavg_dict = {} # {delta: Qavg} 
    Eavg_dict = {} # {delta: Eavg} 
    for delta, runs_iter in itertools.groupby(listOptionsDict,operator.itemgetter('delta')):
        runs_iter = list(runs_iter)

        taskdirs = [r['dirname'] for r in runs_iter]
        if len(taskdirs) == 1:
            taskdir = taskdirs[0]
            logging.info("grouping parsed taskdirs as %s for delta=%f", taskdir, delta)
        else:
            sys.exit("there should not be more than one task directories \
                per value of delta, found %d for delta=%f" %(len(taskdirs), delta))


        with open(os.path.join(taskdir, "Q.log",), 'r') as f:
            Q_dict = eval(f.read())  # {time: Q}
        
        # remove the lowest time entry since the initial parition is a given
        # this also keeps us consistent with Qstar and E below
        del(Q_dict[sorted(Q_dict.keys())[0]])
        logging.debug("Q_dict with first snapshot deleted:%s", str(Q_dict))

        with open(os.path.join(taskdir, "Qstar.log",), 'r') as f:
            Qstar_dict = eval(f.read())  # {time: Qstar}

        with open(os.path.join(taskdir, "Estrangement.log",), 'r') as f:
            E_dict = eval(f.read())  # {time: E}


        dictX["Average loss in Modularity"].append(delta)
        dictY["Average loss in Modularity"].append(numpy.mean(Qstar_dict.values()) - numpy.mean(Q_dict.values()))
        
        dictX["Average Estrangement"].append(delta)
        dictY["Average Estrangement"].append(numpy.mean(E_dict.values()))

    plot_by_param(dictX, dictY, opt, fname='choosing_delta.%s' % opt.image_extension,
        listLinestyles=['bs--', 'ro-',],
        xlabel="$\delta$", ylabel='', title="")


def mit_dates():
    """ return dictionary from time number to version name """
    enddate_file_name = "../../../datasets/realitymining-xml/mitdata-weekly-snapshots-weighted-merged/enddate_dict.txt"
    with open(enddate_file_name, 'r') as f:
        snapshot_dict = eval(f.read())
    return snapshot_dict

def senator_info():
    """ return info about senators from icpsr codes """
    icpsr_dict = {} # key = (congress, icpsr_id), val = list of info about the member
    with open("../../../datasets/voteview/senate_icpsr.txt", 'r') as f:
        for l in f:
            cols = l.split()
            icpsr_dict[int(cols[1])] = ','.join(cols[4:])
    return icpsr_dict    

def congress_years():
    with open("../../../datasets/voteview/congress_to_years.txt", 'r') as f:
        snapshot_dict = eval(f.read())
    return snapshot_dict

def state_names():
    """ return info about senators from icpsr codes """
    state_icpsr_dict = {} # key = state_icpsr_id, val = name
    with open("../../../datasets/voteview/state_codes_icpsr.txt", 'r') as f:
        for l in f:
            cols = l.split()
            state_icpsr_dict[int(cols[0])] = cols[1]
    return state_icpsr_dict    


def preprocess_temporal_communities(nodes_of_interest=[]):
    """ preprocessing to produce the tiled plots for all the runs 

    if nodes_of_interest is not an empty list then show egocentric view of the
    evolution, meaning plot only the nodes which ever share a label with a node
    in the nodes_of_interest 

    """

    listOptionsDict = sorted(dictOptions.values(), key=operator.itemgetter('delta')) 
    logging.debug( "listOptionsDict: %s" , listOptionsDict )
    
    logging.info("nodes_of_interest=%s", str(nodes_of_interest))
    
    # first read all the files and create label_index_dict
    all_labels_set = set()
    appearing_nodes_set = set()
    all_times_set = set()
    
    labels_of_interest_dict = collections.defaultdict(set) # key = delta, val =
                                     # set of labels of interest for that delta
    
    # key = node, val = list of labels the node gets over time for delta = 1.0
    label_time_series_dict = collections.defaultdict(list)

    # key = node, val = list of times at which the node appears
    appearances_dict = collections.defaultdict(list)


    prev_temporal_label_dict = {} # store for alignment across deltas
    for delta, runs_iter in itertools.groupby(listOptionsDict,operator.itemgetter('delta')):
        runs_iter = list(runs_iter)

        taskdirs = [r['dirname'] for r in runs_iter]
        if len(taskdirs) == 1:
            taskdir = taskdirs[0]
            logging.info("grouping parsed taskdirs as %s for delta=%f", taskdir, delta)
        else:
            sys.exit("there should not be more than one task directories \
                per value of delta, found %d for delta=%f" %(len(taskdirs), delta))


        temporal_label_dict = {}  # key = (node, time) val, = label
        
        #with open(os.path.join(taskdir,"simplified_matched_labels.log"), 'r') as label_file:
        with open(os.path.join(taskdir,opt.partition_file), 'r') as label_file:
            for l in label_file:
                # reading dict with key=time t and val = label_dict (key=node_name, val=label
                # at time t
                line_dict = eval(l)
                time = line_dict.keys()[0]
                label_dict = line_dict[time]

                all_times_set.add(time) 
                
                for n,l in label_dict.items():
                    temporal_label_dict[(n,time)] = l

                if delta == opt.delta_to_use_for_node_ordering :
                    for n,l in label_dict.items():
                        label_time_series_dict[n].append(l)
                        appearances_dict[n].append(time) 
        
        #align temporal communities for various deltas for sensible visualization
        matched_temporal_label_dict = utils.match_labels(temporal_label_dict, prev_temporal_label_dict)
        prev_temporal_label_dict = matched_temporal_label_dict 


        logging.info("delta=%f, unmatched_labels=%s \n matched_labels=%s\n", delta,
            str(set(temporal_label_dict.values())),
            str(set(matched_temporal_label_dict.values())))

        all_labels_set.update(matched_temporal_label_dict.values()) 

        with open(os.path.join(taskdir,"temporal_labels.log"), 'w') as f:
            f.write(repr(temporal_label_dict))
   
        with open(os.path.join(taskdir,"matched_temporal_labels.log"), 'w') as f:
            f.write(repr(matched_temporal_label_dict))

        # keep track of all the labels taken over time by nodes_of_interest 
        if nodes_of_interest:
            for (n,t), l in matched_temporal_label_dict.items():
                if n in nodes_of_interest:
                    labels_of_interest_dict[delta].add(l)
            for (n,t), l in matched_temporal_label_dict.items():
                if l in labels_of_interest_dict[delta]:
                    appearing_nodes_set.add(n)

        logging.info("delta=%f, labels_of_interest_dict=%s", delta, str(labels_of_interest_dict))
        
    if opt.nodeorder is not None:
        ordered_nodes = eval(opt.nodeorder)
    else:    
        # node_index_dict,  key = nodename, val=index to use for plotting that node
        # on the y axis using pcolor
        # use the temporal communities for delta=1.0 to get a node ordering for plotting  

        
        label_count_dict = {} # key = node, val = tuple of labels, ordered by freq
        for n , label_list in label_time_series_dict.items():
            label_freq = collections.defaultdict(int) # key = label, val = freq
            for l in label_list:
                label_freq[l] += 1
            label_count_dict[n] = sorted(label_freq.keys(), key=label_freq.get, reverse=True)

        first_appearances_dict = {} # key = node, val = first time that node appears
        for n in appearances_dict.keys():
            first_appearances_dict[n] = min(appearances_dict[n])
        print("first_appearances_dict : ",  str(first_appearances_dict))        

        def node_sorting_function(n):
            return (label_count_dict[n], first_appearances_dict[n])

        print("label_count_dict : ",  str(label_count_dict))        
        #ordered_nodes = sorted(label_count_dict.keys(), key=label_count_dict.get)
        ordered_nodes = sorted(label_count_dict.keys(), key=node_sorting_function)
        print("ordered_nodes : ",  str(ordered_nodes))        
    

    if nodes_of_interest:
        filtered_ordered_nodes = [ n for n in ordered_nodes if n in appearing_nodes_set]
    else:
        filtered_ordered_nodes = ordered_nodes
    node_index_dict =  dict(zip(filtered_ordered_nodes, range(len(filtered_ordered_nodes))))

    logging.info("num_nodes=%d, node_index_dict : %s ", len(node_index_dict), str(node_index_dict))        
    # label_index_dict,  key = label, val=index to use for plotting that label using pcolor
    #shuffle so that initial communites are far away in the colormap space
    #random.shuffle(unique_labels_list)
    # random shuffling changes colors in the egocentric plots, so use some
    # deterministic re-arraning which seprates the labels nevertheless
    if opt.label_sorting_keyfunc == "identity":
        unique_labels_list = sorted(list(all_labels_set))
    elif opt.label_sorting_keyfunc == "random":
        unique_labels_list = list(all_labels_set)
        random.shuffle(unique_labels_list)
    else:
        #unique_labels_list = sorted(list(all_labels_set), key=math.sin)
        unique_labels_list = sorted(list(all_labels_set), key=eval(opt.label_sorting_keyfunc))

    logging.debug("unique_labels_list:%s", str(unique_labels_list))
    label_index_dict = dict(zip(unique_labels_list, range(len(unique_labels_list))))

    
    # t_index_dict,  key = time/snapshot, val=index to use for plotting that # snapshot using pcolor
    t_index_dict = dict(zip(sorted(all_times_set), range(len(all_times_set))))

    return node_index_dict, t_index_dict, label_index_dict, labels_of_interest_dict


def plot_temporal_communities(nodes_of_interest=[]):
    """ the tiled plots for all the runs 
    
    if nodes_of_interest is not an empty list then show egocentric view of the
    evolution, meaning plot only the nodes which ever share a label with a node
    in the nodes_of_interest 
    
    """

    listOptionsDict = sorted(dictOptions.values(), key=operator.itemgetter('delta')) 
    logging.debug( "listOptionsDict: %s" , listOptionsDict )

    node_index_dict, t_index_dict, label_index_dict, labels_of_interest_dict = preprocess_temporal_communities(
        nodes_of_interest=nodes_of_interest)

    t_index_to_label_dict = dict([(v,k) for (k,v) in t_index_dict.items()])

    logging.info("computed labels_of_interest=%s \n -------------------------",
        str(labels_of_interest_dict))

    #logging.debug("node_index_dict:%s", str(node_index_dict))
    logging.debug("t_index_dict:%s", str(t_index_dict))
    logging.debug("t_index_to_label_dict:%s", str(t_index_to_label_dict))
    logging.debug("label_index_dict:%s", str(label_index_dict))



    # now make the tiled plots for all the runs
    deltas_to_plot = eval(opt.deltas_to_plot)

    fig1 = pylab.figure(figsize=eval(opt.tiled_figsize))
    numRows = 1
    if len(deltas_to_plot) is 0:
        numCols = len(listOptionsDict) 
    else:    
        numCols = len(deltas_to_plot) 

    if os.path.exists("merged_label_dict.txt"):
        numCols += 1 # +1 for the merged network

    #http://matplotlib.sourceforge.net/api/colors_api.html#matplotlib.colors.ListedColormap
    if opt.manual_colormap is not None:
        manual_colormap = eval(opt.manual_colormap)
        if len(manual_colormap) != len(label_index_dict):
            logging.error("Error: Length of manual_colormap does not match that of label_index_dict")
            logging.error("manual_color_map = %s, len=%d", str(manual_colormap), len(manual_colormap))
            logging.error("label_index_dict = %s, len=%d", str(label_index_dict), len(label_index_dict))
            sys.exit("Error: Length of manual_colormap does not match that of label_index_dict")

        cmap = matplotlib.colors.ListedColormap([manual_colormap[l] for l in sorted(manual_colormap.keys())],
            name='custom_cmap', N=None)
    else:
        cmap=pylab.cm.get_cmap(opt.label_cmap, len(label_index_dict))

    print "Hello, I am here"


    #control tick locations
    #http://matplotlib.sourceforge.net/examples/pylab_examples/major_minor_demo1.html#pylab-examples-major-minor-demo1
    #majorLocator   = MultipleLocator(5)
    #majorFormatter = FormatStrFormatter('%d')

    # there would only be one run for each delta if we take a max over several
    # runs for each snapshot
    plotNum = 0
    prev_ax = None
    for delta, runs_iter in itertools.groupby(listOptionsDict,operator.itemgetter('delta')):
        runs_iter = list(runs_iter)

        if len(deltas_to_plot) is not 0 and delta not in deltas_to_plot:
            print "Skipping delta: ", delta
            continue

        taskdirs = [r['dirname'] for r in runs_iter]
        taskdir = taskdirs[0]

        plotNum += 1

        if prev_ax is not None:
            ax1 = fig1.add_subplot(numRows, numCols, plotNum, sharex=prev_ax,
                sharey=prev_ax, frameon=opt.frameon)
            ax1.get_yaxis().set_visible(False)
        else:    
            ax1 = fig1.add_subplot(numRows, numCols, plotNum, frameon=opt.frameon)
            prev_ax = ax1

        #ax1.xaxis.set_major_locator(majorLocator)
        ax1.set_xlabel(opt.xlabel, fontsize=opt.label_fontsize)
        if plotNum == 1:
            ax1.set_ylabel(opt.ylabel, fontsize=opt.label_fontsize)
        if opt.show_title is True:
            ax1.set_title("$\delta$=%s"%delta , fontsize=opt.fontsize)
        #ax1.set_title(opt.title, fontsize=opt.fontsize)
        
        # Is not working with this version of mpl
        #ax1.tick_params(right='off')
        #ax1.tick_params(top='off')

        pylab.hold(True)

        x = numpy.array((sorted(t_index_dict.values())))
        y = numpy.array(sorted(node_index_dict.values()))
        Labels = numpy.empty((len(y), len(x)), int)
        Labels.fill(-1)

        
        with open(os.path.join(taskdir,"matched_temporal_labels.log"), 'r') as label_file:
            matched_temporal_label_dict = eval(label_file.read())
           
        for (n,t), l in matched_temporal_label_dict.items():
            if nodes_of_interest and l in labels_of_interest_dict[delta]:
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]
            elif not nodes_of_interest:    
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]


        #numpy.set_printoptions(threshold=400)
        #print("time: %d, Labels: %s" %(time, str(Labels)))

        # mask the nodes not seen in some snapshots
        Labels_masked = numpy.ma.masked_equal(Labels, -1)

        #pylab.pcolor(Labels_masked,
        #    cmap=pylab.cm.get_cmap(opt.label_cmap, len(label_index_dict)),
        #    #cmap=pylab.cm.get_cmap(opt.label_cmap),
        #    vmin = 0,
        #    vmax = len(label_index_dict) - 1,
        #    alpha=opt.alpha,
        #    edgecolors='none')

        im = pylab.imshow(Labels_masked, 
            #cmap=pylab.cm.get_cmap(opt.label_cmap, len(label_index_dict)),
            cmap=cmap,
            vmin = 0,
            vmax = len(label_index_dict) - 1,
            interpolation='nearest',
            aspect='auto',
            origin='lower')

        # @todo apply the right labels using set_xlabel and python unzip
        # specifying x and y leads to last row and col of Labels beting omitted from plotting
        #pylab.pcolor(x, y, Labels_masked, cmap=pylab.cm.get_cmap(opt.label_cmap), alpha=opt.alpha)
        
        if opt.colorbar is True:
            levels = numpy.unique1d(Labels_masked)
            cb = pylab.colorbar(ticks=levels)
            reverse_label_index_dict = dict([(v,k) for (k,v) in label_index_dict.items()])
            level_labels = [ reverse_label_index_dict[l] for l in levels.compressed() ]
            cb.ax.set_yticklabels(level_labels)

        #if opt.colorbar is True:
        #    if opt.label_with_country_names is True:
        #        levels_ccodes = [ COW_ccode_dict[reverse_node_index_dict[l]] for l in levels.compressed() ]
        #        cb = pylab.colorbar(ticks=levels)
        #        cb.ax.set_yticklabels(levels_ccodes)
        #    else:
        #        pylab.colorbar(ticks=levels)


        #if opt.label_with_country_names is True:
        #    ylocs = sorted(node_index_dict.values(), key=int)
        #    ylabs = sorted(node_index_dict.keys(), key=lambda x:int(node_index_dict[x]))
        #    country_names = [COW_ccode_dict[c] for c in ylabs]
        #    pylab.yticks(ylocs, country_names, fontsize=11)

        
        ylocs = sorted(node_index_dict.values(), key=int)
        ylabs = sorted(node_index_dict.keys(), key=node_index_dict.get)

        if opt.show_yticklabels is False:
            ax1.set_yticklabels([])
        else:    
            if opt.nodelabel_func is not None:
                nodelabel_dict = eval(opt.nodelabel_func+'()')
                node_labels = [str(nodelabel_dict[c]) for c in ylabs]
                pylab.yticks(ylocs, node_labels, fontsize=10, rotation=15)
            else:
                pylab.yticks(ylocs, ylabs, fontsize=10)

        # show every 5th label on the x axis
        xlocs = [x for x in sorted(t_index_dict.values(), key=int) if x%opt.xtick_separation == 0]
        #xlabs = sorted(t_index_dict.keys(), key=t_index_dict.get)
        #xlocs = [x for x in sorted(ax1.xaxis.get_ticklocs()) if x%5 == 0]
        logging.debug("xlocs:%s", str(xlocs))
        xlabs = [t_index_to_label_dict[x] for x in xlocs]
        logging.debug("xlabs:%s", str(xlabs))

        if opt.snapshotlabel_func is not None:
            snapshotlabel_dict = eval(opt.snapshotlabel_func+'()')
            snapshot_labels = [snapshotlabel_dict[t] for t in xlabs]
            pylab.xticks(xlocs, snapshot_labels, fontsize=11, rotation=75)
        else:
            pylab.xticks(xlocs, xlabs, fontsize=11)

        suffix=''
        if opt.nodelabel_func is not None:
            nodelabel_dict = eval(opt.nodelabel_func+'()')
            suffix='-'.join([str(nodelabel_dict[n]) for n in nodes_of_interest])
        else:    
            suffix='-'.join([str(n) for n in nodes_of_interest])
            
        xvals = t_index_dict.values()    
        ax1.set_xlim((min(xvals), max(xvals)))
        # end for delta, runs_iter in itertools.groupby(listOptionsDict,operator.itemgetter('delta')):

    fig1.subplots_adjust(wspace=opt.wspace, bottom=opt.bottom)
    
    pylab.savefig('dynconsuper%s.%s'%(suffix,opt.image_extension), dpi=opt.dpi)
    # svg viewers are slow, also save pdf
    pylab.savefig('dynconsuper%s.%s'%(suffix,'pdf'), dpi=opt.dpi)
    #if opt.display_on is True:
    pylab.show()


def main():
    
    global dictOptions

    dictOptions = parseOptions()
    for nodes_of_interest in opt.nodes_of_interest:
        plot_temporal_communities(eval(nodes_of_interest))
    ChoosingDelta()
    plot_function(['Q', 'F',])
    plot_function(['ierr', 'feasible'])
    plot_function(['best_feasible_lambda', 'lambdaopt'])
    plot_function(['numfunc'])
    plot_function(['GD', 'Node_GD'])
    plot_function(['Estrangement'])
    plot_function(['NumConsorts', 'NumEdges', ])
    plot_function(['StrengthConsorts', 'Size'])
    plot_function(['NumComm', 'NumComponents'])
    plot_function(['NumNodes', 'LargestComponentsize'])

if __name__ == '__main__':

    global opt

    (opt, args) = parse_args()

    print opt
    
    logging.basicConfig(level=getattr(logging, opt.loglevel.upper(), None))

    random.seed(opt.seed)

    if opt.profiler_on:
        # run main through the profiler
        cProfile.run('main()', 'postprocess.prof')
        # print 40 most expensive fuctions sorted by tot_time, after we are done
        st = pstats.Stats('postprocess.prof')
        st.sort_stats('time')
        st.print_stats(50)
    else:
        # run without profiler
        main()
    print "############### Postprocess done ################" 
