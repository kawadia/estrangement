#!/usr/bin/env python

import matplotlib
#matplotlib.use('SVG')
#matplotlib.use('WXAgg')
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
import pylab
import sys
#import configparse
import argparse
import os
import numpy
import collections
import itertools
import random
import operator
import cProfile
import pstats
import logging
import numpy
import re
#import csv
#import math
#import re
#import pprint

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


def GetDeltas():
    deltas = []
    dictOptions = {}
    for dirname in os.listdir(os.getcwd()):
        if not os.path.isdir(dirname):
            continue
        if not dirname.startswith("task"):
            continue
	print(dirname)
	infile = open(os.path.join(dirname, "options.log"), 'r')
	for l in infile:
		dictOptions = eval(l)
		delta = dictOptions['delta']
	deltas.append(delta)
    deltas.sort()
    print(deltas)
    return(deltas)


def plot_by_param(dictX, dictY, deltas=[], linewidth=2.0, markersize=15, label_fontsize=20, xfigsize=16.0, yfigsize=12.0, fontsize=28, fname=None, listLinestyles=None, xlabel="", ylabel="", title="", xscale='linear', yscale='linear', dictErr=None, display_on=False):
    """
    Given dicts, dictX with key=label, val = iterable of X values, 
    dictY with key=label, val = iterable of Y values, 
    plots lines for all the labels on the same plot.  """
    pyplot.clf()
    fig2 = pyplot.figure(figsize=(xfigsize,yfigsize))
    ax2 = fig2.add_subplot(111)
    ax2.set_title(title, fontsize=fontsize)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_ylabel(ylabel, fontsize=fontsize)

    ax2.set_xscale(xscale)
    ax2.set_yscale(yscale)

    xticklabels = pyplot.getp(pyplot.gca(), 'xticklabels')
    pyplot.setp(xticklabels, fontsize=label_fontsize)

    yticklabels = pyplot.getp(pyplot.gca(), 'yticklabels')
    pyplot.setp(yticklabels, fontsize=label_fontsize)
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
                label="%s"%str(label), linewidth=linewidth,
                elinewidth=linewidth / 2.0,
                markersize=markersize)
        else:
            line_dict[label] = pyplot.plot(
                arrayX, arrayY, fmt,
                label="%s"%str(label), linewidth=linewidth, 
                markersize=markersize)

    pyplot.legend()

    # magic function to adjust the various spacings in the plot
    # not availabe until matplotlib 1.1.0 which needs dist upgrade to oneric ;(
    # pyplot.tight_layout()

    # But do not grief. Save as .svg and edit the final plots in inkscape. Remember to ungroup.
    
    if fname is not None:
        pyplot.savefig('%s'%fname)
    if display_on is True:
        pyplot.show()

    return ax2

def plot_function(listNames,image_extension="svg"):
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
        pieces = dirname.split('-run_')
        if len(pieces) > 2:
            taskey = pieces[0]+pieces[1][pieces[1].find('-'):]
        else:
            taskey = pieces[0]
        runsdict[taskey].append(dirname)

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

      
    plot_by_param(dictX, dictY, fname='%s.%s'%('-'.join(listNames), image_extension),
        listLinestyles=['bo-', 'ro-', 'go-', 'mo-', 'ko-', 'yo-', 'co-',
                  'bs-', 'rs-', 'gs-', 'ms-', 'ks-', 'ys-', 'cs-',
                  'b*-', 'r*-', 'g*-', 'm*-', 'k*-', 'y*-', 'c*-',],
        xlabel="Time", ylabel=name, title="%s evolution"% ', '.join(listNames))



def ChoosingDelta(image_extension="svg",deltas=[]):
    """ plot avg Q*-Q vs delta to get insights into the best delta """

    dictX = collections.defaultdict(list)
    dictY = collections.defaultdict(list)

    Qavg_dict = {} # {delta: Qavg} 
    Eavg_dict = {} # {delta: Eavg} 

    if(len(deltas) == 0): 
    	deltas = GetDeltas()

    for delta in deltas:
        with open("./task_delta_" + str(delta) + "/Q.log", 'r') as f:
            Q_dict = eval(f.read())  # {time: Q}

        # remove the lowest time entry since the initial parition is a given
        # this also keeps us consistent with Qstar and E below
        del(Q_dict[sorted(Q_dict.keys())[0]])

        with open("./task_delta_" + str(delta) +"/Qstar.log", 'r') as f:
            Qstar_dict = eval(f.read())  # {time: Qstar}

        with open("./task_delta_" + str(delta) +"/Estrangement.log", 'r') as f:
            E_dict = eval(f.read())  # {time: E}

        dictX["Average loss in Modularity"].append(delta)
        dictY["Average loss in Modularity"].append(numpy.mean(Qstar_dict.values()) - numpy.mean(Q_dict.values()))
        
        dictX["Average Estrangement"].append(delta)
        dictY["Average Estrangement"].append(numpy.mean(E_dict.values()))

    plot_by_param(dictX, dictY,deltas=[],fname='choosing_delta.%s' % image_extension,
        listLinestyles=['bs--', 'ro-',], xlabel="$\delta$", ylabel='', title="")



def preprocess_temporal_communities(deltas=[],nodes_of_interest=[],partition_file="matched_labels.log",delta_to_use_for_node_ordering=1.0,label_sorting_keyfunc="random",nodeorder=None):
    """ preprocessing to produce the tiled plots for all the runs 

    if nodes_of_interest is not an empty list then show egocentric view of the
    evolution, meaning plot only the nodes which ever share a label with a node
    in the nodes_of_interest 

    """

    if(len(deltas) == 0):
    	deltas = GetDeltas()
    
    logging.info("nodes_of_interest=%s", str(nodes_of_interest))
    
    # first read all the files and create label_index_dict
    all_labels_set = set()
    appearing_nodes_set = set()
    all_times_set = set()
    
    labels_of_interest_dict = collections.defaultdict(set) 
    # key = delta, val = set of labels of interest for that delta
    
    label_time_series_dict = collections.defaultdict(list)
    # key = node, val = list of labels the node gets over time for delta = 1.0

    appearances_dict = collections.defaultdict(list)
    # key = node, val = list of times at which the node appears


    prev_temporal_label_dict = {} # store for alignment across deltas
    for delta in deltas:

        taskdir = "task_delta_" + str(delta)
        temporal_label_dict = {}  # key = (node, time) val, = label
        
        #with open(os.path.join(taskdir,"simplified_matched_labels.log"), 'r') as label_file:
        with open(os.path.join(taskdir,partition_file), 'r') as label_file:
            for l in label_file:
                # reading dict with key=time t and val = label_dict (key=node_name, val=label
                # at time t
                line_dict = eval(l)
		print(delta)
		print(str(line_dict))
                time = line_dict.keys()[0]
                label_dict = line_dict[time]

                all_times_set.add(time) 
                
                for n,l in label_dict.items():
                    temporal_label_dict[(n,time)] = l

                if delta == delta_to_use_for_node_ordering :
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
        
    if(len(appearances_dict) == 0):
	raise ValueError("The 'delta_to_use_for_node_ordering' parameter must be one of the deltas used in simulation")	

    if nodeorder is not None:
        ordered_nodes = eval(nodeorder)
	print("ordered")
    else:    
        # node_index_dict,  key = nodename, val=index to use for plotting that node
        # on the y axis using pcolor
        # use the temporal communities for delta=1.0 to get a node ordering for plotting  
	print("not ordered") 
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
    print("filtered:")
    print(filtered_ordered_nodes)

    logging.info("num_nodes=%d, node_index_dict : %s ", len(node_index_dict), str(node_index_dict))        
    # label_index_dict,  key = label, val=index to use for plotting that label using pcolor
    #shuffle so that initial communites are far away in the colormap space
    #random.shuffle(unique_labels_list)
    # random shuffling changes colors in the egocentric plots, so use some
    # deterministic re-arraning which seprates the labels nevertheless
    if label_sorting_keyfunc == "identity":
        unique_labels_list = sorted(list(all_labels_set))
    elif label_sorting_keyfunc == "random":
        unique_labels_list = list(all_labels_set)
        random.shuffle(unique_labels_list)
    else:
        #unique_labels_list = sorted(list(all_labels_set), key=math.sin)
        unique_labels_list = sorted(list(all_labels_set), key=eval(label_sorting_keyfunc))

    logging.debug("unique_labels_list:%s", str(unique_labels_list))
    label_index_dict = dict(zip(unique_labels_list, range(len(unique_labels_list))))
    print("unique")
    print(unique_labels_list)  

    
    # t_index_dict,  key = time/snapshot, val=index to use for plotting that # snapshot using pcolor
    t_index_dict = dict(zip(sorted(all_times_set), range(len(all_times_set))))
    print("node index dict:")
    print(str(node_index_dict))
    return node_index_dict, t_index_dict, label_index_dict, labels_of_interest_dict


def plot_temporal_communities(nodes_of_interest=[],deltas=[],tiled_figsize='(36,16)',manual_colormap=None,label_cmap='Paired',frameon=True,xlabel="Time",ylabel="Node id",label_fontsize=20,show_title=True,fontsize=28,colorbar=True,show_yticklabels=False,nodelabel_func=None,xtick_separation=5,snapshotlabel_func=None,wspace=0.2,bottom=0.1,image_extension="svg",dpi=200):
    """ the tiled plots for all the runs 
    
    if nodes_of_interest is not an empty list then show egocentric view of the
    evolution, meaning plot only the nodes which ever share a label with a node
    in the nodes_of_interest 
    
    """
    if(len(deltas) == 0):
    	deltas = GetDeltas()    

    node_index_dict, t_index_dict, label_index_dict, labels_of_interest_dict = preprocess_temporal_communities(
        nodes_of_interest=nodes_of_interest)

    print("node index dict: %s" % str(node_index_dict))

    t_index_to_label_dict = dict([(v,k) for (k,v) in t_index_dict.items()])

    logging.info("computed labels_of_interest=%s \n -------------------------",
        str(labels_of_interest_dict))

    #logging.debug("node_index_dict:%s", str(node_index_dict))
    logging.debug("t_index_dict:%s", str(t_index_dict))
    logging.debug("t_index_to_label_dict:%s", str(t_index_to_label_dict))
    logging.debug("label_index_dict:%s", str(label_index_dict))



    # now make the tiled plots for all the runs
    #deltas_to_plot = eval(deltas_to_plot)

    fig1 = pylab.figure(figsize=eval(tiled_figsize))
    numRows = 1
    numCols = len(deltas) 

    if os.path.exists("merged_label_dict.txt"):
        numCols += 1 # +1 for the merged network

    #http://matplotlib.sourceforge.net/api/colors_api.html#matplotlib.colors.ListedColormap
    if manual_colormap is not None:
        manual_colormap = eval(manual_colormap)
        if len(manual_colormap) != len(label_index_dict):
            logging.error("Error: Length of manual_colormap does not match that of label_index_dict")
            logging.error("manual_color_map = %s, len=%d", str(manual_colormap), len(manual_colormap))
            logging.error("label_index_dict = %s, len=%d", str(label_index_dict), len(label_index_dict))
            sys.exit("Error: Length of manual_colormap does not match that of label_index_dict")

        cmap = matplotlib.colors.ListedColormap([manual_colormap[l] for l in sorted(manual_colormap.keys())],
            name='custom_cmap', N=None)
    else:
        cmap=pylab.cm.get_cmap(label_cmap, len(label_index_dict))

    print "Hello, I am here"


    #control tick locations
    #http://matplotlib.sourceforge.net/examples/pylab_examples/major_minor_demo1.html#pylab-examples-major-minor-demo1
    #majorLocator   = MultipleLocator(5)
    #majorFormatter = FormatStrFormatter('%d')

    # there would only be one run for each delta if we take a max over several
    # runs for each snapshot
    plotNum = 0
    prev_ax = None
    for delta in deltas:

        taskdir = "./task_delta_" + str(delta)

        plotNum += 1

        if prev_ax is not None:
            ax1 = fig1.add_subplot(numRows, numCols, plotNum, sharex=prev_ax,
                sharey=prev_ax, frameon=frameon)
            ax1.get_yaxis().set_visible(False)
        else:    
            ax1 = fig1.add_subplot(numRows, numCols, plotNum, frameon=frameon)
            prev_ax = ax1

        ax1.set_xlabel(xlabel, fontsize=label_fontsize)
        if plotNum == 1:
            ax1.set_ylabel(ylabel, fontsize=label_fontsize)
        if show_title is True:
            ax1.set_title("$\delta$=%s"%delta , fontsize=fontsize)

        pylab.hold(True)

        x = numpy.array((sorted(t_index_dict.values())))
        y = numpy.array(sorted(node_index_dict.values()))
        Labels = numpy.empty((len(y), len(x)), int)
        Labels.fill(-1)

	print("here")
	print(len(y))
        
        with open(os.path.join(taskdir,"matched_temporal_labels.log"), 'r') as label_file:
            matched_temporal_label_dict = eval(label_file.read())
           
        for (n,t), l in matched_temporal_label_dict.items():
            if nodes_of_interest and l in labels_of_interest_dict[str(delta)]:
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]
            elif not nodes_of_interest: 
		print(n)
		print(node_index_dict[n])
		print(l)   
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]



        # mask the nodes not seen in some snapshots
        Labels_masked = numpy.ma.masked_equal(Labels, -1)


        im = pylab.imshow(Labels_masked, 
            cmap=cmap,
            vmin = 0,
            vmax = len(label_index_dict) - 1,
            interpolation='nearest',
            aspect='auto',
            origin='lower')

        # @todo apply the right labels using set_xlabel and python unzip
        # specifying x and y leads to last row and col of Labels beting omitted from plotting
        #pylab.pcolor(x, y, Labels_masked, cmap=pylab.cm.get_cmap(opt.label_cmap), alpha=opt.alpha)
        
        if colorbar is True:
            levels = numpy.unique(Labels_masked)
            cb = pylab.colorbar(ticks=levels)
            reverse_label_index_dict = dict([(v,k) for (k,v) in label_index_dict.items()])
            level_labels = [ reverse_label_index_dict[l] for l in levels.compressed() ]
            cb.ax.set_yticklabels(level_labels)

        ylocs = sorted(node_index_dict.values(), key=int)
        ylabs = sorted(node_index_dict.keys(), key=node_index_dict.get)

        if show_yticklabels is False:
            ax1.set_yticklabels([])
        else:    
            if nodelabel_func is not None:
                nodelabel_dict = eval(nodelabel_func+'()')
                node_labels = [str(nodelabel_dict[c]) for c in ylabs]
                pylab.yticks(ylocs, node_labels, fontsize=10, rotation=15)
            else:
                pylab.yticks(ylocs, ylabs, fontsize=10)

        # show every 5th label on the x axis
        xlocs = [x for x in sorted(t_index_dict.values(), key=int) if x%xtick_separation == 0]
        logging.debug("xlocs:%s", str(xlocs))
        xlabs = [t_index_to_label_dict[x] for x in xlocs]
        logging.debug("xlabs:%s", str(xlabs))

        if snapshotlabel_func is not None:
            snapshotlabel_dict = eval(snapshotlabel_func+'()')
            snapshot_labels = [snapshotlabel_dict[t] for t in xlabs]
            pylab.xticks(xlocs, snapshot_labels, fontsize=11, rotation=75)
        else:
            pylab.xticks(xlocs, xlabs, fontsize=11)

        suffix=''
        if nodelabel_func is not None:
            nodelabel_dict = eval(nodelabel_func+'()')
            suffix='-'.join([str(nodelabel_dict[n]) for n in nodes_of_interest])
        else:    
            suffix='-'.join([str(n) for n in nodes_of_interest])
            
        xvals = t_index_dict.values()    
        ax1.set_xlim((min(xvals), max(xvals)))
        # end for delta, runs_iter in itertools.groupby(listOptionsDict,operator.itemgetter('delta')):

    fig1.subplots_adjust(wspace=wspace, bottom=bottom)
    
    pylab.savefig('dynconsuper%s.%s'%(suffix,image_extension), dpi=dpi)
    # svg viewers are slow, also save pdf
    pylab.savefig('dynconsuper%s.%s'%(suffix,'pdf'), dpi=dpi)
    #if opt.display_on is True:
    pylab.show()


def main():
    

    for nodes_of_interest in opt.nodes_of_interest:
        plot_temporal_communities(nodes_of_interest)
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

#    global opt
#
#   opt = parse_args()

#    print opt
    
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
