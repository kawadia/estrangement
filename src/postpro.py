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
from enthought.mayavi import mlab

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

        

def compute_avg_lifetime_and_size(label_filename_path):    
    

    lifetime_dict = collections.defaultdict(int) # key = label, val = lifetime for the label
                       # lifetime is the max number of consecutive snapshots that
                       # the label appears in
    avg_community_size_dict = collections.defaultdict(float) # key = label,
                           # val=avg size of the community over its lifetime

    community_sizes_dict = collections.defaultdict(list) # key = label, val =
              # list of community size for label for all the snapshots (ordered in time)                   

    label_file = open(label_filename_path, 'r')

    for line in label_file:
        # reading dict with key=time t and val = label_dict (key=node_name, val=label
        # at time t
        line_dict = eval(line)
        time = line_dict.keys()[0]
        label_dict = line_dict[time]
        
        logging.info("################# time  %s", str(time))

        current_community_sizes_dict = {} # key = label, value= size of
                            #community corresponding to label at time t
        for node_iter in itertools.groupby(sorted(label_dict,
                key=label_dict.get), key=label_dict.get):
            label = node_iter[0]
            nodes = [n for n in node_iter[1]]
            # be careful, printing would consume the iterator
            logging.debug("label: %s, nodes: %s", str(node_iter[0]), str(nodes))
            current_community_sizes_dict[label] = float(len(nodes))

        for l in set(community_sizes_dict.keys())|set(current_community_sizes_dict.keys()):
            if l in current_community_sizes_dict:
                community_sizes_dict[l].append(current_community_sizes_dict[l])
            else:
                community_sizes_dict[l].append(0)
                
    logging.info("community_sizes_dict: %s", str(community_sizes_dict))            

    # now find the lifetime and the average size of each community
    # this is messy C-style code, urghh... ,  but I dont see a cleaner way.
    for l, community_sizes_list in community_sizes_dict.items():
        temp_lifetime = 0
        c_sizes = []
        for c in community_sizes_list:
            if c > 0:
                temp_lifetime += 1
                c_sizes.append(c)
                #logging.debug("c= %d, Label: %s, temp_lifetime: %s, c_sizes: %s",
                #    c, str(l), str(temp_lifetime), str(c_sizes))
            else:
                #logging.debug("c= %d, Label: %s, temp_lifetime: %s, c_sizes: %s",
                #    c, str(l), str(temp_lifetime), str(c_sizes))
                if temp_lifetime > lifetime_dict[l]:
                    lifetime_dict[l] = temp_lifetime
                    avg_community_size_dict[l] = numpy.mean(c_sizes)
                temp_lifetime = 0
                c_sizes = []
        #this is to account for any communities that are there until the end            
        if temp_lifetime > lifetime_dict[l]:
            lifetime_dict[l] = temp_lifetime
            avg_community_size_dict[l] = numpy.mean(c_sizes)


    logging.info("avg_community_size_dict: %s", str(avg_community_size_dict))            
    logging.info("lifetime_dict: %s", str(lifetime_dict))            
    label_file.close()
    return avg_community_size_dict, lifetime_dict, community_sizes_dict




def lifetime_size_scatterplot():
    """ do a scatter plot of community size and lifetimes
    input: label_dict_dict, key = time, value = label_dict where
    label_dict has key = node_id, value = node label

    return list of (avg_community_size, community_lifetime) tuple
    """
    
    scatter_markers_pool = [ '+'	, '^'	, '*'	, 'o'	, 's'	, 'D'	, 'p'	, 'H'	, '8'	, 'V'	, ] 

    scatter_color_pool = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow',
      'black', 'Brown', 'Purple', 'DeepPink', 'DarkRed', 'DarkGreen']

    scatter_colors = []
    handles = {}  # key = delta, value = a scatter plot object

    #fig2 = pyplot.figure(figsize=(opt.xfigsize,opt.yfigsize))
    fig2 = pyplot.figure(figsize=(14,20))
    ax2 = fig2.add_subplot(111)
    xlabel="Patch size"
    ylabel="Cumulative frequency"
    title="Histogram of temporal community size"
    ax2.set_title(title, fontsize=opt.fontsize)
    ax2.set_xlabel(xlabel, fontsize=opt.fontsize)
    ax2.set_ylabel(ylabel, fontsize=opt.fontsize)

    xticklabels = pyplot.getp(pyplot.gca(), 'xticklabels')
    pyplot.setp(xticklabels, fontsize=opt.label_fontsize)

    yticklabels = pyplot.getp(pyplot.gca(), 'yticklabels')
    pyplot.setp(yticklabels, fontsize=opt.label_fontsize)

    pylab.hold(True)

    
    listOptionsDict = sorted(dictOptions.values(), key=operator.itemgetter('delta')) 
    logging.debug( "listOptionsDict: %s" , listOptionsDict )

    # for plotting total number of unique communities with and without
    # alignment
    dictX = collections.defaultdict(list)
    dictY = collections.defaultdict(list)

    # for plotting the average size of temporal communities (area of the patch)
    dictX_patchsize = collections.defaultdict(list)
    dictY_patchsize = collections.defaultdict(list)
    # group the task directories by values of delta

    dictX_hist = {}
    dictY_hist = {}

    # there would only be one run for each delta if we take a max over several
    # runs for each snapshot
    for delta, runs_iter in itertools.groupby(listOptionsDict,operator.itemgetter('delta')):
        runs_iter = list(runs_iter)

        taskdirs = [r['dirname'] for r in runs_iter]
        if len(taskdirs) == 1:
            taskdir = taskdirs[0]
            logging.info("grouping parsed taskdirs as %s for delta=%f", taskdir, delta)
        else:
            sys.exit("there should not be more than one task directories \
                per value of delta, found %d for delta=%f" %(len(taskdirs), delta))
        
        avg_community_size_dict, lifetime_dict, community_sizes_dict = compute_avg_lifetime_and_size(os.path.join(taskdir, "labels.log"))
        matched_avg_community_size_dict, matched_lifetime_dict, matched_community_sizes_dict = compute_avg_lifetime_and_size(os.path.join(taskdir,
          opt.partition_file))
          #"simplified_matched_labels.log"))
        
        dictX["unique-labels-raw"].append(delta)
        dictY["unique-labels-raw"].append(len(avg_community_size_dict.keys()))

        #dictX["raw>4"].append(delta)
        #dictY["raw>4"].append(len([k for k in avg_community_size_dict.keys() if
        #  avg_community_size_dict[k]>4]))

        dictX["unique-labels-aligned"].append(delta)
        dictY["unique-labels-aligned"].append(len(matched_avg_community_size_dict.keys()))
        
        #dictX["aligned>4"].append(delta)
        #dictY["aligned>4"].append(len([k for k in matched_avg_community_size_dict.keys() if
        #  avg_community_size_dict[k]>4]))

        patchsizes_list = [sum(size_list) for size_list in matched_community_sizes_dict.values()]

        tmplist = sorted(patchsizes_list, reverse=True)
        top10_patchsize_list = tmplist[:10]
        dictX_patchsize["top10avg-patchsize"].append(delta)
        dictY_patchsize["top10avg-patchsize"].append(numpy.mean(top10_patchsize_list))
      
        #dictX_patchsize["max-patchsize"].append(delta)
        #dictY_patchsize["max-patchsize"].append(max(patchsizes_list))

        dictX_patchsize["median-patchsize"].append(delta)
        dictY_patchsize["median-patchsize"].append(numpy.median(patchsizes_list))

        dictX_patchsize["unique-labels-aligned"].append(delta)
        dictY_patchsize["unique-labels-aligned"].append(len(matched_avg_community_size_dict.keys()))
        

        #for the scatter
        x_data = []
        y_data = []
        for l in matched_lifetime_dict:
            x_data.append(matched_lifetime_dict[l])
            y_data.append(matched_avg_community_size_dict[l])

        scolor = scatter_color_pool.pop(0)
        scatter_colors.append(scolor)
        #handles[delta]=(pyplot.scatter(x_data, y_data, c=scolor, s=opt.markersize,
        #    label="delta=%s"%str(delta)))
        #n, bins, patches = pyplot.hist(patchsizes_list, bins=50, range=(50,500),
        
        n, bins, patches = pyplot.hist(patchsizes_list, bins=100,
            cumulative=True, normed=1, color=scolor, histtype='step',
            label="delta=%s"%str(delta), linewidth=opt.linewidth,)
            #marker=scatter_markers.pop(0))

        #hist, bin_edges = numpy.histogram(patchsizes_list, bins=100, normed=True)
        #hist, bin_edges = numpy.histogram(patchsizes_list, bins=100)
        label="delta=%s"%str(delta)

        logging.info("delta=%s, patchsizes_list=%s", str(delta), str(patchsizes_list))
        hist = numpy.bincount(patchsizes_list)

        #bin_centers = [ (bin_edges[i]+bin_edges[i+1])/2.0 for i in
        #      xrange(len(bin_edges)-1) ]

        #logging.info("delta=%s", str(delta))
        #logging.info("hist=%s \n bin_edges=%s: \n bin_centers=%s", str(hist),
        #      str(bin_edges), str(bin_centers))

        # filter out bins where count is 0
        filtered_hist = []
        filtered_bins = []
        for i in xrange(len(hist)):
            if hist[i] > 0:
                filtered_hist.append(hist[i])
                filtered_bins.append(i)

        #sum_filtered_hist = float(sum(filtered_hist))
        #normed_filtered_hist = [ i/sum_filtered_hist for i in filtered_hist]

        logging.info("filtered_hist=%s \n filtered_bins=%s",
            str(filtered_hist), str(filtered_bins))

        cumu = 0
        inverted_cumulative_hist = []
        for v in reversed(filtered_hist):
            cumu += v
            inverted_cumulative_hist.insert(0, cumu)
      
        #inverted_cumulative_hist = [ 1 - v for v in cumulative_filtered_hist ]

        logging.info("inverted_cumulative_hist=%s", str(inverted_cumulative_hist))

        dictX_hist[label] = filtered_bins
        dictY_hist[label] = inverted_cumulative_hist

    pyplot.legend()

    pyplot.savefig( "patchsize_histogram.%s"%opt.image_extension)
            
    plot_by_param(dictX_hist, dictY_hist, opt, fname="patchsize_histogram_loglog.%s"%opt.image_extension,
        listLinestyles=['bo', 'ro', 'go', 'ms', 'cs', 'ys',  'b*', 'k*', 'r*', 'g*'],
        #listLinestyles=['b-', 'r-', 'g-', 'm-', 'c-', 'y-',  'b--', 'k--', 'r--', 'g--'],
        xlabel="temporal community size", ylabel="Inverted relative occurence",
        title="Log Log Histogram", xscale='log', 
        yscale='log')
    

    plot_by_param(dictX_hist, dictY_hist, opt, fname="patchsize_histogram_loglinear.%s"%opt.image_extension,
        listLinestyles=['bo', 'ro', 'go', 'ms', 'cs', 'ys',  'b*', 'k*', 'r*', 'g*'],
        xlabel="temporal community size", ylabel="Inverted relative occurence",
        title="Linear Log Histogram", xscale='log', 
        yscale='linear')

    
    plot_by_param(dictX_hist, dictY_hist, opt, fname="patchsize_histogram_linearlog.%s"%opt.image_extension,
        listLinestyles=['bo', 'ro', 'go', 'ms', 'cs', 'ys',  'b*', 'k*', 'r*', 'g*'],
        xlabel="temporal community size", ylabel="Inverted relative occurence",
        title="Linear Log Histogram", xscale='linear', 
        yscale='log')



    plot_by_param(dictX, dictY, opt, fname='alignment_effect.%s'%opt.image_extension, 
        listLinestyles=['bo-', 'ro-', 'go-', 'mo-',], 
        xlabel="delta", ylabel="numer of unique lables", title="Effect of alignment")


    plot_by_param(dictX_patchsize, dictY_patchsize, opt, fname='patchsize.%s'%opt.image_extension, 
        listLinestyles=['bo-', 'ro-', 'go-', 'mo-',], 
        xlabel="delta", ylabel="Avg patch size", title="Looking for the best delta?")

    return



def tile3d(x, y, z):
    """use mayavi2 to create a 3d plot"""
    # Create data with x and y random in the [-2, 2] segment, and z a
    # Gaussian function of x and y.

    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    # Visualize the points
    pts = mlab.points3d(x, [i/20.0 for i in y], [i*4 for i in z], z,
        colormap='Paired', mode='sphere',
        resolution=8, scale_mode='none', scale_factor=0.8)

    #mlab.xlabel("Time")
    #mlab.ylabel("Senators")
    
    # Create and visualize the mesh
    #mesh = mlab.pipeline.delaunay2d(pts)
    #surf = mlab.pipeline.surface(mesh)

    #mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
    mlab.show()


def COW_country_codes():
    """ read COW country codes """
    cow_dict = {} # key = code, val = name
    #codefilename = "/home/vkawadia/datasets/COW_IGO/COWStatelist.csv"
    codefilename = "../../../datasets/COW_IGO/COWStatelist.csv"
    with open(codefilename, 'rU') as f:
        reader = csv.reader(f)
        #skip header
        reader.next()
        for row in reader:
            cow_dict[int(row[1])] = row[2]

    return cow_dict


def linux_author_names():
    """ dict of node indices to linux author names"""
    author_file_name = "../../../datasets/linux-kernel-repos/output-usedirauthor-names.txt"
    with open(author_file_name, 'r') as f:
        linux_authornames_dict = eval('{' + f.read() + '}')
            
    # pick first of the multiple names used by an author with the same email        
    first_authornames_dict = {}        
    for i in linux_authornames_dict:
        first_authornames_dict[i] = linux_authornames_dict[i][0]

    return first_authornames_dict

def linux_version_names():
    """ return dictionary from time number to version name """
    version_file_name = "../../../datasets/linux-kernel-repos/snapshot_dict"
    with open(version_file_name, 'r') as f:
        snapshot_dict = eval(f.read())
    return snapshot_dict


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

        
        mayavi_x = []
        mayavi_y = []
        mayavi_z = []

        with open(os.path.join(taskdir,"matched_temporal_labels.log"), 'r') as label_file:
            matched_temporal_label_dict = eval(label_file.read())
           
        for (n,t), l in matched_temporal_label_dict.items():
            if nodes_of_interest and l in labels_of_interest_dict[delta]:
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]
            elif not nodes_of_interest:    
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]
                mayavi_x.append(t_index_dict[t])
                mayavi_y.append(node_index_dict[n])
                mayavi_z.append(label_index_dict[l])

        if opt.mayavi is True and delta == 0.05:
            tile3d(mayavi_x, mayavi_y, mayavi_z)


            

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


    if os.path.exists("merged_label_dict.txt"):
        ## add a chart in the end for the merged network
        # the same partition is repeated for evey snapshot
        # this plotting code is very similar to the one in the for-loop above except
        # that we read the stuff from merged_label_dict.txt
        
        plotNum += 1
        ax1 = fig1.add_subplot(numRows, numCols, plotNum)
        prev_ax = ax1
        
        ax1.set_xlabel("Time")
        ax1.set_title("Merged network" , fontsize=opt.fontsize)
        #ax1.set_title(opt.title, fontsize=opt.fontsize)
        
        x = numpy.array((sorted(t_index_dict.values())))
        y = numpy.array(sorted(node_index_dict.values()))
        Labels = numpy.empty((len(y), len(x)), int)
        Labels.fill(-1)

        
        with open("merged_label_dict.txt", 'r') as label_file:
            merged_label_dict = eval(label_file.read())

        merged_temporal_label_dict = {}
        for n, l in merged_label_dict.items():
            for t in t_index_dict.keys():
                merged_temporal_label_dict[(n, t)] = l
           
        unique_merged_labels_list = set(merged_label_dict.values())
        label_index_dict = dict(zip(unique_merged_labels_list, range(len(unique_merged_labels_list))))

        for (n,t), l in merged_temporal_label_dict.items():
            if nodes_of_interest and l in labels_of_interest_dict[delta]:
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]
            elif not nodes_of_interest:    
                Labels[node_index_dict[n], t_index_dict[t]] = label_index_dict[l]

        # mask the nodes not seen in some snapshots
        Labels_masked = numpy.ma.masked_equal(Labels, -1)

        pylab.pcolor(Labels_masked,
            cmap=pylab.cm.get_cmap(opt.label_cmap, len(label_index_dict)),
            #cmap=pylab.cm.get_cmap(opt.label_cmap),
            vmin = 0,
            vmax = len(label_index_dict) - 1,
            alpha=opt.alpha,
            edgecolors='none')

        if opt.colorbar is True:
            levels = numpy.unique1d(Labels_masked)
            cb = pylab.colorbar(ticks=levels)
            reverse_label_index_dict = dict([(v,k) for (k,v) in label_index_dict.items()])
            level_labels = [ reverse_label_index_dict[l] for l in levels.compressed() ]
            cb.ax.set_yticklabels(level_labels)

        ylocs = sorted(node_index_dict.values(), key=int)
        ylabs = sorted(node_index_dict.keys(), key=node_index_dict.get)

        if opt.nodelabel_func is not None:
            nodelabel_dict = eval(opt.nodelabel_func+'()')
            node_labels = [str(nodelabel_dict[c]) for c in ylabs]
            pylab.yticks(ylocs, node_labels, fontsize=11, rotation=15)
        else:
            pylab.yticks(ylocs, ylabs, fontsize=11)

        xlocs = sorted(t_index_dict.values(), key=int)
        xlabs = sorted(t_index_dict.keys(), key=t_index_dict.get)

        if opt.snapshotlabel_func is not None:
            snapshotlabel_dict = eval(opt.snapshotlabel_func+'()')
            snapshot_labels = [snapshotlabel_dict[t] for t in xlabs]
            pylab.xticks(xlocs, snapshot_labels, fontsize=11, rotation=75)
        else:
            pylab.xticks(xlocs, xlabs, fontsize=11, rotation=75)

    fig1.subplots_adjust(wspace=opt.wspace, bottom=opt.bottom)
    
    pylab.savefig('dynconsuper%s.%s'%(suffix,opt.image_extension), dpi=opt.dpi)
    # svg viewers are slow, also save pdf
    pylab.savefig('dynconsuper%s.%s'%(suffix,'pdf'), dpi=opt.dpi)
    #if opt.display_on is True:
    pylab.show()



def plot_senator_polarization():
    """  party polarization plot """
    ### polarization plot

    dictX = collections.defaultdict(list)
    dictY = collections.defaultdict(list)

    listOptionsDict = sorted(dictOptions.values(), key=operator.itemgetter('delta')) 
    logging.debug( "listOptionsDict: %s" , listOptionsDict )
    
    party_dict = {} # key = (congress, icpsr_id), val = party code
    info_dict = {} # key = (congress, icpsr_id), val = full info
    with open("../../../datasets/voteview/senate_icpsr.txt", 'r') as f:
        for l in f:
            #cols = l.split()
            # keep only numeric fields
            cols = re.findall(r'\d+',l)
            party_dict[(int(cols[0]), int(cols[1]))] = int(cols[4])
            info_dict[(int(cols[0]), int(cols[1]))] = l
    #logging.debug("party_dict: %s", str(party_dict))        

    congress_years_dict = congress_years()

    party_labels_dict = {} # key = (partycode, delta), val = label in that run
    party_labels_dict[(100, 0.05)] = 36008849
    party_labels_dict[(200, 0.05)] = 18009613

    party_labels_dict[(100, 0.1)] = 36009526
    party_labels_dict[(200, 0.1)] = 14004959

    party_labels_dict[(100, 1.0)] = 36009526
    party_labels_dict[(200, 1.0)] = 16005895

    for delta, runs_iter in itertools.groupby(listOptionsDict,operator.itemgetter('delta')):
        runs_iter = list(runs_iter)

        if delta not in [0.05,]:
            continue

        taskdirs = [r['dirname'] for r in runs_iter]
        if len(taskdirs) == 1:
            taskdir = taskdirs[0]
            logging.info("grouping parsed taskdirs as %s for delta=%f", taskdir, delta)
        else:
            sys.exit("there should not be more than one task directories \
                per value of delta, found %d for delta=%f" %(len(taskdirs), delta))


        #with open(os.path.join(taskdir,"simplified_matched_labels.log"), 'r') as label_file:
        with open(os.path.join(taskdir, "matched_labels.log"), 'r') as label_file:
            for l in label_file:
                # reading dict with key=time t and val = label_dict (key=node_name, val=label
                # at time t
                line_dict = eval(l)
                time = line_dict.keys()[0]
                label_dict = line_dict[time]

                pol_dict = collections.defaultdict(int)    
                for n,l in label_dict.items():
                    # democrats voting like republicans  
                    # label values are for the final run, pulled from # label_index_dict
                    for party in [100, 200]:
                        if party_dict[(time,n)] == party and l!= party_labels_dict[(party, delta)]: 
                            pol_dict[(party, delta)] += 1
                            logging.info("Deviant info: %d, %d, %d, %s ",
                                congress_years_dict[time], n, l, info_dict[(time,n)])

                    # republicans voting like democrats
                    #if party_dict[(time,n)] == 200 and l!= 18009613:
                    #    pol_dict[(200, delta)] += 1
                    #    logging.info("Deviant info: %d, %d, %d, %s ",
                    #        congress_years_dict[time], n, l, info_dict[(time,n)])

                #logging.debug("delta=%f, pol_dict=%s", delta, str(pol_dict))    
                for k in sorted(pol_dict.keys()):
                    if k == (100, 0.05):
                        label = "Atypical Democrats"
                    if k == (200, 0.05):
                        label = "Atypical Republicans"
                    dictX[label].append(congress_years_dict[time])                
                    dictY[label].append(pol_dict[k])                

    plot_by_param(dictX, dictY, opt, fname='atypical_senators.%s' % opt.image_extension,
        #listLinestyles=['bs-', 'b*-', 'bo-', 'rs-',  'ro-', 'r*-'],
        listLinestyles=['bo-', 'rs-',],
        xlabel="Time", ylabel='Number of atypical senators', title="")

def main():
    
    global dictOptions

    dictOptions = parseOptions()
    #plot_senator_polarization()
    for nodes_of_interest in opt.nodes_of_interest:
        plot_temporal_communities(eval(nodes_of_interest))
    sys.exit()
    ChoosingDelta()
    lifetime_size_scatterplot()
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
    plot_function(['VI'])

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
