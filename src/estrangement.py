import networkx as nx
import random
import sys
import math
import os
import operator
import collections
import logging
import itertools
import numpy
from scipy import optimize
import pylab
import pprint
#local modules
import lpa
import snapshot_stats
import utils
import agglomerate
#import louvainorig

#have to make this global for access inside g_of_lambda
itrepeats = 0

def make_Zgraph(g0, g1, g0_label_dict):
    """ compute the graph Z which consists of edges in the intersection
    of g0 and g1 that have the same labels on the endpoint nodes."""

    Z = nx.Graph()
    Z.add_weighted_edges_from(
        (e[0], e[1], math.sqrt(float(g0[e[0]][e[1]]['weight'] * g1[e[0]][e[1]]['weight'])))
        for e in g1.edges_iter()
            if g0.has_edge(*e[:2]) and g0_label_dict[e[0]] == g0_label_dict[e[1]]
    )        

    return Z 

def update_Zgraph(Zgraph, g0, g1, g0_label_dict):
    """ update the graph Z which consists of all consort edges upto to time t
    
    This Zgraph may have edges which are not in g1

    The weight on the consort edges has also not been averaged
    
    """

    # add consort edges from g0 to Zgraph, this also updates the weights
    Zgraph.add_weighted_edges_from([(e[0], e[1], e[2]['weight']) for e in g0.edges(data=True) 
      if g0_label_dict[e[0]] == g0_label_dict[e[1]] ])

    # remove non-consort edges in g0 from Zgraph
    Zgraph.remove_edges_from([e for e in g0.edges()
      if g0_label_dict[e[0]] != g0_label_dict[e[1]] ])


    # further remove edges (u, v) Zgraph, if both u and v are present in g0 but
    # are no connected by an edge
    #@todo assumes that there are no zero weight edges
    extra_Zgraph_edges = set(Zgraph.edges()) - set(g0.edges())
    Zgraph.remove_edges_from([e for e in list(extra_Zgraph_edges)
      if e[0] in g0.nodes() and e[1] in g0.nodes() ])



def repeated_runs(g1, opt, lambduh, Zgraph, repeats):
    """ do repeated call to agglomerate lpa to optimize F
    return the label_dict and Q value to help searching for lambduh
    """
    dictPartition = {} # key = run number, val = label_dict
    dictQ = {} # key = run number, val = Q for that run
    dictE = {} # key = run number, val = E for that run
    dictF = {} # key = run number, val = F for that run
    
    # the for loop below does repeat number of runs to find the best F using
    # agglomerate lpa
    for r in range(repeats):
        logging.info('########## [repeat %d] #############', r)
        # best_partition calls agglomerative lpa
        r_partition = agglomerate.best_partition(g1, opt, lambduh, Zgraph)
        dictPartition[r] = r_partition
        dictQ[r] = agglomerate.modularity(r_partition, g1)
        dictE[r] = utils.Estrangement(g1, r_partition, Zgraph, opt.gap_proof_estrangement)
        dictF[r] = dictQ[r] - lambduh*dictE[r] + lambduh*opt.delta
        
    #logging.info("selected Best F among repeats; F = %s, and Q=%s, E=%s, lambduh=%f",
    #    str(best_F), str(best_Q), str(best_E), lambduh)

    return (dictPartition, dictQ, dictE, dictF)


def ERA(graph_reader_fn, opt):
    """
    Estrangement reduction algorithm

    graph_reader_fn is a generator function that yields (t, g) tuples in
    temporal order

    opt is the options parser opt object
    """
    
    #open files to log results
    label_file = open("labels.log", 'w')
    matched_label_file = open("matched_labels.log", 'w')

    snapstats = snapshot_stats.SnapshotStatistics()

    # keeping track of max number of nodes and num snapshots for help in plotting
    nodename_set = set()
    snapshots_list = []

    # to plot the histogram of node appearances in snapshots
    # motivation is to see if there is a persistent core
    # its only a one-time thing, so does not need to be in the sim loop but...
    node_appearances_dict = collections.defaultdict(int) # key = nodeid,
                                              # val=number of appearances
    
    beginning = True
    snapshot_number = 0
    for t, g1, initial_label_dict in graph_reader_fn(opt.graph_reader_fn_arg):
        # initial_label_dict label dict is non None only for the first call to # graph_reader_fn
        
        logging.info("############# snapshot: %d #################", t)
        snapshots_list.append(t)
        nodename_set.update(g1.nodes())
        #logging.info("nodes in g1: %s", str(g1.nodes()))

        if beginning is True:
            g0 = g1
            prev_label_dict = {}
            prev_matched_label_dict = {}

            label_dict = initial_label_dict 
            if len(label_dict) != g1.number_of_nodes():
                sys.exit("initial label_dict does not have the same number of nodes as g1")

            snapstats.Q[t] = agglomerate.modularity(label_dict, g1)
            Zgraph = nx.Graph() 
            beginning = False
        else:
            if opt.gap_proof_estrangement is True:
                update_Zgraph(Zgraph, g0, g1, prev_label_dict)
            else:    
                Zgraph = make_Zgraph(g0, g1, prev_label_dict)

            ## Record Q* for comparison only
            dictlabel_dict0, dictQ0, dictE0, dictF0 = repeated_runs(g1, opt, 0.0, Zgraph, opt.minrepeats)
            snapstats.Qstar[t] = max(dictQ0.values())


            # store some stats for optimization over lambda for a given snapshot
            # this is kept for analysis purposes, not strictly required for solving
            # the problem
            label_dict_lam = {} # key = lambduh, val = dictPartition where key = run number val = label_dict
            Qlam = {} # key = lambduh, val = dictQ where key = run number val = Q
            Elam = {} # key = lambduh, val = dictE where key = run number val = E
            Flam = {} # key = lambduh, val = dictF where key = run number val = F


            def g_of_lambda(lambduh):
                """ scipy.opimize.fminbound needs a function that takes a scalar and
                returns a scalar, so make a function like that"""
                global itrepeats
                logging.info("itrepeats: %d", itrepeats)
                dictPartition, dictQ, dictE, dictF = repeated_runs(g1, opt, lambduh,
                    Zgraph, itrepeats)

                label_dict_lam[lambduh] = dictPartition
                Qlam[lambduh] = dictQ
                Elam[lambduh] = dictE
                Flam[lambduh] = dictF
                itrepeats += opt.increpeats
                
                return max(dictF.values())
            
            global itrepeats
            itrepeats = opt.minrepeats
                    
            lambdaopt, fval, ierr, numfunc = optimize.fminbound(
                g_of_lambda,
                0.0, 10.0, args=(), xtol=opt.convergence_tolerance,
                maxfun=opt.maxfun, full_output=True, disp=2)  
            
            if ierr is 0:
                logging.info("[%d] best lambduh = %f, found in %d function calls", t,
                    lambdaopt, numfunc)
            else:
                logging.error("[%d] No convergence for fminbound", t)
      
            # filter Fs at lambdaopt which are feasible
            dictFeasibleFs = dict([((lambdaopt, r), F) for (r, F) in Flam[lambdaopt].items() if
                Flam[lambdaopt][r] > Qlam[lambdaopt][r]])
            
            logging.info("[%d], lambdaopt=%f, feasibleFs=%s", t, lambdaopt,
                str(dictFeasibleFs))

            listLambduhs = sorted(Flam.keys())
            logging.info("listLambduhs: %s,", str(listLambduhs))
            current_index = listLambduhs.index(lambdaopt)

            while len(dictFeasibleFs) == 0 and current_index < len(listLambduhs) - 1:
                logging.error("No feasibleFs found at current_lambduh=%f, increasing search range of lambduhs",
                    listLambduhs[current_index])

                # get next highest lambda
                # this should thrown an exception if we run out of lambdas
                next_highest_lambduh = listLambduhs[current_index + 1]

                # add those to dictFeasibleFs 
                dictFeasibleFs.update(dict([((next_highest_lambduh, r), F) 
                    for (r, F) in Flam[next_highest_lambduh].items() 
                    if Flam[next_highest_lambduh][r] > Qlam[next_highest_lambduh][r]]))

                logging.info("[%d], next_highest_lambduh=%f, feasibleFs=%s", t,
                    next_highest_lambduh, str(dictFeasibleFs))

                current_index += 1

            if len(dictFeasibleFs) > 0:
                # get best r and best_feasible_lambda
                best_feasible_lambda, bestr = max(dictFeasibleFs, key=dictFeasibleFs.get) 
                #bestr = max(Flam[lambdaopt], key=Flam[lambdaopt].get) 
                snapstats.feasible[t] = 1
            else:
                logging.error("Nothing feasible found, constraint too harsh perhaps.\
                    Using highest lambda wihout worrying about feasibility.")
                best_feasible_lambda = listLambduhs[-1]
                bestr = max(Flam[best_feasible_lambda], key=Flam[best_feasible_lambda].get) 
                snapstats.feasible[t] = 0

            logging.info("best_feasible_lambda=%f, bestr = %d", best_feasible_lambda, bestr)
                  
            label_dict = label_dict_lam[best_feasible_lambda][bestr]
            
            snapstats.lambdaopt[t] = lambdaopt
            snapstats.best_feasible_lambda[t] = best_feasible_lambda
            snapstats.numfunc[t] = numfunc
            snapstats.ierr[t] = ierr
            snapstats.StrengthConsorts[t] = Zgraph.size(weight='weight')
            snapstats.NumConsorts[t] = Zgraph.size()
            snapstats.Estrangement[t] = Elam[best_feasible_lambda][bestr]
            snapstats.Q[t] =  Qlam[best_feasible_lambda][bestr]
            snapstats.F[t] =  Flam[best_feasible_lambda][bestr]
            snapstats.VI[t] = utils.compute_VI(label_dict, prev_label_dict)
            snapstats.VL[t] = utils.compute_VL(label_dict, prev_label_dict)
            snapstats.Qdetails[t] = Qlam
            snapstats.Edetails[t] = Elam
            snapstats.Fdetails[t] = Flam
            # end else of if beginning is True:

        # increment the labels by some huge offset each snapshot and let alignment do the work
        # @todo: shoud use the number of nodes to automatically increment labels
        for n in label_dict:
            label_dict[n] += 1000000*snapshot_number

        if opt.record_matched_labels is True:
            matched_label_dict = utils.match_labels(label_dict, prev_matched_label_dict)
            matched_label_file.write("{%d:%s}\n" % (t,str(matched_label_dict)))


        label_file.write("{%d:%s}\n" % (t,str(label_dict)))

        snapstats.GD[t] = utils.graph_distance(g0, g1, True)
        snapstats.Node_GD[t] = utils.node_graph_distance(g0, g1)
        snapstats.NumComm[t] = len(set((label_dict.values())))
                

        snapstats.NumNodes[t] = g1.number_of_nodes()
        snapstats.NumEdges[t] = g1.number_of_edges()
        snapstats.Size[t] = g1.size(weight='weight')
        snapstats.NumComponents[t] = nx.number_connected_components(g1)
        if g1.number_of_nodes > 0:
            snapstats.LargestComponentsize[t] = len(nx.connected_components(g1)[0])
        else:
            snapstats.LargestComponentsize[t] = 0

        

        for n in g1.nodes():
            node_appearances_dict[n] += 1


        # save graph for layout
        if opt.savefor_layouts:
            glay = g1.copy()
            for n in glay.nodes():
                glay.add_node(n, comlabel=matched_label_dict[n])
            for e in glay.edges(data=True):
                if label_dict[e[1]] == label_dict[e[0]]:
                    e[2]['estranged'] = 0
                else:    
                    e[2]['estranged'] = 1
            nx.write_gexf(glay, "%s.gexf"%(str(t)))    
        
        
        # keep track of prev snaphost graph and labelDict for computing T_dict
        # and VI
        g0 = g1
        prev_label_dict = label_dict
        
        if opt.record_matched_labels is True:
            prev_matched_label_dict = matched_label_dict
        snapshot_number += 1

        # end for t, g1 in ......

    for statname, statobj in vars(snapstats).items():
        with open('%s.log'%statname, 'w') as fout:
            pprint.pprint(statobj, stream=fout) 

    label_file.close()
    matched_label_file.close()
    
    summary_dict = { 
      'num_nodes': len(nodename_set),
      'snapshots_list': snapshots_list,
    }  

    with open("summary.log", 'w') as summary_file:
        summary_file.write(str(summary_dict))


    with open("node_appearances.log", 'w') as freq_file:
        # write nodes appearing most first
        freq_file.write(
          str(sorted(node_appearances_dict.items(), key=operator.itemgetter(1),
            reverse=True))
        )

    return True


