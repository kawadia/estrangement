#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements various functions used to compute and plot temporal communities.
"""

__author__ = """\n""".join(['Vikas Kawadia (vkawadia@bbn.com)',
                                    'Sameet Sreenivasan <sreens@rpi.edu>'])

#   Copyright (C) 2012 by 
#   Vikas Kawadia <vkawadia@bbn.com>
#   Sameet Sreenivasan <sreens@rpi.edu>
#   All rights reserved. 
#   BSD license. 

__all__ = ['graph_distance','node_graph_distance']


import networkx as nx
import collections
import random
import math
import operator
import logging
import sys


def graph_distance(g0, g1, weighted=True):
    """Return the Tanimoto distance between the two input graphs.

    Tanimoto distance between the set of edges is defined as    
    (a.b - aUb)/aUb where a.b is dot product and aUb = a^2 + b^2 - a.b

    Parameters
    ----------
    g0,g1: graph
	Input networkx graphs to be compared
    weighted: bool
	True if the edges of the graph are weighted, False otherwise

    Returns
    -------
    graph_distance: float
	The Tanimoto distance between the nodes of g0 and g1
	
    Note
    ----
    Used only in snapstats to plot. 
    """

    intersection = set(g1.edges_iter()) & set(g0.edges_iter())
    if weighted is False:
        union = set(g1.edges_iter()) | set(g0.edges_iter())
        graph_distance = (len(union) - len(intersection))/float(len(union))
    else:
        g0weights = nx.get_edge_attributes(g0,'weight')
        g1weights = nx.get_edge_attributes(g1,'weight')
        dot_product = sum((g0weights[i]*g1weights[i] for i in intersection))
        e1_norm = sum((g1weights[i]**2 for i in g1.edges_iter()))
        e0_norm = sum((g0weights[i]**2 for i in g0.edges_iter()))
        graph_distance = 1 - dot_product/float(e0_norm + e1_norm - dot_product)

    return graph_distance

def node_graph_distance(g0, g1):
    """Return the Tanimoto distance between the two input graphs.

    Jaccard distance between the set of nodes is defined as    
    (a.b - (aUb - a.b)) /aUb where a.b is dot product and aUb = a^2 + b^2 - a.b

    Parameters
    ----------
    g0,g1: graph
        Input networkx graphs to be compared

    Returns
    -------
    node_graph_distance: float
        The Jaccard distance between the nodes of g0 and g1
        
    Note
    ----
    Used only in snapstats to plot. 
    """

    g1_nodes = set(g1.nodes())
    g0_nodes = set(g0.nodes())
    graph_distance = 1 - len(g0_nodes & g1_nodes)/float(len(g0_nodes | g1_nodes)) 
    
    return graph_distance



def Estrangement(G, label_dict, Zgraph, gap_proof):
    """Return the Estrangement between G and Zgraph

    Compute Q-tauE for the given input parameters

    Parameters
    -----------
    G: graph
	A networkx graph object
    label_dict: dictionary
	key = node_identifier, value = community label
    ZGraph: graph
	<>
    gap_proof: boolean
	<>

    Returns
    -------
    estrangement: float
	the value of Q-tauE for the given input
 
    Note
    ----
    Used in LPA and Agglomerate"""

    consort_edge_set =  set(Zgraph.edges()) & set(G.edges())
    logging.info("Estrangement(): Z edges: %s", str(Zgraph.edges(data=True)))   
    logging.info("Estrangement(): G edges: %s", str(G.edges(data=True)))   
    logging.info("Estrangement(): consort_edge_set: %s", str(consort_edge_set))   
    if len(consort_edge_set) == 0:
        estrangement = 0
    else:    
        if gap_proof is True:
            estrangement = sum([ math.sqrt(float(Zgraph[e[0]][e[1]]["weight"]) * G[e[0]][e[1]]['weight']) 
                for e in consort_edge_set if label_dict[e[0]] != label_dict[e[1]]]) / float(G.size(weight='weight'))
        else:    
            estrangement = sum([e[2]['weight'] for e in Zgraph.edges(data=True) if label_dict[e[0]] !=
            label_dict[e[1]]]) / float(G.size(weight='weight'))
    return estrangement


def compute_VI(label_dict, prev_label_dict):
    """ Compute variation of information

    There is one nagging issue with VI that we have not settled:

    It implicitly assumes that the number of nodes is unchanged when
    comparing two partitions since there is an 'n' in the denominator.


         What we do to be formally correct is to compute partition
    distance between nodes present in both snapshots. I don't remember if we're
    already doing this?

    What I mean is that n_i, n_j and n_ij in the VI definition should be adapted
    to consider only nodes present in both snapshots. So if I call the set of
    nodes present in both snapshots, S
    Then, following Clauset's notation:   n_ij = nodes belonging to S that are
    belong to group i in C and group j in C'
    Similarly, for n_i and n_j.

    This of course does not preclude VI from being large the network changes a
    lot, because then estrangement can (and should) help very little, but at
    least our comparison of partitions will be appropriate.
    
    """
   
    VI = 0
    common_nodes = set(label_dict.keys()) & set(prev_label_dict.keys())

    C0 = collections.defaultdict(list)
    C1 = collections.defaultdict(list)

    for n in common_nodes:
        C0[prev_label_dict[n]].append(n)
        C1[label_dict[n]].append(n)

    #for n, l in prev_label_dict.items():
    #    C0[l].append(n)
    #for n, l in label_dict.items():
    #    C1[l].append(n)

    for i in C1.keys():
        for j in C0.keys():
            n_ij = len(set(C1[i]) & set(C0[j]))
            #logging.debug("n_ij, n_i, n_j ", n_ij, len(C0[j]), len(C1[j]))
            if n_ij > 0: 
                # normalized VI
                #VI += -1*(n_ij/(math.log(n)*float(len(common_nodes))))*math.log(n_ij**2/float(len(C0[j])*len(C1[i])))
                # not normalized
                VI += -1*(n_ij/float(len(common_nodes)))*math.log(n_ij**2/float(len(C0[j])*len(C1[i])))

    return VI


def compute_VL(label_dict, prev_label_dict):
    """ Compute normalized variation of labels"""
    
    common_nodes = set(label_dict.iterkeys()) & set(prev_label_dict.iterkeys())

    VL_dict = {}
    for n in common_nodes:
        if label_dict[n] == prev_label_dict[n]:
            VL_dict[n] = 0
        else:
            VL_dict[n] = 1
    
    if len(VL_dict) > 0:
        VL = sum(VL_dict.itervalues())/float(len(VL_dict))
    else:
        # -1 denotes no common nodes, so VL is undefined
        VL = -1
    return VL



def simplified_match_labels(label_dict, prev_label_dict):
    """ match labels by finding a max weight matching fromt he overlap graph
    
    also check if max_cardinality helps
   
    """
    
    # corner case for the first snapshot
    if prev_label_dict == {}:
        return label_dict

    nodesets_per_label_t = collections.defaultdict(set) # key = label, val = set
                                                    # of nodes with that label

    nodesets_per_label_t_minus_1 = collections.defaultdict(set) # key = label, val = set
                                                    # of nodes with that label
    
    for n,l in label_dict.items():
        nodesets_per_label_t[l].add(n)


    for n,l in prev_label_dict.items():
        nodesets_per_label_t_minus_1[l].add(n)


    logging.debug("nodesets_per_label_t_minus_1: %s",
        str(nodesets_per_label_t_minus_1))
    logging.debug("nodesets_per_label_t: %s", str(nodesets_per_label_t))

    overlap_dict = {} # key = (prev_label, new_label), value = jaccard overlap

    overlap_graph = nx.Graph() # store jaccard overlap between all pairs of
    # labels between t and t-1. Undirected bi-partite graph
    # compute jaccard between all possible directed pairs of labels between
    # snapshopts t and t-1
    for l_t, nodeset_t in nodesets_per_label_t.items():
        for l_t_minus_1, nodeset_t_minus_1 in nodesets_per_label_t_minus_1.items():
            jaccard =  len(nodeset_t_minus_1 & nodeset_t)/float(len((nodeset_t_minus_1 | nodeset_t))) 
            overlap_graph.add_edge(l_t_minus_1, l_t, weight=jaccard)

    logging.debug("overlap_graph nodes: %s", overlap_graph.nodes())
    logging.debug("overlap_graph edges: %s", overlap_graph.edges(data=True))

    mate = nx.max_weight_matching(overlap_graph, maxcardinality=False)
    logging.debug("max_weight_matching mate: %s", str(mate))

    # the matching func is not for a bipartite graph specifically so we dont
    # know if the kets come from the t-1 labels only. make rev_mate.
    #rev_mate = dict([(v,k) for (k,v) in mate.items()])
    #logging.debug("max_weight_matching rev_mate: %s", str(rev_mate))
    
    matched_label_dict = {} # key = node, value = new label
    for l_t in nodesets_per_label_t.keys():
        if mate.has_key(l_t):
            best_matched_label = mate[l_t]
        else:
            best_matched_label = l_t
        for n in nodesets_per_label_t[l_t]:
            matched_label_dict[n] = best_matched_label

    #logging.debug("matched_label_dict %s", str(matched_label_dict))
    return matched_label_dict




def match_labels(label_dict, prev_label_dict):
    """ match labels using Sameet's bipartitie graph based algorithm

    We start by representing the communities at t-1 and t as nodes of a
    bipartite graph.

    from each node at t-1 draw a directed link to the node at t with which it has
    maximum overlap.

    from each node at t draw a directed link to the node at t-1 with which it has
    maximum overlap.

        Basically x,y and z choose who they are most similar to among a and b
    and denote this by arrows directed outward from them. Similarly a and
    b, choose who they are most similar to among x, y and z. Then the rule
    is that labels on the t-1 side of every bidirected (symmetric) link is
    preserved - all other labels on the t-1 side die.

    """
    
    # corner case for the first snapshot
    if prev_label_dict == {}:
        return label_dict

    nodesets_per_label_t = collections.defaultdict(set) # key = label, val = set
                                                    # of nodes with that label

    nodesets_per_label_t_minus_1 = collections.defaultdict(set) # key = label, val = set
                                                    # of nodes with that label
    
    for n,l in label_dict.items():
        nodesets_per_label_t[l].add(n)


    for n,l in prev_label_dict.items():
        nodesets_per_label_t_minus_1[l].add(n)


    #logging.debug("nodesets_per_label_t_minus_1: %s",
    #    str(nodesets_per_label_t_minus_1))
    #logging.debug("nodesets_per_label_t: %s", str(nodesets_per_label_t))

    overlap_dict = {} # key = (prev_label, new_label), value = jaccard overlap

    overlap_graph = nx.Graph() # store jaccard overlap between all pairs of
    # labels between t and t-1. Undirected bi-partite graph
    # compute jaccard between all possible directed pairs of labels between
    # snapshopts t and t-1
    for l_t, nodeset_t in nodesets_per_label_t.items():
        for l_t_minus_1, nodeset_t_minus_1 in nodesets_per_label_t_minus_1.items():
            jaccard =  len(nodeset_t_minus_1 & nodeset_t)/float(len((nodeset_t_minus_1 | nodeset_t))) 
            overlap_graph.add_edge(l_t_minus_1, l_t, weight=jaccard)

    #logging.debug("overlap_graph nodes: %s", overlap_graph.nodes())
    #logging.debug("overlap_graph edges: %s", overlap_graph.edges(data=True))

    max_overlap_digraph = nx.DiGraph() # each label at t-1  and at t is a vertex in
        # this bi-partite graph and a directed edge implies the max overlap with the
        # other side (see comment at the beg of this function)

    for v in overlap_graph.nodes():
        # find the nbr with max weight
        maxwt_nbr = max([(nbrs[0],nbrs[1]['weight']) for nbrs in overlap_graph[v].items()],
            key=operator.itemgetter(1))[0]
        max_overlap_digraph.add_edge(v, maxwt_nbr)

    
    #logging.debug("max_overlap_digraph nodes: %s", max_overlap_digraph.nodes())
    #logging.debug("max_overlap_digraph edges %s", max_overlap_digraph.edges())


    #logging.debug("out_degrees in max_overlap_digraph: %s",
    #    str(max_overlap_digraph.out_degree()))

    #logging.debug("in_degrees in max_overlap_digraph: %s",
    #    str(max_overlap_digraph.in_degree()))

    matched_label_dict = {} # key = node, value = new label
    for l_t in nodesets_per_label_t.keys():
        match_l_t_minus_1 = max_overlap_digraph.successors(l_t)[0]
        # match if it is a bi-directional edge
        if max_overlap_digraph.successors(match_l_t_minus_1)[0] == l_t:
            best_matched_label = match_l_t_minus_1
        else:
            best_matched_label = l_t

        for n in nodesets_per_label_t[l_t]:
            matched_label_dict[n] = best_matched_label

    #logging.debug("matched_label_dict %s", str(matched_label_dict))
    return matched_label_dict



def Tanimoto(g0, g1, weighted=True):
    """ compute Tanimoto similarity between one hop neginbors of all nodes in g0
    and g1"""
    T_dict = {}
    for v in g1.nodes():
        n1 = g1[v]
        if v in g0.nodes():
            n0 = g0[v]
        else:
            n0 = {}

        intersection = set(n1.keys()) & set(n0.keys())
        union = set(n1.keys()) | set(n0.keys())
        if weighted is False:
            T_dict[v] = len(intersection)/float(len(union))
        else:
            dot_product = sum((n1[i]['weight']*n0[i]['weight'] for i in intersection))
            n1_norm = sum((n1[i]['weight']**2 for i in n1.keys()))
            n0_norm = sum((n0[i]['weight']**2 for i in n0.keys()))
            T_dict[v] = dot_product/float(n0_norm + n1_norm - dot_product)

    return T_dict


