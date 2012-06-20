
""" implementation of lpa to maximize Q-lambduh*E for a single snapshot """


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
import pylab
import utils
import snapshot_stats
import pprint
import agglomerate


def lpa(G, opt, lambduh, initial_label_dict=None, Z=nx.Graph()):

    """
    input_graph is a networkx graph object

    initial_label_dict : key node, value: initial label for the node

    """

    if initial_label_dict is None:    
        initial_label_dict = dict(zip(G.nodes(), G.nodes()))
    
    
    logging.debug("initial_labels: %s", str(initial_label_dict)) 
    if sorted(initial_label_dict.keys()) != sorted(G.nodes()):
        sys.exit("Invalid initial_label_dict")

    label_dict = initial_label_dict.copy() # key = node, value = label

    degree_dict = G.degree(weight='weight')
    logging.debug("degree_dict: %s", str(degree_dict)) 
    

    two_m = float(2*G.size(weight='weight'))

    label_volume_dict = collections.defaultdict(float) # key = label, value = volume of that label (K_l)
    term3_dict = collections.defaultdict(float) # key = label, value = volume of that label (K_l)
    for v in G.nodes_iter():
        label_volume_dict[label_dict[v]] += G.degree(v, weight='weight')
        term3_dict[v] = opt.resolution*(degree_dict[v]**2)/two_m

    nodes = G.nodes()
    
    running = True
    iteration = 0

    communities = set((label_dict.values()))
    mod = agglomerate.modularity(label_dict, G)
    E = utils.Estrangement(G, label_dict, Z, opt.gap_proof_estrangement)
    F = mod - lambduh*E + lambduh*opt.delta
    logging.info("iteration=%d, num communities=%d, Q=%f, E=%f, F=%f ",
        iteration, len(communities), mod, E, F)

    while running is True:
        running = False
        iteration += 1
        # shuffle the node visitation order
        random.shuffle(nodes)
        logging.debug("node visitation order %s", str(nodes))
        
        for v in nodes:
            if degree_dict[v] == 0:
                continue

            obj_fn_dict = collections.defaultdict(float) # key = label, value = obj func to maximize

            for nbr,eattr in G[v].items():
                # self loops are not included in the N_vl term
                if nbr != v:
                    obj_fn_dict[label_dict[nbr]] += eattr["weight"]    
                else:    
                    obj_fn_dict[label_dict[nbr]] += 0.0
            
            if v in Z.nodes():
                for nbr,eattr in Z[v].items():
                # @todo do we really need if nbr != v:
                    if opt.gap_proof_estrangement is False:
                        if nbr != v:
                            obj_fn_dict[label_dict[nbr]] += lambduh*float(eattr["weight"]) 
                    else:        
                        if nbr != v and G.has_edge(v,nbr):
                            obj_fn_dict[label_dict[nbr]] += lambduh*math.sqrt(float(eattr["weight"]) * G[v][nbr]['weight']) 

                            
            for l in obj_fn_dict.keys():
                obj_fn_dict[l] -= opt.resolution * degree_dict[v]*label_volume_dict[l]/two_m
                if l == label_dict[v]:
                    obj_fn_dict[l] += term3_dict[v]
                    
            logging.debug("node:%s, obj_fn_dict: %s", v, repr(obj_fn_dict))
            
            # max(obj_fn_dict, key=obj_fn_dict.get)
            maxwt = 0
            maxwt = max(obj_fn_dict.values())
            logging.debug("node:%s, maxwt: %f", str(v), maxwt)
            dominant_labels = [ l for l in obj_fn_dict.keys()
                if abs(obj_fn_dict[l] - maxwt) < opt.tolerance ]
            
            logging.debug("node:%s, dominant_labels: %s", str(v), str(dominant_labels))
            
            if len(dominant_labels) == 1:        
                the_dominant_label = dominant_labels[0]
            elif label_dict[v] in dominant_labels and opt.precedence_tiebreaking is True:
                the_dominant_label = label_dict[v]
            else:    
                # ties are broken randomly to pick THE dominant_label
                the_dominant_label = random.choice(dominant_labels)

            # check for fixed point: is the vertex's label the same as the dominant label
            if label_dict[v] != the_dominant_label :
                my_prev_label = label_dict[v]
                label_dict[v] = the_dominant_label
                # at least one vertex changed labels, so keep running
                running = True
                ### update K_l
                label_volume_dict[my_prev_label] -= degree_dict[v]
                label_volume_dict[the_dominant_label] += degree_dict[v]

                logging.debug("node:%s, label= %s", str(v),
                    label_dict[v] )
            
            #clear the dict to be safe
            obj_fn_dict.clear()

            
        communities = set((label_dict.values()))
        mod = agglomerate.modularity(label_dict, G)
        E = utils.Estrangement(G, label_dict, Z, opt.gap_proof_estrangement)
        F = mod - lambduh*E + lambduh*opt.delta
        logging.info("iteration=%d, num communities=%d, Q=%f, E=%f, F=%f ",
            iteration, len(communities), mod, E, F)


        logging.debug("the communities are : %s", str(communities))
        if iteration > 4*G.number_of_edges():
            sys.exit("Too many iterations: %d" % iteration)
    

    return label_dict

