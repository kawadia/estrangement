#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
Link Propagation Algorithm (LPA).
"""

import networkx as nx
import random
import collections
import logging
import utils
import agglomerate

__all__ = ['lpa']
__author__ = """\n""".join(['Vikas Kawadia (vkawadia@bbn.com)',
	                    'Sameet Sreenivasan <sreens@rpi.edu>'])

#   Copyright (C) 2012 by 
#   Vikas Kawadia <vkawadia@bbn.com>
#   Sameet Sreenivasan <sreens@rpi.edu>
#   All rights reserved. 
#   BSD license. 


def lpa(G, tolerance=0.00001, tiebreaking=False, lambduh=3.0, initial_label_dict=None, Z=nx.Graph()):

    """Returns a graph with fewer distinct labels than the input graph.  

    Each node examines the labels of its neighbors and determines
    which of their label maximizes a specified objective function.
    The process is repeated until there are no more changes in label.
    Final set of labels is returned.  
    

    Parameters
    ----------
    G : graph
	Input NetworkX Graph.
    tolerance: float, optional
        For a label to be considered a dominant label, it must be within this much of the maximum
        value found for the quality function. The smaller the value of tolerance, the fewer dominant 
        labels there will be.
    tiebreaking: boolean, optional
        This is only relevant when there are multiple dominant labels while running the LPA.
        If it is set to 'True', the dominant label is set dominant label most recently seen. 
        If it is set to 'False', the dominant label is randomly chosen from the set of dominant labels.
    Z: networkx graph, optional
        Graph in each edges join nodes belonging to the same community over
        previous snapshots
    lambduh: 
	Lagrange multiplier.
    initial_label_dict : dictionary  {node_identifier:label,....}, optional
	Initial labeling of the nodes in G.

    Returns
    -------
    label_dict : dictionary  {node_identifier:label,....}
	Modified labeling after running LPA on G and the initial labeling.

    Raises
    ------
    NetworkXError
	If the keys in the initial labelling does not match the nodes of the graph. 

    See Also
    --------
    Agglomerate.generate_dendogram.


    References
    ----------  		
    .. [1] V. Kawadia and S. Sreenivasan, "Online detection of temporal communities 
    in evolving networks by estrangement confinement", http://arxiv.org/abs/1203.5126.

    Examples
    --------
    >>> labeling = lpa.lpa(current_graph, tolerance, tiebreaking, lambduh, Z=current_Zgraph)
    >>> new_labeling = lpa.lpa(current_graph, tolerance, tiebreaking, lambduh, labelling, Z=current_Zgraph)
    >>> list(lpa.lpa(current_graph, tolerance, tiebreaking, lambduh, Z=current_Zgraph))
    [(0,1),(1,1),(2,2),(3,1)]

    """

    # If not specifed, each node's initial label is the node's identifier
    if initial_label_dict is None:    
        initial_label_dict = dict(zip(G.nodes(), G.nodes()))
    
    if sorted(initial_label_dict.keys()) != sorted(G.nodes()):
	raise nx.NetworkXError("Invalid initial_label_dict")
 
    two_m = float(2*G.size(weight='weight'))

    nodes = G.nodes()
    label_dict = initial_label_dict.copy()		# key = nodeId, value = label 
    degree_dict = G.degree(weight='weight')		# key = nodeId, value = degree of node
    label_volume_dict = collections.defaultdict(float) 	# key = label, value = volume of that label (K_l)
    term3_dict = collections.defaultdict(float) 	# key = label, value = ??
    for v in G.nodes_iter():
        label_volume_dict[label_dict[v]] += G.degree(v, weight='weight')
        term3_dict[v] = degree_dict[v]**2/two_m		

    logging.debug("initial_labels: %s", str(label_dict)) 
    logging.debug("degree_dict: %s", str(degree_dict))

    running = True
    iteration = 0
    communities = set((label_dict.values()))


    # For multiple orderings of node visitations, calculate the value of
    # the objective function, equation (6) in reference [1]. 
    while running is True:
        running = False
        iteration += 1
        # shuffle the node visitation order
        random.shuffle(nodes)
        logging.debug("node visitation order %s", str(nodes))
        
        for v in nodes:
            if degree_dict[v] == 0:
                continue

            obj_fn_dict = collections.defaultdict(float) 
			# key = label, value = objective function to maximize

            for nbr,eattr in G[v].items():
                # self loops are not included in the N_vl term
                if nbr != v:
                    obj_fn_dict[label_dict[nbr]] += eattr["weight"]    
                else:    
                    obj_fn_dict[label_dict[nbr]] += 0.0
            
            if v in Z.nodes():
                for nbr,eattr in Z[v].items():
                    if nbr != v:
                        obj_fn_dict[label_dict[nbr]] += lambduh*float(eattr["weight"]) 
                                                
            for l in obj_fn_dict.keys():
                obj_fn_dict[l] -= degree_dict[v]*label_volume_dict[l]/two_m
                if l == label_dict[v]:
                    obj_fn_dict[l] += term3_dict[v]
                    
            logging.debug("node:%s, obj_fn_dict: %s", v, repr(obj_fn_dict))
            
	    # get the highest weighted label
            maxwt = 0
            maxwt = max(obj_fn_dict.values())
            logging.debug("node:%s, maxwt: %f", str(v), maxwt)
	
	    # record only those labels with weight sufficiently close the maxwt	
            dominant_labels = [ l for l in obj_fn_dict.keys()
                if abs(obj_fn_dict[l] - maxwt) < tolerance ]
            
            logging.debug("node:%s, dominant_labels: %s", str(v), str(dominant_labels))
            
            if len(dominant_labels) == 1:        
                the_dominant_label = dominant_labels[0]
            elif label_dict[v] in dominant_labels and tiebreaking is True:
                the_dominant_label = label_dict[v]
            else:    
                # ties are broken randomly to pick THE dominant_label
                the_dominant_label = random.choice(dominant_labels)

	    # change the node's label to the dominant label if it is not already
            if label_dict[v] != the_dominant_label :
                my_prev_label = label_dict[v]
                label_dict[v] = the_dominant_label
                # at least one vertex changed labels, so keep running
                running = True
                # update the weights of labels to refect the above change
                label_volume_dict[my_prev_label] -= degree_dict[v]
                label_volume_dict[the_dominant_label] += degree_dict[v]

                logging.debug("node:%s, label= %s", str(v), label_dict[v] )
            
            #clear the dict to be safe
            obj_fn_dict.clear()

        communities = set((label_dict.values()))

        logging.debug("the communities are : %s", str(communities))
        if iteration > 4*G.number_of_edges():
	    raise nx.NetworkXError("Too many iterations: %d" % iteration)
    
    return label_dict

