#!/usr/bin/python 

import networkx as nx
import random
import sys
import os

if len(sys.argv) > 1:
    output_dir = os.path.abspath(sys.argv[1])
else:
    sys.exit("usage: %s output_dir" % sys.argv[0] )


if os.path.isdir(output_dir):
    sys.exit("result dir already exists, will not overwrite")
else:
    os.mkdir(output_dir)


def gen_random_with_alternating_core():
    """ generate a larger random graph first
    generate an additional random set of edges among a core set of nodes
    The core is different in the first half time from the second half time

    """
    merged_graph = nx.Graph()
    for t in xrange(0,40):
        print "t :", t
        g1 = nx.gnm_random_graph(40, 80, seed=100+t)
        
        if t % 2 == 0:
            g2 = nx.gnm_random_graph(10, 20, seed=200+t)
        #else:
        #    mapping = dict([(i, i+10) for i in range(0,10)])
        #    g2_temp = nx.relabel_nodes(g2, mapping)
        #    g2 = g2_temp

        g3 = nx.compose(g1,g2)
        g4 = nx.Graph()
        g4.add_edges_from(g3.edges(), weight=1.0)
        
        nx.write_weighted_edgelist(g4, (os.path.join(output_dir, "%d.ncol"%t)))
        
        merged_graph.add_edges_from(g3.edges(), weight=1.0)
    
    #nx.write_weighted_edgelist(g4, (os.path.join(output_dir, "network.merged")))


if __name__ == "__main__":
    gen_random_with_alternating_core()

