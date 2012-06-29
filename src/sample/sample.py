#!/usr/bin/python

import sys
import os
sys.path.append(os.getcwd() + "/..")
import utils
import agglomerate
import optionsadder
import postpro
import snapshot_stats
import lpa
import main
import string
import estrangement


opt = optionsadder.parse_args()
deltas = [0.01,0.025,0.05,0.10]


print(opt.maxfun)
print(opt.increpeats)

#for d in deltas:
#	estrangement.ERA(dataset_dir='../data',delta=d,increpeats=opt.increpeats,minrepeats=opt.minrepeats)

print(opt.nodes_of_interest)
#postpro.ChoosingDelta()
postpro.plot_temporal_communities()
exit()
postpro.plot_function(['Estrangement'])
postpro.plot_function(['Q', 'F',])
postpro.plot_function(['ierr', 'feasible'])
postpro.plot_function(['best_feasible_lambda', 'lambdaopt'])
postpro.plot_function(['numfunc'])
postpro.plot_function(['GD', 'Node_GD'])
postpro.plot_function(['Estrangement'])
postpro.plot_function(['NumConsorts', 'NumEdges', ])
postpro.plot_function(['StrengthConsorts', 'Size'])
postpro.plot_function(['NumComm', 'NumComponents'])
postpro.plot_function(['NumNodes', 'LargestComponentsize'])



