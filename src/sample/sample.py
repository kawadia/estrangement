#!/usr/bin/python

import sys
import os
sys.path.append(os.getcwd() + "/..")
import utils
import agglomerate
import optionsadder
import postpro
import lpa
import string
import estrangement


opt = optionsadder.parse_args()
#deltas = [0.01,0.025]
deltas = [0.5]
matched_labels_dict = {}

print(opt.maxfun)
print(opt.increpeats)

for d in deltas:
	matched_labels = estrangement.ERA(dataset_dir='../data',delta=d,increpeats=opt.increpeats,minrepeats=opt.minrepeats)
        matched_labels_dict[d] = matched_labels 
postpro.plot_temporal_communities()

print(matched_labels)
exit()

#postpro.ChoosingDelta()
#postpro.plot_function(['Q', 'F',])
postpro.plot_function(['Estrangement'])
exit()
postpro.plot_function(['ierr', 'feasible'])
postpro.plot_function(['best_feasible_lambda', 'lambdaopt'])
postpro.plot_function(['numfunc'])
postpro.plot_function(['GD', 'Node_GD'])
postpro.plot_function(['NumConsorts', 'NumEdges', ])
postpro.plot_function(['StrengthConsorts', 'Size'])
postpro.plot_function(['NumComm', 'NumComponents'])
postpro.plot_function(['NumNodes', 'LargestComponentsize'])



