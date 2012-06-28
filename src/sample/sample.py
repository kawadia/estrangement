#!/usr/bin/python

import sys
import os
sys.path.append(os.getcwd() + "/..")
import utils
import agglomerate
import optionsadder
import visualoptions
import postpro
import snapshot_stats
import lpa
import main
import string
import estrangement


opt = main.parse_args()
print(opt)
opts = str(opt)
opts = string.replace(opts,'Namespace(','{')
opts = string.replace(opts,'convergence_tolerance',"'convergence_tolerance")
opts = string.replace(opts,'=',"':")
opts = string.replace(opts,' '," '")
opts = string.replace(opts,')','}')
print(opts)
print(" whereisthisgoing")
with open("options.log", 'w') as optf:
	optf.write(opts)

estrangement.ERA(opt)




