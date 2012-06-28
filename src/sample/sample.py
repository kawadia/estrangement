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


opt = optionsadder.parse_args()
estrangement.ERA(delta=opt.delta,increpeats=opt.increpeats,minrepeats=opt.minrepeats)




