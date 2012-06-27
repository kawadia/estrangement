#!/usr/bin/env python


# $Id$

import sys
import os
import compiler
import string

#script to read in parameters.conf file and produce task directories with a
#simulation.conf file in each

def cartesian(l2d):
    """input a list of n input lists, each of len l_i
    output: a list of Product(l_i) lists, each of whom has exactly one element
    from the input lists
    Example: 
    input: [ ['a', 'b', 'c'] , [1, 2 ] , ['x', 'y'] ] 
    output: [['a', 1, 'x'], ['b', 1, 'x'], ['c', 1, 'x'], ['a', 2, 'x'], ['b', 2, 'x'], ['c', 2, 'x'], ['a', 1, 'y'], ['b', 1, 'y'], ['c', 1, 'y'], ['a', 2, 'y'], ['b', 2, 'y'], ['c', 2, 'y']]
    """
    if len(l2d) <= 1:
        return [[x] for x in l2d[0]]
    else:
        return [[l] + x for x in cartesian(l2d[1:]) for l in l2d[0]]


def getTaskDir(plist, params_dict):
    """function to construct the name of a task dir from a bunch of arguments"""
    special_chars = """ '"/\()!~`@#$%^&*<>?:;{}[]"""
    base = 'results/task-'
    #only include those parameters which are varying for a run
    sargs = [ x[0] + '_' + str(x[1]) for x in plist if len(params_dict[x[0]]) > 1] 
    good_args = [ ]
    #if no varying parameters (single run), use a reasonable default name
    if len(sargs) == 0:
        good_args.append('single')
    # os.path.basename(x)
    for s in sargs:
        for c in special_chars:
            g = s.replace(c, '')
            s = g
        good_args.append(s)
    taskDir = base + '-'.join(good_args)
    if not os.path.exists(taskDir):
        print "creating dir ", taskDir
        os.makedirs(taskDir)
    else:
        #sys.exit(taskDir + " already exists, please cleanup.\n")
        print taskDir + " already exists, will use it "
    return taskDir

def read_parameter_list_file(plistfilename):
    """read the parameter-list config file
    return the evaluated string in the file, this is expected to be a dict, no
    error checking done.
    """
    try:
    	params_file = open(plistfilename)
    	lineobj = compiler.compile(params_file.read(), 'read.err', 'eval')
    except:
        sys.exit("ERROR: Could not open " + plistfilename)
    return eval(lineobj)


if __name__ == '__main__':
    # read the parameter-lists.conf file
    params_dict = read_parameter_list_file("configs/parameter-lists.conf")
    print "params_dict: ", params_dict

    # convert it into a list of list of tuples of the form (paramname,
    # paramvalue). There is a list for each parameter in the list of lists.
    params_list2 = [[(k,x) for x in params_dict[k]] for k in sorted(params_dict.keys())]

    cartesian_product = cartesian(params_list2)

    for conflist in cartesian_product:
        taskDir = getTaskDir(conflist, params_dict)
        confile = open(taskDir + '/simulation.conf', 'w')
        for param in conflist:
            confile.write("--%s=%s\n" % param) 
        confile.close()

