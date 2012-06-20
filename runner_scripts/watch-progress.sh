#!/bin/sh


# $Id$

watch -n 2 'files="$(find . -name 'simulation.inprogress' -o -name 'simulation.done')" ; if [ -n "${files}" ] ; then tail -n 1 ${files} | tail -n $(( $(tput lines) - 2 )) ; else echo "no status" ; fi'
