#!/bin/bash

#
# $Id$
#
# Automated simulation scenario runner.  Runs the script "scenario" in
# the run directory on each task's config file (one at a time), then
# runs an analysis script on the resulting trace files.  Can be
# invoked more than once to take advantange of multiple processors.
# If a task fails, this script will abort.  The output of the failed
# task (including stderr) will be in the .inprogress file left by the
# failed task.  To resume, delete the .inprogress file and re-run this
# script.

# Configure this script by editing the following files:
#
#     configs/files.conf
#
#     file names for various output files
#


COMPILE_NICENESS="19"
SIMULATION_NICENESS="19"
ANALYSIS_NICENESS="19"

        
function main () {

    # relevant file names (output, .done, .inprogress)
    rundone="simulation.done"
    allrunsdone="simulations_and_analysis.done"
    runinprog="simulation.inprogress"
    result="simulation.log"
    analysis="analysis.txt"
    analysisdone="analysis.done"
    analysisinprog="analysis.inprogress"
    postprocessdone="postprocess.done"
    postprocessinprog="postprocess.inprogress"
    
    cd "$(dirname "$0")"
    RUNDIR="$(pwd -P)"
    CODEDIR="${RUNDIR}/imported-code"
    SRCLINK="${CODEDIR}/src"
    RESULTSDIR="${RUNDIR}/results"
    CONFIGDIR="${RUNDIR}/configs"
    #this contains third party python packages from our repository
    LIBDIR="${RUNDIR}/../../lib/python/"

    PYTHONPATH="${PYTHONPATH}:${LIBDIR}:${SRCLINK}"
    export PYTHONPATH

    # load the output filename choices
    filesFN="${CONFIGDIR}/files.conf"
    if [[ -r "${filesFN}" ]] ; then
        source "${filesFN}"
    fi
    # set relevant file names (output, .done, .inprogress)
    [[ -z "${analysis_input}" ]] &&  analysis_input="labels.log"
    [[ -z "${postprocess_input}" ]] &&  postprocess_input="analysis.done"


    if [[ -f "${RESULTSDIR}/${allrunsdone}" ]] ; then
       logdebug "${tdShort}: Assuming Simulations and Analysis all already done:
       postprocessing only"
    else
        # set up the tasks
        "${CODEDIR}/setup-tasks" || error "\"${CODEDIR}/setup-tasks\""

        for taskDir in $(taskDirs) ; do

            checkforstop
            
            tdShort="$(basename "${taskDir}")"
            logdebug "Processing task:  ${taskDir}"

            cd "${taskDir}"
            scenarioConfig="simulation.conf"

            if [[ ! -f "${scenarioConfig}" ]] ; then
                logwarn "${tdShort}:  No config file (${scenarioConfig}) in this task directory."
                continue
            fi
            
            #################
            # SIMULATION
            # check the status of this task
            if [[ -f "${rundone}" ]] ; then
                logdebug "${tdShort}:  Simulation already done."
            elif [[ -f "${runinprog}" ]] ; then
                logdebug "${tdShort}:  Simulation in progress."
                continue
            else
                # copy initial_label_assignment.txt  if one was provided in the configs dir
                #if [[ -f "${CONFIGDIR}/initial_label_assignment.txt" ]]; then
                #    cp "${CONFIGDIR}/initial_label_assignment.txt" .
                #fi 

                log "${tdShort}:  Running simulation..."
                echo "Host:  ${HOSTNAME}" > "${runinprog}"
                echo  "# run using gdb -x cmds.gdb python to debug using both pdb
                and gdb simultaneously " > cmds.gdb
                echo  "run -m pdb ${CODEDIR}/simulate"  >> cmds.gdb
                #gdb -x cmds.gdb python
                if nice -n ${SIMULATION_NICENESS}  "${CODEDIR}/simulate" >> "${runinprog}" 2>&1; then
                    mv "${runinprog}" "${rundone}"
                else
                    error "nice -n ${SIMULATION_NICENESS} \"${CODEDIR}/simulate\" "
                fi
            fi
            
            checkforstop
            
            #########################
            # ANALYSIS
            
            # copy analysis.conf if one was provided in the configs dir
            if [[ -f "${CONFIGDIR}/analysis.conf" ]]; then
                cp "${CONFIGDIR}/analysis.conf" .
            fi 
            
            

            # if simulation and analysis has already been done then results have
            # been gzipped, so check for that
            if [[ -f "${analysis_input}.gz" ]]; then
                analysis_input="${analysis_input}.gz"
            fi    
            logdebug "using analysis_input ${analysis_input}"

            # check the status of this task
            ANALYSIS_ARGS=""
            if [[ -f "${analysisdone}" ]] ; then
                logdebug "${tdShort}:  Analysis already done."
            elif [[ -f "${analysisinprog}" ]] ; then
                logdebug "${tdShort}:  Analysis in progress."
            elif [[ ( ! -f "${rundone}" ) || ( ! -f "${analysis_input}" ) ]] ; then
                logdebug "${tdShort}:  Analysis can't start -- prerequisite not done."
            else
                log "${tdShort}:  Running analysis..."
                if nice -n ${ANALYSIS_NICENESS} "${CODEDIR}/analyze" ${ANALYSIS_ARGS} &> "${analysisinprog}"; then
                    cp "${analysisinprog}" "${analysis}"
                    mv "${analysisinprog}" "${analysisdone}"
                    ## now that analysis is done, gzip the simulation results
                    ## note gzip is smart enough to not gzip a file which has .gz extension
                    #if [[ -f "${result}" ]]; then
                    #    gzip "${result}"
                    #fi
                else
                    error "\"${CODEDIR}/analyze\" \"${result}\""
                fi
            fi
        done

        checkforstop

    fi
    #########################
    # POSTPROCESS

    cd "${RESULTSDIR}"

    # copy postpro.conf if one was provided in the configs dir
    if [[ -f "${CONFIGDIR}/postpro.conf" ]]; then
        cp "${CONFIGDIR}/postpro.conf" .
    fi 


    POSTPROCESS_ARGS=""
    if [[ -f "${postprocessdone}" ]] ; then
        logdebug "Postprocessing already done."
    elif [[ -f "${postprocessinprog}" ]] ; then
        logdebug "Postprocessing in progress."
    else
        # n is a counter for task directories
        n=0
        for taskDir in $(taskDirs) ; do
            cd "${taskDir}"
            if [[ ! -f "${scenarioConfig}" ]] ; then
                continue
            fi
            checkforstop
            if [[ ! -f "${analysisdone}" ]] ; then
                doPostProcessing="no"
            else
                #pp_args_array is an array of log files to be passed to the
                #postprocessng script. use of array is necessary to handle weird
                #chars in dir and file names correctly
                pp_args_array[${n}]="$(basename ${taskDir})/${postprocess_input}"
                let n=n+1
            fi
        done
        checkforstop
        cd "${RESULTSDIR}"
        #logdebug ${pp_args_array[@]}
        if [[ "${doPostProcessing}" != "no" ]] ; then
            log "Postprocessing..."
            if "${CODEDIR}/postprocess" ${POSTPROCESS_ARGS} ${pp_args_array[@]}  &> "${postprocessinprog}" ; then
                mv "${postprocessinprog}" "${postprocessdone}"
            else
                error "\"${CODEDIR}/postprocess\""
            fi
        else
            logdebug "Postprocessing can't start -- prerequisites not done."
        fi
    fi


    
    logdebug "Nothing left to do.  Check the above output to see if any tasks failed."
}

####################
# Helper functions
####################

# Note -- assumes GNU readlink; OSX and BSD don't support -f canonicalize option
scriptFullName="$(readlink -f "$0")"
scriptName="$(basename "${scriptFullName}")"

# escape sequences for colorizing output
Clear="$(tput sgr0)"
Bold="$(tput bold)"

BlackFG="$(tput setaf 0)"
RedFG="$(tput setaf 1)"
GreenFG="$(tput setaf 2)"
YellowFG="$(tput setaf 3)"
BlueFG="$(tput setaf 4)"
MagentaFG="$(tput setaf 5)"
CyanFG="$(tput setaf 6)"
WhiteFG="$(tput setaf 7)"

BlackBG="$(tput setab 0)"
RedBG="$(tput setab 1)"
GreenBG="$(tput setab 2)"
YellowBG="$(tput setab 3)"
BlueBG="$(tput setab 4)"
MagentaBG="$(tput setab 5)"
CyanBG="$(tput setab 6)"
WhiteBG="$(tput setab 7)"

# logging helper functions
function logdebug () {
    echo "${scriptName} debug:  $*"
}
function log () {
    echo "${Bold}${scriptName} info:   $*${Clear}"
}
function logwarn () {
    echo "${Bold}${MagentaFG}${scriptName} warn:   $*${Clear}" >&2
}
function logerr () {
    echo "${Bold}${RedFG}${scriptName} error:  $*${Clear}" >&2
}

# logs $* and exits
function error () {
    logerr "
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ERROR:  $*

If the above error is due to a failed task, you can look in the
task's corresponding .inprogress file for the task's output (stdout
and stderr).  If you want the failed task to re-run, you must delete
the .inprogress file.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
}

function checkforstop () {
    if [[ -f "${RUNDIR}/stop" ]] ; then
        logwarn "
=====================================================================
Stopped due to presense of file named \"${RUNDIR}/stop\"
====================================================================="
        exit 1
    fi
}

function taskDirs () {
    find "${RESULTSDIR}" -maxdepth 1 -name 'task-*' -type d | sort
}

main
