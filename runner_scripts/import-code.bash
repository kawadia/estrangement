#!/bin/bash

# checks out a tagged version from the repository and sets
# up symlinks in the run directory.
#
# Configure this script by editing the following files:
#
#     configs/vcsrev
#
#         the CVS revision to specify when checking out the code
#
#     configs/scripts.conf
#
#         specifies various scripts in the repository to make
#         symlinks to
#
# Once the code has been imported, run the symlink named "run" to set
# up the tasks directories, and run each task.

function main () {
    
    # get the directory of the run this script is setting up
    RUNDIR="$(directory-expand "$(dirname "$0")")"
    echo "Setting up run in:  ${RUNDIR}"
    [[ -d "${RUNDIR}" ]] || error "No such directory:  ${RUNDIR}"

    CONFIGDIR="${RUNDIR}/configs"
    RESULTSDIR="${RUNDIR}/results"
    
    # get the base directory where all runs live 
    BASEDIR="$(directory-expand "${RUNDIR}/../..")"
    echo "Simulation runs base directory:  ${BASEDIR}"
    
    #the dir where the runner scripts live
    RUNNERSCRIPTSDIR="${BASEDIR}/runner_scripts"
    echo "Runner scripts directory:  ${RUNNERSCRIPTSDIR}"

    # the dirname of the simulation source
    SRCNAME="src"

    #the src dir in the working tree
    CURRENTSRCDIR="${BASEDIR}/${SRCNAME}"

    # get the cvs tag from the run directory
    VCSREVFILE="${CONFIGDIR}/vcsrev"
    [[ -f "${VCSREVFILE}" ]] || error "Could not find file:  ${VCSREVFILE}"
    VCSREV="$(cat "${VCSREVFILE}")"
    
    #if VCSREV is an empty string then links to src in the current working tree
    [[ -z "${VCSREV}" ]] && echo "Warning: vcsrev file empty, linking to current working tree"
    # make sure there is a src directory for that cvs tag
    SRCDIR="${BASEDIR}/${SRCNAME}${VCSREV}"
    [[ -d "${SRCDIR}" ]] || vcs_checkout
    [[ -d "${SRCDIR}" ]] || error "Couldn't check out src from VCS rev \"${VCSREV}\" to directory \"${SRCDIR}\"."
    echo "Simulation source directory:  ${SRCDIR}"
    
    IMPORTEDCODEDIR="${RUNDIR}/imported-code"
    [[ -d "${IMPORTEDCODEDIR}" ]] || mkdir -p "${IMPORTEDCODEDIR}"
    
    # make a symbolic link to the code
    SRCLINK="${IMPORTEDCODEDIR}/src"
    [[ "${SRCDIR}" -ef "${SRCLINK}" ]] || make-relative-symlink "${SRCLINK}" "${SRCDIR}"
    
    # load the script choices
    configScriptsFN="${CONFIGDIR}/scripts.conf"
    if [[ -r "${configScriptsFN}" ]] ; then
        source "${configScriptsFN}"
    fi
    [[ -z "${RUNNER_SCRIPT}" ]] && RUNNER_SCRIPT="simulation-runner.bash"
    [[ -z "${SIMULATION_SCRIPT}" ]] && SIMULATION_SCRIPT="main.py"
    [[ -z "${WATCH_SCRIPT}" ]] && WATCH_SCRIPT="watch-progress.sh"
    [[ -z "${TASK_SETUP_SCRIPT}" ]] && TASK_SETUP_SCRIPT="setup-tasks.py"
    #[[ -z "${MASTER_CONFIG}" ]] && MASTER_CONFIG="configs/slinky.conf"
    [[ -z "${ANALYSIS_SCRIPT}" ]] && ANALYSIS_SCRIPT="plot_labels.py"
    [[ -z "${POSTPROCESS_SCRIPT}" ]] && POSTPROCESS_SCRIPT="postpro.py"
    
    # make a symbolic link to the runner script
    RUNNER_SCRIPT="${RUNNERSCRIPTSDIR}/${RUNNER_SCRIPT}"
    RUNNER_SCRIPTLINK="${RUNDIR}/run"
    [[ "${RUNNER_SCRIPT}" -ef "${RUNNER_SCRIPTLINK}" ]] || make-relative-symlink "${RUNNER_SCRIPTLINK}" "${RUNNER_SCRIPT}"
    
    # make a symbolic link to the simulation script
    SIMULATION_SCRIPT="${SRCDIR}/${SIMULATION_SCRIPT}"
    SIMULATION_SCRIPTLINK="${IMPORTEDCODEDIR}/simulate"
    [[ "${SIMULATION_SCRIPT}" -ef "${SIMULATION_SCRIPTLINK}" ]] || make-relative-symlink "${SIMULATION_SCRIPTLINK}" "${SIMULATION_SCRIPT}"

    # make a symbolic link to the watch-progress script
    WATCH_SCRIPT="${RUNNERSCRIPTSDIR}/${WATCH_SCRIPT}"
    WATCH_SCRIPTLINK="${RUNDIR}/watch"
    [[ "${WATCH_SCRIPT}" -ef "${WATCH_SCRIPTLINK}" ]] || make-relative-symlink "${WATCH_SCRIPTLINK}" "${WATCH_SCRIPT}"

    # make a symbolic link to the task setup script
    TASK_SETUP_SCRIPT="${RUNNERSCRIPTSDIR}/${TASK_SETUP_SCRIPT}"
    TASK_SETUP_SCRIPTLINK="${IMPORTEDCODEDIR}/setup-tasks"
    [[ "${TASK_SETUP_SCRIPT}" -ef "${TASK_SETUP_SCRIPTLINK}" ]] || make-relative-symlink "${TASK_SETUP_SCRIPTLINK}" "${TASK_SETUP_SCRIPT}"

#     skip this as it is not used for Slinky
#     # make a symbolic link to the master config file. 
#     MASTER_CONFIG="${SRCDIR}/${MASTER_CONFIG}"
#     MASTER_CONFIGLINK="${IMPORTEDCODEDIR}/master.conf"
#     [[ "${MASTER_CONFIG}" -ef "${MASTER_CONFIGLINK}" ]] || make-relative-symlink "${MASTER_CONFIGLINK}" "${MASTER_CONFIG}"
    
    # make a symbolic link to the analysis script
    ANALYSIS_SCRIPT="${SRCDIR}/${ANALYSIS_SCRIPT}"
    ANALYSIS_SCRIPTLINK="${IMPORTEDCODEDIR}/analyze"
    [[ "${ANALYSIS_SCRIPT}" -ef "${ANALYSIS_SCRIPTLINK}" ]] || make-relative-symlink "${ANALYSIS_SCRIPTLINK}" "${ANALYSIS_SCRIPT}"
    
    # make a symbolic link to the postprocessing script
    #POSTPROCESS_SCRIPT="${IMPORTEDCODEDIR}/${POSTPROCESS_SCRIPT}"
    POSTPROCESS_SCRIPT="${SRCDIR}/${POSTPROCESS_SCRIPT}"
    POSTPROCESS_SCRIPTLINK="${IMPORTEDCODEDIR}/postprocess"
    [[ "${POSTPROCESS_SCRIPT}" -ef "${POSTPROCESS_SCRIPTLINK}" ]] || make-relative-symlink "${POSTPROCESS_SCRIPTLINK}" "${POSTPROCESS_SCRIPT}"
}


####################
# helper functions
####################


function error () {
    echo "Error:  $*" >&2
    exit 1
}

function make-relative-symlink () {
    local LINKNAME="$1"
    local TARGETNAME="$2"

    ###############################
    # figure out the relative path

    # first lop off the common directories close to the root directory
    local TMPLINKNAME="$(reverse-pathname "$(directory-expand "$(dirname "${LINKNAME}")")/$(basename "${LINKNAME}")")"
    if [[ -d "${TARGETNAME}" ]] ; then
        local TMPTARGETNAME="$(reverse-pathname "$(directory-expand "${TARGETNAME}")")"
    elif [[ -d "$(dirname "${TARGETNAME}")" ]] ; then
        local TMPTARGETNAME="$(reverse-pathname "$(directory-expand "$(dirname "${TARGETNAME}")")/$(basename "${TARGETNAME}")")"
    else
        error "for symlink ${LINKNAME}: couldn't expand directory of symlink target \"${TARGETNAME}\""
    fi
    while [[ "$(basename "${TMPTARGETNAME}")" == "$(basename "${TMPLINKNAME}")" ]] ; do
        TMPTARGETNAME="$(dirname "${TMPTARGETNAME}")"
        TMPLINKNAME="$(dirname "${TMPLINKNAME}")"
    done
    TMPTARGETNAME="$(reverse-pathname "${TMPTARGETNAME}")"
    TMPLINKNAME="$(reverse-pathname "${TMPLINKNAME}")"

    # next, figure out how many levels up are necessary to get to the
    # target
    local UPPART="/"
    while [[ "$(dirname "${TMPLINKNAME}")" != "/" ]] ; do
        #echo "Before 1:  TMPLINKNAME=${TMPLINKNAME} UPPART=${UPPART}" >&2
        UPPART="${UPPART}../"
        TMPLINKNAME="$(dirname "${TMPLINKNAME}")"
        #echo "  After1:  TMPLINKNAME=${TMPLINKNAME} UPPART=${UPPART}" >&2
    done

    # next figure out the levels down to the target
    local DOWNPART=""
    while [[ "${TMPTARGETNAME}" != "/" ]] ; do
        #echo "Before 2:  TMPTARGETNAME=${TMPTARGETNAME} DOWNPART=${DOWNPART}" >&2
        DOWNPART="/$(basename "${TMPTARGETNAME}")${DOWNPART}"
        TMPTARGETNAME="$(dirname "${TMPTARGETNAME}")"
        #echo "  After2:  TMPTARGETNAME=${TMPTARGETNAME} DOWNPART=${DOWNPART}" >&2
    done

    # combine the two
    #echo "Before 3:  ${UPPART}${DOWNPART}" >&2
    if [[ "${UPPART:0:1}" == "/" ]] ; then UPPART="${UPPART:1}" ; fi
    if [[ "${DOWNPART:0:1}" == "/" ]] ; then DOWNPART="${DOWNPART:1}" ; fi
    #echo "  After3:  ${UPPART}${DOWNPART}" >&2
    TARGETNAME="${UPPART}${DOWNPART}"

    # done finding the relative path
    ###############################

    echo "Linking \"${LINKNAME}\" to \"${TARGETNAME}\""
    [[ -L "${LINKNAME}" ]] && rm "${LINKNAME}"
    [[ -e "${LINKNAME}" ]] && error "File \"${LINKNAME}\" already exists"
    ln -s "${TARGETNAME}" "${LINKNAME}" || error "Couldn't create symlink"
}

function directory-expand () {
    #echo "Expanding \"$1\"" >&2
    [[ -d "$1" ]] || error "no such directory $1"
    pushd "$1" > /dev/null
    local RETVAL=$(pwd)
    popd > /dev/null
    echo "${RETVAL}"
    #echo "Expanded to \"${RETVAL}\"" >&2
}

function reverse-pathname () {
    #echo "Reversing \"$1\"" >&2
    [[ "${1:0:1}" == "/" ]] || error "path must be absolute"
    local FWDPATH="$1"
    local REVPATH=""
    while [[ "${FWDPATH}" != "/" ]] ; do
        REVPATH="${REVPATH}/$(basename "${FWDPATH}")"
        FWDPATH="$(dirname "${FWDPATH}")"
    done
    echo "${REVPATH}"
    #echo "Reversed to \"${REVPATH}\"" >&2
}

function vcs_checkout () {
    pushd "${BASEDIR}" > /dev/null
    #svn co --username ${USER} -r ${VCSREV} "${SRCNAME}" ${SRCDIR} || error "svn checkout failed"
    # todo update with appropriate git checkout
    popd > /dev/null
}


main
