import os


def add_options(parser, reader_functions={}):
    """define all the program options here"""
    parser.add_option("-d", "--dataset_dir",
            dest="dataset_dir", 
            help="dir where all the datasets are stored, passed to the graph_reader_fcns. [default: %default]",
            type="string",
            default=os.path.join(os.path.expanduser("~"), "datasets"),
            config='true'
    )


    parser.add_option("-e", "--precedence_tiebreaking",
            dest="precedence_tiebreaking", 
            help="turn on precedence tiebreaking which keeps a node's current label if it is one of the dominant lables. [default: %default]",
            action="store_true",
            default=False,
            config='true'
    )

 
    parser.add_option("--tolerance",
            dest="tolerance", 
            help="tolerance as a fraction of the max when picking dominant labels [default: %default]",
            type="float",
            default=0.00001,
            config='true'
    )
    
    parser.add_option("--convergence_tolerance",
            dest="convergence_tolerance", 
            help="lambda (X) tolerance for scipy optimize [default: %default]",
            type="float",
            default=0.01,
            config='true'
    )

    parser.add_option("--maxfun",
            dest="maxfun", 
            help="max number of iterations allowed for scipy optimize [default: %default]",
            type="int",
            default=500,
            config='true'
    )


    parser.add_option("--delta",
            dest="delta", 
            help="constraint on estrangement [default: %default]",
            type="float",
            default=0.05,
            config='true'
    )
    
    
    parser.add_option("--minrepeats",
            dest="minrepeats", 
            help="min number of repeats for each snapshot [default: %default]",
            type="int",
            default=10,
            config='true'
    )

    parser.add_option("--increpeats",
            dest="increpeats", 
            help="number of repeats in increased by this amount for every call to g_of_lambda [default: %default]",
            type="int",
            default=10,
            config='true'
    )

    parser.add_option("--graph_reader_fn",
            dest="graph_reader_fn",
            type="choice",
            choices=reader_functions.keys(),
            help="generator fcn to read graph snapshots. Choices are " + str(reader_functions.keys()) + " [default: %default]",
            default="read_general",
            config='true')
    
    
    parser.add_option("--graph_reader_fn_arg",
            dest="graph_reader_fn_arg",
            type="str",
            help="string argument for graph_reader_fn [default: %default]",
            default="",
            config='true')



    parser.add_option("--profiler_on",
            dest="profiler_on",
            action="store_true",
            help="Turn on profiling [default: %default]",
            default=False,
            config='true')

    parser.add_option("--savefor_layouts",
            dest="savefor_layouts",
            action="store_true",
            help="Save gexf for layouts [default: %default]",
            default=False,
            config='true')


    parser.add_option("--loglevel",
            dest="loglevel",
            type="choice",
            choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], 
            help="Set console logging level to LEVEL [default: %default]",
            metavar="LEVEL",
            default="DEBUG", 
            config='true') 

