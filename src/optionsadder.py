import os


def add_options(parser):
    """define all the program options here"""
    parser.add_argument("-d", "--dataset_dir",
            dest="dataset_dir", 
            help="dir where all the datasets are stored [default: %default]",
            type=str,
            default=os.path.join(os.path.expanduser("~"), "datasets")
    )


    parser.add_argument("-e", "--precedence_tiebreaking",
            dest="precedence_tiebreaking", 
            help="turn on precedence tiebreaking which keeps a node's current label if it is one of the dominant lables. [default: %default]",
            action="store_true",
            default=False
    )

 
    parser.add_argument("--tolerance",
            dest="tolerance", 
            help="tolerance as a fraction of the max when picking dominant labels [default: %default]",
            type=float,
            default=0.00001
    )
    
    parser.add_argument("--convergence_tolerance",
            dest="convergence_tolerance", 
            help="lambda (X) tolerance for scipy optimize [default: %default]",
            type=float,
            default=0.01
    )

    parser.add_argument("--delta",
            dest="delta", 
            help="constraint on estrangement [default: %default]",
            type=float,
            default=0.05
    )
    
    
    parser.add_argument("--minrepeats",
            dest="minrepeats", 
            help="min number of repeats for each snapshot [default: %default]",
            type=int,
            default=10
    )

    parser.add_argument("--increpeats",
            dest="increpeats", 
            help="number of repeats in increased by this amount for every call to g_of_lambda [default: %default]",
            type=int,
            default=10
    )


    parser.add_argument("--savefor_layouts",
            dest="savefor_layouts",
            type=bool,
            help="Save gexf for layouts [default: %default]",
            default=False
    )

