

def add_options(parser):
    """define all the program options here"""
    parser.add_option("-x", "--xfigsize",
            dest="xfigsize", 
            help="fig width in inches [default: %default]",
            type="float",
            default=16.0,
            config='true'
    )

    parser.add_option("-y", "--yfigsize",
            dest="yfigsize", 
            help="fig height in inches [default: %default]",
            type="float",
            default=12.0,
            config='true'
    )

    parser.add_option("--tiled_figsize",
            dest="tiled_figsize", 
            help="tuple of figsize for the tiled plots [default: %default]",
            type="string",
            default='(36,16)',
            config='true'
    )


    parser.add_option("-m", "--markersize",
            dest="markersize", 
            help="marker size in points [default: %default]",
            type="float",
            default=15,
            config='true'
    )

    parser.add_option("--markerheight",
            dest="markerheight", 
            help="marker height as a fraction of markersize [default: %default]",
            type="float",
            default=0.2,
            config='true'
    )

    parser.add_option("--xtick_separation",
            dest="xtick_separation", 
            help="separation of labels on the x axis [default: %default]",
            type="int",
            default=5,
            config='true'
    )


    parser.add_option("--linewidth",
            dest="linewidth", 
            help="linewidth in pts [default: %default]",
            type="float",
            default=2.0,
            config='true'
    )


    parser.add_option("--seed",
            dest="seed", 
            help="seed for randomizing label indices [default: %default]",
            type="int",
            default=1378389,
            config='true'
    )
    
    parser.add_option("--label_sorting_keyfunc",
            dest="label_sorting_keyfunc", 
            help="keyfunc for sorting label indices so that patches dont get nearby colors in the the colormap [default: %default]. Identity implies sorting by label values. Also see the seed option above.",
            type="str",
            default="random",
            config='true'
    )

    parser.add_option("--fontsize",
            dest="fontsize", 
            help="fontsize for figure text [default: %default]",
            type="float",
            default=28,
            config='true'
    )

    parser.add_option("--label_fontsize",
            dest="label_fontsize", 
            help="fontsize for axis labels [default: %default]",
            type="float",
            default=20,
            config='true'
    )


    parser.add_option("--alpha",
            dest="alpha", 
            help="figure transparency [default: %default]",
            type="float",
            default=1.0,
            config='true'
    )


    parser.add_option("--dpi",
            dest="dpi", 
            help="dpi for savefig [default: %default]",
            type="int",
            default=200,
            config='true'
    )

    parser.add_option("--wspace",
            dest="wspace", 
            help="whitespace between subplots [default: %default]",
            type="float",
            default=0.2,
            config='true'
    )
    
    parser.add_option("--bottom",
            dest="bottom", 
            help="whitesapce below subplots [default: %default]",
            type="float",
            default=0.1,
            config='true'
    )

    parser.add_option("--frameon",
            dest="frameon", 
            help="set frameon on each axis True or False [default: %default]",
            action="store_false",
            default=True,
            config='true'
    )

    parser.add_option("--delta_to_use_for_node_ordering",
            dest="delta_to_use_for_node_ordering", 
            help="delta value to use for node ordering on the tiles plots [default: %default]",
            type="float",
            default=1.0,
            config='true'
    )
    
    parser.add_option("--deltas_to_plot",
            dest="deltas_to_plot", 
            help="deltas_to_plot in the dynconsuper plot [default: %default]",
            type="str",
            default="[]",
            config='true'
    )

    parser.add_option("--manual_colormap",
            dest="manual_colormap", 
            help="dict of label_indices to colors to use for the patches in the dynconsuper plot [default: %default]",
            type="str",
            default=None,
            config='true'
    )


    parser.add_option("--title",
            dest="title", 
            help="title text [default: %default]",
            type="string",
            default="Dynamic communities",
            config='true'
    )

    parser.add_option("--xlabel",
            dest="xlabel", 
            help="xlabel [default: %default]",
            type="string",
            default="Time",
            config='true'
    )

    parser.add_option("--ylabel",
            dest="ylabel", 
            help="ylabel [default: %default]",
            type="string",
            default="Node id",
            config='true'
    )

    parser.add_option("--label_cmap",
            dest="label_cmap", 
            help="color map for communities [default: %default]",
            type="string",
            default='Paired',
            config='true'
    )


    parser.add_option("-d", "--display_on",
            dest="display_on", 
            help="display label plot after each task is done [default: %default]",
            action="store_true",
            default=False,
            config='true'
    )
    
    parser.add_option("--mayavi",
            dest="mayavi", 
            help="use mayavi to do a 3d plot [default: %default]",
            action="store_true",
            default=False,
            config='true'
    )

    parser.add_option("--show_title",
            dest="show_title", 
            help="show title on plots [default: %default]",
            action="store_true",
            default=True,
            config='true'
    )


    parser.add_option("--use_separate_label_indices",
            dest="use_separate_label_indices", 
            help="set to true if labels are not from the space of nodenames [default: %default]",
            action="store_true",
            default=False,
            config='true'
    )

    parser.add_option("--colorbar",
            dest="colorbar", 
            help="display colorbar if true [default: %default]",
            action="store_true",
            default=True,
            config='true'
    )
    
    parser.add_option("--show_yticklabels",
            dest="show_yticklabels", 
            help="display yticklabels if true [default: %default]",
            action="store_true",
            default=False,
            config='true'
    )

    
    parser.add_option("--label_with_country_names",
            dest="label_with_country_names", 
            help="use COW copuntry names for nodenames [default: %default]",
            action="store_true",
            default=False,
            config='true'
    )

    parser.add_option("--loglevel",
            dest="loglevel",
            type="choice",
            choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], 
            help="Set console logging level to LEVEL [default: %default]",
            metavar="LEVEL",
            default="INFO", 
            config='true') 
    
    parser.add_option("--node_indexing",
            dest="node_indexing", 
            help="node indexing scheme [default: %default]",
            type="string",
            default='fancy',
            config='true'
    )

    parser.add_option("--nodeorder",
            dest="nodeorder", 
            help="psecify node order for tiled plots [default: %default]",
            type="string",
            default=None,
            config='true'
    )



    parser.add_option("--nodelabel_func",
            dest="nodelabel_func", 
            help="func which provides node names to display, it should return a dict from nodes in the graph to node name strings to display[default: %default]",
            type="string",
            default=None,
            config='true'
    )

    parser.add_option("--nodes_of_interest",
            dest="nodes_of_interest", 
            help="only show labels ever taken by the node in this list, can be specified multiple times[default: %default]",
            type="string",
            action="append",
            #default='[]',
            config='true'
    )


    parser.add_option("--snaphotlabel_func",
            dest="snapshotlabel_func", 
            help="func which provides snapshot names to display, it should return a dict from nodes in the graph to snapshot name strings to display[default: %default]",
            type="string",
            default=None,
            config='true'
    )

    parser.add_option("--image_extension",
            dest="image_extension", 
            help="image_extension [default: %default]",
            type="string",
            default="svg",
            config='true'
    )

    parser.add_option("--partition_file",
            dest="partition_file", 
            help="name of the file from which to read matched partitions (simplified_matched_labels.log is max weight matching and matched_labels.log is max mutual overlap) [default: %default]",
            type="string",
            default="matched_labels.log",
            config='true'
    )

    parser.add_option("--profiler_on",
        dest="profiler_on", 
        help="turn profiling using cProfile on [default: %default], saves output to profile_output_filename",
        action="store_true",
        default=True,
        config='true')
