# setup for estrangement

from distutils.core import setup
setup(
    name = "estrangement",
    packages = ["estrangement", "estrangement.tests","bin"],
    version = "0.1.0",
    description = "Temporal Community Detection and Plotting",
    author = ["Vikas Kawadia, Sameet Sreenivasan, Stephen Dabideen"],
    author_email = "vkawadia@bbn.com",
    url = "https://github.com/kawadia/estrangement",
    keywords = ["graph","community","temporal","estrangement"],
    classifiers = ["Development Status :: 5 - Production/Stable",
		   "Operating System :: OS Independent",
		   "Programming Language :: Python :: 2.7",
		   "Topic :: Scientific/Engineering :: Mathematics",
		   "Topic :: Scientific/Engineering :: Visualization",
		   "Topic :: Software Development :: Libraries :: Python Modules"], 
    long_description = """\
Temporal Community Detection using estrangement Confinement.

See reference: 
[1] V. Kawadia and S. Sreenivasan, "Online detection of temporal communities in evolving networks by 
                                    estrangement confinement", http://arxiv.org/abs/1203.5126 

"""
)
