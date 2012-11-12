
To build documentation:

PYTHONPATH=~/repos/estrangement make html

The docs are stored in ../ecdocs which is on a separate branch gh-pages so that
it can be served directly.

So to update the docs we need a way to commit to gh-pages branch the docs built
using master. Write now this is cumbersome, but there should be an easier way.


