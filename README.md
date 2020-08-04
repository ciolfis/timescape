# timescape

Timescape is a simple spatiotemporal modelling algorithm.
It takes as input (the source) a number of records and outputs a 3D voxel lattice (the target).

Author: Marco Ciolfi - marco.ciolfi@cnr.it - https://www.cnr.it/people/marco.ciolfi
The current version is released as a Python3 module

The Timescape algorithm treats the time as a third spatial dimension, with the addition of a causal constraint.

The source should be a plain ascii file containing the model's parameters and input values.

The target consists in a binary object: a 'tsm' or TimeScapeModel, that can be manipulated through a set of dedicated methods.
