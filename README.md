# timescape

Timescape is a simple spatiotemporal modelling algorithm.
It takes as input (the source) a number of records and outputs a 3D voxel lattice (the target).

Author: Marco Ciolfi - marco.ciolfi@cnr.it - https://www.cnr.it/people/marco.ciolfi
The current version is released as a Python3 module

The Timescape algorithm, in a nutshell, treats the time as a third spatial dimension, with the addition of a causal constraint.

Full details can be found in the following articles:
- Ciolfi M, Chiocchini F, Mattioni M, Lauteri M: TimescapeGlobal spacetime interpolation tool - geographic coordinates Java standalone application, Smart eLab vol.11 (2018), ISSN 2282-2259, doi:0.30441/smart-elab.v11i0.202,
- Ciolfi M, Chiocchini F, Mattioni M, Lauteri M: TimescapeLocal spacetime interpolation tool - projected coordinates Java standalone application, Smart eLab vol.10 (2017), ISSN 2282-2259, doi:0.30441/smart-elab.v10i0.201.
(these are referred to the Java implementation of the Timescape algoritm)

The source should be a plain ascii file containing the model's parameters and input values.

The target consists in a binary object: a 'tsm' or TimeScapeModel, that can be manipulated through a set of dedicated methods.
