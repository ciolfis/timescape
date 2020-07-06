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

The source should be an ascii file formatted as follows:
  #any comment in the header section follows a # sign
  #model parameters - these are the model parameters
  ALGORITHM=KRIG, METRIC=EUCLID
  C=1.5, K=1.0, NEIGH=0
  #bulk bounds - these are the target bounds and number of cells in time, x and y coordinates
  NT=64, MINT=0.0, MAXT=80.0
  NX=128, MINX=0, MAXX=144.01
  NY=128, MINY=0.0, MAXY=122.59
  #source events
  ID,T,X,Y,VAL #this header string is mandatory, the format is LABEL, time, x-coord, y-coord, value
  SAMPLE_LABEL,11.11,22.22,33.33,99.99
    ...
    ...

The target consists in a binary object: a 'tsm' or TimeScapeModel, that can be manipulated through a set of dedicated methods.

The Timescape distribution comes with a set of four sample datasets:
  d2H: World deuterium vs 1H abundance, 1975 to 1984, from IAEA GNIP - https://nucleus.iaea.org/Pages/GNIPR.aspx
  d15N: Fungi 15N vs 14N isotopic abundance, from CNR-IRET - http://www.iret.cnr.it
  d18O: Olive oil 18O vs 16O isotopic abundance, from CNR-IRET - http://www.iret.cnr.it - https://doi.org/10.1016/j.foodchem.2016.01.146
  tmin: Umbria (central Italy) minumum temperatures by month, 1980 to 1999, https://servizioidrografico.regione.umbria.it - 
each sample dataset consists in a descriptive pdf file (example.pdf), the ascii source input file (example_source.txt) and an R object containing the source and the target as SpatialPointsDataFrame object, and the spatial reference object (example.RData).
