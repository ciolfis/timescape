"""
   Timescape module

   Credits & license
   
     Coding: Marco Ciolfi - marco.ciolfi@cnr.it
     
     CNR-IRET, via Marconi 2, Porano 05010 IT
     
     https://www.cnr.it/people/marco.ciolfi
     
   GitHub https://github.com/ciolfis/timescape
     
   Rev. 1.1 - Mar.2020 - GNU-GPLv3 license

   needs python 3.7
"""

__author__ = 'Marco Ciolfi'
__email__ = 'marco.ciolfi@cnr.it'
__version__ = '1.1'
__license__ = 'GPL'


import os, sys, math
import numpy as np
from copy import copy
from osgeo import gdal
import pykrige as PyKrige
import statsmodels.api as sm
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from datetime import datetime
from dataclasses import dataclass
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7, 'axes.linewidth':.5})
from mpl_toolkits import mplot3d #noqa: F401 unused import
from mpl_toolkits.mplot3d import Axes3D #noqa: F401 unused import
try: import cPickle as pickle
except ModuleNotFoundError: import pickle
import bz2

_SEPARATOR = ','
_DATE_FORMAT = '%Y%m%d:%H:%M:%S'
_METRICS = {'EUCLID':0, 'DIAMOND':1, 'SQUARE':2, 'SPHERE':-1}
_METRIC_KEY = 'METRIC'
_METRIC_DEFAULT = 'EUCLID' #default meric
_ALG_DEFAULT = 'KRIG' #default algorithm (method of TimescapeEvaluator)
_NEIGH_DEFAULT = 0
_RADIUS_DEFAULT = 6378100.0 #avg Earth radius in m
_EVENTS_LIST_FIELDS = 'ID,T,X,Y,VAL'
_HEADER_STRING = '---------- START ----------'
_FOOTER_STRING = '----------- END -----------'
_OUT_COMM = '#---------------------------------\n'
_OUT_FIELDS = ('K', 'I', 'J', 'T', 'X', 'Y', 'VAL', 'STDEV', 'NEIGH')
_DEG2RAD = math.pi / 180.0 #supposing input angles are in deg
_REM_START = '#' #skip input file comment lines
_MYPAR_PREFIX = 'MYPAR_' #user-defined params prefix
_TIFF_PROJ = '' #tiff proj string
_TIFF_NODATA = -9999 #tiff nodata value
_TIFF_VAL_SUFFIX = '_val.tiff' #tiff values suffix
_TIFF_ACC_SUFFIX = '_acc.tiff' #tiff accuracy suffix
_TIFF_NUM_SUFFIX = '_num.tiff' #tiff neighborhood suffix
_VAR_BINS = 11 #variogram bins - 1

FILEPATH_INP = './ts_in.txt' #default input file
FILEPATH_TXT = './ts_out.txt' #default output txt file
FILEPATH_TSM = './ts_out.tsm' #default output tsm file
FILEPATH_TIF = './ts_out' #default output tiff files prefix
FILEPATH_RES = './ts_res.txt' #default residuals file
FILEPATH_TND = './ts_trend.txt' #default trend file
FILEPATH_TME = './ts_time_series.txt' #default trend file
FILEPATH_LOG = './timescape.log' #log file
LOG_TO_FILE = True #log timescape() run
PLOT_COLORS = 'plasma' #values color ramp
PLOT_ALPHA = .5 #plot transparency
PLOT_DPI = 150 #images resoluition (dpi)
ROUND_FLOATS = True
ROUND_DIGITS = 4
FIELD_T = 'T' #time field
FIELD_X = 'X' #x-coord field
FIELD_Y = 'Y' #y-coord field
FIELD_VAL = 'VAL' #value field
FIELD_ACC = 'ACC' #accuracy field
FIELD_NUM = 'NUM' #neighborhood field


@dataclass
class Model: #Timescape model
    """
    The class of a Timescape model.
    
    Attributes
    ----------
      
    source: [Event], the source events
      
    target: [[[Voxel]]], the target events (i.e. the bulk)
      
    params: {}, the model parameters
        
        ALGORITHM: str, default KRIG, the interpolator method, one of KRIG, IDW, SIDW or a user-defined method
        
        NEIGH: int, default 0 (all), tne maximum number of neighborhoods
        
        METRIC: str, default EUCLID, the spatial metric function, one of
        
            EUCLID = 0, the Euclidean distance
            
            DIAMOND = 1, the diamond metric d = x + y
            
            SQUARE = 2, the square metric d = max{x, y}
            
            SPHERE = -1, the geodesic arc length on the sphere
        
        RADIUS: float, default Earth radus, the radius of the SPHERE metric
        
        C: float, the Timescape c parameter
        
        K: float, the Timescape k parameter
        
        KPERIOD: float, optional, the causal cone period
        
        NT: int, the number of bulk sheets
        
        NX: int, the number of bulk rows
        
        NY: int, the number of bulk columns
        
        MINT, MAXT: float, the bulk time bounds
        
        MINX, MAXX: float, the bulk x (rows) bounds
        
        MINY, MAXY: float, the bulk y (columns) bounds
        
        MYPAR_*: str, any user-defined parameter
      
    straight: bool, the shape of the causal cone, straight / periodic
      
    omega: float, default 0, the frequency of KPERIOD
      
    null_vxl: int, number of null target events
      
    bad_vxl: int, number of bad target events
    
    Methods
    ----------
    time(k:int): the time of the k sheet
      
    place(i:int, j:int): the place of the (i, j) core
      
    coordinates(k:int, i:int, j:int): the centroid of the target voxel of indices (k, i, j)
   
    voxel(t:float, x:float, y:float): the voxel containing (t, x, y)
   
    write_tsm(picklefile=FILEPATH_TSM): saves the model as a pickle binary file
    
    write_txt(txtfile=FILEPATH_TXT): saves the model as a text file (csv and metadata)
    
    write_tiff(tiffilespref=FILEPATH_TIF): saves the model as three tiff files (value, accuracy and neighborhood)
    
    extract_bulk(self, field=FIELD_VAL): extracts the bulk as a numpy array
    
    extract_sheet(self, k, field=FIELD_VAL): extracts a sheet as a numpy array
    
    extract_core(self, i, j, field=FIELD_VAL): extracts a core as a numpy array
    
    time_series(self, i, j, outfile=FILEPATH_TME): writes a time series on a text file
    
    desc(self): prints a model description
    
    show(self, field=FIELD_VAL): shows the bulk values, accuracies or neighborhood as volume isosurfaces
    
    histogram(self, field=FIELD_VAL): plots the values, accuracy or neighborhood histogram
    
    plot_bulk(self): plot_bulk(self): plots the bulk as a 3d scatterplot
    
    plot_sheet(self, k): plot_sheet(self, k): plots a sheet as points and contour lines
    
    plot_core(self, i, j): plot_core(self, i, j): plots a core as a time series
    """
    def __init__(self):
        self.params = {'ALGORITHM':_ALG_DEFAULT, 'NEIGH':_NEIGH_DEFAULT, 
              'METRIC':_METRICS[_METRIC_DEFAULT], 'RADIUS':_RADIUS_DEFAULT,
              'C':None, 'K':None, 'KPERIOD':None, 
              'NT':None, 'MINT':None, 'MAXT':None, 
              'NX':None, 'MINX':None, 'MAXX':None, 
              'NY':None, 'MINY':None, 'MAXY':None}
        self.source = []
        self.target = None #target events as [[[Voxel]]]
        self.null_vxl = 0 #number of null target events
        self.bad_vxl = 0 #number of bad target events (evaluation errors)
        self.straight = True #straight or periodic causal cone
        self.omega = None #pi * frequency of the causal cone
    
    def time(self, k:int):
        """
        Finds the time of the k sheet

        Parameters
        ----------
        k: int, sheet index

        Raises
        ------
        TFEx if out range

        Yields
        -------
        t: float, time
        """
        if k<0 or k>= self.params.get('NT'): raise TFEx('Out of range k={}'.format(k))
        return _clean_float(_index2coord(k, self.params.get('NT'), self.params.get('MINT'), self.params.get('MAXT')))

    def place(self, i:int, j:int):
        """
        Finds the place of the (i, j) core

        Parameters
        ----------
        i: int, row index
        
        j: int, column index

        Raises
        ------
        TFEx if out range

        Yields
        -------
        x: float, x-coordinate

        y: float, y-coordinate
        """
        if i<0 or i>= self.params.get('NX'): raise TFEx('Out of range i={}'.format(i))
        if j<0 or j>= self.params.get('NY'): raise TFEx('Out of range j={}'.format(j))
        x = _clean_float(_index2coord(i, self.params.get('NX'), self.params.get('MINX'), self.params.get('MAXX')))
        y = _clean_float(_index2coord(j, self.params.get('NY'), self.params.get('MINY'), self.params.get('MAXY')))
        return x, y

    def coordinates(self, k:int, i:int, j:int):
        """
        Finds the centroid of the target voxel of indices (k, i, j)

        Parameters
        ----------
        k: int, sheet index
        
        i: int, row index
        
        j: int, column index

        Raises
        ------
        TFEx if out range

        Yields
        -------
        t: float, time

        x: float, x-coordinate

        y: float, y-coordinate
        """
        if k<0 or k>= self.params.get('NT'): raise TFEx('Out of range k={}'.format(k))
        if i<0 or i>= self.params.get('NX'): raise TFEx('Out of range i={}'.format(i))
        if j<0 or j>= self.params.get('NY'): raise TFEx('Out of range j={}'.format(j))
        t = _clean_float(_index2coord(k, self.params.get('NT'), self.params.get('MINT'), self.params.get('MAXT')))
        x = _clean_float(_index2coord(i, self.params.get('NX'), self.params.get('MINX'), self.params.get('MAXX')))
        y = _clean_float(_index2coord(j, self.params.get('NY'), self.params.get('MINY'), self.params.get('MAXY')))
        return t, x, y

    def voxel(self, t:float, x:float, y:float):
        """
        Returns voxel contaning the (t, x, y) coordinates

        Parameters
        ----------
        t: float, time
        
        x: float, x-coordinate
        
        y: float, y-coordinate

        Raises
        ------
        TFEx if out range

        Yields
        -------
        The voxel (t, x, y) belong to 
        """
        if t < self.params.get('MINT') or t > self.params.get('MAXT'): raise TFEx('Out of range t={}'.format(t))
        if x < self.params.get('MINX') or x > self.params.get('MAXX'): raise TFEx('Out of range x={}'.format(x))
        if y < self.params.get('MINY') or y > self.params.get('MAXY'): raise TFEx('Out of range y={}'.format(y))
        k = int(self.params.get('NT') * (t - self.params.get('MINT')) / (self.params.get('MAXT') - self.params.get('MINT')))
        i = int(self.params.get('NX') * (x - self.params.get('MINX')) / (self.params.get('MAXX') - self.params.get('MINX')))
        j = int(self.params.get('NY') * (y - self.params.get('MINY')) / (self.params.get('MAXY') - self.params.get('MINY')))
        k, i, j = min(k, self.params.get('NT')-1), min(i, self.params.get('NX')-1), min(j, self.params.get('NY')-1)
        return self.target[k][i][j]

    def write_tsm(self, picklefile=FILEPATH_TSM):
        """ calls write_tsm(self, picklefile) """
        write_tsm(self, picklefile)

    def write_txt(self, txtfile=FILEPATH_TXT):
        """ calls write_txt(self, txtfile) """
        write_txt(self, txtfile)

    def write_tiff(self, tiffilespref=FILEPATH_TIF):
        """ calls write_tiff(self, tiffilespref) """
        write_tiff(self, tiffilespref)

    def extract_bulk(self, field=FIELD_VAL):
        """ calls extract_bulk(self, field) """
        return extract_bulk(self, field)

    def extract_sheet(self, k, field=FIELD_VAL):
        """ calls extract_sheet(self, k, field) """
        return extract_sheet(self, k, field)

    def extract_core(self, i, j, field=FIELD_VAL):
        """ calls extract_core(self, i, j, field) """
        return extract_core(self, i, j, field)

    def time_series(self, i, j, outfile=FILEPATH_TME):
        """ calls time_series(self, i, j, outfile) """
        time_series(self, i, j, outfile)

    def plot_bulk(self):
        """ calls plot_bulk(self, field) """
        plot_bulk(self)

    def plot_sheet(self, k):
        """ calls plot_sheet(self, k, field) """
        plot_sheet(self, k)

    def plot_core(self, i, j):
        """ calls plot_core(self, i, j, field) """
        plot_core(self, i, j)

    def show(self, field=FIELD_VAL):
        """ calls target_show(self, field) """
        target_show(self, field)

    def desc(self):
        """ calls describe_model(self) """
        describe_model(self)

    def histogram(self, field=FIELD_VAL):
        """ calls target_histogram(self, field) """
        target_histogram(self, field)

class TEvalEx(Exception):
    """ The Timescape evaluation exception, not blocking. """
    def __init__(self, ex):
        super().__init__('{} @line_{}'.format(ex, sys.exc_info()[-1].tb_lineno))

class TFEx(Exception):
    """ The Timescape Fatal exception. """
    def __init__(self, ex):
        if type(ex) is str: super().__init__('{}'.format(ex))
        else: super().__init__('{} @line_{}'.format(ex, sys.exc_info()[-1].tb_lineno))

@dataclass
class Event:
    """ a source event """
    label: str = None
    t: float = None
    ct: float = None
    x: float = None
    y: float = None
    val: float = None

@dataclass
class EventWrapper:
    """ a source event with distance to target, ordered by distance """
    def __init__(self, event, dist):
        self.t = event.t
        self.ct = event.ct
        self.x = event.x
        self.y = event.y
        self.val = event.val
        self.dist = dist
    def __lt__(self, other): return self.dist < other.dist

@dataclass
class Voxel:
    """ a target event """
    k: int = None
    i: int = None
    j: int = None
    t: float = None
    ct: float = None
    x: float = None
    y: float = None
    val: float = None
    stdev: float = None
    neigh: int = None
    bad: bool = False
    def label(self): return 'T{}-X{}-Y{}{}'.format(self.k, self.i, self.j, '-BAD' if self.bad else '')

@dataclass
class _Residual:
    c: float = None
    k: float = None
    sqres: float = None
    respevt: float = None
    null: int = 0
    bad: int = 0
    thput: float = 0.
    
@dataclass
class _ResEnsemble:
    numc: int = None
    cmin: float = None
    cmax: float = None
    numk: int = None
    kmin: float = None
    kmax: float = None
    residuals: [] = None

@dataclass
class _ResVoxel:
    label: str = None
    t: float = None
    ct: float = None
    x: float = None
    y: float = None
    exact: float = None
    val: float = None
    stdev: float = None
    neigh: int = 0
    bad: bool = False

# -----------------------------------------------------------------
    
@dataclass
class TimescapeEvaluator:
    """
    The interpolators class, one method per interpolator.
    
    Attributes
    ----------
      
    mod: Model, the Timescape model
    
    ews: [EventWrapper], the causally connected source events with respect to vox
    
    vox: Voxel, the target event to be interpolated
    
    Methods
    -------
      
    KRIG: A simple 3d universal Kriging based on PyKrige
    
    IDW: The sharp Inverse Distance Weighting

    SIDW: A smoother Inverse Distance Weighting

    MYINTERP: A skeleton method
    """
    mod: Model = None #timescape model
    ews: [] = None #source events as [EventWrapper]
    vox: Voxel = None #target event as Voxel

    def KRIG(self):
        """
        A simple 3d universal Kriging based on PyKrige
        
        Reference https://github.com/bsmurphy/PyKrige

        Raises
        ------
        TEvalEx if the target event evluation fails

        Yields
        -------
        val: float, the event value
        
        acc: float, the event accuracy
        """
        try:
            if len(self.ews) < 3: return None, None #too few source events
            ct = self.vox.ct #target event time in length units
            x, y = self.vox.x, self.vox.y #target event x, y
            ctt = [ew.ct for ew in self.ews] #source events times
            xx = [ew.x for ew in self.ews] #source events xs
            yy = [ew.y for ew in self.ews] #source events ys
            vv = [ew.val for ew in self.ews] #source events values
            UK3D = PyKrige.UniversalKriging3D(xx, yy, ctt, vv)
            values, variances = UK3D.execute('points', x, y, ct)
            return values[0], math.sqrt(variances[0])
        except Exception as ex: raise TEvalEx(ex)

    def IDW(self):
        """
        The sharp Inverse Distance Weighting

        The weight function is 1/d

        Raises
        ------
        TEvalEx if the target event evluation fails

        Yields
        -------
        val: float, the event value
        
        acc: float, the event accuracy, None unless distance = 0
        """
        try:
            norm, value = 0.0, 0.0
            for ew in self.ews:
                if ew.dist == 0.0: return ew.val, 0.0
                weight = 1.0 / ew.dist
                norm += weight
                value += weight * ew.val
            return (value / norm), None
        except Exception as ex: raise TEvalEx(ex)

    def SIDW(self):
        """
        A smoother Inverse Distance Weighting
        
        The weight function is 1/(d^2 + m^2)
        
        where m^2 is the MYPAR_SIDW_SQMASS user-defined parameter

        Raises
        ------
        TEvalEx if the target event evluation fails

        Yields
        -------
        val: float, the event value
        
        acc: None, cannot estimate the event accuracy
        """
        try:
            sq_mass = self.mod.params.get('MYPAR_SIDW_SQMASS', 1.0)
            norm, value = 0.0, 0.0
            for ew in self.ews:
                weight = 1.0 / (ew.dist * ew.dist + sq_mass)
                norm += weight
                value += weight * ew.val
            return (value / norm), None
        except Exception as ex: raise TEvalEx(ex)

    def MYINTERP(self):
        """
        A mock interpolator - use as a skeleton for adding new interpolators
        
        Keep everything enclosed in the try-except construct to log the exception and to mark the event as BAD
        
        Use _logline('{} {}'.format(self.vox, 'text_to_log')) to log the 'text_to_log' string

        Raises
        ------
        TEvalEx if the target event evluation fails

        Yields
        -------
        val: float, the event value
        
        acc: float, the event accuracy
        """
        try:
            foo, bar = 123.45, 6.78
            #logging a message
            _logline('{} {}'.format(self.vox, 'text_to_log'))
            #your favourite code goes here
            the_value, the_stdev = foo, bar
            return the_value, the_stdev
        except Exception as ex: raise TEvalEx(ex)
        
# -----------------------------------------------------------------


def build_model(source=FILEPATH_INP):
    """
    Parameters
    ----------
       source: str, the source file. The default is FILEPATH_INP
       
    Yields
    ------
       mod: Model - the finished Timescape model
    """
    timestamp = datetime.now().timestamp()
    _logline(_HEADER_STRING, True)
    mod = Model()
    try:
        _read_input(mod, source)
        _eval_model(mod)
        delta_timestamp = round(datetime.now().timestamp() - timestamp, 2)
        thput = round((mod.params.get('NT') * mod.params.get('NX') * mod.params.get('NY')) / delta_timestamp, 2)
        _logline('Timescape finished in {} seconds\nThroughput = {} events / s'.format(delta_timestamp, thput), True)
        return mod
    except TFEx as tfex:
        _logline('Fatal error: {}'.format(tfex), True)
    finally:
        _logline(_FOOTER_STRING, True)

def _empty_voxel(mod, k, i, j):
    vox = Voxel()
    vox.k, vox.i, vox.j = k, i, j
    vox.t = _index2coord(k, mod.params.get('NT'), mod.params.get('MINT'), mod.params.get('MAXT'))
    vox.ct = mod.params.get('C') * vox.t
    vox.x = _index2coord(i, mod.params.get('NX'), mod.params.get('MINX'), mod.params.get('MAXX'))
    vox.y = _index2coord(j, mod.params.get('NY'), mod.params.get('MINY'), mod.params.get('MAXY'))
    return vox

def _index2coord(k: int, n: int, low: float, hi: float): return low + (hi - low) * (0.5 + k) / n

def _eval_model(mod):
    nt, nx, ny = mod.params.get('NT'), mod.params.get('NX'), mod.params.get('NY')
    bulk_size = nt * nx * ny
    try: mod.target = [[[_empty_voxel(mod,k,i,j) for j in range(ny)] for i in range(nx)] for k in range(nt)]
    except Exception as ex: raise TFEx('Model allocation error: {}'.format(ex))
    _logline('Bulk space allocated {} bytes'.format(format(bulk_size * sys.getsizeof(Voxel), ',d')), True)
    _progbar('Timescaping', 0, bulk_size)
    sheets_done = 0
    for sheet in mod.target:
        rows_done = 0
        for row in sheet:
            for voxel in row:
                try: _eval_voxel(mod, voxel)
                except TEvalEx as evalex:
                    _logline('{} evaluation error: {}'.format(voxel.label(), evalex))
                    voxel.bad = True #marks the target event as BAD
                    mod.bad_vxl += 1 #not blocking, damage limited to the single event
                if voxel.val == None: mod.null_vxl += 1
            rows_done += 1
            _progbar('Timescaping', sheets_done * nx * ny + rows_done * ny, bulk_size)
        sheets_done += 1
    _logline('Null events count is {} out of {} ({}%)'.format(mod.null_vxl, bulk_size, int(100.0 * mod.null_vxl / bulk_size)), True)
    if mod.bad_vxl: _logline('Warning: {} errors occurred, events marked as -BAD'.format(mod.bad_vxl), True)
    else: _logline('Zero errors occurred in target evaluation', True)

def _eval_voxel(mod, voxel):
    try:
        causes = [EventWrapper(event, _distance(mod, event, voxel)) for event in mod.source] #instances tentative causes for each target
        causes[:] = [cause for cause in causes if cause.dist is not None] #removes null distances (no causal connection)
        if len(causes) > 0:
            causes.sort() #sorts source events by distance
            if not mod.params.get('NEIGH') == _NEIGH_DEFAULT:
                causes = causes[:min(len(causes), mod.params.get('NEIGH'))] #prunes neighborhood
            if len(causes) > 0:
                voxel.neigh = len(causes)
                voxel_evaluator = TimescapeEvaluator(mod, causes, voxel) #instances TimescapeEvaluator
                voxel.val, voxel.stdev = getattr(voxel_evaluator, mod.params.get('ALGORITHM'))() #calls algorithms metod
    except TEvalEx: raise
    except Exception as ex: raise TEvalEx(ex)

def _distance(mod, event, voxel):
    try:
        deltat = voxel.ct - event.ct
        if deltat < 0.0: return None
        psi = 1.0 if mod.omega==None else math.cos(mod.omega * deltat)**2
        deltax, deltay, deltas = math.fabs(voxel.x - event.x), math.fabs(voxel.y - event.y), None
        if mod.params.get(_METRIC_KEY) == _METRICS['EUCLID']: deltas = math.sqrt(deltax * deltax + deltay * deltay)
        elif mod.params.get(_METRIC_KEY) == _METRICS['DIAMOND']: deltas = deltax + deltay
        elif mod.params.get(_METRIC_KEY) == _METRICS['SQUARE']: deltas = max(deltax, deltay)
        elif mod.params.get(_METRIC_KEY) == _METRICS['SPHERE']:
            phi1, lambda1 = _DEG2RAD * voxel.y, _DEG2RAD * voxel.x
            phi2, lambda2 = _DEG2RAD * event.y, _DEG2RAD * event.x
            arg = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(lambda1 - lambda2)
            deltas = mod.params.get('RADIUS') * math.acos(round(arg, 8))
        if deltas > psi * mod.params.get('K') * deltat: return None
        return math.sqrt(deltat * deltat + deltas * deltas)
    except Exception as ex: raise TEvalEx(ex)


def _read_input(mod, src):
    if not os.path.isfile(src): raise TFEx('file {} does not exist'.format(src))
    keys = set(mod.params.keys())
    event_cnt, event_parsing = 0, False
    with open(src, 'r') as inp:
        _logline('Reading {}...'.format(src))
        for cnt, line in enumerate(inp):
            try:
                strip = line.strip().upper().replace('\t', '').replace(' ', '')
                if strip == '' or strip.startswith(_REM_START): continue
                if event_parsing:
                    _parse_event_line(mod, strip)
                    event_cnt += 1
                else:
                    if strip == _EVENTS_LIST_FIELDS:
                        _check_params(mod, keys)
                        event_parsing = True
                    else: _parse_params_line(mod, strip)
            except Exception as ex: raise TFEx('{} in input line {} \'{}\''.format(ex, cnt + 1, line))
    _logline('Input file parsed, {} source events found'.format(event_cnt), True)
    _logline('Model params checked: {}'.format(mod.params))
    
def _check_params(mod, keys):
    if mod.params.keys() != keys:
        bad_keys = [key for key in set(mod.params.keys()).difference(keys) if not key.startswith(_MYPAR_PREFIX)]
        if bad_keys: raise TFEx('Illegal parameters found {}'.format(bad_keys))
    if not hasattr(TimescapeEvaluator(), mod.params.get('ALGORITHM')):
        raise TFEx('Algorithm {} does not exist'.format(mod.params.get('ALGORITHM')))
    _cast_int(mod, 'NEIGH', False, True)
    if mod.params[_METRIC_KEY] == _METRICS['SPHERE']: _cast_float(mod, 'RADIUS', True, True)
    else: mod.params.pop('RADIUS', None)
    _cast_float(mod, 'C', False, True)
    _cast_float(mod, 'K', True, True)
    if mod.params.get('KPERIOD'):
        mod.straight = False
        _cast_float(mod, 'KPERIOD', True, True)
        mod.omega = float(math.pi / (mod.params.get('C') * mod.params.get('KPERIOD')))
    else: mod.params.pop('KPERIOD', None)
    for coord in ('T', 'X', 'Y'):
        _cast_int(mod, 'N' + coord, True, True)
        _cast_float(mod, 'MIN' + coord)
        _cast_float(mod, 'MAX' + coord)
        inf, sup, num = mod.params.get('MIN' + coord), mod.params.get('MAX' + coord), mod.params.get('N' + coord)
        if inf > sup: raise TFEx('Bad {} interval [{},{}]'.format(coord, inf, sup))
        if inf == sup and num > 1: _logline('Warning: {} interval is shrunk to {} and replicated {} times'.format(coord, inf, num), True)

def _parse_event_line(mod, strip):
    chunks = strip.split(_SEPARATOR)
    mod.source.append(Event(str(chunks[0]), float(chunks[1]), mod.params.get('C') * float(chunks[1]), float(chunks[2]), float(chunks[3]), float(chunks[4])))
    _logline('{} added'.format(mod.source[-1]))

def _parse_params_line(mod, strip):
    for chunk in strip.split(_SEPARATOR):
        if chunk == '': continue
        key = chunk.split('=')[0]
        val = chunk.split('=')[1]
        if key == _METRIC_KEY:
            mod.params[key] = _METRICS.get(val)
            if mod.params[key] == None: raise Exception('Unknown metric {}'.format(val))
        else: mod.params[key] = val
        _logline('Param {} set'.format(chunk))

def _cast_int(mod, key, not_zero=False, not_neg=False):
    if mod.params.get(key) == None: raise TFEx('Missing param {}'.format(key))
    try: mod.params[key] = int(mod.params.get(key))
    except: raise TFEx('Invalid {} = {}'. format(key, mod.params.get(key)))
    if not_zero == True and mod.params.get(key) == 0: raise TFEx('{} cannot be zero'. format(key))
    if not_neg == True and mod.params.get(key) < 0: raise TFEx('{} cannot be negative'. format(key))

def _cast_float(mod, key, not_zero=False, not_neg=False):
    if mod.params.get(key) == None: raise TFEx('Missing param {}'.format(key))
    try: mod.params[key] = float(mod.params.get(key))
    except: raise TFEx('Invalid {} = {}'. format(key, mod.params.get(key)))
    if not_zero and mod.params.get(key) == 0.0: raise TFEx('{} cannot be zero'. format(key))
    if not_neg and mod.params.get(key) < 0.0: raise TFEx('{} cannot be negative'. format(key))


def describe_model(mod: Model):
    """
    Prints a textual description of the Timescape model

    Parameters
    ----------
    mod: Model, the model to be described

    Yields
    ------
    str, the descripton

    """
    print(_format_header(mod))


def read_tsm(picklefile=FILEPATH_TSM):
    """
    Parameters
    ----------
    picklefile : str, the saved Timescape pickle object file name. The default is FILEPATH_TSM.

    Raises
    ------
    TFEx

    Yields
    -------
    mod : Model, the Timescape model
    """
    mod = None
    try:
        if not os.path.isfile(picklefile): raise TFEx('Missing {} file'.format(picklefile))
        with bz2.BZ2File(picklefile, 'rb') as inp:
            mod = pickle.load(inp)
    except Exception as ex: raise TFEx('Error reading {}: {}'.format(picklefile, ex))
    finally: return mod

def write_tsm(mod, picklefile=FILEPATH_TSM):
    """
    Saves the Timescape model as a pickle file     

    Parameters
    ----------
    mod : Model, the Timescape model
    
    picklefile : str, the output file name. The default is FILEPATH_TSM.

    Raises
    ------
    TFEx
    """
    try:
        if os.path.isfile(picklefile): print('File {} is being superseded'.format(picklefile))
        with bz2.BZ2File(picklefile, 'wb') as out:
            pickle.dump(mod, out)
    except Exception as ex: raise TFEx('Error writing output: {}'.format(ex))


def write_txt(mod, txtfile=FILEPATH_TXT):
    """
    Saves the Timescape model as a human-readable text file, with a commented metadata header

    Parameters
    ----------
    mod : Model, the Timescape model
    
    txtfile : str, the output file name. The default is FILEPATH_TXT.

    Raises
    ------
    TFEx
    """
    if os.path.isfile(txtfile): print('File {} is being superseded'.format(txtfile))
    try:
        with open(txtfile, 'w') as out:
            out.write(_OUT_COMM)
            out.write(_format_header(mod, True))
            out.write(_OUT_COMM + 'LABEL')
            for field in _OUT_FIELDS: out.write('{}{}'.format(_SEPARATOR, field))
            out.write('\n')
            for sheet in mod.target:
                for row in sheet:
                    for voxel in row:
                        out.write(_format_voxel(voxel))
    except Exception as ex: raise TFEx('Error writing output: {}'.format(ex))

def _format_voxel(voxel):
    record = '{}{}'.format(voxel.label(), _SEPARATOR)
    record += '{}{}'.format(voxel.k, _SEPARATOR)
    record += '{}{}'.format(voxel.i, _SEPARATOR)
    record += '{}{}'.format(voxel.j, _SEPARATOR)
    record += '{}{}'.format(_clean_float(voxel.t), _SEPARATOR)
    record += '{}{}'.format(_clean_float(voxel.x), _SEPARATOR)
    record += '{}{}'.format(_clean_float(voxel.y), _SEPARATOR)
    record += '{}{}'.format(_clean_float(voxel.val), _SEPARATOR)
    record += '{}{}'.format(_clean_float(voxel.stdev), _SEPARATOR)
    record += '{}{}'.format(_clean_int(voxel.neigh) , '\n')
    return record

def _format_header(mod, onfile=False):
    cone = 'straight' if mod.straight else 'periodic'
    period = '' if mod.straight else ', KPERIOD {}'.format(mod.params.get('KPERIOD'))
    radius = ' R={}'.format(mod.params.get('RADIUS')) if mod.params[_METRIC_KEY] == _METRICS['SPHERE'] else ''
    neigh = 'all' if mod.params.get('NEIGH') == 0 else 'nearest {}'.format(mod.params.get('NEIGH'))
    mypar_keys = [key for key in mod.params.keys() if key.startswith(_MYPAR_PREFIX)]
    nt, nx, ny = mod.params.get('NT'), mod.params.get('NX'), mod.params.get('NY')
    mint, maxt = mod.params.get('MINT'), mod.params.get('MAXT')
    minx, maxx = mod.params.get('MINX'), mod.params.get('MAXX')
    miny, maxy = mod.params.get('MINY'), mod.params.get('MAXY')
    sizet, sizex, sizey = (maxt-mint)/nt, (maxx-minx)/nx, (maxy-miny)/ny
    sizect = mod.params.get('C') * sizet
    theta_rad = _clean_float(math.atan(mod.params.get('K')))
    omega_srad = _clean_float(2 * math.pi * (1.0 - math.cos(theta_rad)))
    if onfile: header = '  Output time {}\n'.format(format(datetime.now().strftime(_DATE_FORMAT)))
    else: header = 'Timescape description\n'
    header += '  {} Source events\n'.format(len(mod.source))
    header += '  {} Target events\n'.format(nt * nx * ny)
    header += '  Null target events count is {}\n'.format(mod.null_vxl)
    if mod.bad_vxl: header += '  Warning: {} errors found\n'.format(mod.bad_vxl)
    header += '  Model parameters:\n'
    header += '    Algorithm: {}, Neighborhood: {}\n'.format(mod.params.get('ALGORITHM'), neigh)
    header += '    Metric: {}{}\n'.format(list(_METRICS.keys())[list(_METRICS.values()).index(mod.params[_METRIC_KEY])], radius)
    header += '    Time to space conversion factor C={}\n'.format(mod.params.get('C'))
    header += '    Causal cone is {} with K={}{}\n'.format(cone, mod.params.get('K'), period)
    header += '       tip angle={} rad, Omega={} srad\n'.format(round(2.0 * theta_rad, 4), round(omega_srad, 4))
    header += '       {} coverage is {}% of half-plane\n'.format(('cone' if mod.straight else 'envelope'), int(round(50.0 * omega_srad / math.pi, 0)))
    header += '    T from {} to {} in {} sheets: Tk, k={}...{}\n'.format(mint, maxt, nt, 0, nt -1)
    header += '    X from {} to {} in {} rows: Xi, i={}...{}\n'.format(minx, maxx, nx, 0, nx -1)
    header += '    Y from {} to {} in {} cells: Yj, j={}...{}\n'.format(miny, maxy, ny, 0, ny - 1)
    if mypar_keys:
        header += '    User-defined parameters:\n'
        for key in mypar_keys: header += "       {}={}\n".format(key, mod.params[key])
    header += '  Target events voxel size (each):\n    dT={} (time units) or {} (length units)\n'.format(_clean_float(sizet), _clean_float(sizect))
    header += '    dX={}, dY={}, Area={}\n    Volume={} (length^3 units)\n'.format(_clean_float(sizex), _clean_float(sizey), _clean_float(sizex * sizey), _clean_float(sizect * sizex * sizey))
    return header


def write_tiff(mod, tiffilespref=FILEPATH_TIF):
    """
    Saves the Timescape model as three geotiff files, with the following suffixes:
        
        _val.tiff - the target values (Voxel.val)

        _acc.tiff - the target accuracies (Voxel.stdev)

        _num.tiff - the target neighborhood (Voxel.neigh)
        
    Each geotiff consists in NT layers, one per mod.target's sheet
    
    Each layer is commented with its time value

    Parameters
    ----------
    mod : Model, the Timescape model
    
    tiffilespref : str, the output files prefix. The default is FILEPATH_TIF.

    Raises
    ------
    TFEx
    """
    try:
        events = []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    events.append(voxel)
        nt, nx, ny = int(mod.params.get('NT')), int(mod.params.get('NX')), int(mod.params.get('NY'))
        mint = float(mod.params.get('MINT'))
        ulx, uly = float(mod.params.get('MINX')), float(mod.params.get('MAXY'))
        deltat = (float(mod.params.get('MAXT')) - mint) / float(nt)
        deltax = (float(mod.params.get('MAXX')) - ulx) / float(nx)
        deltay = (uly - float(mod.params.get('MINY'))) / float(ny)
        vv = np.array([event.val for event in events])
        vv = np.where(vv == None, _TIFF_NODATA, vv)
        vv = vv.reshape(nt, nx, ny)    
        aa = np.array([event.stdev for event in events])
        aa = np.where(aa == None, _TIFF_NODATA, aa)
        aa = aa.reshape(nt, nx, ny)    
        nn = np.array([event.neigh for event in events])
        nn = nn.reshape(nt, nx, ny)    
        val_gtiff = gdal.GetDriverByName('GTiff').Create(tiffilespref + _TIFF_VAL_SUFFIX, nx, ny, nt, gdal.GDT_Float32)
        val_gtiff.SetGeoTransform((ulx, deltax, 0, uly, 0, -deltay))
        val_gtiff.SetProjection(_TIFF_PROJ)    
        acc_gtiff = gdal.GetDriverByName('GTiff').Create(tiffilespref + _TIFF_ACC_SUFFIX, nx, ny, nt, gdal.GDT_Float32)
        acc_gtiff.SetGeoTransform((ulx, deltax, 0, uly, 0, -deltay))
        acc_gtiff.SetProjection(_TIFF_PROJ)    
        num_gtiff = gdal.GetDriverByName('GTiff').Create(tiffilespref + _TIFF_NUM_SUFFIX, nx, ny, nt, gdal.GDT_Int16)
        num_gtiff.SetGeoTransform((ulx, deltax, 0, uly, 0, -deltay))
        num_gtiff.SetProjection(_TIFF_PROJ)    
        for k in range(nt):
            _write_layer(val_gtiff, k, vv[k,:,:], nx, ny, mint + float(k) * deltat)
            _write_layer(acc_gtiff, k, aa[k,:,:], nx, ny, mint + float(k) * deltat)
            _write_layer(num_gtiff, k, nn[k,:,:], nx, ny, mint + float(k) * deltat)
        val_gtiff.FlushCache()
        acc_gtiff.FlushCache()
        num_gtiff.FlushCache()
    except Exception as ex: raise TFEx('Error writing {}: {}'.format(tiffilespref, ex))

def _write_layer(gtiff, band, data, nx, ny, time):
    data.reshape(nx, ny)
    data = np.transpose(data)
    data = np.flip(data, 0)
    gtiff.GetRasterBand(band+1).WriteArray(data)
    gtiff.GetRasterBand(band+1).SetDescription('TIME={}'.format(time))
    gtiff.GetRasterBand(band+1).SetNoDataValue(_TIFF_NODATA)


def extract_bulk(mod, field=FIELD_VAL):
    """
    Extracts all the model target as a numpy array

    Parameters
    ----------
    mod: Model, the Timescape model

    field: str, the field to be extracted. The default is FIELD_VAL.
        The available fields are:
            
            FIELD_T = 'T': the time - voxel.t
            
            FIELD_X = 'X': the x-coordinate - voxel.x
            
            FIELD_Y = 'Y': the y-coordinate - voxel.y
            
            FIELD_VAL = 'VAL': the value (default) - voxel.val
            
            FIELD_ACC = 'ACC': the accuracy - voxel.stdev
            
            FIELD_NUM = 'NUM': the neighborhood - voxel.neigh

    Raises
    ------
    TFEx

    Yields
    ------
    numpy array
    """
    try:
        nt, nx, ny = int(mod.params.get('NT')), int(mod.params.get('NX')), int(mod.params.get('NY'))
        values = []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    values.append(_extract(voxel, field))
        return np.array(values).reshape(nt, nx, ny)
    except Exception as ex: raise TFEx('Error extracting bulk: {}'.format(ex))

def extract_sheet(mod, k, field=FIELD_VAL):
    """
    Extracts a the model sheet as a numpy array

    Parameters
    ----------
    mod: Model, the Timescape model

    k: int, the sheet number

    field: str, the field to be extracted. The default is FIELD_VAL.
        The available fields are:
            
            FIELD_T = 'T': the time - voxel.t
            
            FIELD_X = 'X': the x-coordinate - voxel.x
            
            FIELD_Y = 'Y': the y-coordinate - voxel.y
            
            FIELD_VAL = 'VAL': the value (default) - voxel.val
            
            FIELD_ACC = 'ACC': the accuracy - voxel.stdev
            
            FIELD_NUM = 'NUM': the neighborhood - voxel.neigh

    Raises
    ------
    TFEx

    Yields
    ------
    numpy array
    """
    try:
        nx, ny = int(mod.params.get('NX')), int(mod.params.get('NY'))
        values = []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    if voxel.k == k: values.append(_extract(voxel, field))
        return np.array(values).reshape(nx, ny)
    except Exception as ex: raise TFEx('Error extracting core: {}'.format(ex))

def extract_core(mod, i, j, field=FIELD_VAL):
    """
    Extracts a model core as a numpy array

    Parameters
    ----------
    mod: Model, the Timescape model

    i: int, the core row number

    j: int, the core column number

    field: str, the field to be extracted. The default is FIELD_VAL.
        The available fields are:
            
            FIELD_T = 'T': the time - voxel.t
            
            FIELD_X = 'X': the x-coordinate - voxel.x
            
            FIELD_Y = 'Y': the y-coordinate - voxel.y
            
            FIELD_VAL = 'VAL': the value (default) - voxel.val
            
            FIELD_ACC = 'ACC': the accuracy - voxel.stdev
            
            FIELD_NUM = 'NUM': the neighborhood - voxel.neigh

    Raises
    ------
    TFEx

    Yields
    ------
    numpy array
    """
    try:
        values = []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    if voxel.i == i and voxel.j == j: values.append(_extract(voxel, field))
        return np.array(values)
    except Exception as ex: raise TFEx('Error extracting core: {}'.format(ex))

def _extract(voxel: Voxel, field):
    if field == FIELD_VAL: return voxel.val
    elif field == FIELD_ACC: return voxel.stdev
    elif field == FIELD_NUM: return voxel.neigh
    elif field == FIELD_T: return voxel.t
    elif field == FIELD_X: return voxel.x
    elif field == FIELD_Y: return voxel.y
    else: return None


def time_series(mod, i, j, outfile=FILEPATH_TME):
    """
    Writes a time series on a text file

    Parameters
    ----------
    mod: Model, the Timescape model

    i: int, the core row number

    j: int, the core column number

    outfile : str, the output file name. The default is FILEPATH_TME.
    
    Raises
    ------
    TFEx

    Yields
    ------
    nothing, writes the output file
    """
    if i < 0 or i >= mod.params.get('NX'): raise TFEx('Index out of range')
    if j < 0 or j >= mod.params.get('NY'): raise TFEx('Index out of range')
    if os.path.isfile(outfile): print('File {} is being superseded'.format(outfile))
    try:
        tt = mod.extract_core(i, j, field=FIELD_T)
        vv = mod.extract_core(i, j, field=FIELD_VAL)
        aa = mod.extract_core(i, j, field=FIELD_ACC)
        nn = mod.extract_core(i, j, field=FIELD_NUM)
        with open(outfile, 'w') as out:
            out.write('{},{},{},{}\n'.format(FIELD_T, FIELD_VAL, FIELD_ACC, FIELD_NUM))
            for ind in range(len(tt)):
                out.write('{},{},{},{}\n'.format(_clean_float(tt[ind]), _clean_float(vv[ind]), _clean_float(aa[ind]), _clean_int(nn[ind])))
    except Exception as ex: raise TFEx('Error extracting time series: {}'.format(ex))


def target_show(mod, field=FIELD_VAL):
    """
    Plots the Timescape target bulk as volume isosurfaces via plotly

    Parameters
    ----------
    mod: Model, the Timescape model

    field: str, the field to be extracted. The default is FIELD_VAL.
        
        The available fields are:
            
            FIELD_VAL = 'VAL': the value (default) - voxel.val
            
            FIELD_ACC = 'ACC': the accuracy - voxel.stdev
            
            FIELD_NUM = 'NUM': the neighborhood - voxel.neigh

    Raises
    ------
    TFEx

    Yields
    ------
    Nothing, opens the volume in a browser
    """
    if field == FIELD_VAL: colors, titlefield = PLOT_COLORS, 'Values'
    elif field == FIELD_ACC: colors, titlefield = 'reds', 'Accuracy'
    elif field == FIELD_NUM: colors, titlefield = 'Bluered_r', 'Neighborhood'
    else: raise TFEx('Cannot show field {}'.format(field))
    try:
        vals = []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    if voxel.val:
                        if field == FIELD_VAL: vals.append(voxel.val)
                        elif field == FIELD_ACC: vals.append(voxel.stdev)
                        else: vals.append(voxel.neigh)
        minv, maxv = min(vals), max(vals)
        T, X, Y = np.mgrid[0:mod.params.get('NT'), 0:mod.params.get('NX'), 0:mod.params.get('NY')]
        V = mod.extract_bulk(field)
        fig = go.Figure(data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=T.flatten(),
            value=V.flatten(), isomin=minv, isomax=maxv,
            name = '', hovertemplate = 'K=%{z}  I=%{x}  J=%{y}',
            opacity=0.1, surface_count=14, colorscale=colors))
        fig.update_layout(title='Timescape {}'.format(titlefield), scene = dict(
            xaxis_title='I ROW (X)',
            yaxis_title='J COLUMN (Y)',
            zaxis_title='K SHEET (T)'))
        fig.show()
    except Exception as ex: raise TFEx('Error showing bulk: {}'.format(ex))


def target_histogram(mod, field=FIELD_VAL):
    """
    Plots an histogram of the selected field 

    Parameters
    ----------
    mod: Model, the Timescape model

    field: str, the field to be exmined. The default is FIELD_VAL.
        
        The available fields are:
            
            FIELD_VAL = 'VAL': the value (default) - voxel.val
            
            FIELD_ACC = 'ACC': the accuracy - voxel.stdev
            
            FIELD_NUM = 'NUM': the neighborhood - voxel.neigh

    Raises
    ------
    TFEx

    Yields
    ------
    Nothing, shows the plot
    """
    if field == FIELD_VAL: titlefield = 'Value'
    elif field == FIELD_ACC: titlefield = 'Accuracy'
    elif field == FIELD_NUM: titlefield = 'Neighborhood'
    else: raise TFEx('Cannot show field {}'.format(field))
    try:
        vals = []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    if voxel.val:
                        if field == FIELD_VAL: vals.append(voxel.val)
                        elif field == FIELD_ACC: vals.append(voxel.stdev)
                        else: vals.append(voxel.neigh)
        minv, maxv = min(vals), max(vals)
        if field == FIELD_NUM: histbins = range(min(100, maxv))
        else: histbins = min(100, int(len(vals)/5))
        plt.figure(figsize = plt.figaspect(.5), dpi=PLOT_DPI)
        plt.title('Histogram of Events {}'.format(titlefield))
        plt.xlabel(titlefield)
        plt.ylabel('Number')
        plt.hist(vals, bins=histbins, range=(minv, maxv),
            density=False, edgecolor='black', color='lightgrey')
        plt.show()
    except Exception as ex: raise TFEx('Error making histogram: {}'.format(ex))


def plot_bulk(mod):
    """
    Plots the Timescape target bulk as a 3d scatterplot
    
    Warning: this function is in trouble with too many target events

    Parameters
    ----------
    mod: Model, the Timescape model

    Raises
    ------
    TFEx

    Yields
    ------
    Nothing, shows the plot

    """
    try:
        events = []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    events.append(voxel)
        events[:] = [event for event in events if event.val is not None]
        xx = [event.x for event in events]
        yy = [event.y for event in events]
        tt = [event.t for event in events]
        vv = [event.val for event in events]
        mint, maxt = min(tt), max(tt)
        minx, maxx = min(xx), max(xx)
        miny, maxy = min(yy), max(yy)
        fig = plt.figure(dpi=PLOT_DPI)
        ax3d = fig.add_subplot(1, 1, 1, projection='3d')
        _scatterplot(plt, ax3d, tt, xx, yy, vv, mint, maxt, minx, maxx, miny, maxy, 'Target events')
        cbar = plt.colorbar()
        cbar.set_label('Event values')
        plt.show()
    except Exception as ex: raise TFEx('Error plotting bulk: {}'.format(ex))

def plot_sheet(mod, k):
    """
    Plots a Timescape target sheet as points on a plane and contour lines

    Parameters
    ----------
    mod: Model, the Timescape model

    k: int, the sheet number

    Raises
    ------
    TFEx

    Yields
    ------
    Nothing, shows the plot

    """
    try:
        nx, ny = int(mod.params.get('NX')), int(mod.params.get('NY'))
        xx, yy, vv = [], [], []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    if voxel.k == k:
                        xx.append(voxel.x)
                        yy.append(voxel.y)
                        vv.append(voxel.val)
        xi = np.linspace(min(xx), max(xx), nx)
        yi = np.linspace(min(yy), max(yy), ny)
        vvv = griddata((xx, yy), vv, (xi[None,:], yi[:,None]), method='linear')
        plt.figure(dpi=PLOT_DPI)
        plt.title('SHEET {}'.format(k))
        plt.xlabel(FIELD_X)
        plt.ylabel(FIELD_Y)
        lines = plt.contour(xi, yi, vvv, levels=12, linewidths=0.3, cmap=PLOT_COLORS)
        plt.clabel(lines, inline=1, fontsize=3)
        plt.scatter(xx, yy, marker='o', alpha=PLOT_ALPHA, cmap=PLOT_COLORS, c=vv, s=1)
        cbar = plt.colorbar()
        cbar.set_label(FIELD_VAL)
        plt.show()
    except Exception as ex: raise TFEx('Error plotting sheet: {}'.format(ex))


def plot_core(mod, i, j):
    """
    Plots a Timescape target core as a time series

    Parameters
    ----------
    mod: Model, the Timescape model

    i: int, the core row number

    j: int, the core column number

    Raises
    ------
    TFEx

    Yields
    ------
    Nothing, shows the plot

    """
    try:
        tt, vv = [], []
        for sheet in mod.target:
            for row in sheet:
                for voxel in row:
                    if voxel.i == i and voxel.j == j:
                        tt.append(voxel.t)
                        vv.append(voxel.val)
        plt.figure(figsize = plt.figaspect(.5), dpi=PLOT_DPI)
        plt.title('CORE {} {}'.format(i, j))
        plt.xlabel(FIELD_T)
        plt.ylabel(FIELD_VAL)
        plt.plot(tt, vv, color='black', linestyle=':', linewidth=.25)
        plt.scatter(tt, vv, marker='o', alpha=PLOT_ALPHA, cmap=PLOT_COLORS, c=vv, s=7)
        cbar = plt.colorbar()
        cbar.set_label(FIELD_VAL)
        plt.show()
    except Exception as ex: raise TFEx('Error plotting core: {}'.format(ex))


def source_trend(src=FILEPATH_INP, out=FILEPATH_TND):
    """
    Source analysis: evaluates the linear trend of the source's events values
    with respect to time, x- and y-coordinate using the statsmodels.api module

    Parameters
    ----------
    src: str, the source file. The default is FILEPATH_INP.
    
    out: str, the output file. The default is FILEPATH_TND.

    Yields
    ------
    Nothing. Shows the plot and writes the output
    """
    tt, xx, yy, vv = [], [], [], []
    print(_HEADER_STRING)
    try:
        mod = Model()
        _read_input(mod, src)
        for evt in mod.source:
            tt.append(evt.t)
            xx.append(evt.x)
            yy.append(evt.y)
            vv.append(evt.val)
        mint, maxt = min(tt), max(tt)
        minx, maxx = min(xx), max(xx)
        miny, maxy = min(yy), max(yy)
        minv, maxv = min(vv), max(vv)
        trendt = sm.OLS(vv, sm.add_constant(tt)).fit()
        trendx = sm.OLS(vv, sm.add_constant(xx)).fit()
        trendy = sm.OLS(vv, sm.add_constant(yy)).fit()
        _write_trend(mod, mint, maxt, minx, maxx, miny, maxy, minv, maxv, trendt, trendx, trendy, out)
        fig = plt.figure(figsize = plt.figaspect(.25), dpi=PLOT_DPI)
        axt = fig.add_subplot(1, 3, 1)
        _plot_trend(axt, tt, vv, trendt, mint, maxt, FIELD_T, FIELD_VAL)
        axx = fig.add_subplot(1, 3, 2)
        _plot_trend(axx, xx, vv, trendx, minx, maxx, FIELD_X, '')
        axy = fig.add_subplot(1, 3, 3)
        _plot_trend(axy, yy, vv, trendy, miny, maxy, FIELD_Y, '')
        plt.show()
    except TFEx as tfex: print('Error: {}'.format(tfex))
    finally: print(_FOOTER_STRING)

def _plot_trend(ax, x, y, trend, minx, maxx, xlbl, yllbl):
    ax.set_xlabel(xlbl)
    ax.set_ylabel(yllbl)
    xx = np.linspace(minx, maxx, 100)
    yy = [(x * trend.params[1] + trend.params[0]) for x in xx]
    ax.scatter(x, y, marker='o', color='grey', alpha=PLOT_ALPHA, s=7)
    ax.plot(xx, yy, color='black', linestyle=':', linewidth=.5)

def _write_trend(mod, mint, maxt, minx, maxx, miny, maxy, minv, maxv, trendt, trendx, trendy, outfile):
    with open(outfile, 'w') as out:
        out.write('{} source events found within\n'.format(len(mod.source)))
        out.write('   {} < T < {}\n'.format(mint, maxt))
        out.write('   {} < X < {}\n'.format(minx, maxx))
        out.write('   {} < Y < {}\n'.format(miny, maxy))
        out.write('  {} < VAL < {}\n'.format(minv, maxv))
        out.write('Metric:{}\nc = {}, k = {}\n'.format(list(_METRICS.keys())[list(_METRICS.values()).index(mod.params[_METRIC_KEY])], mod.params.get('C'), mod.params.get('K')))
        if mod.params.get('KPERIOD'): out.write('The causal cone is periodic with k period = {}\n'.format(mod.params.get('KPERIOD')))
        else: out.write('The causal cone is straight\n')
        out.write('\n\n\nTrend of VAL vs T:' + str(trendt.summary()))
        out.write('\n\n\nTrend of VAL vs X:' + str(trendx.summary()))
        out.write('\n\n\nTrend of VAL vs Y:' + str(trendy.summary()))
    print('File {} output complete'.format(outfile))


def source_footprint(src=FILEPATH_INP):
    """
    Source analysis: plots the footprint of the source events
    
    The spatial and temporal bounds are read from the input file
    
    The resulting plot consists in two subplots:
        
        The events spatial distribution (i.e. the footprint)
        
        The time histogram of the events

    Parameters
    ----------
    src: str, the source file. The default is FILEPATH_INP.

    Yields
    ------
    Nothing. Shows the plot
    """
    tt, xx, yy = [], [], []
    print(_HEADER_STRING)
    try:
        mod = Model()
        _read_input(mod, src)
        for evt in mod.source:
            tt.append(evt.t)
            xx.append(evt.x)
            yy.append(evt.y)
        mint, maxt = min(tt), max(tt)
        minx, maxx = min(xx), max(xx)
        miny, maxy = min(yy), max(yy)
        deltax = .05 * (maxx - minx)
        deltay = .05 * (maxy - miny)
        ttt = np.linspace(mint, maxt, 100)
        tdens = gaussian_kde(tt)
        tdens.covariance_factor = lambda : .25
        tdens._compute_covariance()
        fig = plt.figure(figsize = plt.figaspect(.5), dpi=PLOT_DPI)
        axfp = fig.add_subplot(1, 2, 1)
        axfp.set_title('Events Footprint')
        axfp.set_xlabel(FIELD_X)
        axfp.set_ylabel(FIELD_Y)
        axfp.set_xlim(minx - deltax, maxx + deltax)
        axfp.set_ylim(miny - deltay, maxy + deltay)
        axfp.scatter(xx, yy, marker='o', color='grey', alpha=PLOT_ALPHA, s=14)
        axhg = fig.add_subplot(1, 2, 2)
        axhg.set_title('Temporal Density of {} Events'.format(len(tt)))
        axhg.set_xlabel(FIELD_T)
        axhg.set_ylabel('Density')
        axhg.yaxis.set_label_position('right')
        axhg.yaxis.tick_right()
        axhg.hist(tt, bins=int(min(len(tt)/5, 100)), range=(mint, maxt),
            density=True, edgecolor='black', color='lightgrey')
        axhg.plot(ttt, tdens(ttt), color='black', linewidth=2)
        plt.show()
    except TFEx as tfex: print('Error: {}'.format(tfex))
    finally: print(_FOOTER_STRING)


def source_dist(src=FILEPATH_INP):
    """
    Source analysis: evaluates the spatiotempral variogram and the distribution of the values.    
    The resulting plot consists in three subplots:
        
        The varioram
        
        A 3d-scatterplot
        
        A boxplot as box-and-whisker plus violin plot, superimposed

    Parameters
    ----------
    src: str, the source file. The default is FILEPATH_INP.

    Yields
    ------
    The variogram points as (h, gamma) pairs in stout lines
    
    Shows the plot
    """
    tt, xx, yy, vv = [], [], [], []
    print(_HEADER_STRING)
    try:
        mod = Model()
        _read_input(mod, src)
        for evt in mod.source:
            tt.append(evt.t)
            xx.append(evt.x)
            yy.append(evt.y)
            vv.append(evt.val)
        mint, maxt = min(tt), max(tt)
        minx, maxx = min(xx), max(xx)
        miny, maxy = min(yy), max(yy)
        minv, maxv = min(vv), max(vv)
        fig = plt.figure(figsize = plt.figaspect(.3), dpi=PLOT_DPI)
        axvg = fig.add_subplot(1, 3, 1)
        _variogram(mod, axvg, tt, xx, yy, vv)
        ax3d = fig.add_subplot(1, 3, 2, projection='3d')
        _scatterplot(plt, ax3d, tt, xx, yy, vv, mint, maxt, minx, maxx, miny, maxy, 'Source Events')
        axvp = fig.add_subplot(1, 3, 3)
        axvp.axis('off')
        axvp.set_ylim([minv, maxv])
        vp = axvp.violinplot(vv, showmeans=True, showextrema=False)
        for pc in vp['bodies']:
            pc.set_facecolor('grey')
            pc.set_edgecolor('black')
            pc.set_alpha(.2)
        axvp.boxplot(vv, notch=True, medianprops=dict(color='black'), flierprops=dict(marker='o', markersize=2))
        cbar = plt.colorbar()
        cbar.set_label('Event values')
        plt.show()
    except TFEx as tfex: print('Error: {}'.format(tfex))
    finally: print(_FOOTER_STRING)

def _scatterplot(plt, ax3d, tt, xx, yy, vv, mint, maxt, minx, maxx, miny, maxy, title):
    ax3d.set_title(title)
    ax3d.set_xlim3d(minx, maxx)
    ax3d.set_ylim3d(miny, maxy)
    ax3d.set_zlim3d(mint, maxt)
    ax3d.set_xlabel(FIELD_X)
    ax3d.set_ylabel(FIELD_Y)
    ax3d.set_zlabel(FIELD_T)
    plt.scatter(xx, yy, tt, cmap = PLOT_COLORS, c=vv, marker='')
    ax3d.scatter(xx, yy, [min(tt) for k in range(len(tt))], color = 'grey', marker='.')
    for i, j, k, h in zip(xx, yy, [mint] * len(tt), tt):
        ax3d.plot([i,i], [j,j], [k,h], color = 'grey', linestyle='dotted', linewidth=.5)
    ax3d.scatter(xx, yy, tt, cmap = PLOT_COLORS, c=vv, marker='o')

def _variogram(mod, ax, tt, xx, yy, vv):
    ax.set_xlabel('$h$')
    ax.set_ylabel('$\gamma(h)$')
    ax.set_title('Spatiotemporal Variogram')
    d, h, col = [], [], []
    for i in range(len(tt)):
        for j in range(len(tt)):
            if i != j:
                vxl0, vxl1 = Voxel(), Voxel()
                vxl0.k, vxl0.i, vxl0.j = 0, 0, 0
                vxl0.t, vxl0.ct = tt[i], mod.params.get('C') * tt[i]
                vxl0.x, vxl0.y = xx[i], yy[i]
                vxl1.k, vxl1.i, vxl1.j = 0, 0, 0
                vxl1.t, vxl1.ct = tt[j], mod.params.get('C') * tt[j]
                vxl1.x, vxl1.y = xx[j], yy[j]
                dist = _distance(mod, vxl0, vxl1)
                if dist is not None:
                    d.append(dist)
                    h.append(math.fabs(vv[i] - vv[j])) #absolute value variogran
                    col.append(vv[j])
    mind, maxd = min(d), max(d)
    delta = (maxd - mind) / float(_VAR_BINS)
    hist_d, hist_n, hist_h, hist_s = [], [0] * (_VAR_BINS + 1), [0.] * (_VAR_BINS + 1), [1.] * (_VAR_BINS + 1)
    for i in range(len(d)):
        hist_n[math.floor(d[i] / delta)] += 1
        hist_h[math.floor(d[i] / delta)] += h[i]
    for i in range(len(hist_n)):
        hist_d.append((i + .5) * delta)
        if hist_n[i]: hist_h[i] /= float(hist_n[i])
        hist_s[i] += 100. * hist_n[i] / sum(hist_n)
    print('Variogram\nh,gamma')
    for i in range(len(hist_n)):
        print('{},{}'.format(hist_d[i], hist_h[i]))
    ax.scatter(d, h, marker='.', alpha=PLOT_ALPHA, color='grey', s=1)
    ax.scatter(hist_d, hist_h, marker='o', alpha=PLOT_ALPHA, color='red', s=hist_s)


def source_ensemble(nc, cmin, cmax, nk, kmin, kmax, src=FILEPATH_INP, out=FILEPATH_RES):
    """
    Source analysis: ensemble analysis of the residuals of the source vs itself.
    The ensemble is build on a regular lattice of c and k values
    
    The output consists in a text file of resuduals, null and bad events for each (c,k)
    and a plot consisting of three 3d surfaces vs (c,k):
        
        RESpEVT - average residual per event in value units

        NULL - number of null events

        BAD - number of bad events

    Parameters
    ----------
    nc: int, the number of c values (2 or more)
    
    cmin: float, the minimum c value
    
    cmax: float, the maximum c value
    
    nk: int, the number of k values (2 or more)
    
    kmin: float, the minimum k value
    
    kmax: float, the maximum k value
    
    src: str, the source file. The default is FILEPATH_INP.
    
    out: str, the output file. The default is FILEPATH_RES.

    Yields
    ------
    Nothing, shows the plot and writes the output
    """
    print(_HEADER_STRING)
    try:
        mod = Model()
        ensemble = _build_ensemble(nc, cmin, cmax, nk, kmin, kmax)
        _read_input(mod, src)
        source = copy(mod.source)
        target = copy(mod.source)
        count = 0
        for res in ensemble.residuals:
            count += 1
            _eval_residual(mod, res, source, target)
            _progbar('Evaluating residuals', count, ensemble.numc * ensemble.numk)
        _write_residuals(ensemble, mod, out, src)
        _plot_residuals(float(len(mod.source)) / 100., ensemble)
    except TFEx as tfex: print('Error: {}'.format(tfex))
    finally: print(_FOOTER_STRING)

def _eval_residual(mod, res, source, target):
    mod.params['C'] = res.c
    mod.params['K'] = res.k
    res.sqres = 0.
    timestamp = datetime.now().timestamp()
    for tar in target:
        voxel = _ResVoxel(tar.label, tar.t, res.c * tar.t, tar.x, tar.y, tar.val)
        mod.source = []
        for src in source:
            if(voxel.label != src.label):
                mod.source.append(src)
        try: _eval_voxel(mod, voxel)
        except TEvalEx: res.bad += 1
        if voxel.val == None: res.null += 1
        else: res.sqres += (voxel.exact - voxel.val) * (voxel.exact - voxel.val)
    if len(mod.source) != res.null: res.respevt = math.sqrt(res.sqres / float(len(mod.source) - res.null))
    res.thput = float(len(target)) / (datetime.now().timestamp() - timestamp)

def _build_ensemble(nc, cmin, cmax, nk, kmin, kmax):
    try: nc = max(2, nc)
    except ValueError: raise TFEx('Invalid nc = {}'.format(nc))
    try: nk = max(2, nk)
    except ValueError: raise TFEx('Invalid nk = {}'.format(nk))
    try: cmin = float(cmin)
    except ValueError: raise TFEx('Invalid cmin = {}'.format(cmin))
    try: cmax = float(cmax)
    except ValueError: raise TFEx('Invalid cmax = {}'.format(cmax))
    if cmin > cmax: raise TFEx('Bad cmin > cmax')
    try: kmin = float(kmin)
    except ValueError: raise TFEx('Invalid kmin = {}'.format(kmin))
    try: kmax = float(kmax)
    except ValueError: raise TFEx('Invalid kmax = {}'.format(kmax))
    if kmin > kmax: raise TFEx('Bad kmin > kmax')
    dc = (cmax - cmin) / float(nc - 1)
    dk = (kmax - kmin) / float(nk - 1)
    ress = []
    for i in range(nc):
        for j in range(nk):
            ress.append(_Residual((cmin + dc * float(i)), (kmin + dk * float(j))))
    print('Ensemble elements: {}'.format(nc * nk))
    return _ResEnsemble(nc, cmin, cmax, nk, kmin, kmax, ress)

def _write_residuals(ensemble, mod, outfile, src):
    avthp = 0.
    for res in ensemble.residuals: avthp += res.thput
    avthp /= float(len(ensemble.residuals))
    print('Average throughput: {} vx/s'.format(round(avthp, 2)))
    period = '' if mod.straight else ', period={}'.format(mod.params.get('KPERIOD'))
    radius = ' radius={}'.format(mod.params.get('RADIUS')) if mod.params[_METRIC_KEY] == _METRICS['SPHERE'] else ''
    neigh = 'all' if mod.params.get('NEIGH') == 0 else 'nearest {}'.format(mod.params.get('NEIGH'))
    metric = list(_METRICS.keys())[list(_METRICS.values()).index(mod.params[_METRIC_KEY])]
    try:
        with open(outfile, 'w') as out:
            out.write('# Residuals of {}:\n#\n'.format(src))
            out.write('# Algorithm: {}\n'.format(mod.params.get('ALGORITHM')))
            out.write('# Neighborhood: {}\n'.format(neigh))
            out.write('# Metric: {}{}{}\n'.format(metric, radius, period))
            out.write('# Source events: {}\n'.format(len(mod.source) + 1))
            out.write('# Ensemble elements: {}\n'.format(ensemble.numc * ensemble.numk))
            out.write('# Avg throughput: {} vx/s\n'.format(avthp))
            out.write('# Cmin={} Cmax={} in {} steps\n'.format(ensemble.cmin, ensemble.cmax, ensemble.numc))
            out.write('# Kmin={} Kmax={} in {} steps\n#\n'.format(ensemble.kmin, ensemble.kmax, ensemble.numk))
            out.write('{}\n'.format('C,K,SQRES,RESpEVT,NULL,BAD,VXpS'))
            for res in ensemble.residuals:
                out.write('{},{},{},{},{},{},{}\n'.format(res.c, res.k, res.sqres, res.respevt, res.null, res.bad, res.thput))
            print('{} output complete'.format(outfile))
    except Exception as ex: raise TFEx('Error writing output: {}'.format(ex))

def _plot_residuals(num, ensemble):
    cc = np.linspace(ensemble.cmin, ensemble.cmax, ensemble.numc)
    kk = np.linspace(ensemble.kmin, ensemble.kmax, ensemble.numk)
    ress, nuls, bads = [], [], []
    for res in ensemble.residuals:
        ress.append(res.respevt)
        nuls.append(float(res.null) / num)
        bads.append(float(res.bad) / num)
    C, K = np.meshgrid(cc, kk)
    Z = np.array(ress).reshape(ensemble.numk, ensemble.numc)
    N = np.array(nuls).reshape(ensemble.numk, ensemble.numc)
    B = np.array(bads).reshape(ensemble.numk, ensemble.numc)
    fig = plt.figure(figsize = plt.figaspect(.3), dpi=PLOT_DPI)
    axr = fig.add_subplot(1, 3, 1, projection='3d')
    axr.plot_surface(C, K, Z, color='yellow', edgecolor='black', alpha=PLOT_ALPHA)
    axr.set_title('RESpEVT - Avg Res per not-null Event')
    axr.set_xlabel('C')
    axr.set_ylabel('K')
    axn = fig.add_subplot(1, 3, 2, projection='3d')
    axn.plot_surface(C, K, N, color='blue', edgecolor='black', alpha=PLOT_ALPHA)
    axn.set_title('NULL - Null Events %')
    axn.set_xlabel('C')
    axn.set_ylabel('K')
    axn.set_zlim(0, 50)
    axb = fig.add_subplot(1, 3, 3, projection='3d')
    axb.plot_surface(C, K, B, color='red', edgecolor='black', alpha=PLOT_ALPHA)
    axb.set_title('BAD - Bad Events %')
    axb.set_xlabel('C')
    axb.set_ylabel('K')
    axb.set_zlim(0, 10)
    plt.show()


def _clean_int(n):
    if n == None: return 0
    return int(n)

def _clean_float(x):
    if x == None: return ''
    return round(x, ROUND_DIGITS) if ROUND_FLOATS else x

def _logline(line, stdout_too=False):
    if LOG_TO_FILE:
        with open(FILEPATH_LOG, 'a') as log: 
            log.write('{} {}\n'.format(datetime.now().strftime(_DATE_FORMAT), str(line)))
    if stdout_too: print('{}'.format(line))

def _progbar(mess: str, num: int, tot: int):
    percent, done = int(100 * num / tot), int(40 * num // tot)
    progstr = ('|' * done) + ('-' * (40 - done))
    print('\r{} |{}| {}% done'.format(mess, progstr, percent), end=('\n' if num == tot else '\r'))
