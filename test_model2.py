# Need to reload modules so the updated code gets run.
from homework2 import planner
from homework2 import controller
from importlib import reload
from homework2 import utils
import os
reload(utils)
reload(controller)
reload(planner)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Hack for notebooks.
if 'pytux' not in locals():
    pytux = utils.PyTux()

tracks = []
tracks.append('cocoa_temple')
tracks.append('lighthouse')
tracks.append('zengarden')
tracks.append('hacienda')
tracks.append('snowtuxpeak')
tracks.append('cornfield_crossing')
tracks.append('scotland')

verbose=True
# verbose=False

planner.test_planner(pytux, tracks, verbose=verbose)