import os
from homework2 import controller
from importlib import reload
from homework2 import utils
reload(utils)
reload(controller)

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

# verbose = True
verbose = False

controller.test_controller(pytux, tracks, verbose=verbose)
