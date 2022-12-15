# Need to reload modules so the updated code gets run.
import os
from homework import utils
from importlib import reload
from homework import controller
reload(controller)
reload(utils)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hack for notebooks.
if 'pytux' not in locals():
    pytux = utils.PyTux()

# Modify any of the arguments below to tweak the generated dataset.
tracks = []
tracks.append('cocoa_temple')
tracks.append('lighthouse')
tracks.append('zengarden')
tracks.append('hacienda')
tracks.append('snowtuxpeak')
tracks.append('cornfield_crossing')
tracks.append('scotland')

verbose = False

# Dataset will be collected in a directory called "drive_data".
utils.main(pytux, tracks, n_images=10000, steps_per_track=20000,
           aim_noise=0.1, vel_noise=5, verbose=verbose)
