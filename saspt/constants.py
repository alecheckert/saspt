#!/usr/bin/env python
import numpy as np

#############################
## DEFAULT PARAMETER GRIDS ##
#############################

DEFAULT_DIFF_COEFS = np.logspace(-2.0, 2.0, 100)
DEFAULT_LOC_ERRORS = np.arange(0.0, 0.072, 0.002)
DEFAULT_HURST_PARS = np.arange(0.05, 1.0, 0.05)

###############################################
## DEFAULT PREPROCESSING AND HYPERPARAMETERS ##
###############################################

# When trajectories are too long, split them into smaller trajectories.
# *splitsize* defines the maximum trajectory length (in # jumps) before
# splitting.
DEFAULT_SPLITSIZE = 10

# Default concentration parameter for the prior distribution over state
# occupations.
DEFAULT_CONC_PARAM = 1.0

# Default number of iterations to do when inferring posterior
DEFAULT_MAX_ITER = 200

# Default first frame to consider
DEFAULT_START_FRAME = 0

# Maximum number of trajectories to consider when running state arrays
DEFAULT_SAMPLE_SIZE = 10000

################################
## DETECTION-LEVEL ATTRIBUTES ##
################################

# Column in the detections DataFrame encoding frame index
FRAME = "frame"

# Column in the detections DataFrame encoding trajectory index
TRACK = "trajectory"

# Column in the detections DataFrame encoding trajectory length (in frames)
TRACK_LENGTH = "track_length"

# Column in the detections DataFrame encoding y-position in pixels
PY = "y"

# Column in the detections DataFrame encoding x-position in pixels
PX = "x"

###########################
## JUMP-LEVEL ATTRIBUTES ##
###########################

# Column in the jumps DataFrame encoding number of frames over which
# the jump happened
DFRAMES = "dframes"

# Column in the jumps DataFrame encoding the change in y-position in microns
DY = "dy"

# Column in the jumps DataFrame encoding the change in x-position in microns
DX = "dx"

# Column in the jumps DataFrame encoding the squared 2D radial jump length
# in squared microns
DR2 = "dr2"

# Column in the jumps DataFrame encoding the number of jumps per trajectory
JUMPS_PER_TRACK = "jumps_per_track"

####################################
## AVAILABLE LIKELIHOOD FUNCTIONS ##
####################################

# Names of likelihood functions
RBME = "rbme"
RBME_MARGINAL = "rbme_marginal"
GAMMA = "gamma"
FBME = "fbme"

# All available likelihood functions
LIKELIHOOD_TYPES = [RBME, RBME_MARGINAL, GAMMA, FBME]

###########
## OTHER ##
###########

# Bucket condition column for StateArrayDatasets without an experimental condition
DEFAULT_CONDITION_COL = "default_condition"
DEFAULT_CONDITION = "no_condition"
