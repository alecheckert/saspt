import numpy as np
from typing import Tuple
from .constants import (
    DEFAULT_MAX_ITER,
    DEFAULT_CONC_PARAM,
    DEFAULT_SPLITSIZE,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_START_FRAME,
)

class StateArrayParameters:
    """ Struct encapsulating settings for a complete state array analysis.

    init
    ----
        pixel_size_um       :   size of camera pixels in microns

        frame_interval      :   time between frames in seconds

        focal_depth         :   focal depth in microns

        splitsize           :   maximum number of jumps to tolerate per trajectory
                                before splitting into smaller trajectories

        sample_size         :   maximum number of trajectories to consider when 
                                running state arrays. If exceeded, we subsample.

        start_frame         :   disregard all jumps observed before this frame

        max_iter            :   maximum number of iterations of state array inference
                                to run

        conc_param          :   concentration parameter for prior distribution over
                                state occupations; number of pseudocounts per element
                                in the state array

        progress_bar        :   show a progress bar, where relevant

        num_workers         :   number of parallel processes to use

    """
    def __init__(self, pixel_size_um: float, frame_interval: float,
        focal_depth: float=np.inf, splitsize: int=DEFAULT_SPLITSIZE,
        sample_size: int=DEFAULT_SAMPLE_SIZE, start_frame: int=DEFAULT_START_FRAME,
        max_iter: int=DEFAULT_MAX_ITER, conc_param: float=DEFAULT_CONC_PARAM,
        progress_bar: bool=False, num_workers: int=1, **kwargs):

        self.pixel_size_um = pixel_size_um
        self.frame_interval = frame_interval
        self.focal_depth = focal_depth
        self.splitsize = splitsize
        self.sample_size = sample_size 
        self.start_frame = start_frame
        self.max_iter = max_iter
        self.conc_param = conc_param 
        self.progress_bar = progress_bar
        self.num_workers = num_workers

    @property
    def parameters(self) -> Tuple[str]:
        return ("pixel_size_um", "frame_interval", "focal_depth", "splitsize",
            "sample_size", "start_frame", "max_iter", "conc_param")

    @property
    def units(self) -> dict:
        return dict(pixel_size_um="µm", frame_interval="sec", focal_depth="µm",
            splitsize="jumps", sample_size="tracks", start_frame="frames", 
            max_iter="iterations", conc_param="pseudocounts per state")

    def __eq__(self, other) -> bool:
        """ Test for equality of two StateArrayParameters objects """
        return all(map(lambda a: getattr(self, a) == getattr(other, a), self.parameters))

    def __repr__(self) -> str:
        """ String representation of this StateArrayParameters object """
        return "StateArrayParameters:\n  {}".format("\n  ".join([
            f"{a}:\t{getattr(self, a)}" for a in self.parameters
        ]))       
