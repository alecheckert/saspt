import os, unittest, numpy as np
from saspt.parameters import StateArrayParameters
from saspt.constants import RBME

class TestStateArrayParameters(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(pixel_size_um=0.16, frame_interval=0.01,
            focal_depth=0.7, splitsize=10, sample_size=200,
            start_frame=10, max_iter=50, conc_param=2.0, 
            progress_bar=False)

    def test_init(self):
        SAP = StateArrayParameters(**self.kwargs, nothing=0)
        for k, v in self.kwargs.items():
            assert getattr(SAP, k) == v, k
        
    def test_eq(self):
        SAP1 = StateArrayParameters(**self.kwargs)
        SAP2 = StateArrayParameters(**self.kwargs)
        kwargs = self.kwargs.copy()
        kwargs.update(max_iter=70)
        SAP3 = StateArrayParameters(**kwargs)
        assert SAP1 == SAP2
        assert SAP1 != SAP3
