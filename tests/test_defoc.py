import os, unittest, numpy as np, pandas as pd 
from saspt import defoc
from saspt.constants import GAMMA, RBME, RBME_MARGINAL, FBME

# All likelihood functions supported by *defoc_corr*
LIKELIHOOD_TYPES = [GAMMA, RBME_MARGINAL, RBME, FBME]

class TestDefocCorr(unittest.TestCase):
    def setUp(self):
        """ Establish support for likelihood functions """
        self.kwargs = dict(frame_interval=0.00748, dz=0.7, normalize=True)
        self.diff_coefs = np.logspace(-2.0, 2.0, 100)
        self.loc_errors = np.arange(0.2, 0.625, 0.025)
        self.hurst_pars = np.array([0.35, 0.50, 0.65])
        self.n_dc = len(self.diff_coefs)
        self.n_le = len(self.loc_errors)
        self.n_hp = len(self.hurst_pars)
        self.n_tracks = 50
        self.supports = [
            (self.diff_coefs,),
            (self.diff_coefs,),
            (self.diff_coefs, self.loc_errors),
            (self.diff_coefs, self.hurst_pars),
        ]
        self.occ_sets = [
            np.random.random(size=(self.n_dc, 0)),
            np.random.random(size=(self.n_dc, 0)),
            np.random.random(size=(self.n_dc, self.n_le, 0)),
            np.random.random(size=(self.n_dc, self.n_hp, 0)),
        ]

        np.random.seed(66666)

    def tearDown(self):
        pass

    def test_normalization(self):
        """ Make sure that *defoc_corr* normalizes its output if the *normalize*
        flag is set, for all likelihood types """
        # GAMMA
        support = (self.diff_coefs,)
        occs = np.random.random(size=(self.n_dc, self.n_tracks))
        result = defoc.defoc_corr(occs, support=support, likelihood=GAMMA, **self.kwargs)
        assert (np.abs(result.sum(axis=0) - 1.0) <= 1.0e-6).all()
        assert (defoc.defoc_corr(occs[:,0], support=support, likelihood=GAMMA, **self.kwargs).sum() - 1.0) <= 1.0e-6

        # RBME_MARGINAL
        support = (self.diff_coefs,)
        occs = np.random.random(size=(self.n_dc, self.n_tracks))
        result = defoc.defoc_corr(occs, support=support, likelihood=RBME_MARGINAL, **self.kwargs)
        assert (np.abs(result.sum(axis=0) - 1.0) <= 1.0e-6).all()
        assert (defoc.defoc_corr(occs[:,0], support=support, likelihood=RBME_MARGINAL, **self.kwargs).sum() - 1.0) <= 1.0e-6

        # RBME
        support = (self.diff_coefs, self.loc_errors)
        occs = np.random.random(size=(self.n_dc, self.n_le, self.n_tracks))
        result = defoc.defoc_corr(occs, support=support, likelihood=RBME, **self.kwargs)
        assert (np.abs(result.sum(axis=(0,1)) - 1.0) <= 1.0e-6).all()
        assert (defoc.defoc_corr(occs[:,:,0], support=support, likelihood=RBME, **self.kwargs).sum() - 1.0) <= 1.0e-6

        # FBME
        support = (self.diff_coefs, self.hurst_pars)
        occs = np.random.random(size=(self.n_dc, self.n_hp, self.n_tracks))
        result = defoc.defoc_corr(occs, support=support, likelihood=FBME, **self.kwargs)
        assert (np.abs(result.sum(axis=(0,1)) - 1.0) <= 1.0e-6).all()
        assert (defoc.defoc_corr(occs[:,:,0], support=support, likelihood=FBME, **self.kwargs).sum() - 1.0) <= 1.0e-6

    def test_inf_focal_depth(self):
        """ All defocalization function should return the input unmodified if passed 
        an infinite focal depth """
        kwargs = self.kwargs.copy()
        kwargs.update(dz=np.inf, normalize=False)

        for likelihood, support, occs in zip(LIKELIHOOD_TYPES, self.supports, self.occ_sets):
            result = defoc.defoc_corr(occs, support=support, likelihood=likelihood, **kwargs)
            assert (np.abs(result - occs) <= 1.0e-6).all()

    def test_empty(self):
        """ Make sure all defocalization functions tolerate being passed input with 
        no trajectories """
        for likelihood, support, occs in zip(LIKELIHOOD_TYPES, self.supports, self.occ_sets):
            result = defoc.defoc_corr(occs, support=support, likelihood=likelihood, **self.kwargs)
            assert result.shape == occs.shape
            assert result.shape[-1] == 0
