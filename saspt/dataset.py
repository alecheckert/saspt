import os, dask, warnings, numpy as np, pandas as pd 
from typing import Tuple, List, Callable
from dask.diagnostics import ProgressBar
pbar = ProgressBar()

from .constants import DEFAULT_CONDITION_COL, DEFAULT_CONDITION
from .io import load_detections
from .lik import make_likelihood, make_likelihood_from_params, Likelihood
from .parameters import StateArrayParameters
from .sa import StateArray 
from .trajectory_group import TrajectoryGroup
from .plot import (
    heatmap_by_condition,
    lineplot_by_condition,
)
from .utils import normalize_2d

class StateArrayDataset:
    """ A collection of state arrays for each of a collection of sets of 
    trajectories. Each set of trajectories is assumed to be stored in a 
    distinct file.

    Implements methods to:
        - parallelize state array computations across multiple files
        - visualize variability in state array output between different sets
            of trajectories
        - visualize chnages between arbitrary experimental conditions

    init
    ----
        paths           :   pandas.DataFrame encapsulating target files and
                            associated experimental conditions. Contains at
                            minimum the column *path_col* and optionally a
                            column indicating the experimental condition to
                            which each file belongs (*condition_col*)

        likelihood      :   likelihood function on which to define the state
                            array

        params          :   options for StateArray inference. Also determines
                            the degree of parallelism in the StateArrayDataset

        path_col        :   column in *paths* indicating the path to the 
                            corresponding file

        condition_col   :   column in *paths* indicating experimental condition

    example initialization
    ----------------------
        from saspt import StateArrayParameters, make_likelihood, StateArrayDataset

        # Specify target files and experimental conditions
        paths = pd.DataFrame({
            'filepath':  ['some_tracks_1.csv', 'some_tracks_2.csv'],
            'condition': ['control', 'transfection'],
        })

        # Make likelihood function
        likelihood = make_likelihood(likelihood_type=RBME, **kwargs)

        # Make StateArrayParameters, including the number of parallel processes
        # (*num_workers*) to use in inference
        params = StateArrayParameters(pixel_size_um=0.16, frame_interval=0.00748,
            focal_depth=0.7, num_workers=2)

        # Initialize the StateArrayDataset
        with StateArrayDataset(paths, likelihood, params, path_col="filepath",
            condition_col="condition") as SAD:

            # do stuff

    """
    def __init__(
        self,
        paths: pd.DataFrame,
        likelihood: Likelihood, 
        params: StateArrayParameters,
        path_col: str,
        condition_col: str=None,
        **kwargs
    ):
        # Check inputs
        if path_col not in paths.columns:
            raise ValueError(f"input DataFrame missing path column {path_col}")
        if (condition_col is not None) and (condition_col not in paths.columns):
            raise ValueError(f"column {condition_col} not found in input DataFrame")

        # Exclude invalid paths
        valid = paths[path_col].map(os.path.isfile).astype(bool)
        if not valid.all():
            warnings.warn(f"only {valid.sum()}/{len(paths)} input files exist")
        paths = paths[valid].reset_index(drop=True)

        # If the user does not specify experiment condition, assume that all input
        # files are in the same experimental condition
        if condition_col is None:
            condition_col = DEFAULT_CONDITION_COL
            paths[condition_col] = DEFAULT_CONDITION

        # All unique experimental conditions
        self.conditions = paths[condition_col].unique()

        self.paths = paths 
        self.likelihood = likelihood
        self.params = params 
        self.path_col = path_col
        self.condition_col = condition_col

        # Prevent child StateArray objects from using their own progress bars
        self.progress_bar = self.params.progress_bar
        self.params.progress_bar = False

        self.condition_map = self.paths.groupby(self.path_col)[self.condition_col].first()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, traceback):
        return etype is None

    def __repr__(self):
        return "StateArrayDataset:\n  " + "\n  ".join([
            f"{k : <18} : {getattr(self, k)}" for k in [
                "likelihood_type", "shape", "n_files", "path_col", "condition_col",
                "conditions",
            ]
        ])

    @classmethod
    def from_kwargs(cls, paths: pd.DataFrame, likelihood_type: str, **kwargs):
        """ Convenience constructor; generate a StateArrayDataset from keyword
        arguments rather than from objects.

        args
        ----
            paths               :    see StateArrayDataset docstring
            likelihood_type     :    type of likelihood function to use
            path_col            :    see StateArrayDataset docstring
            condition_col       :    see StateArrayDataset docstring
            pixel_size_um       :    see StateArrayParameters docstring
            frame_interval      :    see StateArrayParameters docstring
            focal_depth         :    see StateArrayParameters docstring
            etc.                :    see StateArrayParameters docstring

        returns
        -------
            new instance of StateArrayDataset
        """
        params = StateArrayParameters(**kwargs)
        likelihood = make_likelihood_from_params(likelihood_type, params, **kwargs)
        return cls(paths, likelihood, params, **kwargs)

    ################
    ## PROPERTIES ##
    ################

    @property 
    def n_files(self) -> int:
        """ Number of input files in this StateArrayDataset """
        return len(self.paths)

    @property
    def shape(self) -> Tuple[int]:
        """ Shape of the parameter grid on which this state array is defined """
        return self.likelihood.shape 

    @property 
    def likelihood_type(self) -> str:
        """ Name of the likelihood function """
        return self.likelihood.name

    @property
    def n_diff_coefs(self) -> int:
        """ The number of distinct diffusion coefficients in the parameter
        grid on which this state array is defined. If the underlying likelihood
        function does NOT use diffusion coefficient as a parameter, returns 0. """
        if not hasattr(self, "_n_diff_coefs"):
            self._n_diff_coefs = len(self.likelihood.diff_coefs) if \
                hasattr(self.likelihood, "diff_coefs") else 0
        return self._n_diff_coefs

    @property 
    def jumps_per_file(self) -> np.ndarray:
        """ Total number of jumps observed in each file. 

        returns
        -------
            1D numpy.ndarray of shape (n_files,), the number of jumps
                observed in each file
        """
        if not hasattr(self, "_jumps_per_file"):
            self._jumps_per_file = np.asarray(self.processed_track_statistics["n_jumps"])
        return self._jumps_per_file

    @property
    def raw_track_statistics(self) -> pd.DataFrame:
        """ Statistics on the raw trajectories from each file.

        returns
        -------
            pandas.DataFrame, where each row corresponds to one file
        """
        if not hasattr(self, "_raw_track_statistics"):
            self._raw_track_statistics = self._get_raw_track_statistics()
        return self._raw_track_statistics   

    @property
    def processed_track_statistics(self) -> pd.DataFrame:
        """ Statistics on preprocessed trajectories from each file.

        Differences between *processed_track_statistics* and *raw_track_statistics*
        reflect the effect of TrajectoryGroup.preprocess, which removes singlets
        and breaks large trajectories into smaller pieces.

        returns
        -------
            pandas.DataFrame, where each row corresponds to one file
        """
        if not hasattr(self, "_processed_track_statistics"):
            self._processed_track_statistics = self._get_processed_track_statistics()
        return self._processed_track_statistics

    @property 
    def naive_occs(self) -> np.ndarray:
        """ Naive estimate for the occupations of each state on the parameter grid
        for each file in this StateArrayDataset.

        Unnormalized, so that the total "occupation" across all states per
        file is equal to the number of jumps observed in that file.

        returns
        -------
            numpy.ndarray of shape (self.n_files, *self.shape)
        """
        if not hasattr(self, "_naive_occs"):
            if self.n_files > 0:
                self._naive_occs = np.asarray(self.parallel_map(
                    self.calc_naive_occs,
                    self.paths[self.path_col],
                    progress_bar=self.progress_bar,
                ))
            else:
                self._naive_occs = np.zeros((self.n_files, *self.shape), dtype=np.float64)
        return self._naive_occs

    @property 
    def posterior_occs(self) -> np.ndarray:
        """ Posterior mean estimate for the occupations of each state on the
        parameter grid for each file in this StateArrayDataset.

        Unnormalized, so that the total "occupation" across all states per
        file is equal to the number of jumps observed in that file.

        returns
        -------
            numpy.ndarray of shape (self.n_files, *self.shape)
        """
        if not hasattr(self, "_posterior_occs"):
            if self.n_files > 0:
                self._posterior_occs = np.asarray(self.parallel_map(
                    self.calc_posterior_occs,
                    self.paths[self.path_col],
                    progress_bar=self.progress_bar,
                ))
            else:
                self._posterior_occs = np.zeros((self.n_files, *self.shape), dtype=np.float64)
        return self._posterior_occs

    @property 
    def marginal_naive_occs(self) -> np.ndarray:
        """ Likelihood functions for each movie in this dataset, marginalized
        on the diffusion coefficient. This provides a naive estimate of the 
        occupation of each state.

        Unnormalized, so that the total "occupation" across all states per
        file is equal to the number of jumps observed in that file.

        returns
        -------
            2D numpy.ndarray of shape (n_files, n_diff_coefs)
        """
        if not hasattr(self, "_marginal_naive_occs"):
            if "diff_coef" not in self.likelihood.parameter_names:
                self._marginal_naive_occs = np.zeros((self.n_files, 0), dtype=np.float64)
            else:
                i = self.likelihood.parameter_names.index("diff_coef")
                axes = [j+1 for j in range(len(self.likelihood.parameter_names)) if j != i]
                self._marginal_naive_occs = self.naive_occs.sum(axis=tuple(axes))
        return self._marginal_naive_occs

    @property 
    def marginal_posterior_occs(self) -> np.ndarray:
        """ Posterior mean state occupations marginalized on diffusion coefficient
        for each file in this dataset.

        Unnormalized, so that the total "occupation" across all states per
        file is equal to the number of jumps observed in that file.

        returns
        -------
            numpy.ndarray of shape (n_files, n_diff_coefs). Note that the occupations
                for file *i* are scaled by the total number of jumps in that file.
        """
        if not hasattr(self, "_marginal_posterior_occs"):
            if "diff_coef" not in self.likelihood.parameter_names:
                self._marginal_posterior_occs = np.zeros((self.n_files, 0), dtype=np.float64)
            else:
                i = self.likelihood.parameter_names.index("diff_coef")
                axes = [j+1 for j in range(len(self.likelihood.parameter_names)) if j != i]
                self._marginal_posterior_occs = self.posterior_occs.sum(axis=tuple(axes))
        return self._marginal_posterior_occs

    @property
    def marginal_posterior_occs_dataframe(self) -> pd.DataFrame:
        """ pandas.DataFrame representation of the naive and posterior state
        occupations for each file in this StateArrayDataset, marginalized on
        diffusion coefficient.

        returns
        -------
            pandas.DataFrame. Each column corresponds to one of the files in
                this dataset and each row to the marginal occupation of a 
                distinct diffusion coefficient.
        """
        if not hasattr(self, "_marginal_posterior_occs_dataframe"):
            cols = ['diff_coef', 'naive_occupation', 'posterior_occupation', 'n_jumps'] \
                + list(self.paths.columns)

            # Likelihood does not support diffusion coefficient
            if self.n_diff_coefs == 0:
                df = pd.DataFrame(index=np.array([]), columns=cols, dtype=np.float64)

            # Likelihood supports diffusion coefficient
            else:
                i = self.likelihood.parameter_names.index("diff_coef")
                diff_coefs = self.likelihood.parameter_values[i]
                M = self.n_diff_coefs * self.n_files
                index = np.arange(M)
                df = pd.DataFrame(index=index, columns=cols, dtype=np.float64)
                if self.n_files > 0:
                    df['diff_coef'] = np.tile(diff_coefs, self.n_files)
                    df['naive_occupation'] = normalize_2d(self.marginal_naive_occs, axis=1).ravel()
                    df['posterior_occupation'] = normalize_2d(self.marginal_posterior_occs, axis=1).ravel()
                    df['n_jumps'] = np.repeat(self.jumps_per_file, self.n_diff_coefs)
                    for c in self.paths.columns:
                        df[c] = np.repeat(self.paths[c], self.n_diff_coefs).reset_index(drop=True)

            self._marginal_posterior_occs_dataframe = df 
        return self._marginal_posterior_occs_dataframe

    def infer_posterior_by_condition(self, col: str, normalize: bool=False
        ) -> (np.ndarray, List[str]):
        """ Aggregate trajectories across files by grouping on an arbitrary
        column in *self.paths*. Run state array inference on each group. 

        args
        ----
            col         :   str, a column in *self.paths* to group by
            normalize   :   bool, normalize posterior occupations after running

        returns
        -------
            (
                2D numpy.ndarray of shape (n_conditions, n_diff_coefs), 
                    posterior occupations for each condition marginalized
                    on diffusion coefficient;

                list of str of length n_conditions, the names of the conditions
                    corresponding to the first axis
            )
        """
        posterior_occs, conditions = self.apply_by(col,
            self.calc_marginal_posterior_occs, is_variadic=True)
        posterior_occs = np.asarray(posterior_occs)
        if normalize:
            posterior_occs = normalize_2d(posterior_occs, axis=1)
        return posterior_occs, conditions

    #############
    ## METHODS ##
    #############

    def clear(self):
        """ Delete expensive cached attributes """
        for attr in ["_n_files", "_naive_occs", "_posterior_occs"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def calc_naive_occs(self, *track_paths: str) -> np.ndarray:
        """
        args
        ----
            track_paths :   paths to files with trajectories, readable by
                            saspt.utils.load_detections

        returns
        -------
            numpy.ndarray of shape *self.shape*, occupations scaled by the
                total number of jumps observed for each SPT experiment
        """
        SA = self._init_state_array(*track_paths)
        return SA.naive_occs

    def calc_posterior_occs(self, *track_paths: str) -> np.ndarray:
        """
        args
        ----
            track_paths :   paths to files with trajectories, readable by
                            saspt.utils.load_detections

        returns
        -------
            numpy.ndarray of shape *self.shape*, mean posterior occupations
                scaled by the total number of jumps observed for each SPT experiment
        """
        SA = self._init_state_array(*track_paths)
        return SA.n_jumps * SA.posterior_occs

    def calc_marginal_naive_occs(self, *track_paths: str) -> np.ndarray:
        """ Calculate the likelihood function for a particular set of 
        trajectories, marginalized on the diffusion coefficient.

        args
        ----
            track_paths :   paths to files with trajectories readable
                            by saspt.utils.load_detections

        returns
        -------
            numpy.ndarray of shape *n_diff_coefs*, occupations scaled by the
                total number of jumps observed in these trajectories
        """
        return self.likelihood.marginalize_on_diff_coef(
            self.calc_naive_occs(*track_paths))

    def calc_marginal_posterior_occs(self, *track_paths: str) -> np.ndarray:
        """ Calculate the posterior mean state occupations for a particular
        set of trajectories, marginalized on diffusion coefficient.

        args
        ----
            track_paths :   paths to files with trajectories readable
                            by saspt.utils.load_detections

        returns
        -------
            numpy.ndarray of shape *n_diff_coefs*, occupations scaled
                by the total number of jumps observed in this set of 
                trajectories
        """
        return self.likelihood.marginalize_on_diff_coef(
            self.calc_posterior_occs(*track_paths))

    ##############
    ## PLOTTING ##
    ##############

    def naive_heat_map(self, out_png: str, normalize: bool=True, 
        order_by_size: bool=True, **kwargs):
        """ Plot the naive state occupation estimates (marginalized on
        diffusion coefficient) for each file in this StateArrayDataset,
        grouping by experimental condition.

        args
        ----
            out_png     :   path to output plot
            normalize   :   normalize the distribution over each file
                            to unit intensity. If False, each file has 
                            intensity proportional to the number of jumps
                            observed in that file
            order_by_size:  within each condition, order the files by 
                            decreasing total number of jumps
        """
        if "diff_coef" not in self.likelihood.parameter_names:
            warnings.warn(f"naive heat map not available for " \
                "likelihood {self.likelihood.name}")
        elif self.n_files == 0:
            warnings.warn(f"number of files is 0; cannot produce naive heat map")
        else:
            dists = {c: self.marginal_naive_occs[df.index,:].copy() \
                for c, df in self.paths.groupby(self.condition_col)}
            heatmap_by_condition(
                out_png=out_png,
                diff_coefs=self.likelihood.diff_coefs,
                dists=dists,
                ylabel="File",
                cbar_label="Naive\noccupation",
                normalize=normalize,
                order_by_size=order_by_size,
                **kwargs
            )

    def naive_line_plot(self, out_png: str, **kwargs):
        """ Plot the naive state occupation estimates (marginalized on
        diffusion coefficient) for each file in this StateArrayDataset,
        grouping by experimental condition.

        This is essentially an alternative representation of the same 
        information in *naive_heat_map*.

        args
        ----
            out_png     :   path to output plot
        """
        if "diff_coef" not in self.likelihood.parameter_names:
            warnings.warn(f"naive line plot not available for " \
                "likelihood {self.likelihood.name}")
        elif self.n_files == 0:
            warnings.warn(f"number of files is 0; cannot produce naive line plot")
        else:
            dists = {c: self.marginal_naive_occs[df.index,:].copy() \
                for c, df in self.paths.groupby(self.condition_col)}
            lineplot_by_condition(
                out_png=out_png,
                diff_coefs=self.likelihood.diff_coefs,
                dists=dists,
                normalize=True,
                ylabel="Naive\noccupation",
                **kwargs
            )

    def posterior_heat_map(self, out_png: str, normalize: bool=True, 
        order_by_size: bool=True, **kwargs):
        """ Plot the mean posterior state occupations (marginalized on
        diffusion coefficient) for each file in this StateArrayDataset,
        grouping by experimental condition.

        args
        ----
            out_png     :   path to output plot
            normalize   :   normalize the distribution over each file
                            to unit intensity. If False, each file has 
                            intensity proportional to the number of jumps
                            observed in that file
            order_by_size:  within each condition, order the files by 
                            decreasing total number of jumps
        """
        if "diff_coef" not in self.likelihood.parameter_names:
            warnings.warn(f"posterior heat map not available for " \
                "likelihood {self.likelihood.name}")
        elif self.n_files == 0:
            warnings.warn(f"number of files is 0; cannot produce posterior heat map")
        else:
            dists = {c: self.marginal_posterior_occs[df.index,:].copy() \
                for c, df in self.paths.groupby(self.condition_col)}
            heatmap_by_condition(
                out_png=out_png,
                diff_coefs=self.likelihood.diff_coefs,
                dists=dists,
                ylabel="File",
                cbar_label="Posterior\noccupation",
                normalize=normalize,
                order_by_size=order_by_size,
                **kwargs
            )

    def posterior_line_plot(self, out_png: str, **kwargs):
        """ Plot the mean posterior state occupations (marginalized on
        diffusion coefficient) for each file in this StateArrayDataset,
        grouping by experimental condition.

        This is essentially an alternative representation of the same 
        information in *posterior_heat_map*.

        args
        ----
            out_png     :   path to output plot
        """
        if "diff_coef" not in self.likelihood.parameter_names:
            warnings.warn(f"posterior line plot not available for " \
                "likelihood {self.likelihood.name}")
        elif self.n_files == 0:
            warnings.warn(f"number of files is 0; cannot produce posterior line plot")
        else:
            dists = {c: self.marginal_posterior_occs[df.index,:].copy() \
                for c, df in self.paths.groupby(self.condition_col)}
            lineplot_by_condition(
                out_png=out_png,
                diff_coefs=self.likelihood.diff_coefs,
                dists=dists,
                normalize=True,
                ylabel="Posterior\noccupation",
                **kwargs
            )

    ######################
    ## OBJECT UTILITIES ##
    ######################

    def _load_tracks(self, *track_paths: str) -> TrajectoryGroup:
        """ Load trajectories from one or more files directly into a
        TrajectoryGroup object. """
        T = TrajectoryGroup.from_params(load_detections(*track_paths), self.params)
        return T if (T.n_tracks <= self.params.sample_size) else \
            T.subsample(self.params.sample_size)

    def _init_state_array(self, *track_paths: str) -> StateArray:
        """ Load trajectories from one or more files, and initialize a 
        StateArray over them """
        return StateArray(self._load_tracks(*track_paths), self.likelihood, self.params)

    def _get_processed_track_statistics(self) -> pd.DataFrame:
        """ Calculate some statistics on the preprocessed trajectories for each 
        file in this StateArrayDataset.

        returns
        -------
            pandas.DataFrame with each row corresponding to one file. Columns
                correspond to different statistics
        """
        @dask.delayed
        def g(filepath: str) -> dict:
            T = self._load_tracks(filepath)
            stats = T.processed_track_statistics
            stats[self.path_col] = filepath
            return stats
        result = pd.DataFrame(self.parallel_map(g, self.paths[self.path_col]))

        # Conceivable that there are zero files in this dataset
        if len(result) == 0:
            result[self.path_col] = self.paths[self.path_col]
            for stat in TrajectoryGroup.statistic_names:
                result[stat] = pd.Series([], dtype=np.float64)
                
        # Sanity check
        assert (result[self.path_col] == self.paths[self.path_col]).all()

        # Map all metadata from the input paths DataFrame to the track statistics dataframe
        for c in filter(lambda c: c!=self.path_col, self.paths.columns):
            result[c] = self.paths[c]

        return result

    def _get_raw_track_statistics(self) -> pd.DataFrame:
        """ Calculated some statistics on the raw trajectories for each file in 
        this StateArrayDataset.

        returns
        -------
            pandas.DataFrame with each row corresponding to one file. Columns
                correspond to different statistics
        """
        @dask.delayed
        def g(filepath: str) -> dict:
            T = TrajectoryGroup.from_params(load_detections(filepath), self.params)
            stats = T.raw_track_statistics
            stats[self.path_col] = filepath
            return stats
        result = pd.DataFrame(self.parallel_map(g, self.paths[self.path_col]))

        # Conceivable that there are zero files in this dataset
        if len(result) == 0:
            result[self.path_col] = self.paths[self.path_col]
            for stat in TrajectoryGroup.statistic_names:
                result[stat] = pd.Series([], dtype=np.float64)

        # Sanity check
        assert (result[self.path_col] == self.paths[self.path_col]).all()

        # Map all metadata from the input paths DataFrame to the track statistics dataframe
        for c in filter(lambda c: c!=self.path_col, self.paths.columns):
            result[c] = self.paths[c]

        return result       

    def parallel_map(self, func, args, msg: str=None, progress_bar: bool=False):
        """ Parallelize a function across multiple arguments using a process-based
        dask scheduler.

        args
        ----
            func    :   function to apply
            args    :   operands to *func*
            msg     :   a message to show to the user

        returns
        -------
            list of *func(arg)* for each *arg* in *args*
        """
        if progress_bar:
            if msg: print(msg)
            pbar.register()
        result = dask.compute(*map(dask.delayed(func), args),
            num_workers=self.params.num_workers, scheduler="processes")
        if progress_bar:
            pbar.unregister()
        return result 

    def apply_by(self, col: str, func: Callable, is_variadic: bool=False, 
        **kwargs) -> (list, List[str]):
        """ Apply a function in parallel to groups of files in *self.paths*.

        args
        ----
            col         :   str, column in *self.paths* to group by
            is_variadic :   bool, *func* is variadic
            func        :   Callable, function to apply by group.

        note on usage
        -------------
        If *is_variadic* is *True*, then *func* should have the signature
            func(*paths: str, **kwargs) -> result

        If *is_variadic* is *False*, then *func* should have the signature
            func(paths: List[str], **kwargs) -> result

        returns
        -------
            (
                list, outputs of *func* on each file group,
                list of str, names of conditions matching file groups 
            )

        """
        if col not in self.paths.columns:
            raise ValueError(f"column {col} not present in self.paths; " \
                "available columns are {', '.join(self.paths.columns)}")

        G = self.paths.groupby(col)

        # List of string, conditions
        conditions = G.apply(lambda i: i.name).tolist()

        # List of list of filepaths
        file_groups = G[self.path_col].apply(list).tolist()

        # Run the function in parallel on this group
        if is_variadic:
            result = self.parallel_map(lambda paths: func(*paths, **kwargs), file_groups)
        else:
            result = self.parallel_map(lambda paths: func(paths, **kwargs), file_groups)

        return result, conditions
