import os, numpy as np, pandas as pd
from glob import glob
from saspt import StateArray, StateArrayDataset, RBME, load_detections

# Parent directory for this file
SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

def exampleStateArray():
    """ Demonstrates some features of the saspt.StateArray class,
    which runs state arrays on a single group of trajectories. """

    # Input files containing trajectories
    input_files = glob(os.path.join(SAMPLE_DIR, "u2os_ht_nls_7.48ms", "*.csv"))

    # Load and concatenate detected particles across files
    detections = load_detections(*input_files)

    # Configuration for the state array, including microscope parameters
    settings = dict(
        # Type of likelihood function to use
        likelihood_type = RBME,

        # Camera pixel size (after magnification) in microns
        pixel_size_um = 0.16,

        # Frame interval in seconds
        frame_interval = 0.00748,

        # Objective focal depth in microns
        focal_depth = 0.7,

        # Ignore detections before this frame
        start_frame = 2000,

        # Show progress and be a little verbose, where relevant
        progress_bar = True,
    )

    # Create a StateArray from these detections
    with StateArray.from_detections(detections, **settings) as SA:

        # Show some statistics about this StateArray, including the
        # number of trajectories, the shape of the parameter grid 
        # on which the array is defined, etc.
        print(SA)

        # Show some statistics about the trajectories after preprocessing,
        # including mean trajectory length, fraction of singlets, etc.
        print("Trajectory statistics:")
        for k, v in SA.trajectories.processed_track_statistics.items():
            print(f"{k : <20} : {v}")

        # Get the occupation of each state under the prior distribution
        prior_occs = SA.prior_occs

        # Get the naive estimate for the occupations of each state
        naive_occs = SA.naive_occs

        # Get the posterior estimate for the occupations of each state
        posterior_occs = SA.posterior_occs

        # Marginalize the posterior occupations over localization error
        marginal_posterior_occs = SA.marginalize_on_diff_coef(posterior_occs)

        # Example calculation: estimated fraction of trajectories that 
        # have diffusion coefficient less than 0.5 µm/sec
        p0 = marginal_posterior_occs[SA.diff_coefs < 0.5].sum()

        # Example calculation: estimated fraction of trajectories that
        # have diffusion coefficient between 1.0 and 5.0 µm2/sec
        p1 = marginal_posterior_occs[np.logical_and(
            SA.diff_coefs>=1.0, SA.diff_coefs<5.0
        )].sum()

        # Save the posterior state occupations to a CSV
        SA.occupations_dataframe.to_csv("posterior_occupations.csv", index=False)

        ###############
        ## PLOT DEMO ##
        ###############

        # Plot state occupations
        SA.plot_occupations("posterior_occupations.png")

        # Plot the probabilities for each trajectory-state assignment
        SA.plot_assignment_probabilities("assignment_probabilities.png")

        # Plot the probabilities for each trajectory-state
        # assignment as a function of frame index
        SA.plot_temporal_assignment_probabilities("assignment_probabilities_by_frame.png")


def exampleStateArrayDataset():
    """ Demonstrates some features of the saspt.StateArrayDataset class.

    This class offers methods to:
        - parallelize state array computations

        - visualize state array output across multiple target files

        - compare between experimental conditions

        - conveniently marginalize the posterior distribution onto
            diffusion coefficient, which is often the main parameter
            of interest
    """

    # In this experiment, we're comparing two HaloTag constructs: HaloTag-NLS
    # and RARA-HaloTag. The file 'experiment_conditions.csv' contains two
    # columns:
    #
    #   'filepath':  path to the file containing trajectories from the SPT
    #                experiment
    #
    #   'condition': the experimental condition that each file belongs to;
    #                either 'ht-nls' or 'rara-ht'
    experiment_conditions = pd.read_csv('experiment_conditions.csv')

    settings = dict(
        # Type of likelihood function to use
        likelihood_type = RBME,

        # Camera pixel size (after magnification) in  microns
        pixel_size_um = 0.16,

        # Time between frames in seconds
        frame_interval = 0.00748,

        # Microscope focal depth in microns
        focal_depth = 0.7,

        # Ignore trajectories before this frame
        start_frame = 1000,

        # Show a progress bar
        progress_bar = True,

        # Number of parallel processes to use. Recommended not to set
        # this greater than the number of CPUs
        num_workers = 4,

        # Column in *experiment_conditions* that encodes the path to the 
        # target file
        path_col = "filepath",

        # Column in *experiment_conditions* that encodes the condition 
        # to which each file belongs
        condition_col = "condition",
    )

    with StateArrayDataset.from_kwargs(experiment_conditions, **settings) as SAD:

        # Show some basic information about this StateArrayDataset
        print(SAD)

        # Save some statistics on each group of trajectories, including the mean
        # trajectory length, fraction of singlets, etc.
        print("\nTrajectory statistics:")
        print(SAD.processed_track_statistics)
        SAD.processed_track_statistics.to_csv("track_statistics.csv", index=False)

        # Get the posterior state occupations for each file in the dataset
        # as a pandas.DataFrame
        posterior_df = SAD.marginal_posterior_occs_dataframe
        posterior_df.to_csv("posterior_occupations.csv", index=False)

        # Example calculation: Estimate the fraction of particles with
        # diffusion coefficients lower than 0.5 µm2/sec for each file in this
        # StateArrayDataset
        print("\nExample calculation: fraction of particles with diff coef < 0.5 µm2/sec:")
        print(posterior_df.loc[posterior_df['diff_coef']<0.5].groupby(
            'filepath')['posterior_occupation'].sum())

        ###############
        ## PLOT DEMO ##
        ###############

        # Make a heat map of the posterior occupations marginalized on 
        # diffusion coefficient, grouped by experimental condition
        SAD.posterior_heat_map("posterior_heat_map_by_condition.png")

        # Make the same plot, but leave the posterior occupations
        # unnormalized. The unnormalized posterior distribution for each 
        # SPT experiment is proportional to the number of particle-particle
        # jumps observed in that experiment. This lets us see the 
        # heterogeneity in dataset size between SPT experiments.
        SAD.posterior_heat_map("posterior_heat_map_by_condition_unnormalized.png",
            normalize=False)

        # Make the same plot, but use the "naive" state occupations which
        # provide a quicker, less precise way to estimate the posterior distribution
        SAD.naive_heat_map("naive_heat_map_by_condition.png")
        SAD.naive_line_plot("naive_line_plot_by_condition.png")

        # Make a line plot of the posterior occupations marginalized on
        # diffusion coefficient. This is essentially the same information
        # as the previous plot, but in a different format.
        SAD.posterior_line_plot("posterior_line_plot_by_condition.png")

if __name__ == '__main__':
    exampleStateArray() 
    exampleStateArrayDataset() 
