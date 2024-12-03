import pyemma
from pyemma.msm import bayesian_hidden_markov_model
import numpy as np

"""
This script estimates a Hidden MSM from cluster trajectories.
"""


def main():
    # Load clustering of MD:
    Z = 5.0
    clustering = f"20ns_interval_100ps_ACE_torch_Z_{Z}_isHalo_False_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
    trajectories_array = np.load(
        "/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/clustering/dpa_cluster/results/dpa_inter/"
        + clustering,
        allow_pickle=True,
    )
    # Convert trajectories so that Pyemma can take the input:
    num_atoms = 16080
    trajectories_array = np.reshape(trajectories_array, (-1, num_atoms))
    print(trajectories_array, trajectories_array.shape)
    trajectories = []
    for i in range(trajectories_array.shape[1]):
        trajectories.append(trajectories_array[:, i])

    # Set the number of hidden states.
    num_states = 20
    # Estimate the msm at a lag time tau=1
    model = bayesian_hidden_markov_model(
        trajectories, num_states, 1, show_progress=False, store_hidden=True
    )

    # Save the transition matrix and samples from the posterior
    np.save("transition_matrix.npy", model.transition_matrix)
    np.save("sample_mean.npy", model.sample_mean("transition_matrix"))
    np.save("sample_std.npy", model.sample_std("transition_matrix"))

    # Save the clusters which actually become part of the MSM.
    np.save("active_states.npy", model.active_states)

    # Logging
    print("Transition matrix", model.transition_matrix)
    print("Sample mean", model.sample_mean("transition_matrix"))
    print("Sample std", model.sample_std("transition_matrix"))
    print("Active states", model.active_set)

    model.save("./model.file")


if __name__ == "__main__":
    main()
