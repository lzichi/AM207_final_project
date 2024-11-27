import pyemma
from pyemma.msm import bayesian_markov_model, its
import numpy as np
import matplotlib.pyplot as plt


# Script computes intrinsic time scale of trajectories.


def main():
    # clustering="all_ACE_torch_Z_1.65_isHalo_False_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
    # clustering="first_1ns_ACE_torch_Z_1.65_isHalo_False_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
    Z = 1.65
    clustering = f"20ns_interval_100ps_ACE_torch_Z_{Z}_isHalo_False_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
    trajectories_array = np.load(
        "/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/clustering/dpa_cluster/results/dpa_inter/"
        + clustering,
        allow_pickle=True,
    )

    num_atoms = 16080
    trajectories_array = np.reshape(trajectories_array, (-1, num_atoms))

    print(trajectories_array, trajectories_array.shape)
    trajectories = []
    for i in range(trajectories_array.shape[1]):
        trajectories.append(trajectories_array[:, i] + 1)
    # trajectories = trajectories.astype(int)
    ts = its(trajectories, [1, 2, 3, 4, 5])

    pyemma.plots.plot_implied_timescales(ts)
    plt.savefig("its.png", dpi=300)


if __name__ == "__main__":
    main()
