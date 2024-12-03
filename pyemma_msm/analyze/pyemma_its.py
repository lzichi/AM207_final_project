import pyemma
from pyemma.msm import its
import numpy as np
import matplotlib.pyplot as plt

"""
Examine the intrinsic timescales of clustered trajectories at different lag times.
"""


def main():
    # Load and process cluster trajectories for PyEmma
    Z = 5.0
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
        trajectories.append(np.ascontiguousarray(trajectories_array[:, i]))

    # Perform the implied timescale computation.
    ts = its(
        trajectories,
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
        ],
    )

    pyemma.plots.plot_implied_timescales(ts, show_mle=False, nits=50)
    plt.savefig("its_nomle_50.png", dpi=300)
    plt.close()

    # Compute the implied timescales with Bayesian estimation as well to get error bars.
    bayes_ts = its(
        trajectories, [1, 2, 4, 6, 8, 12, 16, 20, 25], errors="bayes", nits=50
    )
    pyemma.plots.plot_implied_timescales(bayes_ts, show_mle=False)
    plt.savefig("bayes_its_nomle_50.png", dpi=300)


if __name__ == "__main__":
    main()
