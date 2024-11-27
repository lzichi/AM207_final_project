import numpy as np
import pyemma

# Script saves the list of active states given an estimated markov model
# and prints some information on the clusters and resulting transition matrix

Z = 1.65
clustering = f"20ns_interval_100ps_64_ACE_torch_Z_{Z}_isHalo_False_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
trajectories_array = np.load(
    "/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/clustering/dpa_cluster/results/dpa_inter/"
    + clustering,
    allow_pickle=True,
)

print("Max state", np.max(trajectories_array))
print("Min state", np.min(trajectories_array))
transition_matrix = np.load("./transition_matrix.npy", allow_pickle=True)
print("Transition_matrix size", transition_matrix.shape)


model = pyemma.load("model.file")
np.save("active_states.npy", model.active_set)
print("active states", model.active_set)
