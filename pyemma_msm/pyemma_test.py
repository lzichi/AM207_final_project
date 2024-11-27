import pyemma
from pyemma.msm import bayesian_markov_model
import numpy as np

# Script estimates a bayesian markov model based on trajectories.


# clustering="first_1ns_ACE_torch_Z_1.65_isHalo_True_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
# clustering="all_ACE_torch_Z_1.65_isHalo_True_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
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
    trajectories.append(trajectories_array[:, i])
# trajectories = trajectories.astype(int)


model = bayesian_markov_model(trajectories, 1)
model.save("./model.file")
np.save("transition_matrix.npy", model.transition_matrix)
np.save("sample_mean.npy", model.sample_mean("transition_matrix"))
np.save("sample_std.npy", model.sample_std("transition_matrix"))
np.save("active_states.npy", model.active_states)

print("Transition matrix", model.transition_matrix)

print("Sample mean", model.sample_mean("transition_matrix"))

print("Sample std", model.sample_std("transition_matrix"))
print("Active states", model.active_states)
