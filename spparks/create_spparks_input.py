import numpy as np

def save_spparks_reaction_file(transition_matrix, initial_counts, output_file):
    """
    Save a transition matrix to a SPPARKS-compatible chemistry input file.

    Parameters:
        transition_matrix (numpy.ndarray): The transition matrix.
        initial_counts (numpy.ndarray): Array of initial counts for each species.
        output_file (str): The name of the output file.
    """

    num_states = transition_matrix.shape[0]

    # generate species names
    species = [f"S{i+1}" for i in range(num_states)]

    with open(output_file, "w") as f:
        f.write("# SPPARKS chemistry application\n")
        f.write("# Generated from transition matrix\n\n")
        f.write("seed 12345\n\n")
        f.write("app_style chemistry\n")
        f.write("solve_style linear\n\n")
        f.write("volume 1.0\n\n") 

        # add species
        for s in species:
            f.write(f"add_species {s}\n")
        f.write("\n")

        # add reactions
        reaction_id = 1
        print(num_states)
        for i in range(num_states):
            for j in range(num_states):
                # no self reactons or reactions with nonzero rates
                if i!=j and transition_matrix[i, j] > 0:  
                    f.write(f"add_reaction {reaction_id} {species[i]} {transition_matrix[i, j]/(100e-12):.6e} {species[j]}\n")
                    reaction_id += 1
        f.write("\n")

        # add initial counts
        for s, count in zip(species, initial_counts):
            if(count > 0):
                f.write(f"count {s} {count}\n")
        f.write("\n")

        f.write("stats 1e-10\n\n")
        f.write("run 2e-8\n")

    print(f"SPPARKS chemistry input file saved as {output_file}")

def get_initial_state_hidden(labels_initial):
    """
    Generate initial state for kMC based on MD trajectory.

    Parameters:
        labels_initial (numpy.ndarray): labels of HMSM for every ACE for initial state.
    """
    max_cluster = np.max(labels_initial)
    initial_condition = [np.where(labels_initial == i)[0].shape[0] for i in range(max_cluster+1)]
    return initial_condition

def get_initial_state(file_in):
    """
    Generate initial state for kMC based on MD trajectory.

    Parameters:
        file_in (str): path to file with labels for ACE descs for MD trajectory.
    
    """
    # load and reshape labels
    labels = np.load(file_in)
    labels = labels.reshape(200, 16080)

    # only need first MD frame
    max_cluster = np.max(labels.flatten())
    labels_initial = labels[0]
    initial_condition = [np.where(labels_initial == i)[0].shape[0] for i in range(max_cluster+1)]

    assert(np.sum(initial_condition) == 16080)
    return initial_condition


if __name__ == "__main__":

    # Z = 3.5
    labels_in = "/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/clustering/dpa_cluster/results/dpa_inter/20ns_interval_100ps_64_ACE_torch_Z_3.5_isHalo_False_maxk_50_rmax_6.0_lmax_3_nmax_2.npy"
    initial_counts = np.array(get_initial_state(labels_in))
    connected_indices = np.load("/n/netscratch/kozinsky_lab/Lab/mdescoteaux/24_11_21_Pyemmacluster_LiOnly/20ns_interval_100ps_differentZ/nohalo_20ns_every100ps_3.5/active_states.npy")

    initial_counts = initial_counts[connected_indices]
    print(np.sum(initial_counts), initial_counts.shape) # sanity check
    transition_matrix_file = "/n/netscratch/kozinsky_lab/Lab/mdescoteaux/24_11_21_Pyemmacluster_LiOnly/20ns_interval_100ps_differentZ/nohalo_20ns_every100ps_3.5/transition_matrix.npy"
    transition_matrix = np.load(transition_matrix_file)
    output_file = "spparks_gill_in/20ps_1ps_3.5_spparks_gill_correct.in"
    save_spparks_reaction_file(transition_matrix, initial_counts, output_file)

    # Z = 5.0 
    labels_in = np.load("HiddenBayesian_5.0_hidden_trajectory.npy")[:, 0]
    initial_counts = np.array(get_initial_state_hidden(labels_in))

    print(np.sum(initial_counts), initial_counts.shape) # sanity check
    transition_matrix_file = "/n/netscratch/kozinsky_lab/Lab/mdescoteaux/24_11_21_Pyemmacluster_LiOnly/20ns_interval_100ps_differentZ/hidden/nohalo_100ps_every1ps_hidden_5.0/transition_matrix.npy"
    transition_matrix = np.load(transition_matrix_file)
    output_file = "spparks_gill_in/20ps_1ps_5.0_spparks_gill_1s_new.in"
    save_spparks_reaction_file(transition_matrix, initial_counts, output_file)