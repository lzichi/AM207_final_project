import numpy as np
import torch
import ase
import sys
from flarelite.ace_descriptor import AtomicClusterExpansionDescriptor

torch.set_default_dtype(torch.float64)

def create_ACE(file, store_file, rmax_in, nmax_in, lmax_in, 
               range, env, frames_interval=1, start=0):
    """
    Generate and save Atomic Cluster Expansion (ACE) descriptors from atomic trajectory data.

    Parameters:
        file (str): Path to the input atomic trajectory file.
        store_file (str): Path and base name for the output descriptor file.
        rmax_in (float): Maximum cutoff radius.
        nmax_in (int): Highest order of radial basis functions.
        lmax_in (int): Highest order of angular basis functions.
        range (int): Number of atomic environments to include in the descriptor slice.
        env (str): Label for the environment, used for annotation in the output file.
        frames_interval (int, optional): Interval for reading frames from the trajectory.
        start (int, optional): Index of the starting frame.
    """
    # read all frames
    atoms_all = ase.io.read(file, '%s::%s'%(start, frames_interval))

    # construct ACE basis
    num_species = 4
    cutoff_matrix = torch.full((num_species, num_species), rmax_in)
    species = torch.tensor([3, 15, 16, 17])
    ace = AtomicClusterExpansionDescriptor(
            species=species,
            cutoff_matrix=cutoff_matrix,
            nmax=nmax_in,
            lmax=lmax_in,
            )
    atomic_data_dict = ace.atomic_data.Atoms2AtomicData(atoms_all[0])
    B20 = ace(atomic_data_dict, derivatives=False)

    # save first frame's desc to get size of ACE
    desc_slice = B20[:range]
    norm = np.linalg.norm(desc_slice, axis=1, keepdims=True) # normalize
    desc_slice_reshape = desc_slice/norm
    desc_all = np.zeros((0, B20.shape[1]))
    desc_all = np.vstack((desc_all, desc_slice_reshape))

    # save ACE for all frames
    for atoms in atoms_all[1:]:
        atomic_data_dict = ace.atomic_data.Atoms2AtomicData(atoms)
        B20 = ace(atomic_data_dict, derivatives=False)
        desc_slice = B20[:range]
        norm = np.linalg.norm(desc_slice, axis=1, keepdims=True) # normalize
        desc_slice_reshape = desc_slice/norm
        desc_all = np.vstack((desc_all, desc_slice_reshape))

    # save descs and label
    labels_all = np.ones(desc_all.shape[0])*label_dict[env]
    np.savez_compressed("%s_ACE_rmax_%s_nmax_%s_lmax_%s_start_%s_every_%s.npz"%(store_file, rmax_in, nmax_in, lmax_in, start, frames_interval), ACE=desc_all, label=labels_all)
    print("finished: ", file)
    print("stored: ", store_file)
    print(desc_all.shape)

def ASSLMB(rmax_in, lmax_in, nmax_in):
    """
    Run the ACE descriptor generation pipeline for predefined trajectory files.

    Parameters:
        rmax_in (float): Maximum cutoff radius.
        nmax_in (int): Highest order of radial basis functions.
        lmax_in (int): Highest order of angular basis functions.
    """
    # create ACE for every frame in 20 ns for 100 ps trajectory
    base_dir = "/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/clustering/process_data"
    for file in ["20ns_interval_100ps.xyz"]:
        file_in = f"{base_dir}/{file}"
        store_file = f"/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/data/descs/{file}"
        create_ACE(file_in, store_file, rmax_in, nmax_in, lmax_in, 16080, "unknown")

if __name__ == "__main__":
    rmax_in = float(sys.argv[1])
    lmax_in = int(sys.argv[2])
    nmax_in = int(sys.argv[3])

    label_dict = {'unknown':-1}
    ASSLMB(rmax_in, lmax_in, nmax_in)
