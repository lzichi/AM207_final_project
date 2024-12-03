import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List
import sys

sys.path.append('/n/holyscratch01/kozinsky_lab/Users/lzichi/projects/classifier/software/')
from DADApy.dadapy import *

class DensityPeakAdvancedClusteringACE:

    def __init__(self, isVerbose):
        """
        Class for applying density peak advanced (DPA) clustering to atomic cluster expansion (ACE) descriptors.
        
        Parameters:
                isVerbose (bool): Print (or not) helpful statements.
        """
        self.isVerbose = isVerbose

    def load_data(self, labeled_list_path, ACE_shape):
        """
        Load the ACE descriptors into an array for DPA clustering.
        
        Parameters:
                labeled_list_path (List[str]): List of strings to paths of .npz files of ACE descs
                                               the .npz file with key 'ACE' (with associated ACE descs).
                ACE_shape (int): Shape of ACE descs.
        """
        if(self.isVerbose):
                print("loading data")

        ## LOAD DATA ##
        data_all_labeled = np.zeros((0, ACE_shape))
        for label_path in labeled_list_path:
            if(self.isVerbose):
                print(label_path)
            data_labeled = np.load(label_path)['ACE']
            data_descs = data_labeled
            data_all_labeled = np.vstack((data_all_labeled, data_descs))
        print(data_all_labeled.shape)

        self.labelled_data_shape = np.shape(data_all_labeled)

        ## CREATE ARRAYS OF UNIQUE DATA FOR CLUSTERING ##
        _, index_unique, index_inverse = np.unique(np.round(data_all_labeled, 5), axis=0, return_inverse=True,
                                                                                                return_index = True)
        self.data_all_unique_labelled = data_all_labeled[index_unique]
        self.index_inverse_labelled = index_inverse
        self.data_all_labelled = data_all_labeled

    def cluster_dpa(self, maxk, Z, isVisualNeigh, saveDistName=None, loadDistName=None):
        """
        Use DPA clustering method to find intrinsic dimension and predict clusters on ACE descs.
        
        Parameters:
            maxk (int): Max neighbors in DPA.
            Z_in (float): Smoothing parameter in DPA.
            isVisualNeigh (bool): Visualize (or not) the number of neighbors for each ACE desc.
            saveDistName (str, optional): Path to save distances between ACE to.
            loadDistName (str, optional): Path to load saved distances between ACE from.
        
        Returns:
                cluster_assignments (numpy.ndarray[num ACE descs]): Cluster assigned to each ACE desc.
                cluster_assignments_halo (numpy.ndarray[num ACE descs]): Cluster assigned to each ACE desc with halo imposed.
        """

        if(self.isVerbose):
                print('calculating DPA clusters')
        
        # compute, load, or save distances
        data = Data(self.data_all_unique_labelled, verbose=self.isVerbose, maxk=maxk)
        if(loadDistName):
            data.distances = np.load(loadDistName)['distances']
            data.dist_indices = np.load(loadDistName)['dist_indices']
        else:
            data.compute_distances(maxk = maxk, n_jobs = 64)
            if(saveDistName):
                np.savez_compressed(saveDistName, distances=data.distances, dist_indices=data.dist_indices)
        
        # calculate densities and cluster ACE
        data.compute_id_2NN()
        data.compute_density_kstarNN()
        _ = data.compute_clustering_ADP(Z = Z, halo=False)

        if(isVisualNeigh):
                plt.hist(data.kstar)
                plt.xlabel('Number of neighbors')
                plt.ylabel('Frequency')
                plt.savefig(f'neighbors_of_data_Z_{Z}_maxk_{maxk}.png')

        cluster_assignments = data.cluster_assignment[self.index_inverse_labelled]
        return cluster_assignments

if __name__ == "__main__":

        Z = float(sys.argv[1])

        rmax_in = 4.0
        lmax_in = 3
        nmax_in = 2

        maxk = 50                        
        isVisualNeigh_in = True 
        isVerbose = True         

        # cluster 20 ns at 100ps interval trajectory
        base_dir = "/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/data/descs/"
        name_end = f".xyz_ACE_rmax_{rmax_in}_nmax_{nmax_in}_lmax_{lmax_in}_start_0_every_1.npz"
        labeled_list_path = []
        for name in ["20ns_interval_100ps"]:
            labeled_list_path.append(f"{base_dir}{name}{name_end}")

        ACE_shape = 324
        dpa_class = DensityPeakAdvancedClusteringACE(isVerbose)
        dpa_class.load_data(labeled_list_path, ACE_shape)

        # save descs and distances
        prefix = "20ns_interval_100ps_rcut_4.0"
        saveName = f"/n/netscratch/kozinsky_lab/Lab/lzichi/lzichi/projects/AM207_final_project/AM207_final_project/clustering/dpa_cluster/results/distances/{prefix}_entire_traj_distances.npz"
        cluster_assignments = dpa_class.cluster_dpa(maxk, Z, isVisualNeigh_in, loadDistName=saveName)
        name = f"results/dpa_inter/{prefix}_ACE_torch_Z_{Z}_isHalo_{False}_maxk_{maxk}_rmax_{rmax_in}_lmax_{lmax_in}_nmax_{nmax_in}.npy"
        print(name)
        np.save(name, cluster_assignments)