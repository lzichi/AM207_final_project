import ase.io as ase

## create xyf frames from Jingxuan's 1ps interval trajectory every 30ns ##
# /n/holystore01/LABS/kozinsky_lab/Lab/User/classifier_data/MD-traj-30ns/trajectory.lammpstrj_30ns

def create_xyz(file, start, stop, interval, save_name):
    frames = ase.read(file, index=f"{start}:{stop}:{interval}", format="lammps-dump-text")
    for frame in frames:
        frame.symbols = "Li16080S3600P720Cl720" 
    ase.write(save_name, frames)
    print(save_name)
    print(len(frames))

file = "/n/holystore01/LABS/kozinsky_lab/Lab/User/classifier_data/MD-traj-30ns/trajectory.lammpstrj_30ns"

# first 10 ps in 1 ps intervals, don't use first frame
save_name = "10ps_interval_1ps.xyz"
create_xyz(file, 1, 11, 1, save_name)

# 10ps to 100ps in 10 ps intervals
save_name = "100ps_interval_10ps.xyz"
create_xyz(file, 10, 101, 10, save_name)

# 100 ps to 1 ns in 100 ps intervals
save_name = "1ns_interval_100ps.xyz"
create_xyz(file, 100, 1001, 100, save_name)

# 30 ns in 1 ns intervals
save_name = "30ns_interval_500ps.xyz"
create_xyz(file, 1000, 30001, 500, save_name)
