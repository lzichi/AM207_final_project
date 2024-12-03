import ase.io as ase

def create_xyz(file, start, stop, interval, save_name):
    """
    Create and save XYZ frames from a LAMMPS trajectory file at specified intervals.

    Parameters:
        file (str): Path to the input LAMMPS trajectory file.
        start (int): The starting frame index (inclusive).
        stop (int): The ending frame index (exclusive).
        interval (int): The interval between frames to extract.
        save_name (str): The name of the output XYZ file.
    """
    frames = ase.read(file, index=f"{start}:{stop}:{interval}", format="lammps-dump-text")
    for frame in frames:
        frame.symbols = "Li16080S3600P720Cl720" 
    ase.write(save_name, frames)
    print(save_name)
    print(len(frames))

if __name__ == "__main__":
    file = "/n/holystore01/LABS/kozinsky_lab/Lab/User/classifier_data/MD-traj-30ns/trajectory.lammpstrj_30ns"

    # 20 ns in 100ps intervals
    save_name = "20ns_interval_100ps.xyz"
    create_xyz(file, 1, 20001, 100, save_name) # do not include first frame (artifical)
