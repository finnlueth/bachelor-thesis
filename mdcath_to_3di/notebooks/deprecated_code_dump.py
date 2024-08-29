# with h5py.File(file_path, "r") as file:
#     tag = 'pdbProteinAtoms'
#     print(file['1a0rP01'][tag][()].decode("utf-8"))
#     text = file['1a0rP01'][tag][()].decode("utf-8")
    
#     # out_dir = '../data/output2.pdb'
#     # with open(out_dir, 'w') as f:
#     #     f.write(text)








traj_name = '1a0rP01'
file_path = f"../data/mdcath/mdcath_dataset_{traj_name}.h5"

with h5py.File(file_path, "r") as file:
    tag0 = '320'
    tag1 = '0'
    tag2 = 'coords'
    item = file[traj_name][tag0][tag1][tag2][()]
item, item.shape



# [x.decode("utf-8") for x in item[0]]


p_traj = pt.load(nv.datafiles.TRR, top=nv.datafiles.PDB)
# p_view = nv.show_pytraj(p_traj)
# p_view



with h5py.File(file_path, "r") as file:
    tag0 = '320'
    tag1 = '0'
    tag2 = 'coords'
    xyz = file[traj_name][tag0][tag1][tag2][()]
    top = file[traj_name]['pdb'][()].decode("utf-8")
    traj_1 = pt.Trajectory(xyz=xyz, top=traj.top)
item, item.shape




traj_1 = pt.Trajectory(xyz=xyz, top=traj.top)