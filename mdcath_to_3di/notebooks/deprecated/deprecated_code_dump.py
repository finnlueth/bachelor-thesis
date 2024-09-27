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





# @dataclass
# class HDF5:
#     decode: bool = True

#     def __repr__(self):
#         return 'dict'

#     def decode_example(self, value: dict, token_per_repo_id=None):
#         print('decoding...')
#         return 'decoded example'

# dataset_mdcath = dataset_mdcath.cast_column("image", HDF5())
# dataset_mdcath = dataset_mdcath.rename_column("image", "hdf5")


# item_name = "dataset_mdcath_1avyB00.h5"
# matching_item = dataset_mdcath.filter(lambda example: item_name in example['file_name'])


# dataset_mdcath = dataset_mdcath.cast_column("image", Image(decode=False))
# dataset_mdcath = dataset_mdcath.cast_column("image", dict)
# dataset_mdcath = dataset_mdcath.rename_column("image", "mdcath")


# item = item.cast_column("image", Image(decode=False))
# dataset_mdcath = dataset_mdcath.rename_column("image", "mdcath")


# item.info.builder_name= None


# ds_item = next(iter(dataset_mdcath))


# dataset_mdcath.info


# "dataset_mdcath_1avyB00.h5"





# # dataset_mdcath = Dataset.from_dict({"a": range(10)}).to_iterable_dataset(num_shards=3)

# for idx, data in enumerate(dataset_mdcath):
#     data = h5py.File(data, 'r')
#     print(data.keys())
#     # data = extract_dataset_information(
#     #     file_path=f"../data/mdcath/dataset_mdcath_{item}.h5",
#     # )

#     # items = []
#     # for x in range(0, len(data[item]["coords"])):
#     #     new = replace_coordinates_in_pdb(
#     #         original_pdb=data[item]["pdb"], new_coordinates=data[item]["coords"][x]
#     #     )
#     #     items.append(new)

#     # file_path = f"../data/3Di/{item}.fasta"

#     # with open(file_path, "w") as file:
#     #     file.write(f">{item}|{data[item]['seq']}\n" + "\n".join(get_3di_sequences_from_memory(pdb_files=items)))

#     if idx == 0:
#         state_dict = dataset_mdcath.state_dict()
#         break


# with open('../data/3Di/_state_dict.pkl', 'wb') as f:
#     pickle.dump(state_dict, f)







# iterable_dataset = Dataset.from_dict({"a": range(12)}).to_iterable_dataset(num_shards=3)

# for idx, example in enumerate(iterable_dataset):
#     print(example)
#     if idx == 4:
#         break
# state_dict = iterable_dataset.state_dict()
# print(state_dict)

# iterable_dataset = Dataset.from_dict({"a": range(12)}).to_iterable_dataset(num_shards=3)
# iterable_dataset.load_state_dict(state_dict)
# print(f"restart from checkpoint")
# for idx, example in enumerate(iterable_dataset):
#     print(example)
#     if idx == 1:
#         break
# state_dict = iterable_dataset.state_dict()
# print(state_dict)

# iterable_dataset = Dataset.from_dict({"a": range(12)}).to_iterable_dataset(num_shards=3)
# iterable_dataset.load_state_dict(state_dict)
# print(f"restart from checkpoint")
# for example in iterable_dataset:
#     print(example)
#     if idx == 5:
#         break
# state_dict = iterable_dataset.state_dict()
# print(state_dict)
