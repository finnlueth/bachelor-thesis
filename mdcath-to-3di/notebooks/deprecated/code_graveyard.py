
    def use_trajectory_location(self, idx, lock):
        """
        Mark a trajectory location as used.
        """
        with lock:
            with open(self.save_path + f"/{self.cache_name}.json", "r") as f:
                used_trajectory_locations = json.load(f)
            used_trajectory_locations.append(self.trajectory_locations[idx])
            with open(self.save_path + f"/{self.cache_name}.json", "w") as f:
                json.dump(used_trajectory_locations, f)
        self.used_trajectory_locations.add(self.trajectory_locations[idx])
        self.blocked_trajectory_locations.discard(self.trajectory_locations[idx])
        self._save_used_trajectory_locations()

    # def reset(self):
    #     """
    #     Reset the used indices to allow reusing the dataset.
    #     """
    #     self.used_trajectory_locations.clear()
    #     self.blocked_trajectory_locations.clear()
    #     self._save_used_trajectory_locations()

    # def get_unused_indices(self):
    #     """
    #     Get a list of indices that have not been accessed yet.
    #     """
    #     return [
    #         i
    #         for i in range(len(self.trajectory_locations))
    #         if self.trajectory_locations[i] not in self.used_trajectory_locations
    #     ]

    # def is_used_index(self, idx):
    #     """
    #     Check if a specific index has been accessed.
    #     """
    #     return self.trajectory_locations[idx] in self.used_trajectory_locations

    # def _save_used_trajectory_locations(self):
    #     """
    #     Save the used indices to a JSON file.
    #     """
    #     with open(self.save_path + "/used_trajectory_locations.json", "w") as f:
    #         fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    #         try:
    #             json.dump(list(self.used_trajectory_locations), f)
    #         finally:
    #             fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # def _load_used_trajectory_locations(self):
    #     """
    #     Load the used indices from a JSON file.
    #     """
    #     with open(self.save_path + "/used_trajectory_locations.json", "r") as f:
    #         self.used_trajectory_locations = set(json.load(f))

    # def load_used_trajectory_locations(self, file_path):
    #     """
    #     Load used indices from a specified file and update the dataset.
    #     :param file_path: Path to the file containing the used indices.
    #     """
    #     if os.path.exists(file_path):
    #         with open(file_path, "r") as f:
    #             loaded_indices = set(json.load(f))
    #             self.used_trajectory_locations.update(loaded_indices)
    #             self._save_used_trajectory_locations()  # Save the combined result to the default save_path
    #     else:
    #         raise FileNotFoundError(f"File {file_path} does not exist.")



    def __getitem__(self, idx):
        """Retrieve a data sample for the given index."""
        # if self.trajectory_locations[idx] in self.used_trajectory_locations:
        #     raise ValueError(f"Trajectory {self.trajectory_locations[idx]} has already been accessed and processed.")
        # if self.trajectory_locations[idx] in self.blocked_trajectory_locations:
        #     raise ValueError(f"Trajectory {self.trajectory_locations[idx]} has been blocked and cannot be accessed.")
        # self.blocked_trajectory_locations.add(self.trajectory_locations[idx])
        with open(self.save_path + f"/{self.cache_name}.json", "r") as f:
            used_trajectory_locations = json.load(f)
        if self.trajectory_locations[idx] in used_trajectory_locations:
            raise TrajectoryAlreadyProcessedError(self.trajectory_locations[idx])
        return self._get_item(idx)
    
    
    
    
# def get_3di_sequences_from_file(pdb_files: T.List[str], foldseek_path="foldseek"):
#     pdb_file_string = " ".join([str(p) for p in pdb_files])
#     pdb_dir_name = hash(pdb_file_string)

#     with tempfile.TemporaryDirectory() as tmpdir:
#         FSEEK_BASE_CMD = f"{foldseek_path} createdb {pdb_file_string} {tmpdir}/{pdb_dir_name}"
#         # log(FSEEK_BASE_CMD)
#         proc = sp.Popen(
#             shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE
#         )
#         out, err = proc.communicate()

#         with open(f"{tmpdir}/{pdb_dir_name}_ss", "r") as seq_file:
#             seqs = [i.strip().strip("\x00") for i in seq_file]

#         with open(f"{tmpdir}/{pdb_dir_name}.lookup", "r") as name_file:
#             names = [i.strip().split()[1].split(".")[0] for i in name_file]

#         seq_records = {
#             n: SeqRecord.SeqRecord(Seq.Seq(s), id=n, description=n)
#             for (n, s) in zip(names, seqs)
#         }

#         return seq_records


# def translate_pdb_to_3di(pbds: dict) -> dict:
#     """
#     Translates a dictionary of PDB files into 3Di sequences.

#     Args:
#         pbds (dict): A dictionary where keys are identifiers and values are lists of PDB file paths.

#     Returns:
#         dict: A dictionary where keys are the same identifiers and values are the corresponding 3DI sequences.
#     """
#     items = {}
#     for key, values in pbds.items():
#         items[key] = get_3di_sequences_from_memory(pdb_files=values)
#     return items


# def generate_fasta(extraced_traj: dict, processed_3Di: dict) -> str:
#     """
#     Generates a FASTA file from the extracted trajectory and the processed 3Di sequences.

#     Args:
#         extraced_traj (dict): A dictionary containing the extracted trajectory data.
#                               It should have the following structure:
#                               {
#                                   "name": trajectory_name,
#                                   "seq": amino_acid_sequence,
#                                 }
#         processed_3Di (dict): A dictionary containing the processed 3Di sequences.
#                                 It should have the following structure:
#                                 {
#                                     "temp|replica": [sequence1, sequence2, ...],
#                                 }
#     Returns:
#         str: A string containing the FASTA formatted data.
#     """
#     items = []
#     for name, sequences in processed_3Di.items():
#         items.append(f">{name}|{extraced_traj['seq']}")
#         items.extend(sequences)
#     return "\n".join(items)


# def read_3Di_fasta(fasta: str) -> dict:
#     """
#     Read a FASTA file containing 3Di sequences.
#     Args:
#         fasta (str): The FASTA formatted string.
#     Returns:
#         dict: A dictionary containing the 3Di sequences.
#     """
#     items = {}
#     for line in fasta.split("\n"):
#         if line.startswith(">"):
#             name = line[1:]
#             items[name] = []
#         else:
#             items[name].append(line)
#     return items


# def fastas_to_hf_dataset(fasta: str, output_path: str) -> None:
#     """
#     Convert a FASTA file to an HDF5 dataset.
#     Args:
#         fasta (str): The FASTA formatted string.
#         output_path (str): The path to the output HDF5 file.
#     """
#     pass



from typing import List
import time

def decode_bytes_comprehension(byte_strings: List[bytes]) -> List[str]:
    return [b.decode('utf-8') for b in byte_strings]

def decode_bytes_map(byte_strings: List[bytes]) -> List[str]:
    return list(map(lambda x: x.decode('utf-8'), byte_strings))

def benchmark_decoding(byte_strings: List[bytes], iterations: int = 1):
    start = time.perf_counter()
    for _ in range(iterations):
        decode_bytes_comprehension(byte_strings)
    comp_time = (time.perf_counter() - start) / iterations
    
    start = time.perf_counter()
    for _ in range(iterations):
        decode_bytes_map(byte_strings)
    map_time = (time.perf_counter() - start) / iterations
    
    return {'comprehension': comp_time, 'map': map_time}

test_data = [b'hello'*40,] * 400 * 25 * 5000
results = benchmark_decoding(test_data)
print(f"List comprehension: {results['comprehension']:.6f} seconds")
print(f"Map: {results['map']:.6f} seconds")



# with h5py.File('../tmp/data/tokenized/mdcath/tokenized_data.h5', 'r') as file:
#     strings = {
#         f'{domain}_{trajectory}': rust_modules.decode_bytestrings(file['foldseek'][domain][trajectory])
#         for domain in tqdm(list(file['foldseek']))
#         for trajectory in file['foldseek'][domain]
#     }

# with open('../tmp/mdcath_foldseek_trajectories.pkl', 'wb') as f:
#     pickle.dump(strings, f)

# print(len(strings))

# assert strings == strings_new





class T5PSSMHead1(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()

        config.d_model = 1024

        self.conv1 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.bn1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.bn2(hidden_states)
        hidden_states = torch.relu(hidden_states)
        # hidden_states = self.global_pool(hidden_states)
        # hidden_states = hidden_states.view(hidden_states.size(0), -1)
        return hidden_states


class T5PSSMHead2(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        embedding_size = 1024
        num_3Di_classes = 27
        num_filters = 256
        kernel_size = 5

        self.conv1 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same",
        )

        self.batch_norm = nn.BatchNorm1d(num_filters)

        self.fc = nn.Linear(num_filters, num_3Di_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.fc(hidden_states)


class T5PSSMHead3(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.conv1 = nn.Conv1d(1024, 512, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=5, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 20, kernel_size=5, padding="same")

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.bn3(x)
        x = self.conv4(x)
        return x


class T5PSSMHead4(nn.Module):
    def __init__(self, config: T5Config):
        super(T5PSSMHead4, self).__init__()
        self.conv1 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 20, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        return x


class T5PSSMHeadFromMichael(nn.Module):
    def __init__(self):
        super(T5PSSMHeadFromMichael, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

        self.dssp8_classifier = torch.nn.Sequential(nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0)))
        self.diso_classifier = torch.nn.Sequential(nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0)))

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        return d3_Yhat, d8_Yhat
