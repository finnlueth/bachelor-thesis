


# def process_pipeline(md_cath_hdf5: bytes, config: dict = None, save_path_pdb: str = None, save_path_fasta: str = None) -> str:
#     extraced_traj = extract_mdcath_information(file_path=md_cath_hdf5, config=config)
#     processed_traj = rmsd_align_all(extraced_traj)
#     processed_traj = generate_mdcath_coordiante_pdbs(extraced_trajectroy=processed_traj)

#     if save_path_pdb:
#         save_PDBs(
#             template_PDB=extraced_traj["pdb"],
#             output_dir=f"{save_path_pdb}/{extraced_traj['name']}",
#             name=f"{extraced_traj['name']}",
#         )

#         for cathid_temp_repl, pdb_frames in processed_traj.items():
#             save_PDBs(PDBs=pdb_frames, output_dir=f"{save_path_pdb}/{extraced_traj['name']}", name=cathid_temp_repl)

#     processed_traj = translate_pdb_to_3di(processed_traj)

#     if save_path_fasta:
#         processed_fasta = generate_fasta(extraced_traj=extraced_traj, processed_3Di=processed_traj)
#         with open(f"{save_path_fasta}/{extraced_traj['name']}.fasta", "w", encoding="UTF-8") as file:
#             file.write(processed_fasta)
#         del processed_fasta

#     processed_traj = generate_pssms(processed_traj)

#     return processed_traj


# def download_process_pipeline(
#     url: str = None, path: str = None, config: dict = None, save_path_pdb: str = None, save_path_fasta: str = None
# ):
#     bytes_ = download_open(url=url, path=path, config=config)
#     return process_pipeline(md_cath_hdf5=bytes_, config=config, save_path_pdb=save_path_pdb, save_path_fasta=save_path_fasta)
