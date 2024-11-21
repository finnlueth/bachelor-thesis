from huggingface_hub import hf_hub_download


def main():
    with open("./tmp/data/mdCATH/mdcath_domains.txt", "r") as file:
        domain_ids = [line.strip() for line in file.readlines()]
    # add snapshot_download()
    for domain_id in domain_ids:
        hf_hub_download(
            repo_id="compsciencelab/mdCATH",
            filename=f"mdcath_dataset_{domain_id}.h5",
            subfolder="data",
            local_dir="./tmp/data/mdCATH/",
            repo_type="dataset",
        )

if __name__ == "__main__":
    main()
