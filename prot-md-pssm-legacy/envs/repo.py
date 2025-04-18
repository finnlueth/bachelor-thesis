import os
import yaml


def main():
    with open('configs/project.yml', 'r') as file:
        FILE_PATHS = yaml.safe_load(file)['paths']

    for name, path in FILE_PATHS.items():
        print(f"Creating directory for {name} at {path}")
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    main()
