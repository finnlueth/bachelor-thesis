import datasets
from collections import defaultdict

def _group_by_protein(dataset: datasets.Dataset, identifier_key: str):
    """Group dataset entries by protein."""
    protein_groups = defaultdict(list)
    for i, entry in enumerate(dataset):
        protein = entry[identifier_key]
        protein_groups[protein].append(i)
    return protein_groups
