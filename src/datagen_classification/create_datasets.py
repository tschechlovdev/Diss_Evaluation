# %%
from pathlib import Path

from DataGenerator import Generator
from Taxonomy import EngineTaxonomy
from anytree import RenderTree
import numpy as np
from Taxonomy import Node

np.random.seed(123456)

generated_data_path = Path("./generated")
generated_data_path.mkdir(parents=True, exist_ok=True)


# %%
def generate_dataset(n_instances, n_features, n_classes, taxonomy):
    generator = Generator(root=taxonomy, n=n_instances, c=n_classes, n_features=n_features)
    df = generator.generate_data_from_taxonomy()
    return df


def check_tree_viability(min_nodes_current_level, current_level, max_level, max_leaf_nodes):
    """when randomly creating trees check if it is possible to have leafnodes = n_clusters based on the number of nodes on the current level
    (assuming a minimkum split of 2)
    """
    min_nodes = min_nodes_current_level * 2 ** (max_level - current_level)
    return min_nodes <= max_leaf_nodes


def generate_taxonomy(min_lvl=2, max_lvl=3, n_clusters=30, engine_perc=10):
    """Randomly generates a taxonomy for data generation or returns the engine taxonomy with a given chance

    Args:
        min_lvl (int, optional): Min Hierarchie level of taxonomy. Defaults to 2.
        max_lvl (int, optional): Max Hierarchie level of taxonomy. Defaults to 3.
        n_clusters (int, optional): Number of clusters (leaf nodes of result). Defaults to 30.
        engine_perc (int, optional): Percentage of cases in which the default 'EngineTaxonomy' should be returned. Defaults to 10.
    """
    # check if engine taxonomy should be returned
    if np.random.rand() <= engine_perc / 100:
        taxonomy = EngineTaxonomy().create_taxonomy()
        return taxonomy
    # create random taxonomy object
    levels = np.random.randint(min_lvl, max_lvl + 1)
    taxonomy_root = Node("Root Node")
    nodes_curr_level = 0
    curr_level_nodes = [taxonomy_root]
    for level in range(levels):
        generated_nodes = []
        for index, node in enumerate(curr_level_nodes):
            min_nodes_current_level = 99999
            while not check_tree_viability(min_nodes_current_level, level + 1, levels, n_clusters):
                # get splits
                if level == 0:
                    splits = np.random.randint(2, 5)
                elif level == levels - 1:
                    nodes_to_iterate = len(curr_level_nodes) - (index + 1)
                    already_generated = len(generated_nodes)
                    # budget is the maximum amount of splits currently possible
                    budget = n_clusters - (nodes_to_iterate * 2 + already_generated)
                    # if current node is the last node to split split maximum ammount
                    if index == len(curr_level_nodes) - 1:
                        splits = budget
                    else:
                        splits = np.random.randint(2, budget + 1)
                else:
                    splits = np.random.randint(2, 4)
                nodes_to_iterate = len(curr_level_nodes) - (index + 1)
                already_generated = len(generated_nodes)
                min_nodes_current_level = nodes_to_iterate * 2 + already_generated + splits
            # generate child nodes for current node
            for i in range(splits):
                child = Node(f"child_{level + 1}_{index + 1}_{i + 1}", parent=node)
                generated_nodes.append(child)
        curr_level_nodes = generated_nodes
    return taxonomy_root


# %%
# parameter configuration for the datasets to generate
n_instances = 10000
n_features = [10, 30, 50]
n_clusters = [10, 30, 50]
# gs default 1
group_separation = [0, 0.25, 0.5, 0.75, 1]
# sg default 'medium'
group_imbalance = [0, 0.25, 0.5, 0.75, 1]
# %%
# taxonomy configuration for the datasets to generate
taxonomies = {}
for n_c in n_clusters:
    np.random.seed(0)
    taxonomies[n_c] = generate_taxonomy(min_lvl=3, max_lvl=3, n_clusters=n_c, engine_perc=0)

# %%
# generate datasets with specified parameters
count = 0
for n_c in n_clusters:
    taxonomy_t = taxonomies[n_c]
    for n_f in n_features:
        for i in range(len(group_imbalance)):
            import copy

            taxonomy = copy.deepcopy(taxonomy_t)
            generator = Generator(root=taxonomy, n=n_instances, c=30, n_features=n_f, gs=0.5, sG=group_imbalance[i])
            df = generator.generate_data_from_taxonomy()
            df.to_csv(
                generated_data_path / f"dataset{n_instances}_clusters{n_c}_features{n_f}_gs05_sg{str(group_imbalance[i]).replace('.', '')}.csv",
                encoding='utf-8', index=False)
            generator = Generator(root=taxonomy, n=n_instances, c=30, n_features=n_f, gs=group_separation[i], sG=1)
            df = generator.generate_data_from_taxonomy()
            df.to_csv(
                generated_data_path / f"dataset{n_instances}_clusters{n_c}_features{n_f}_gs{str(group_separation[i]).replace('.', '')}_sg1.csv",
                encoding='utf-8', index=False)
            print(f"success: conf: clusters-{n_c}, n_features-{n_f}, sg-{group_imbalance[i]}, gs-{group_separation[i]}")

# %%
# remove nan rows
import os
import glob
import pandas as pd
import numpy as np

# os.chdir("C:/Users/rappjs/Documents/data_gen_automl/DataGenerator-main")
datasets, labels, dataset_names = [], [], []
for file in glob.glob(str(generated_data_path / "*.csv")):
    df = pd.read_csv(file)
    n_rows = df.shape[0]
    df = df.dropna()
    if df.shape[0] != n_rows:
        df.to_csv(file, encoding='utf-8', index=False)

# %%
# save taxonomies
with open("taxnomies.txt", "a+", encoding='utf-8') as f:
    for tax in taxonomies.values():
        f.write(f"Taxonomy:\n")
        f.write(str(RenderTree(tax)))
        f.write(f"\n\n")
