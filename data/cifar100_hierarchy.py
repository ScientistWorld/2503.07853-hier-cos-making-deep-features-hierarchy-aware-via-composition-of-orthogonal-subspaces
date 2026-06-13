"""CIFAR-100 hierarchy utilities.

The paper evaluates CIFAR-100 with a deeper hierarchy from prior work.  For
this compact reproduction we use the public CIFAR-100 fine/coarse taxonomy:
20 coarse classes, each with 5 fine classes.  The root node is omitted, matching
the paper's convention that the universal root is never used as a class.
"""

from __future__ import annotations

from dataclasses import dataclass


FINE_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab",
    "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
    "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower",
    "sweet_pepper", "table", "tank", "telephone", "television",
    "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]

COARSE_LABELS = [
    "aquatic_mammals", "fish", "flowers", "food_containers",
    "fruit_and_vegetables", "household_electrical_devices",
    "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores", "medium_mammals",
    "non-insect_invertebrates", "people", "reptiles", "small_mammals",
    "trees", "vehicles_1", "vehicles_2",
]

FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11,
    5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4,
    18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9,
    13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0,
    17, 8, 14, 13,
]


@dataclass(frozen=True)
class Cifar100Hierarchy:
    node_names: list[str]
    node_levels: list[int]
    leaf_node_indices: list[int]
    fine_to_coarse: list[int]
    coarse_node_indices: list[int]
    fine_node_indices: list[int]
    node_to_subspace: list[list[int]]
    leaf_to_subspace: list[list[int]]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_node_indices)

    @property
    def num_coarse(self) -> int:
        return len(self.coarse_node_indices)


def build_hierarchy() -> Cifar100Hierarchy:
    coarse_nodes = [f"coarse:{name}" for name in COARSE_LABELS]
    fine_nodes = [f"fine:{name}" for name in FINE_LABELS]
    node_names = coarse_nodes + fine_nodes
    coarse_node_indices = list(range(len(COARSE_LABELS)))
    fine_node_indices = list(range(len(COARSE_LABELS), len(COARSE_LABELS) + len(FINE_LABELS)))
    node_levels = [0] * len(COARSE_LABELS) + [1] * len(FINE_LABELS)

    coarse_to_fine_nodes: dict[int, list[int]] = {i: [] for i in range(len(COARSE_LABELS))}
    for fine_idx, coarse_idx in enumerate(FINE_TO_COARSE):
        coarse_to_fine_nodes[coarse_idx].append(fine_node_indices[fine_idx])

    node_to_subspace: list[list[int]] = []
    for coarse_idx in range(len(COARSE_LABELS)):
        node_to_subspace.append([coarse_idx] + coarse_to_fine_nodes[coarse_idx])
    for fine_idx, coarse_idx in enumerate(FINE_TO_COARSE):
        node_to_subspace.append([coarse_idx, fine_node_indices[fine_idx]])

    leaf_to_subspace = [node_to_subspace[idx] for idx in fine_node_indices]
    return Cifar100Hierarchy(
        node_names=node_names,
        node_levels=node_levels,
        leaf_node_indices=fine_node_indices,
        fine_to_coarse=FINE_TO_COARSE.copy(),
        coarse_node_indices=coarse_node_indices,
        fine_node_indices=fine_node_indices,
        node_to_subspace=node_to_subspace,
        leaf_to_subspace=leaf_to_subspace,
    )


def lca_distance(fine_a: int, fine_b: int) -> int:
    if fine_a == fine_b:
        return 0
    if FINE_TO_COARSE[fine_a] == FINE_TO_COARSE[fine_b]:
        return 1
    return 2
