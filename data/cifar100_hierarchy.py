"""CIFAR-100 hierarchy utilities.

The benchmark uses CIFAR-100 with either the five-level taxonomy used by the
paper or the public CIFAR-100 coarse/fine taxonomy for ablations. This module is
solution-agnostic: it exposes labels, paths, and level-to-leaf relationships
needed by data loading and evaluation, but no method-specific representation.
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

# Rows are [fine, level4, level3, level2, level1] from the released CIFAR-100
# level-5 target table used for the paper's benchmark.
PAPER_L5_ROWS = [
    [0, 4, 5, 3, 0], [1, 1, 6, 0, 0], [2, 14, 3, 0, 0],
    [3, 8, 3, 0, 0], [4, 0, 6, 0, 0], [5, 6, 0, 1, 1],
    [6, 7, 4, 0, 0], [7, 7, 4, 0, 0], [8, 18, 7, 1, 1],
    [9, 3, 0, 1, 1], [10, 3, 0, 1, 1], [11, 14, 3, 0, 0],
    [12, 9, 1, 1, 1], [13, 18, 7, 1, 1], [14, 7, 4, 0, 0],
    [15, 11, 3, 0, 0], [16, 3, 0, 1, 1], [17, 9, 1, 1, 1],
    [18, 7, 4, 0, 0], [19, 11, 3, 0, 0], [20, 6, 0, 1, 1],
    [21, 11, 3, 0, 0], [22, 5, 0, 1, 1], [23, 10, 2, 2, 1],
    [24, 7, 4, 0, 0], [25, 6, 0, 1, 1], [26, 13, 4, 0, 0],
    [27, 15, 4, 0, 0], [28, 3, 0, 1, 1], [29, 15, 4, 0, 0],
    [30, 0, 6, 0, 0], [31, 11, 3, 0, 0], [32, 1, 6, 0, 0],
    [33, 10, 2, 2, 1], [34, 12, 3, 0, 0], [35, 14, 3, 0, 0],
    [36, 16, 3, 0, 0], [37, 9, 1, 1, 1], [38, 11, 3, 0, 0],
    [39, 5, 0, 1, 1], [40, 5, 0, 1, 1], [41, 19, 7, 1, 1],
    [42, 8, 3, 0, 0], [43, 8, 3, 0, 0], [44, 15, 4, 0, 0],
    [45, 13, 4, 0, 0], [46, 14, 3, 0, 0], [47, 17, 5, 3, 0],
    [48, 18, 7, 1, 1], [49, 10, 2, 2, 1], [50, 16, 3, 0, 0],
    [51, 4, 5, 3, 0], [52, 17, 5, 3, 0], [53, 4, 5, 3, 0],
    [54, 2, 5, 3, 0], [55, 0, 6, 0, 0], [56, 17, 5, 3, 0],
    [57, 4, 5, 3, 0], [58, 18, 7, 1, 1], [59, 17, 5, 3, 0],
    [60, 10, 2, 2, 1], [61, 3, 0, 1, 1], [62, 2, 5, 3, 0],
    [63, 12, 3, 0, 0], [64, 12, 3, 0, 0], [65, 16, 3, 0, 0],
    [66, 12, 3, 0, 0], [67, 1, 6, 0, 0], [68, 9, 1, 1, 1],
    [69, 19, 7, 1, 1], [70, 2, 5, 3, 0], [71, 10, 2, 2, 1],
    [72, 0, 6, 0, 0], [73, 1, 6, 0, 0], [74, 16, 3, 0, 0],
    [75, 12, 3, 0, 0], [76, 9, 1, 1, 1], [77, 13, 4, 0, 0],
    [78, 15, 4, 0, 0], [79, 13, 4, 0, 0], [80, 16, 3, 0, 0],
    [81, 19, 7, 1, 1], [82, 2, 5, 3, 0], [83, 4, 5, 3, 0],
    [84, 6, 0, 1, 1], [85, 19, 7, 1, 1], [86, 5, 0, 1, 1],
    [87, 5, 0, 1, 1], [88, 8, 3, 0, 0], [89, 19, 7, 1, 1],
    [90, 18, 7, 1, 1], [91, 1, 6, 0, 0], [92, 2, 5, 3, 0],
    [93, 15, 4, 0, 0], [94, 6, 0, 1, 1], [95, 0, 6, 0, 0],
    [96, 17, 5, 3, 0], [97, 8, 3, 0, 0], [98, 14, 3, 0, 0],
    [99, 13, 4, 0, 0],
]


@dataclass(frozen=True)
class Cifar100Hierarchy:
    name: str
    node_names: list[str]
    node_levels: list[int]
    leaf_node_indices: list[int]
    fine_node_indices: list[int]
    level_node_indices: list[list[int]]
    fine_to_path: list[list[int]]
    fine_to_level_labels: list[list[int]]
    level_to_leaf_sets: list[list[list[int]]]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_node_indices)

    @property
    def num_coarse(self) -> int:
        return len(self.level_node_indices[0])

    @property
    def coarse_node_indices(self) -> list[int]:
        return self.level_node_indices[0]

    @property
    def fine_to_coarse(self) -> list[int]:
        return [path[0] - self.level_node_indices[0][0] for path in self.fine_to_path]


def build_hierarchy(name: str = "public") -> Cifar100Hierarchy:
    if name in {"public", "coarse", "two_level"}:
        paths = [[FINE_TO_COARSE[fine], fine] for fine in range(100)]
        level_names = [COARSE_LABELS, FINE_LABELS]
        return _build_from_level_labels("public", paths, level_names)
    if name in {"paper_l5", "l5", "paper"}:
        paths = [list(reversed(row[1:])) + [row[0]] for row in PAPER_L5_ROWS]
        level_names = [
            [f"l1:{i}" for i in sorted({row[4] for row in PAPER_L5_ROWS})],
            [f"l2:{i}" for i in sorted({row[3] for row in PAPER_L5_ROWS})],
            [f"l3:{i}" for i in sorted({row[2] for row in PAPER_L5_ROWS})],
            [f"l4:{i}" for i in sorted({row[1] for row in PAPER_L5_ROWS})],
            FINE_LABELS,
        ]
        return _build_from_level_labels("paper_l5", paths, level_names)
    raise ValueError(f"unknown CIFAR-100 hierarchy: {name}")


def _build_from_level_labels(
    name: str,
    fine_to_level_labels: list[list[int]],
    level_names: list[list[str]],
) -> Cifar100Hierarchy:
    level_value_to_node: list[dict[int, int]] = []
    level_node_indices: list[list[int]] = []
    node_names: list[str] = []
    node_levels: list[int] = []
    next_node = 0

    for level, names in enumerate(level_names):
        values = sorted({path[level] for path in fine_to_level_labels})
        mapping = {value: next_node + offset for offset, value in enumerate(values)}
        level_value_to_node.append(mapping)
        level_nodes = [mapping[value] for value in values]
        level_node_indices.append(level_nodes)
        for value in values:
            label = names[value] if value < len(names) else str(value)
            node_names.append(f"level{level + 1}:{label}")
            node_levels.append(level)
        next_node += len(values)

    fine_to_path = [
        [level_value_to_node[level][value] for level, value in enumerate(path)]
        for path in fine_to_level_labels
    ]
    fine_node_indices = level_node_indices[-1]
    leaf_node_indices = fine_node_indices.copy()

    level_to_leaf_sets: list[list[list[int]]] = [[[] for _ in nodes] for nodes in level_node_indices]
    for fine, path in enumerate(fine_to_path):
        for level, node in enumerate(path):
            local_idx = level_node_indices[level].index(node)
            level_to_leaf_sets[level][local_idx].append(fine)

    return Cifar100Hierarchy(
        name=name,
        node_names=node_names,
        node_levels=node_levels,
        leaf_node_indices=leaf_node_indices,
        fine_node_indices=fine_node_indices,
        level_node_indices=level_node_indices,
        fine_to_path=fine_to_path,
        fine_to_level_labels=fine_to_level_labels,
        level_to_leaf_sets=level_to_leaf_sets,
    )


def lca_distance(fine_a: int, fine_b: int, hierarchy_name: str = "public") -> int:
    hierarchy = build_hierarchy(hierarchy_name)
    path_a = hierarchy.fine_to_path[fine_a]
    path_b = hierarchy.fine_to_path[fine_b]
    common_depth = -1
    for depth, (node_a, node_b) in enumerate(zip(path_a, path_b)):
        if node_a != node_b:
            break
        common_depth = depth
    return len(path_a) - 1 - common_depth
