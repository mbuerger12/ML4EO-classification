import sys
import random
from collections import Counter

import numpy as np
import tifffile
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

print(f"Python interpreter being used: {sys.executable}")

# --- Constants ---
LCZ_MAP_PATH = "../dataset/milan/LCZ_MAP.tif"

PATCH_SIZE = 64
STRIDE = 32
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
TARGET_MEDIAN_MULTIPLIER_OVERSAMPLE = 2
TARGET_MEDIAN_MULTIPLIER_SYNTHETIC = 3
N_REGIONS_ROW = 2
N_REGIONS_COL = 2
N_CLUSTERS = 5


# --- Helper Functions ---
def print_class_distribution(title: str, unique_classes: np.ndarray, class_counts: np.ndarray) -> None:
    """Prints the class distribution with percentages."""
    total_pixels = np.sum(class_counts)
    class_percentages = (class_counts / total_pixels) * 100
    print(f"\n--- {title} ---")
    for cls, count, percentage in zip(unique_classes, class_counts, class_percentages):
        print(f"Class {cls}: {count} occurrences ({percentage:.2f}%)")


def get_patch_coords(lcz_map: np.ndarray, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Generates all possible top-left coordinates for patches."""
    H, W = lcz_map.shape
    coords = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            coords.append((r, c))
    return coords


def calculate_label_distribution(tile_coords: list[tuple[int, int]], lcz_map: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the label distribution for a given set of tile coordinates."""
    labels = []
    for r, c in tile_coords:
        lcz_patch = lcz_map[r : r + patch_size, c : c + patch_size]
        label_counts = Counter(lcz_patch.flatten())
        most_common_label = label_counts.most_common(1)[0][0]
        labels.append(most_common_label)
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    return unique_classes, class_counts


# --- Dataset Class ---
class SimpleLCZDataset(Dataset):
    """
    A simplified PyTorch Dataset for extracting LCZ labels from patches.
    """

    def __init__(self, lcz_map_path: str, patch_size: int, tile_coords: list[tuple[int, int]]):
        self.lcz_map_path = lcz_map_path
        self.patch_size = patch_size
        self.tile_coords = tile_coords
        self.lcz_map = tifffile.imread(lcz_map_path)

    def __len__(self) -> int:
        return len(self.tile_coords)

    def __getitem__(self, idx: int) -> dict[str, int]:
        r, c = self.tile_coords[idx]
        lcz_patch = self.lcz_map[r : r + self.patch_size, c : c + self.patch_size]
        label_counts = Counter(lcz_patch.flatten())
        most_common_label = label_counts.most_common(1)[0][0]
        # Check if lcz_patch is truly 2D or potentially 3D (height, width, bands)
        if lcz_patch.ndim == 2:
            num_bands = 1
        elif lcz_patch.ndim == 3:
            num_bands = lcz_patch.shape[2] # Assuming (height, width, bands)
        else:
            num_bands = "Unknown (not 2D or 3D)"

        print(f"Tile {idx}: Coordinates ({r}, {c}), Label: {most_common_label}, Bands in this patch: {num_bands}")
        return {"label": most_common_label}



# --- Splitting Functions ---
def create_stratified_split_coords(
    lcz_map: np.ndarray, patch_size: int, stride: int, train_ratio: float = TRAIN_RATIO, seed: int = RANDOM_SEED
) -> list[tuple[int, int]]:
    """
    Creates a stratified split of tile coordinates based on the top-left pixel's class.
    Tiles are shuffled within each class before splitting.
    """
    unique_classes = np.unique(lcz_map)
    train_tile_coords = []
    rng = np.random.RandomState(seed)
    tile_coords_by_class = {cls: [] for cls in unique_classes}

    for r in range(0, lcz_map.shape[0] - patch_size + 1, stride):
        for c in range(0, lcz_map.shape[1] - patch_size + 1, stride):
            top_left_class = lcz_map[r, c]
            tile_coords_by_class[top_left_class].append((r, c))

    for cls in unique_classes:
        coords = tile_coords_by_class[cls]
        rng.shuffle(coords)
        train_split = int(train_ratio * len(coords))
        train_tile_coords.extend(coords[:train_split])
    return train_tile_coords


def create_geo_stratified_split_coords(
    lcz_map: np.ndarray,
    patch_size: int,
    stride: int,
    train_ratio: float = TRAIN_RATIO,
    n_regions_row: int = N_REGIONS_ROW,
    n_regions_col: int = N_REGIONS_COL,
    seed: int = RANDOM_SEED,
) -> list[tuple[int, int]]:
    """
    Simulates a geographically stratified split by dividing the map into regions
    and performing stratified sampling within each region.
    """
    unique_classes = np.unique(lcz_map)
    train_tile_coords_geo = []
    rng = np.random.RandomState(seed)
    H, W = lcz_map.shape
    region_h = H // n_regions_row
    region_w = W // n_regions_col

    for r_reg in range(n_regions_row):
        for c_reg in range(n_regions_col):
            r_start = r_reg * region_h
            r_end = min((r_reg + 1) * region_h, H)  # Ensure within bounds
            c_start = c_reg * region_w
            c_end = min((c_reg + 1) * region_w, W)  # Ensure within bounds

            region_coords_by_class = {cls: [] for cls in unique_classes}
            for r in range(r_start, r_end - patch_size + 1, stride):
                for c in range(c_start, c_end - patch_size + 1, stride):
                    top_left_class = lcz_map[r, c]
                    region_coords_by_class[top_left_class].append((r, c))

            for cls in unique_classes:
                coords = region_coords_by_class[cls]
                rng.shuffle(coords)
                train_split = int(train_ratio * len(coords))
                train_tile_coords_geo.extend(coords[:train_split])
    return train_tile_coords_geo


def create_cluster_stratified_split_coords(
    lcz_map: np.ndarray,
    patch_size: int,
    stride: int,
    train_ratio: float = TRAIN_RATIO,
    n_clusters: int = N_CLUSTERS,
    seed: int = RANDOM_SEED,
) -> list[tuple[int, int]]:
    """
    Performs a conceptual cluster-based splitting where clusters of tile coordinates
    are first identified using K-Means, and then a subset of these clusters are
    selected for the training set.
    """
    tile_coords = get_patch_coords(lcz_map, patch_size, stride)
    rng = np.random.RandomState(seed)
    rng.shuffle(tile_coords)  # Shuffle to ensure random cluster assignment if coords are ordered
    coords_array = np.array(tile_coords)

    # Handle cases where n_clusters might be greater than the number of samples
    if len(coords_array) < n_clusters:
        print(f"Warning: n_clusters ({n_clusters}) is greater than the number of available tiles ({len(coords_array)}). Setting n_clusters to {len(coords_array)}.")
        n_clusters = len(coords_array)
        if n_clusters == 0:
            return [] # No tiles available

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    cluster_labels = kmeans.fit_predict(coords_array)

    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) == 0:
        return [] # No clusters found

    num_train_clusters = int(train_ratio * len(unique_clusters))
    if num_train_clusters == 0 and train_ratio > 0: # Ensure at least one cluster if train_ratio is positive
        num_train_clusters = 1

    train_clusters = rng.choice(unique_clusters, size=num_train_clusters, replace=False)

    train_tile_coords_cluster = [
        coords for i, coords in enumerate(tile_coords) if cluster_labels[i] in train_clusters
    ]
    return train_tile_coords_cluster


# --- Resampling Functions ---
def undersample_classes_target(labels: list[int], target_samples_per_class: int) -> list[int]:
    """
    Undersamples classes to a specified target number of samples per class.
    """
    class_counts = Counter(labels)
    undersampled_labels = []
    indices_by_class = {cls: [i for i, l in enumerate(labels) if l == cls] for cls in class_counts}

    for cls, indices in indices_by_class.items():
        n_samples = len(indices)
        samples_to_take = min(n_samples, target_samples_per_class)
        if samples_to_take > 0:
            undersampled_indices = random.sample(indices, samples_to_take)
            undersampled_labels.extend([labels[i] for i in undersampled_indices])
    return undersampled_labels


def conceptual_oversample(labels: list[int], oversampling_target: int) -> list[int]:
    """
    Conceptually oversamples classes to a specified target.
    In a real scenario, this would involve generating new data.
    """
    class_counts = Counter(labels)
    oversampled_labels = []
    for cls, count in class_counts.items():
        oversample_amount = max(0, oversampling_target - count)
        oversampled_labels.extend([cls] * count)  # Add existing samples
        oversampled_labels.extend([cls] * oversample_amount)  # Add conceptual oversamples
    return oversampled_labels


# --- Main Execution ---
if __name__ == "__main__":
    # Load the full LCZ map
    lcz_map_full = tifffile.imread(LCZ_MAP_PATH)
    unique_classes_full, class_counts_full = np.unique(lcz_map_full, return_counts=True)
    print_class_distribution("Full LCZ Map Class Distribution (Before Stratification)", unique_classes_full, class_counts_full)

    # --- Perform Stratified Split ---
    train_tile_coords_stratified = create_stratified_split_coords(
        lcz_map_full, PATCH_SIZE, STRIDE, TRAIN_RATIO, RANDOM_SEED
    )
    train_ds_stratified = SimpleLCZDataset(LCZ_MAP_PATH, PATCH_SIZE, train_tile_coords_stratified)
    train_labels_stratified = [train_ds_stratified[i]["label"] for i in range(len(train_ds_stratified))]
    unique_classes_train_stratified, class_counts_train_stratified = np.unique(
        train_labels_stratified, return_counts=True
    )
    print_class_distribution(
        "Approximate Class Distribution in Stratified Training Tiles",
        unique_classes_train_stratified,
        class_counts_train_stratified,
    )

    # --- Simulate Geographic Stratification ---
    train_tile_coords_geo_stratified = create_geo_stratified_split_coords(
        lcz_map_full, PATCH_SIZE, STRIDE, TRAIN_RATIO, N_REGIONS_ROW, N_REGIONS_COL, RANDOM_SEED
    )
    train_ds_geo_stratified = SimpleLCZDataset(
        LCZ_MAP_PATH, PATCH_SIZE, train_tile_coords_geo_stratified
    )
    train_labels_geo_stratified = [
        train_ds_geo_stratified[i]["label"] for i in range(len(train_ds_geo_stratified))
    ]
    unique_classes_train_geo_stratified, class_counts_train_geo_stratified = np.unique(
        train_labels_geo_stratified, return_counts=True
    )
    print_class_distribution(
        "Approximate Class Distribution in Geographically Stratified Training Tiles",
        unique_classes_train_geo_stratified,
        class_counts_train_geo_stratified,
    )

    # --- Undersampling with Target Level ---
    median_count = (
        int(np.median(class_counts_train_stratified))
        if len(class_counts_train_stratified) > 0
        else 30
    )
    train_labels_undersampled_target = undersample_classes_target(
        train_labels_stratified, median_count
    )
    unique_classes_train_undersampled_target, class_counts_train_undersampled_target = np.unique(
        train_labels_undersampled_target, return_counts=True
    )
    print_class_distribution(
        f"Class Distribution After Undersampling to {median_count} samples per class (max)",
        unique_classes_train_undersampled_target,
        class_counts_train_undersampled_target,
    )

    # --- Conceptual Spatially-Aware Oversampling ---
    oversampling_target_val = median_count * TARGET_MEDIAN_MULTIPLIER_OVERSAMPLE
    oversampled_labels_spatial = conceptual_oversample(
        train_labels_stratified, oversampling_target_val
    )
    unique_classes_train_oversampled_spatial, class_counts_train_oversampled_spatial = np.unique(
        oversampled_labels_spatial, return_counts=True
    )
    print_class_distribution(
        f"Conceptual Class Distribution After Spatially-Aware Oversampling (Target: {oversampling_target_val})",
        unique_classes_train_oversampled_spatial,
        class_counts_train_oversampled_spatial,
    )

    # --- Cluster-Based Splitting with Stratification (Conceptual) ---
    train_tile_coords_cluster_stratified = create_cluster_stratified_split_coords(
        lcz_map_full, PATCH_SIZE, STRIDE, TRAIN_RATIO, N_CLUSTERS, RANDOM_SEED
    )
    train_ds_cluster_stratified = SimpleLCZDataset(
        LCZ_MAP_PATH, PATCH_SIZE, train_tile_coords_cluster_stratified
    )
    train_labels_cluster_stratified = [
        train_ds_cluster_stratified[i]["label"] for i in range(len(train_ds_cluster_stratified))
    ]
    unique_classes_train_cluster_stratified, class_counts_train_cluster_stratified = np.unique(
        train_labels_cluster_stratified, return_counts=True
    )
    print_class_distribution(
        "Approximate Class Distribution in Cluster-Based Stratified Training Tiles",
        unique_classes_train_cluster_stratified,
        class_counts_train_cluster_stratified,
    )

    # --- Conceptual Synthetic Data Generation with Spatial Awareness ---
    synthetic_oversampling_target_val = median_count * TARGET_MEDIAN_MULTIPLIER_SYNTHETIC
    synthetic_labels_spatial = conceptual_oversample(
        train_labels_stratified, synthetic_oversampling_target_val
    )
    unique_classes_train_synthetic_spatial, class_counts_train_synthetic_spatial = np.unique(
        synthetic_labels_spatial, return_counts=True
    )
    print_class_distribution(
        f"Conceptual Synthetic Data Generation with Spatial Awareness (Target: {synthetic_oversampling_target_val})",
        unique_classes_train_synthetic_spatial,
        class_counts_train_synthetic_spatial,
    )


    # --- Plotting the Comparison Bar Chart ---
    def plot_class_distribution_comparison(
        data_to_plot: dict[str, tuple[np.ndarray, np.ndarray]], title: str, median_count: int
    ) -> None:
        """Plots a comparison of class distributions from various techniques."""
        all_unique_classes = sorted(
            list(
                set.union(
                    *(
                        set(unique_cls)
                        for unique_cls, _ in data_to_plot.values()
                        if unique_cls.size > 0
                    )
                )
            )
        )
        if not all_unique_classes:
            print("No classes to plot.")
            return

        n_classes = len(all_unique_classes)
        bar_width = 0.1
        index = np.arange(n_classes)
        plt.figure(figsize=(24, 12))

        # Dynamically set bar positions and labels
        positions = [-3, -2, -1, 0, 1, 2, 3]  # Relative positions for the bars
        labels = [
            "Original",
            "Stratified",
            "Geo-Stratified (Sim.)",
            f"Undersampled (Target: {median_count})",
            f"Oversampled (Conceptual, Target: {median_count * TARGET_MEDIAN_MULTIPLIER_OVERSAMPLE})",
            "Cluster-Stratified (Sim.)",
            f"Synthetic Spatial (Conceptual, Target: {median_count * TARGET_MEDIAN_MULTIPLIER_SYNTHETIC})",
        ]
        colors = ["b", "g", "c", "purple", "orange", "lime", "red"]

        for i, (key, (unique_cls, counts)) in enumerate(data_to_plot.items()):
            percentages = np.zeros(n_classes)
            if counts.size > 0:
                current_percentages = (counts / np.sum(counts)) * 100
                for j, cls in enumerate(unique_cls):
                    if cls in all_unique_classes:
                        idx = all_unique_classes.index(cls)
                        percentages[idx] = current_percentages[j]
            plt.bar(index + positions[i] * bar_width, percentages, bar_width, label=labels[i], color=colors[i])


        plt.xlabel("LCZ Class")
        plt.ylabel("Percentage")
        plt.title(title)
        plt.xticks(index, [str(cls) for cls in all_unique_classes])
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Prepare data for plotting
    data_for_plotting = {
        "Original": (unique_classes_full, class_counts_full),
        "Stratified": (unique_classes_train_stratified, class_counts_train_stratified),
        "Geo-Stratified": (unique_classes_train_geo_stratified, class_counts_train_geo_stratified),
        "Undersampled": (
            unique_classes_train_undersampled_target,
            class_counts_train_undersampled_target,
        ),
        "Oversampled": (
            unique_classes_train_oversampled_spatial,
            class_counts_train_oversampled_spatial,
        ),
        "Cluster-Stratified": (
            unique_classes_train_cluster_stratified,
            class_counts_train_cluster_stratified,
        ),
        "Synthetic_Spatial": (
            unique_classes_train_synthetic_spatial,
            class_counts_train_synthetic_spatial,
        ),
    }

    plot_class_distribution_comparison(
        data_for_plotting,
        "Comparison of Class Distributions with Advanced Techniques (Conceptual) Milan Dataset",
        median_count,
    )