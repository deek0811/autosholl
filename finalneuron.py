# Current Best Script with the centralized node in soma + Sholl Analysis
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import exposure, filters, io, measure, morphology, restoration, util
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects

try:
    from skan import Skeleton, summarize
    SKAN_AVAILABLE = True
except Exception:
    SKAN_AVAILABLE = False


@dataclass
class PipelineConfig:
    mode: str = "single_neuron"          # single_neuron | spine | cilia | network
    channel: Optional[int] = None
    z_project: str = "max"               # max | mean
    gaussian_sigma: float = 0.0
    median_radius: int = 0
    threshold_method: str = "auto"       # auto | otsu | local
    local_block_size: int = 51
    min_object_size: int = 64
    hole_area_threshold: int = 64
    background_disk_radius: int = 0
    center_weight: float = 1.0
    area_weight: float = 0.25
    prune_short_branches_px: float = 10.0
    cilia_min_eccentricity: float = 0.85
    cilia_min_area: int = 30
    network_use_frangi: bool = True
    save_intermediates: bool = True
    soma_fill_disk: int = 5
    soma_detect_disk: int = 20
    # Sholl analysis parameters
    sholl_step_px: float = 5.0           # radial step between concentric rings (pixels)
    sholl_start_px: float = 0.0          # inner radius offset from soma center (0 = soma edge)
    sholl_pixel_size_um: float = 1.0     # µm per pixel (for reporting radii in µm)


# Pipeline
class MorphologyPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    # Public API
    def run(self, image_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = self.load_image(image_path)
        gray = self.select_channel_and_project(image)
        pre = self.preprocess(gray)
        mask = self.segment(pre)
        skeleton = morphology.skeletonize(mask > 0)

        soma_root: Optional[Tuple[int, int]] = None
        sholl_df = pd.DataFrame()
        sholl_metrics: Dict[str, Any] = {}

        if self.config.mode in {"single_neuron", "spine"}:
            mask = self.keep_central_structure(mask)
            skeleton = morphology.skeletonize(mask)
            skeleton, soma_root = self.force_soma_root(skeleton, mask)
            if self.config.save_intermediates:
                self.debug_soma_region(mask, output_dir, image_path.stem)

            # Sholl Analysis
            soma_center = self._find_soma_center(skeleton)
            if soma_center is not None:
                sholl_df = self.compute_sholl(skeleton, soma_center)
                sholl_metrics = self._sholl_summary_stats(sholl_df)
                sholl_metrics["soma_center_row"] = int(soma_center[0])
                sholl_metrics["soma_center_col"] = int(soma_center[1])

        elif self.config.mode == "cilia":
            mask = self.filter_cilia_objects(mask)
            skeleton = morphology.skeletonize(mask)
        elif self.config.mode == "network":
            skeleton = morphology.skeletonize(mask)

        branch_df = self.extract_branch_table(skeleton)
        if not branch_df.empty and self.config.mode in {"single_neuron", "spine"}:
            branch_df = branch_df[
                branch_df["branch_length_px"] >= self.config.prune_short_branches_px
            ].copy()

        summary = self.compute_summary(mask, skeleton, branch_df)
        summary.update(sholl_metrics)

        self.save_results(
            output_dir=output_dir,
            image_name=image_path.stem,
            gray=gray,
            pre=pre,
            mask=mask,
            skeleton=skeleton,
            branch_df=branch_df,
            sholl_df=sholl_df,
            summary=summary,
        )

        return {
            "summary": summary,
            "branch_table": branch_df,
            "sholl_table": sholl_df,
            "config": asdict(self.config),
        }

    # Input
    def load_image(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            image = tifffile.imread(path)
        else:
            image = io.imread(path)
        return np.asarray(image)

    def select_channel_and_project(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)

        if arr.ndim == 4:
            if arr.shape[-1] in (3, 4):
                if self.config.channel is not None:
                    arr = arr[..., self.config.channel]
                arr = arr.mean(axis=0) if self.config.z_project == "mean" else arr.max(axis=0)
            else:
                if self.config.channel is not None and self.config.channel < arr.shape[0]:
                    arr = arr[self.config.channel]
                arr = arr.mean(axis=0) if self.config.z_project == "mean" else arr.max(axis=0)

        elif arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                if self.config.channel is None:
                    arr = rgb2gray(arr)
                else:
                    arr = arr[..., self.config.channel]
            else:
                arr = arr.mean(axis=0) if self.config.z_project == "mean" else arr.max(axis=0)

        arr = util.img_as_float(arr)
        arr = np.nan_to_num(arr, copy=False)
        return arr

    # Preprocessing
    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        img = gray.copy()

        if self.config.median_radius > 0:
            img = filters.median(img, morphology.disk(self.config.median_radius))

        if self.config.gaussian_sigma > 0:
            img = filters.gaussian(img, sigma=self.config.gaussian_sigma, preserve_range=True)

        if self.config.background_disk_radius > 0:
            img = self._soma_aware_background_subtract(img)

        if self.config.mode == "network" and self.config.network_use_frangi:
            img = filters.frangi(img)

        p_low, p_high = np.percentile(img, (1, 98))
        img = np.clip(img, p_low, p_high)
        img = exposure.rescale_intensity(img, out_range=(0, 1))

        return img

    def _soma_aware_background_subtract(self, img: np.ndarray) -> np.ndarray:
        sigma = self.config.background_disk_radius * 2

        soma_thresh = np.percentile(img, 99.9)
        soma_mask = img > soma_thresh
        soma_mask = morphology.remove_small_objects(soma_mask, min_size=self.config.min_object_size)
        soma_mask = morphology.binary_closing(soma_mask, morphology.disk(3))

        halo_mask = morphology.binary_dilation(
            soma_mask,
            morphology.disk(self.config.background_disk_radius)
        )

        inpainted = img.copy()
        inpainted[halo_mask] = np.median(img[~halo_mask])
        inpainted = filters.gaussian(inpainted, sigma=sigma)

        background = filters.gaussian(
            np.where(halo_mask, inpainted, img),
            sigma=sigma
        )

        return np.clip(img - background, 0, None)

    # Segmentation
    def segment(self, img: np.ndarray) -> np.ndarray:
        method = self.config.threshold_method
        if method == "auto":
            method = "local" if self.config.mode in {"cilia", "network", "spine"} else "otsu"

        if method == "otsu":
            thresh = filters.threshold_otsu(img)
            mask = img > thresh
        elif method == "local":
            block_size = self._ensure_odd(self.config.local_block_size)
            local_thresh = filters.threshold_local(img, block_size=block_size)
            mask = img > local_thresh
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        mask = remove_small_objects(mask, min_size=self.config.min_object_size)
        mask = morphology.binary_opening(mask, morphology.disk(1))
        mask = morphology.binary_closing(mask, morphology.disk(1))
        return mask.astype(bool)

    def keep_central_structure(self, mask: np.ndarray) -> np.ndarray:
        labels = label(mask)
        props = regionprops(labels)
        if not props:
            return mask

        center = np.array(mask.shape) / 2.0
        best_label = None
        best_score = -np.inf

        for p in props:
            centroid = np.array(p.centroid)
            dist = np.linalg.norm(centroid - center)
            center_score = -dist * self.config.center_weight
            area_score = float(p.area) * self.config.area_weight
            score = center_score + area_score
            if score > best_score:
                best_score = score
                best_label = p.label

        soma_mask = (labels == best_label)

        soma_core = morphology.opening(soma_mask, morphology.disk(self.config.soma_fill_disk))
        soma_labeled = label(soma_core)
        if soma_labeled.max() > 0:
            regions = regionprops(soma_labeled)
            largest = max(regions, key=lambda r: r.area)
            soma_core = soma_labeled == largest.label
        else:
            fallback_disk = max(self.config.soma_fill_disk // 2, 3)
            soma_core = morphology.opening(soma_mask, morphology.disk(fallback_disk))
            soma_labeled = label(soma_core)
            if soma_labeled.max() > 0:
                regions = regionprops(soma_labeled)
                largest = max(regions, key=lambda r: r.area)
                soma_core = soma_labeled == largest.label

        soma_filled = ndi.binary_fill_holes(soma_core)
        soma_mask = soma_mask | soma_filled
        return soma_mask

    def _detect_soma_core(self, mask: np.ndarray) -> np.ndarray:
        soma_core = morphology.opening(mask, morphology.disk(self.config.soma_detect_disk))
        soma_labeled = label(soma_core)

        if soma_labeled.max() > 0:
            regions = regionprops(soma_labeled)
            largest = max(regions, key=lambda r: r.area)
            return soma_labeled == largest.label
        else:
            fallback_disk = max(self.config.soma_detect_disk // 2, 5)
            soma_core = morphology.opening(mask, morphology.disk(fallback_disk))
            soma_labeled = label(soma_core)
            if soma_labeled.max() > 0:
                regions = regionprops(soma_labeled)
                largest = max(regions, key=lambda r: r.area)
                return soma_labeled == largest.label

        return np.zeros_like(mask, dtype=bool)

    def debug_soma_region(self, mask: np.ndarray, output_dir: Path, image_name: str) -> None:
        soma_core = self._detect_soma_core(mask)
        soma_filled = ndi.binary_fill_holes(soma_core)

        base = util.img_as_ubyte(mask.astype(np.uint8) * 255)
        overlay = np.stack([base, base, base], axis=-1)
        overlay[soma_filled] = [255, 165, 0]  # orange = filled soma
        overlay[soma_core]   = [255, 0,   0]  # red    = soma core

        io.imsave(
            output_dir / f"{image_name}_debug_soma_region.png",
            overlay,
            check_contrast=False
        )
        print(f"  Soma core area (detect disk={self.config.soma_detect_disk}): {soma_core.sum()} px")
        print(f"  Soma filled area:                                             {soma_filled.sum()} px")

    def force_soma_root(
        self,
        skeleton: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        soma_core = self._detect_soma_core(mask)
        soma_filled = ndi.binary_fill_holes(soma_core)

        if soma_filled.sum() == 0:
            cy, cx = ndi.center_of_mass(mask)
            return skeleton, (int(cy), int(cx))

        cy, cx = ndi.center_of_mass(soma_filled)
        root = (int(cy), int(cx))

        skeleton_clean = skeleton.copy()
        skeleton_clean[soma_filled] = False
        skeleton_clean[root] = True

        soma_dilated = morphology.binary_dilation(soma_filled, morphology.disk(2))
        entry_mask = soma_dilated & skeleton_clean
        entry_coords = np.argwhere(entry_mask)

        if len(entry_coords) == 0:
            return skeleton_clean, root

        cost = np.where(mask, 1.0, 1000.0)

        for entry in entry_coords:
            entry_t = tuple(entry)
            if entry_t == root:
                continue
            try:
                path, _ = route_through_array(
                    cost,
                    root,
                    entry_t,
                    fully_connected=True,
                    geometric=True,
                )
                for r, c in path:
                    skeleton_clean[r, c] = True
            except Exception:
                from skimage.draw import line as skline
                rr, cc = skline(root[0], root[1], entry_t[0], entry_t[1])
                rr = np.clip(rr, 0, skeleton.shape[0] - 1)
                cc = np.clip(cc, 0, skeleton.shape[1] - 1)
                skeleton_clean[rr, cc] = True

        return skeleton_clean, root

    def filter_cilia_objects(self, mask: np.ndarray) -> np.ndarray:
        labels = label(mask)
        out = np.zeros_like(mask, dtype=bool)
        for p in regionprops(labels):
            if p.area < self.config.cilia_min_area:
                continue
            if getattr(p, "eccentricity", 0.0) < self.config.cilia_min_eccentricity:
                continue
            out[labels == p.label] = True
        out = remove_small_objects(out, min_size=self.config.cilia_min_area)
        return out

    # Branch extraction
    def extract_branch_table(self, skeleton: np.ndarray) -> pd.DataFrame:
        if skeleton.sum() == 0:
            return pd.DataFrame(columns=[
                "branch_id", "branch_length_px", "euclidean_length_px",
                "branch_type", "mean_pixel_value"
            ])

        if SKAN_AVAILABLE:
            sk = Skeleton(skeleton)
            df = summarize(sk)
            out = pd.DataFrame({
                "branch_id": np.arange(len(df), dtype=int),
                "branch_length_px": df["branch-distance"].astype(float),
                "euclidean_length_px": df["euclidean-distance"].astype(float),
                "branch_type": df["branch-type"].astype(str),
                "mean_pixel_value": df.get(
                    "mean-pixel-value", pd.Series(np.nan, index=df.index)
                ).astype(float),
            })
            return out

        labels = label(skeleton)
        rows: List[Dict[str, Any]] = []
        for idx, p in enumerate(regionprops(labels), start=1):
            coords = p.coords
            rows.append({
                "branch_id": idx,
                "branch_length_px": float(len(coords)),
                "euclidean_length_px": float(self._coords_extent(coords)),
                "branch_type": "unknown",
                "mean_pixel_value": np.nan,
            })
        return pd.DataFrame(rows)

    # Sholl Analysis
    def _find_soma_center(self, skeleton: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return (row, col) of the junction with the most neighbors, or None."""
        if skeleton.sum() == 0:
            return None

        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        conv = ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbor_count = conv - 10 * skeleton.astype(np.uint8)

        junction_mask = skeleton & (neighbor_count >= 3)
        if junction_mask.sum() == 0:
            # No junctions — use skeleton pixel closest to image center
            skel_coords = np.argwhere(skeleton)
            image_center = np.array(skeleton.shape) / 2.0
            dists = np.linalg.norm(skel_coords - image_center, axis=1)
            best = skel_coords[np.argmin(dists)]
            return (int(best[0]), int(best[1]))

        junction_neighbors = neighbor_count * junction_mask
        flat_idx = np.argmax(junction_neighbors)
        row, col = np.unravel_index(flat_idx, skeleton.shape)
        return (int(row), int(col))

    def compute_sholl(
        self,
        skeleton: np.ndarray,
        soma_center: Tuple[int, int],
    ) -> pd.DataFrame:
        """Compute Sholl intersection profile and branching index."""
        if skeleton.sum() == 0:
            return pd.DataFrame(columns=["radius_px", "radius_um", "intersections", "branching_index"])

        step = max(1.0, self.config.sholl_step_px)

        # Distance map from soma center
        rows_idx, cols_idx = np.indices(skeleton.shape)
        cy, cx = soma_center
        dist_map = np.sqrt((rows_idx - cy) ** 2 + (cols_idx - cx) ** 2)

        max_r = float(dist_map[skeleton].max())
        start = self.config.sholl_start_px if self.config.sholl_start_px > 0.0 else step
        radii = np.arange(start, max_r + step, step)

        intersections = np.zeros(len(radii), dtype=int)
        for i, r in enumerate(radii):
            # Crossing detector: skeleton pixels just outside radius r that
            # have at least one 8-connected neighbor inside radius r
            outside = dist_map >= r
            inside = dist_map < r
            inner_boundary = ndi.binary_dilation(inside, structure=np.ones((3, 3))) & outside
            crossing_pixels = skeleton & inner_boundary
            if crossing_pixels.any():
                intersections[i] = int(label(crossing_pixels, connectivity=2).max())

        branching_index = np.zeros(len(radii), dtype=float)
        for i in range(len(radii) - 1):
            branching_index[i] = (intersections[i + 1] - intersections[i]) / 2.0

        return pd.DataFrame({
            "radius_px": radii,
            "radius_um": radii * self.config.sholl_pixel_size_um,
            "intersections": intersections,
            "branching_index": branching_index,
        })

    @staticmethod
    def _sholl_summary_stats(sholl_df: pd.DataFrame) -> Dict[str, Any]:
        """Derive AUC, peak intersections, critical radius, and ramification index."""
        if sholl_df.empty or sholl_df["intersections"].sum() == 0:
            return {
                "sholl_auc": 0.0,
                "sholl_max_intersections": 0,
                "sholl_critical_radius_um": 0.0,
                "sholl_branching_index": 0.0,
                "sholl_num_shells": 0,
                "sholl_max_radius_um": 0.0,
            }

        radii = sholl_df["radius_um"].values
        counts = sholl_df["intersections"].values.astype(float)

        auc = float(np.trapezoid(counts, radii))
        peak_idx = int(np.argmax(counts))
        max_int = int(counts[peak_idx])
        active = sholl_df[sholl_df["intersections"] > 0]
        mean_int = float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0

        return {
            "sholl_auc": round(auc, 4),
            "sholl_max_intersections": max_int,
            "sholl_critical_radius_um": round(float(radii[peak_idx]), 4),
            "sholl_branching_index": round(mean_int / max_int if max_int > 0 else 0.0, 4),
            "sholl_num_shells": int(len(active)),
            "sholl_max_radius_um": round(float(active["radius_um"].max()), 4) if not active.empty else 0.0,
        }

    # Metrics
    def compute_summary(
        self,
        mask: np.ndarray,
        skeleton: np.ndarray,
        branch_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        junctions, endpoints = self._count_skeleton_nodes(skeleton)
        return {
            "mode": self.config.mode,
            "foreground_area_px": int(mask.sum()),
            "skeleton_length_px": int(skeleton.sum()),
            "num_components": int(label(mask).max()),
            "num_branches": int(len(branch_df)),
            "num_junctions": int(junctions),
            "num_endpoints": int(endpoints),
            "mean_branch_length_px": float(branch_df["branch_length_px"].mean()) if not branch_df.empty else 0.0,
            "total_branch_length_px": float(branch_df["branch_length_px"].sum()) if not branch_df.empty else 0.0,
        }

    # Script to save output image or statistics
    def save_results(
        self,
        output_dir: Path,
        image_name: str,
        gray: np.ndarray,
        pre: np.ndarray,
        mask: np.ndarray,
        skeleton: np.ndarray,
        branch_df: pd.DataFrame,
        sholl_df: pd.DataFrame,
        summary: Dict[str, Any],
    ) -> None:
        pd.DataFrame([summary]).to_csv(output_dir / f"{image_name}_summary.csv", index=False)
        branch_df.to_csv(output_dir / f"{image_name}_branches.csv", index=False)

        if not sholl_df.empty:
            sholl_df.to_csv(output_dir / f"{image_name}_sholl.csv", index=False)

        if self.config.save_intermediates:
            def save(name, arr):
                io.imsave(
                    output_dir / f"{image_name}_{name}.png",
                    util.img_as_ubyte(exposure.rescale_intensity(arr, out_range=(0, 1))),
                    check_contrast=False,
                )
            save("gray", gray)
            save("preprocessed", pre)
            io.imsave(output_dir / f"{image_name}_mask.png", util.img_as_ubyte(mask), check_contrast=False)
            io.imsave(output_dir / f"{image_name}_skeleton.png", util.img_as_ubyte(skeleton), check_contrast=False)
            io.imsave(output_dir / f"{image_name}_overlay.png", self.make_overlay(gray, mask, skeleton), check_contrast=False)

    def make_overlay(self, gray: np.ndarray, mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        base = util.img_as_ubyte(exposure.rescale_intensity(gray, out_range=(0, 1)))
        rgb = np.stack([base, base, base], axis=-1)
        rgb[mask] = [0, 255, 0]
        rgb[skeleton] = [255, 0, 0]
        return rgb

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _coords_extent(coords: np.ndarray) -> float:
        if len(coords) < 2:
            return float(len(coords))
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return float(np.linalg.norm(maxs - mins))

    @staticmethod
    def _count_skeleton_nodes(skeleton: np.ndarray) -> Tuple[int, int]:
        if skeleton.sum() == 0:
            return 0, 0
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        conv = ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbor_count = conv - 10 * skeleton.astype(np.uint8)
        endpoints = int(np.logical_and(skeleton, neighbor_count == 1).sum())
        junctions = int(np.logical_and(skeleton, neighbor_count >= 3).sum())
        return junctions, endpoints


# Run a batch of images in a folder
def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    patterns: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"),
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = MorphologyPipeline(config)
    summaries: List[Dict[str, Any]] = []

    image_paths: List[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(input_dir.glob(pattern)))

    for image_path in image_paths:
        sample_dir = output_dir / image_path.stem
        result = pipeline.run(image_path, sample_dir)
        row = result["summary"].copy()
        row["image_name"] = image_path.name
        summaries.append(row)

    all_summary = pd.DataFrame(summaries)
    all_summary.to_csv(output_dir / "all_samples_summary.csv", index=False)
    return all_summary

# Run

if __name__ == "__main__":
    neuron_cfg = PipelineConfig(
        mode="single_neuron",
        channel=None,
        gaussian_sigma=0.5,
        threshold_method="local",
        local_block_size=61,
        min_object_size=120,
        hole_area_threshold=80,
        background_disk_radius=12,
        prune_short_branches_px=12.0,
        soma_fill_disk=5,
        soma_detect_disk=20,
        # Can be changed depending on image specs
        sholl_step_px=5.0,          # ring spacing in pixels
        sholl_start_px=0.0,         # 0 = auto-detect soma edge as starting radius
        sholl_pixel_size_um=1.0,    # set to your actual µm/px to get radii in µm
    )

    cilia_cfg = PipelineConfig(
        mode="cilia",
        gaussian_sigma=1.0,
        threshold_method="local",
        local_block_size=51,
        min_object_size=30,
        hole_area_threshold=20,
        background_disk_radius=15,
        cilia_min_eccentricity=0.88,
        cilia_min_area=30,
    )

    network_cfg = PipelineConfig(
        mode="network",
        gaussian_sigma=1.0,
        threshold_method="local",
        local_block_size=71,
        min_object_size=150,
        hole_area_threshold=120,
        background_disk_radius=10,
        network_use_frangi=True,
    )

    INPUT_DIR = Path("./data")
    OUTPUT_DIR = Path("./sholl")

    chosen_cfg = neuron_cfg
    run_batch(INPUT_DIR, OUTPUT_DIR, chosen_cfg)
