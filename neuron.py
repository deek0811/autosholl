from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore")

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
    mode: str = "single_neuron"  # single_neuron | spine | cilia | network
    channel: Optional[int] = None
    z_project: str = "max"  # max | mean
    gaussian_sigma: float = 0.0
    median_radius: int = 0
    threshold_method: str = "auto"  # auto | otsu | local
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

    sholl_step: float = 10.0
    sholl_max_radius: float = 0.0
    spine_max_length_px: float = 15.0


class MorphologyPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    # -----------------------------
    # Public API
    # -----------------------------
    def run(self, image_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = self.load_image(image_path)
        gray = self.select_channel_and_project(image)
        pre = self.preprocess(gray)
        mask = self.segment(pre)
        skeleton = morphology.skeletonize(mask > 0)

        if self.config.mode in {"single_neuron", "spine"}:
            mask = self.keep_central_structure(mask)
            skeleton = morphology.skeletonize(mask)
        elif self.config.mode == "cilia":
            mask = self.filter_cilia_objects(mask)
            skeleton = morphology.skeletonize(mask)
        elif self.config.mode == "network":
            skeleton = morphology.skeletonize(mask)

        branch_df = self.extract_branch_table(skeleton)
        if not branch_df.empty and self.config.mode in {"single_neuron", "spine"}:
            branch_df = branch_df[branch_df["branch_length_px"] >= self.config.prune_short_branches_px].copy()

        summary = self.compute_summary(mask, skeleton, branch_df)

        sholl_df = pd.DataFrame()
        sholl_skeleton = skeleton.copy()
        if self.config.mode in {"single_neuron", "spine"}:
            sholl_skeleton = self._filter_spines(skeleton)
            soma_center = self._find_soma_center(sholl_skeleton)
            if soma_center is not None:
                sholl_df = self.compute_sholl(sholl_skeleton, soma_center)
                sholl_stats = self._sholl_summary_stats(sholl_df)
                summary.update(sholl_stats)
                summary["soma_center_row"] = int(soma_center[0])
                summary["soma_center_col"] = int(soma_center[1])

        self.save_results(
            output_dir=output_dir,
            image_name=image_path.stem,
            gray=gray,
            pre=pre,
            mask=mask,
            skeleton=skeleton,
            sholl_skeleton=sholl_skeleton,
            branch_df=branch_df,
            summary=summary,
            sholl_df=sholl_df,
        )

        return {
            "summary": summary,
            "branch_table": branch_df,
            "sholl_table": sholl_df,
            "config": asdict(self.config),
        }

    # -----------------------------
    # IO
    # -----------------------------
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

        # Handle stacks
        if arr.ndim == 4:
            # Assume Z, Y, X, C or C, Z, Y, X. Use a simple heuristic.
            if arr.shape[-1] in (3, 4):
                if self.config.channel is not None:
                    arr = arr[..., self.config.channel]
                if self.config.z_project == "mean":
                    arr = arr.mean(axis=0)
                else:
                    arr = arr.max(axis=0)
            else:
                if self.config.channel is not None and self.config.channel < arr.shape[0]:
                    arr = arr[self.config.channel]
                if self.config.z_project == "mean":
                    arr = arr.mean(axis=0)
                else:
                    arr = arr.max(axis=0)

        elif arr.ndim == 3:
            # RGB image
            if arr.shape[-1] in (3, 4):
                if self.config.channel is None:
                    arr = rgb2gray(arr)
                else:
                    arr = arr[..., self.config.channel]
            # Z stack without channels
            else:
                if self.config.z_project == "mean":
                    arr = arr.mean(axis=0)
                else:
                    arr = arr.max(axis=0)

        arr = util.img_as_float(arr)
        arr = np.nan_to_num(arr, copy=False)
        return arr

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        img = gray.copy()

        if self.config.median_radius > 0:
            img = filters.median(img, morphology.disk(self.config.median_radius))

        if self.config.gaussian_sigma > 0:
            img = filters.gaussian(img, sigma=self.config.gaussian_sigma, preserve_range=True)

    #Altered the script from opening --> gaussian to hopefully threshold out some of these wandering axons
        if self.config.background_disk_radius > 0:
            background = filters.gaussian(
                img,
                sigma=self.config.background_disk_radius * 2
            )
            img = img - background
            img = np.clip(img, 0, None)

        if self.config.mode == "network" and self.config.network_use_frangi:
            img = filters.frangi(img)

        p_low, p_high = np.percentile(img, (1, 98))  #Intensity compression to drop the HUGE differences between soma and surr.
        img = np.clip(img, p_low, p_high)

        img = exposure.rescale_intensity(img, out_range=(0, 1))  #Normalization

        return img


    # -----------------------------
    # Segmentation
    # -----------------------------
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
        #mask = remove_small_holes(mask, area_threshold=self.config.hole_area_threshold)
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

    #Tried to isolate the soma and fill holes, here, but didnt actually fix the problem
    #Based on the hole size that I needed to fill soma, the hole is too big.
    #Created stange blobs along the neuron that will effect skeletonization in other neurons
        # Use that label to solate soma
        soma_mask = (labels == best_label)

        # Fill holes throughout the model, but apparently we have little loops throughout (?)
            #Essentially created HUGE blobs along the dendritic structures
        #soma_mask = remove_small_holes(soma_mask, area_threshold=500)

        #This ended up working better because it incorporates an erosion step (removes neurites)
            #This takes out the smaller structures and keeps only the large soma structure, then fills
        soma_core = morphology.opening(soma_mask, morphology.disk(5))
            #I think a disk size of 5 should filter out most neurites
                #To be an issue, there would have to be two VERY thick, overlapping (looping) dendrites
                    #This is not something I have seen commonly -- if two thick proximal dend = opposite sides
            #Food for thought later: soma halo intensity may distort the somas final morphology :'(
        soma_filled = ndi.binary_fill_holes(soma_core)

        # Then we merge it back with neurite mask and end up with a filled soma before skeletonization
        soma_mask = soma_mask | soma_filled

        return soma_mask

        return labels == best_label

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

    # -----------------------------
    # Graph / Branch extraction
    # -----------------------------
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
                "mean_pixel_value": df.get("mean-pixel-value", pd.Series(np.nan, index=df.index)).astype(float),
            })
            return out

        # Fallback: very simple connected-component skeleton fragments
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

    # -----------------------------
    # Spine filter for Sholl
    # -----------------------------
    def _filter_spines(self, skeleton: np.ndarray) -> np.ndarray:
        if skeleton.sum() == 0:
            return skeleton.copy()

        skel = (skeleton > 0).astype(np.uint8)

        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ])
        conv = ndi.convolve(skel, kernel, mode="constant", cval=0)
        neighbor_count = conv - 10 * skel

        endpoint_mask = (skel == 1) & (neighbor_count == 1)
        endpoint_coords = np.argwhere(endpoint_mask)

        filtered = skel.copy()

        for ep in endpoint_coords:
            path = [tuple(ep)]
            prev = None
            current = tuple(ep)

            for _ in range(int(self.config.spine_max_length_px) + 1):
                r, c = current
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]:
                            if filtered[nr, nc] and (nr, nc) != prev:
                                neighbors.append((nr, nc))

                if len(neighbors) == 0:
                    break

                prev = current
                current = neighbors[0]
                path.append(current)

                cr, cc = current
                if neighbor_count[cr, cc] >= 3:
                    if len(path) - 1 <= self.config.spine_max_length_px:
                        for pr, pc in path[:-1]:
                            filtered[pr, pc] = False
                    break

                if len(neighbors) > 1:
                    break

        return filtered.astype(bool)

    # -----------------------------
    # Sholl Analysis
    # -----------------------------
    def _find_soma_center(self, skeleton: np.ndarray) -> Optional[Tuple[int, int]]:
        if skeleton.sum() == 0:
            return None

        skel = (skeleton > 0).astype(np.uint8)

        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ])
        conv = ndi.convolve(skel, kernel, mode="constant", cval=0)
        neighbor_count = conv - 10 * skel

        junction_mask = (skel == 1) & (neighbor_count >= 3)

        if junction_mask.sum() == 0:
            skel_coords = np.argwhere(skel)
            image_center = np.array(skel.shape) / 2.0
            dists = np.linalg.norm(skel_coords - image_center, axis=1)
            best = skel_coords[np.argmin(dists)]
            return (int(best[0]), int(best[1]))

        junction_neighbors = neighbor_count * junction_mask
        flat_idx = np.argmax(junction_neighbors)
        row, col = np.unravel_index(flat_idx, skel.shape)
        return (int(row), int(col))

    def compute_sholl(
        self,
        skeleton: np.ndarray,
        soma_center: Tuple[int, int],
    ) -> pd.DataFrame:
        if skeleton.sum() == 0:
            return pd.DataFrame(columns=["radius", "intersections", "branching_index"])

        step = max(1.0, self.config.sholl_step)
        if self.config.sholl_max_radius > 0:
            max_r = self.config.sholl_max_radius
        else:
            max_r = float(min(skeleton.shape) / 2.0)

        rows_idx, cols_idx = np.where(skeleton)
        dists = np.sqrt(
            (rows_idx - soma_center[0]) ** 2 + (cols_idx - soma_center[1]) ** 2
        )

        radii = np.arange(step, max_r + step, step)
        intersections = np.zeros(len(radii), dtype=int)

        for i, r in enumerate(radii):
            r_inner = r - step
            in_shell = np.sum((dists > r_inner) & (dists <= r))
            intersections[i] = int(in_shell)

        branching_index = np.zeros(len(radii), dtype=float)
        for i in range(len(radii) - 1):
            branching_index[i] = (intersections[i + 1] - intersections[i]) / 2.0
        branching_index[-1] = 0.0

        return pd.DataFrame({
            "radius": radii,
            "intersections": intersections,
            "branching_index": branching_index,
        })

    @staticmethod
    def _sholl_summary_stats(sholl_df: pd.DataFrame) -> Dict[str, Any]:
        if sholl_df.empty:
            return {
                "sholl_auc": 0.0,
                "sholl_max_intersections": 0,
                "sholl_radius_at_max": 0.0,
                "sholl_num_radii": 0,
            }

        auc = float(np.trapz(sholl_df["intersections"], sholl_df["radius"]))
        max_idx = sholl_df["intersections"].idxmax()
        return {
            "sholl_auc": auc,
            "sholl_max_intersections": int(sholl_df.loc[max_idx, "intersections"]),
            "sholl_radius_at_max": float(sholl_df.loc[max_idx, "radius"]),
            "sholl_num_radii": int(len(sholl_df)),
        }

    # -----------------------------
    # Metrics
    # -----------------------------
    def compute_summary(
        self,
        mask: np.ndarray,
        skeleton: np.ndarray,
        branch_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        junctions, endpoints = self._count_skeleton_nodes(skeleton)

        summary = {
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
        return summary

    # -----------------------------
    # Save
    # -----------------------------
    def save_results(
        self,
        output_dir: Path,
        image_name: str,
        gray: np.ndarray,
        pre: np.ndarray,
        mask: np.ndarray,
        skeleton: np.ndarray,
        sholl_skeleton: np.ndarray,
        branch_df: pd.DataFrame,
        summary: Dict[str, Any],
        sholl_df: Optional[pd.DataFrame] = None,
    ) -> None:
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_dir / f"{image_name}_summary.csv", index=False)
        branch_df.to_csv(output_dir / f"{image_name}_branches.csv", index=False)

        if sholl_df is not None and not sholl_df.empty:
            sholl_df.to_csv(output_dir / f"{image_name}_sholl.csv", index=False)

        if self.config.save_intermediates:
            io.imsave(output_dir / f"{image_name}_gray.png", util.img_as_ubyte(exposure.rescale_intensity(gray, out_range=(0, 1))), check_contrast=False)
            io.imsave(output_dir / f"{image_name}_preprocessed.png", util.img_as_ubyte(exposure.rescale_intensity(pre, out_range=(0, 1))), check_contrast=False)
            io.imsave(output_dir / f"{image_name}_mask.png", util.img_as_ubyte(mask), check_contrast=False)
            io.imsave(output_dir / f"{image_name}_skeleton.png", util.img_as_ubyte(skeleton), check_contrast=False)
            io.imsave(output_dir / f"{image_name}_skeleton_sholl.png", util.img_as_ubyte(sholl_skeleton), check_contrast=False)
            overlay = self.make_overlay(gray, mask, skeleton)
            io.imsave(output_dir / f"{image_name}_overlay.png", overlay, check_contrast=False)

    def make_overlay(self, gray: np.ndarray, mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        base = util.img_as_ubyte(exposure.rescale_intensity(gray, out_range=(0, 1)))
        rgb = np.stack([base, base, base], axis=-1)
        rgb[mask] = [0, 255, 0]
        rgb[skeleton] = [255, 0, 0]
        return rgb

    # -----------------------------
    # Utilities
    # -----------------------------
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
        skel = (skeleton > 0).astype(np.uint8)
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ])
        conv = ndi.convolve(skel, kernel, mode="constant", cval=0)
        neighbor_count = conv - 10 * skel
        endpoints = np.logical_and(skel, neighbor_count == 1).sum()
        junctions = np.logical_and(skel, neighbor_count >= 3).sum()
        return int(junctions), int(endpoints)


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


if __name__ == "__main__":
    # Example 1: central-neuron mode
    neuron_cfg = PipelineConfig(
        mode="single_neuron",
        channel=None,
        gaussian_sigma=0.5, #Can talk about this, but somewhere between 0.5 - 0.7 seems to be the sweet spot!
        threshold_method="local", #Otsu is also good but it seems a little too "strict" and we lose some fainter processes
        local_block_size=61,
        min_object_size=120,
        hole_area_threshold=80,
        background_disk_radius=12,
        prune_short_branches_px=12.0,
        sholl_step=10.0,
        sholl_max_radius=0.0,
        spine_max_length_px=15.0,
    )

    # Example 2: cilia mode
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

    # Example 3: network mode for retina / dense mesh
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

    # Change these paths before running.
    INPUT_DIR = Path("./images_Chu")
    OUTPUT_DIR = Path("./results")

    # Pick one config to run.
    chosen_cfg = neuron_cfg
    run_batch(INPUT_DIR, OUTPUT_DIR, chosen_cfg)
