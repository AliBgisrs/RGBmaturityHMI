# main.py
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from uuid import uuid4

import numpy as np
import cv2
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator  # monotone, no overshoot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pandas as pd

from fastapi import FastAPI, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONSTANTS ----------------
HUE_BINS = 120
SAVE_DEBUG = True
PLOT_ID_FIELD = "PlotID"

# HMI parameters
HUE_GREEN_MIN, HUE_GREEN_MAX = 60.0, 120.0
HUE_YELL_MIN,  HUE_YELL_MAX  = 20.0, 60.0
HMI_T1_TRANSITION = 0.40
HMI_T2_MATURE     = 0.80
MIN_VEG_PIXELS    = 500

# NEW: ExG area threshold
EXG_AREA_THRESHOLD = 0.05

OUTPUT_BASE = os.path.abspath("./Output")

# sklearn optional
try:
    from sklearn.linear_model import LinearRegression  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# ---------------- FastAPI ----------------
app = FastAPI(title="HMI Web App", version="2.8.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
JOBS: Dict[str, Dict] = {}  # in-memory job tracker

# ---------------- Utilities ----------------
def extract_date(path: str) -> datetime:
    m = re.search(r"(\d{8})", path)
    if m:
        try:
            return datetime.strptime(m.group(1), "%m%d%Y")
        except ValueError:
            return datetime.max
    return datetime.max

def find_image_files(images_root: str) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(images_root):
        for f in files:
            fl = f.lower()
            if fl.endswith(("group1.tif", "group1.tiff", "group1.png", "group1.jpg")):
                matches.append(os.path.join(root, f))
    return sorted(matches, key=extract_date)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def unique_plotids(gdf: gpd.GeoDataFrame) -> List[str]:
    vals = gdf[PLOT_ID_FIELD].dropna().astype(str).unique().tolist()
    return sorted(vals, key=lambda x: (len(x), x))

def geoms_for_pid(gdf: gpd.GeoDataFrame, pid: str) -> List[BaseGeometry]:
    return list(gdf.loc[gdf[PLOT_ID_FIELD].astype(str) == pid, "geometry"])

def reproj_geoms(geoms: List[BaseGeometry], src_crs, dst_crs) -> List[BaseGeometry]:
    if src_crs == dst_crs:
        return geoms
    gs = gpd.GeoSeries(geoms, crs=src_crs).to_crs(dst_crs)
    return list(gs.geometry)

def overlaps_any(geoms: List[BaseGeometry], src: rasterio.io.DatasetReader, gdf_crs) -> Tuple[bool, List[BaseGeometry]]:
    geoms_t = reproj_geoms(geoms, gdf_crs, src.crs)
    rb = box(*src.bounds)
    for g in geoms_t:
        try:
            if g.intersects(rb):
                return True, geoms_t
        except Exception:
            try:
                if g.buffer(0).intersects(rb):
                    return True, geoms_t
            except Exception:
                continue
    return False, geoms_t

# ---------- HMI helpers ----------
def extract_hue_histogram(rgb_array: np.ndarray, hue_bins: int = HUE_BINS):
    rgb = np.transpose(rgb_array[:3, :, :], (1, 2, 0))
    R, G, B = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    exg = 2 * G - R - B
    veg_mask = exg > 20
    masked_rgb = rgb[veg_mask]
    if masked_rgb.size == 0:
        return np.zeros(hue_bins), None
    masked_rgb = np.clip(masked_rgb, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(masked_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    h_raw = hsv[:, 0, 0].astype(np.float32) * 2.0  # 0-179 -> 0-360
    hist, _ = np.histogram(h_raw, bins=hue_bins, range=(0.0, 120.0))
    return hist, masked_rgb

def compute_hmi_from_hist(hist, hue_bins, green_rng, yellow_rng):
    edges = np.linspace(0.0, 120.0, hue_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    g_mask = (centers >= green_rng[0]) & (centers < green_rng[1])
    y_mask = (centers >= yellow_rng[0]) & (centers < yellow_rng[1])
    g = float(hist[g_mask].sum())
    y = float(hist[y_mask].sum())
    veg = g + y
    if veg < 1:
        return np.nan, g, y, veg
    return y / veg, g, y, veg

def maturity_status(hmi, t1=HMI_T1_TRANSITION, t2=HMI_T2_MATURE):
    if np.isnan(hmi): return "NoData"
    if hmi < t1: return "Immature"
    if hmi < t2: return "Transition"
    return "Mature"

# ---------- ExG helpers ----------
def compute_exg_mean(rgb_array: np.ndarray) -> float:
    rgb = np.transpose(rgb_array[:3, :, :], (1, 2, 0))
    if rgb.size == 0: return np.nan
    R, G, B = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    exg = 2 * G - R - B
    return float(np.nanmean(exg))

def _norm01_if_needed(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB to 0..1 if values look like 0..255."""
    m = float(np.nanmax(rgb)) if rgb.size else 0.0
    if m > 1.5:  # likely 0..255
        return rgb / 255.0
    return rgb

def compute_veg_area_m2(clipped_img: np.ndarray, transform: rasterio.Affine, threshold: float = EXG_AREA_THRESHOLD) -> float:
    """Compute vegetated area (m²) where ExG > threshold."""
    if clipped_img.shape[0] < 3 or clipped_img.size == 0:
        return 0.0
    rgb = np.transpose(clipped_img[:3, :, :], (1, 2, 0)).astype(np.float32)
    rgb = _norm01_if_needed(rgb)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * G - R - B
    mask_veg = np.isfinite(exg) & (exg > threshold)
    pix_area = abs(transform.a * transform.e)  # m² if CRS units are meters
    return float(np.count_nonzero(mask_veg)) * float(pix_area)

def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    if len(x) < 2: return np.nan, np.array([])
    if _HAS_SKLEARN:
        model = LinearRegression().fit(x, y)
        return float(model.coef_[0]), model.predict(x)
    b1, b0 = np.polyfit(x.ravel(), y, 1)
    return float(b1), b1 * x.ravel() + b0

def nearest_index(arr: np.ndarray, value: float) -> int:
    arr = np.asarray(arr, dtype=float)
    return int(np.argmin(np.abs(arr - value)))

def tpmax_tpmin_pchip(dap_arr: np.ndarray, exg_arr: np.ndarray) -> Tuple[int, float, int, float, np.ndarray, np.ndarray]:
    x = np.asarray(dap_arr, dtype=float)
    y = np.asarray(exg_arr, dtype=float)
    xs = np.linspace(x.min(), x.max(), 400)
    pchip = PchipInterpolator(x, y, extrapolate=False)
    ys = pchip(xs)
    dydx = PchipInterpolator(x, y).derivative()(xs)
    sign = np.sign(dydx)
    candidates = []
    for i in range(1, len(xs)-1):
        if sign[i-1] > 0 and sign[i+1] < 0:
            candidates.append(i)
    if candidates:
        i_star = int(sorted(candidates, key=lambda i: ys[i], reverse=True)[0])
        tpmax_est = xs[i_star]
    else:
        tpmax_est = float(xs[np.nanargmax(ys)])
    after = xs >= tpmax_est
    if np.any(after):
        tpmin_est = xs[after][np.nanargmin(ys[after])]
    else:
        tpmin_est = tpmax_est
    i_max_obs = nearest_index(x, tpmax_est)
    i_min_obs = nearest_index(x, tpmin_est)
    return int(x[i_max_obs]), float(y[i_max_obs]), int(x[i_min_obs]), float(y[i_min_obs]), xs, ys

def plot_area_m2_from_geoms(geoms: List[BaseGeometry], crs) -> float:
    """Compute plot area in m² from vector geometries; reproject to a metric CRS if needed."""
    if geoms is None or len(geoms) == 0:
        return 0.0
    gs = gpd.GeoSeries(geoms, crs=crs)
    if crs is None or not getattr(crs, "is_projected", False):
        try:
            gs = gs.to_crs(3857)  # Web Mercator meters (approx)
        except Exception:
            pass
    return float(gs.unary_union.area)

# ---------------- Core pipeline ----------------
def run_pipeline(job_id: str, images_dir: str, vector_path: str, layer_name: str, sowing_date_str: str):
    try:
        try:
            sowing_date = datetime.strptime(sowing_date_str.strip(), "%m_%d_%Y")
        except Exception:
            raise ValueError("Invalid Sowing Date. Use format MM_DD_YYYY, e.g., 06_03_2025")

        JOBS[job_id] = {"percent": 0, "status": "running", "logs": [], "error": None}
        logs = JOBS[job_id]["logs"]

        # Images
        image_files = find_image_files(images_dir)
        if not image_files:
            raise FileNotFoundError("No orthomosaic files found with suffix 'group1.(tif|tiff|png|jpg)'.")
        dts = [extract_date(f) for f in image_files]
        image_daps = [(dt - sowing_date).days if dt != datetime.max else None for dt in dts]
        usable_images = [(p, d) for p, d in zip(image_files, image_daps) if d is not None]
        usable_images.sort(key=lambda x: x[1])
        num_images = len(usable_images)
        logs.append(f"Images with valid dates: {num_images}")

        # Vector
        if vector_path.lower().endswith(".gdb"):
            gdf = gpd.read_file(vector_path, layer=layer_name)
            layer_label = layer_name
        else:
            gdf = gpd.read_file(vector_path)
            layer_label = layer_name.strip() or Path(vector_path).stem

        if PLOT_ID_FIELD not in gdf.columns:
            raise ValueError(f"'{PLOT_ID_FIELD}' field not found in provided layer.")
        if gdf.empty:
            raise ValueError("No features found in the provided vector layer.")
        if gdf.crs is None:
            raise ValueError("Vector layer has no CRS. Please define a CRS and try again.")

        pids = unique_plotids(gdf)
        logs.append(f"Unique PlotIDs to process: {len(pids)}")

        output_root = os.path.join(OUTPUT_BASE, layer_label)
        ensure_dir(output_root)

        # ---------- NEW: prepare summary accumulators ----------
        all_daps_sorted = sorted({d for _, d in usable_images})
        veg_area_columns = [f"DAP_{d}" for d in all_daps_sorted]
        summary_rows: List[Dict] = []  # per-plot rows with maturity date & slope
        veg_area_rows: List[Dict] = []  # per-plot wide table: PlotID, PlotArea_m2, DAP_<d> ...

        total_units = max(1, len(pids) * max(1, num_images) * 2)  # HMI + ExG
        done_units = 0

        edges = np.linspace(0, 120, HUE_BINS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        gdf_crs = gdf.crs

        for pid in pids:
            geoms_list = geoms_for_pid(gdf, pid)
            if not geoms_list:
                logs.append(f"PlotID {pid}: no geometries; skipping.")
                continue

            plot_folder = os.path.join(output_root, f"Plot_{pid}")
            ensure_dir(plot_folder)
            logs.append(f"\nProcessing PlotID {pid} - parts: {len(geoms_list)}")

            # pre-compute plot area (m²)
            plot_area_m2 = plot_area_m2_from_geoms(geoms_list, gdf_crs)

            # -------- HMI --------
            valid_hists, valid_daps, debug_images = [], [], []
            for img_path, dap in usable_images:
                try:
                    with rasterio.open(img_path) as src:
                        hit, geoms_t = overlaps_any(geoms_list, src, gdf_crs)
                        if not hit:
                            continue
                        shapes = [mapping(g) for g in geoms_t]
                        clipped_img, _ = mask(src, shapes, crop=True)
                        if clipped_img.size == 0:
                            continue
                        hist, _ = extract_hue_histogram(clipped_img, HUE_BINS)
                        if hist.sum() != 0:
                            valid_hists.append(hist)
                            valid_daps.append(dap)
                        if SAVE_DEBUG:
                            rgb = np.transpose(clipped_img[:3, :, :], (1, 2, 0))
                            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                            out_preview = os.path.join(plot_folder, f"debug_DAP{dap}.png")
                            cv2.imwrite(out_preview, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                            debug_images.append((out_preview, dap))
                except Exception as e:
                    logs.append(f"Warning (HMI) {img_path}: {e}")
                finally:
                    done_units += 1
                    JOBS[job_id]["percent"] = min(100, int(100 * done_units / total_units))

            maturity_dap = np.nan  # will fill from HMI
            if valid_hists:
                order = np.argsort(np.asarray(valid_daps))
                valid_daps = list(np.asarray(valid_daps)[order])
                hue_matrix = np.asarray(valid_hists)[order, :]

                hmi_list, g_list, y_list, veg_list, status_list = [], [], [], [], []
                for h in hue_matrix:
                    hmi, gpx, ypx, veg = compute_hmi_from_hist(
                        h, HUE_BINS,
                        green_rng=(HUE_GREEN_MIN, HUE_GREEN_MAX),
                        yellow_rng=(HUE_YELL_MIN, HUE_YELL_MAX)
                    )
                    if veg < MIN_VEG_PIXELS: hmi = np.nan
                    hmi_list.append(hmi); g_list.append(gpx); y_list.append(ypx); veg_list.append(veg)
                    status_list.append(maturity_status(hmi))

                df_hmi = pd.DataFrame({
                    "DAP": valid_daps, "HMI": hmi_list, "Status": status_list,
                    "GreenPixels": g_list, "YellowPixels": y_list, "VegPixels": veg_list
                }).sort_values("DAP")

                # maturity date = first DAP where HMI >= T2 (Mature)
                try:
                    idx_mature = np.where(df_hmi["HMI"].to_numpy(dtype=float) >= HMI_T2_MATURE)[0]
                    if len(idx_mature) > 0:
                        maturity_dap = float(df_hmi.iloc[idx_mature[0]]["DAP"])
                except Exception:
                    maturity_dap = np.nan

                # Excel (per-plot, existing behavior)
                xls_path = os.path.join(plot_folder, f"HMI_{pid}.xlsx")
                with pd.ExcelWriter(xls_path, engine="xlsxwriter") as writer:
                    df_hmi.to_excel(writer, index=False, sheet_name="HMI")

                # HMI trajectory (matplotlib)
                plt.figure(figsize=(6, 3.2))
                plt.plot(df_hmi["DAP"], df_hmi["HMI"], marker="o", lw=1.8)
                plt.axhline(HMI_T1_TRANSITION, linestyle="--", linewidth=1)
                plt.axhline(HMI_T2_MATURE, linestyle="--", linewidth=1)
                plt.xlabel("Days After Planting (DAP)")
                plt.ylabel("HMI (yellow / (green+yellow))")
                plt.ylim(0, 1)
                plt.title(f"HMI Trajectory - PlotID {pid}")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_folder, f"HMI_{pid}.png"), dpi=300)
                plt.close()

                # Smooth histograms
                hue_matrix_smooth = gaussian_filter1d(hue_matrix, sigma=2, axis=1)

                # 1) Histograms
                fig, axes = plt.subplots(len(valid_daps), 1, figsize=(6, 1.5 * len(valid_daps)), sharex=True)
                if len(valid_daps) == 1: axes = [axes]
                colors = plt.cm.viridis(np.linspace(0, 1, len(valid_daps)))
                for ax, hist, dap, c in zip(axes, hue_matrix_smooth, valid_daps, colors):
                    hist_rel = hist / hist.sum()
                    ax.plot(centers, hist_rel, color=c, lw=1.5, label=f"{dap} DAP")
                    ax.set_ylabel("Rel. Freq."); ax.legend(loc="upper right", fontsize=8, frameon=False)
                axes[-1].set_xlabel("Hue (degrees)")
                fig.suptitle(f"Hue Histograms - PlotID {pid}", fontsize=14, fontweight="bold")
                plt.tight_layout(); plt.savefig(os.path.join(plot_folder, f"hue_histograms_{pid}.png"), dpi=300); plt.close()

                # 2) 2D Contour
                plt.figure(figsize=(8, 6))
                X, Y = np.meshgrid(centers, valid_daps)
                contour = plt.contourf(X, Y, hue_matrix_smooth, levels=50, cmap="plasma")
                cbar = plt.colorbar(contour); cbar.set_label("Pixel Count")
                plt.xlabel("Hue (degrees)"); plt.ylabel("DAP"); plt.title(f"Hue Intensity 2D Contour - PlotID {pid}")
                plt.tight_layout(); plt.savefig(os.path.join(plot_folder, f"hue_contour_{pid}.png"), dpi=600); plt.close()

                # 3) 3D Stacked
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                max_val = hue_matrix_smooth.max()
                for hist, dap in zip(hue_matrix_smooth, valid_daps):
                    xs = centers
                    verts = [(xs[0], dap, 0)] + [(x, dap, y / hist.sum()) for x, y in zip(xs, hist)] + [(xs[-1], dap, 0)]
                    color = cm.inferno(np.mean(hist) / max_val if max_val > 0 else 0.1)
                    ax.add_collection3d(Poly3DCollection([verts], facecolors=color, alpha=0.7))
                ax.set_xlabel("Hue (degrees)"); ax.set_ylabel("DAP"); ax.set_zlabel("Rel. Freq.")
                ax.set_title(f"Hue-Time Intensity (Stacked) - PlotID {pid}")
                plt.tight_layout(); plt.savefig(os.path.join(plot_folder, f"hue_3D_stacked_{pid}.png"), dpi=300); plt.close()

                # 4) Histograms with shaded band
                fig, axes = plt.subplots(len(valid_daps), 1, figsize=(6, 1.5 * len(valid_daps)), sharex=True)
                if len(valid_daps) == 1: axes = [axes]
                colors = plt.cm.viridis(np.linspace(0, 1, len(valid_daps)))
                for ax, hist, dap, c in zip(axes, hue_matrix_smooth, valid_daps, colors):
                    hist_rel = hist / hist.sum()
                    ax.axvspan(30, 60, color="orange", alpha=0.25, zorder=0)
                    ax.plot(centers, hist_rel, color=c, lw=1.5, label=f"{dap} DAP")
                    ax.set_ylabel("Rel. Freq."); ax.legend(loc="upper right", fontsize=8, frameon=False)
                axes[-1].set_xlabel("Hue (degrees)")
                fig.suptitle(f"Hue Histograms - PlotID {pid}", fontsize=14, fontweight="bold")
                plt.tight_layout(); plt.savefig(os.path.join(plot_folder, f"hue_histograms_{pid}.png"), dpi=300); plt.close()

            # -------- ExG (Matplotlib + PCHIP TPmax) --------
            dap_list, exg_list = [], []
            # NEW: per-DAP vegetation area for this plot
            veg_area_dict: Dict[int, float] = {d: 0.0 for d in all_daps_sorted}

            for img_path, dap in usable_images:
                try:
                    with rasterio.open(img_path) as src:
                        hit, geoms_t = overlaps_any(geoms_list, src, gdf_crs)
                        if not hit:
                            continue
                        shapes = [mapping(g) for g in geoms_t]
                        clipped_img, _ = mask(src, shapes, crop=True)
                        if clipped_img.shape[0] < 3:
                            continue

                        # ExG mean for curve
                        exg_val = compute_exg_mean(clipped_img)
                        if not np.isnan(exg_val):
                            dap_list.append(dap); exg_list.append(exg_val)

                        # NEW: vegetation area (m²) where ExG > 0.05
                        veg_area_m2 = compute_veg_area_m2(clipped_img, src.transform, EXG_AREA_THRESHOLD)
                        veg_area_dict[dap] = veg_area_dict.get(dap, 0.0) + float(veg_area_m2)

                except Exception as e:
                    logs.append(f"Warning (ExG) {img_path}: {e}")
                finally:
                    done_units += 1
                    JOBS[job_id]["percent"] = min(100, int(100 * done_units / total_units))

            slope = np.nan
            tpmax_dap = tpmin_dap = np.nan
            if exg_list:
                df_exg = pd.DataFrame({"DAP": dap_list, "ExG": exg_list}) \
                            .groupby("DAP", as_index=False)["ExG"].mean() \
                            .sort_values("DAP")
                dap_arr = df_exg["DAP"].to_numpy(dtype=float)
                exg_arr = df_exg["ExG"].to_numpy(dtype=float)

                tpmax_dap, tpmax_exg, tpmin_dap, tpmin_exg, xs, ys = tpmax_tpmin_pchip(dap_arr, exg_arr)

                if tpmax_dap <= tpmin_dap:
                    mask_obs = (dap_arr >= tpmax_dap) & (dap_arr <= tpmin_dap)
                    subset_x = dap_arr[mask_obs]; subset_y = exg_arr[mask_obs]
                    slope, line_pred = fit_linear_regression(subset_x, subset_y)
                else:
                    slope, line_pred, subset_x = np.nan, np.array([]), np.array([])

                # Pure Matplotlib plots (no seaborn)
                x_plot = np.linspace(dap_arr.min(), dap_arr.max(), 400)
                y_plot = PchipInterpolator(dap_arr, exg_arr)(x_plot)

                # Plot A: Full curve + TP points + slope segment
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(x_plot, y_plot, linewidth=2.2, label="PCHIP Spline")
                ax.scatter(dap_arr, exg_arr, s=32, c="black", zorder=5, label="Observed")
                ax.scatter([tpmax_dap, tpmin_dap], [tpmax_exg, tpmin_exg], s=50, c="orange", zorder=6)
                if subset_x.size and not np.isnan(slope):
                    ax.plot(subset_x, line_pred, linewidth=2.2, c="red", label=f"Slope = {slope:.3f}")
                ax.set_xlabel("Days After Planting (DAP)"); ax.set_ylabel("ExG Value")
                ax.set_title(f"ExG Trend - Plot {pid}"); ax.legend()
                fig.tight_layout(); fig.savefig(os.path.join(plot_folder, f"ExG_curve_{pid}.png"), dpi=300)
                plt.close(fig)

                # Plot B: Regression focus
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(x_plot, y_plot, linewidth=2.0, alpha=0.65, label="PCHIP Spline")
                ax.scatter(dap_arr, exg_arr, s=32, c="black")
                if subset_x.size and not np.isnan(slope):
                    ax.plot(subset_x, line_pred, linewidth=2.2, c="red", label=f"Slope = {slope:.3f}")
                    ax.scatter([tpmax_dap, tpmin_dap], [tpmax_exg, tpmin_exg], s=50, c="orange")
                ax.set_xlabel("Days After Planting (DAP)"); ax.set_ylabel("ExG Value")
                ax.set_title(f"ExG Subset Regression - Plot {pid}"); ax.legend()
                fig.tight_layout(); fig.savefig(os.path.join(plot_folder, f"ExG_regression_{pid}.png"), dpi=300)
                plt.close(fig)

                # CSV (per-plot, existing behavior)
                out = df_exg.copy()
                out["TPmax_DAP"] = tpmax_dap; out["TPmax_ExG"] = tpmax_exg
                out["TPmin_DAP"] = tpmin_dap; out["TPmin_ExG"] = tpmin_exg
                out["Slope_TPmax_to_TPmin"] = slope
                out.to_csv(os.path.join(plot_folder, f"ExG_values_{pid}.csv"), index=False)
                logs.append(f"ExG -> TPmax {tpmax_dap}, TPmin {tpmin_dap}, slope {slope:.3f}")
            else:
                logs.append("ExG: no valid values for this plot")

            # ---------- NEW: add to summary accumulators ----------
            summary_rows.append({
                "PlotID": pid,
                "Maturity_DAP": maturity_dap,                 # from HMI crossing T2
                "Slope_TPmax_to_TPmin": slope                 # from ExG subset regression
            })
            row_veg = {"PlotID": pid, "PlotArea_m2": plot_area_m2}
            row_veg.update({f"DAP_{d}": float(veg_area_dict.get(d, 0.0)) for d in all_daps_sorted})
            veg_area_rows.append(row_veg)

            logs.append(f"Saved outputs -> {plot_folder}")

        # ---------- NEW: write combined SUMMARY.xlsx ----------
        summary_df = pd.DataFrame(summary_rows).sort_values("PlotID")
        veg_df = pd.DataFrame(veg_area_rows).sort_values("PlotID")
        # Ensure consistent column order for veg_df
        veg_df = veg_df[["PlotID", "PlotArea_m2"] + [f"DAP_{d}" for d in all_daps_sorted]]

        summary_xlsx = os.path.join(output_root, "SUMMARY.xlsx")
        with pd.ExcelWriter(summary_xlsx, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Maturity_Summary")
            veg_df.to_excel(writer, index=False, sheet_name="Vegetation_Area")

        logs.append(f"SUMMARY.xlsx written at: {summary_xlsx}")

        JOBS[job_id]["percent"] = 100
        JOBS[job_id]["status"] = "done"

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

# ---------------- Simple UI ----------------
PAGE_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>HMI Web App</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --border:#e5e7eb; --muted:#6b7280; --bg:#0b1220; --text:#e5e7eb; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .card { max-width: 960px; margin: 0 auto; padding: 20px; border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
    label { display:block; margin:14px 0 6px; font-weight:600; }
    input[type=text] { width:100%; padding:10px; border:1px solid #d1d5db; border-radius:8px; }
    button { margin-top:18px; padding:10px 16px; border:0; border-radius:10px; background:#111827; color:white; cursor:pointer; }
    button:hover { background:#0b1220; }
    pre { white-space: pre-wrap; background: var(--bg); color:var(--text); padding:16px; border-radius:10px; overflow:auto; }
    .error { color:#b91c1c; font-weight:600; }
    .progress { position: relative; height: 18px; width: 100%; background: #f3f4f6; border-radius: 10px; margin-top: 16px; border: 1px solid #e5e7eb; display:none; }
    .bar { height: 100%; width: 0%; border-radius: 10px; transition: width 0.4s ease, background 0.4s ease; }
    .percent-label { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 12px; color:#111827; font-weight:600; }
    footer { margin-top: 18px; font-size: 12px; color: var(--muted); text-align:center; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Hue Maturity Index (HMI) - Web App</h1>
    <p>This app is designed to estimate the maturity date using a time-series analysis of RGB imagery based on the Hue maturity index.</p>

    <form id="hmi-form">
      <label for="images_dir">Images Directory (root to scan recursively)</label>
      <input type="text" id="images_dir" name="images_dir" placeholder="E.g., D:/UAS_Beans/2025/SVREC" required>

      <label for="vector_path">Vector Path (Shapefile .shp or FileGDB .gdb)</label>
      <input type="text" id="vector_path" name="vector_path" placeholder="E.g., D:/.../New File Geodatabase.gdb OR .../plots.shp" required>

      <label for="layer_name">Layer Name (required for .gdb; ignored for .shp)</label>
      <input type="text" id="layer_name" name="layer_name" placeholder="E.g., SEVREC2502">

      <label for="sowing_date">Sowing Date (MM_DD_YYYY)</label>
      <input type="text" id="sowing_date" name="sowing_date" placeholder="06_03_2025" required>

      <button type="submit">Run Pipeline (HMI + ExG)</button>
    </form>

    <div class="progress" id="progress">
      <div class="bar" id="bar"></div>
      <div class="percent-label" id="percent">0%</div>
    </div>

    <div id="error" class="error" style="display:none;"></div>
    <div id="results" style="margin-top:16px;"></div>

    <footer>
      Developed by <strong>Aliasghar Bazrafkan</strong>, <a href="mailto:bazrafka@msu.edu">bazrafka@msu.edu</a>
    </footer>
  </div>

  <script>
    const form = document.getElementById('hmi-form');
    const progress = document.getElementById('progress');
    const bar = document.getElementById('bar');
    const percent = document.getElementById('percent');
    const errorBox = document.getElementById('error');
    const results = document.getElementById('results');

    function colorFor(p){ if(p < 33) return '#ef4444'; if(p < 66) return '#f59e0b'; return '#10b981'; }
    function updateBar(p){ progress.style.display = 'block'; bar.style.width = p + '%'; bar.style.background = colorFor(p); percent.textContent = p + '%'; }
    async function poll(jobId){
      let done = false; updateBar(0);
      while(!done){
        const r = await fetch(`/progress/${jobId}`); const data = await r.json(); updateBar(data.percent || 0);
        if(data.status !== 'running'){
          done = true;
          if(data.status === 'done'){
            const lr = await fetch(`/logs/${jobId}`); const ldata = await r.json().catch(()=>({logs: []}));
            const pre = document.createElement('pre'); pre.textContent = (ldata.logs || []).join('\\n');
            results.innerHTML = '<h2>Run Log</h2>'; results.appendChild(pre);
          } else { errorBox.style.display = 'block'; errorBox.textContent = 'Error: ' + (data.error || 'Unknown error'); }
        }
        await new Promise(res => setTimeout(res, 1000));
      }
    }
    form.addEventListener('submit', async (e) => {
      e.preventDefault(); errorBox.style.display = 'none'; results.innerHTML = ''; updateBar(0);
      const fd = new FormData(form);
      const r = await fetch('/start', { method: 'POST', body: fd });
      if(!r.ok){ const d = await r.json().catch(()=>({detail:'Failed to start'})); errorBox.style.display = 'block'; errorBox.textContent = d.detail || 'Failed to start'; return; }
      const data = await r.json(); poll(data.job_id);
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE_HTML)

@app.post("/start")
def start_job(
    background_tasks: BackgroundTasks,
    images_dir: str = Form(...),
    vector_path: str = Form(...),
    layer_name: str = Form(""),
    sowing_date: str = Form(...),  # MM_DD_YYYY
):
    if not images_dir or not vector_path or not sowing_date:
        raise HTTPException(status_code=400, detail="images_dir, vector_path and sowing_date are required.")
    if vector_path.lower().endswith(".gdb") and not layer_name.strip():
        raise HTTPException(status_code=400, detail="layer_name is required when using a FileGDB (.gdb).")

    job_id = uuid4().hex
    JOBS[job_id] = {"percent": 0, "status": "running", "logs": [], "error": None}
    background_tasks.add_task(run_pipeline, job_id, images_dir, vector_path, layer_name, sowing_date)
    return {"job_id": job_id}

@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    return {"percent": job.get("percent", 0), "status": job.get("status", "running"), "error": job.get("error")}

@app.get("/logs/{job_id}")
def get_logs(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    return {"logs": job.get("logs", [])}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002)
