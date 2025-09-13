# main.py
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from uuid import uuid4

import numpy as np
import cv2
import rasterio
from rasterio.mask import mask
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pandas as pd

from fastapi import FastAPI, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONSTANTS (same logic as original) ----------------
HUE_BINS = 120
SAVE_DEBUG = True
PLOT_ID_FIELD = "PlotID"

# HMI parameters
HUE_GREEN_MIN, HUE_GREEN_MAX = 60.0, 120.0
HUE_YELL_MIN,  HUE_YELL_MAX  = 20.0, 60.0
HMI_T1_TRANSITION = 0.40
HMI_T2_MATURE     = 0.80
MIN_VEG_PIXELS    = 500

# Output base
OUTPUT_BASE = os.path.abspath("./Output")

# ---------------- FastAPI app ----------------
app = FastAPI(title="HMI Web App", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# In-memory job store for progress
JOBS: Dict[str, Dict] = {}
# JOBS[job_id] = {"percent": int, "status": "running"|"done"|"error", "logs": [str], "error": str|None}

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
    """Recursively find files matching *group1.(tif|tiff|png|jpg)*."""
    matches: List[str] = []
    for root, _, files in os.walk(images_root):
        for f in files:
            fl = f.lower()
            if fl.endswith(("group1.tif", "group1.tiff", "group1.png", "group1.jpg")):
                matches.append(os.path.join(root, f))
    return sorted(matches, key=extract_date)

def extract_hue_histogram(rgb_array: np.ndarray, hue_bins: int = HUE_BINS):
    # rgb_array shape: (bands, H, W) — use first 3 bands
    rgb = np.transpose(rgb_array[:3, :, :], (1, 2, 0))

    # Vegetation mask (Excess Green)
    R, G, B = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    exg = 2 * G - R - B
    veg_mask = exg > 20  # same threshold
    masked_rgb = rgb[veg_mask]

    if masked_rgb.size == 0:
        return np.zeros(hue_bins), None

    masked_rgb = np.clip(masked_rgb, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(masked_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    h_raw = hsv[:, 0, 0].astype(np.float32) * 2.0  # OpenCV 0–179 => 0–360

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
    hmi = y / veg
    return hmi, g, y, veg

def maturity_status(hmi, t1=HMI_T1_TRANSITION, t2=HMI_T2_MATURE):
    if np.isnan(hmi):
        return "NoData"
    if hmi < t1:
        return "Immature"
    elif hmi < t2:
        return "Transition"
    else:
        return "Mature"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ---------------- Core pipeline (ALL PlotIDs) ----------------
def run_pipeline(job_id: str, images_dir: str, vector_path: str, layer_name: str, sowing_date_str: str):
    try:
        # Validate sowing date (MM_DD_YYYY)
        try:
            sowing_date = datetime.strptime(sowing_date_str.strip(), "%m_%d_%Y")
        except Exception:
            raise ValueError("Invalid Sowing Date. Use format MM_DD_YYYY, e.g., 06_03_2025")

        JOBS[job_id] = {"percent": 0, "status": "running", "logs": [], "error": None}
        logs = JOBS[job_id]["logs"]

        # Gather + sort images
        image_files = find_image_files(images_dir)
        if not image_files:
            raise FileNotFoundError("No orthomosaic files found with suffix 'group1.(tif|tiff|png|jpg)'.")
        image_daps = [(extract_date(f) - sowing_date).days if extract_date(f) != datetime.max else None
                      for f in image_files]
        usable_images = [(p, d) for p, d in zip(image_files, image_daps) if d is not None]
        num_images = len(usable_images)

        # Load plot boundaries (ALL PlotIDs)
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

        # Output root mirrors original pattern but under ./Output/<layer_label>/
        output_root = os.path.join(OUTPUT_BASE, layer_label)
        ensure_dir(output_root)

        total_units = max(1, len(gdf) * max(1, num_images))
        done_units = 0

        for _, row in gdf.iterrows():
            plot_id = row[PLOT_ID_FIELD]
            logs.append(f"\n▶ Processing PlotID {plot_id}")
            plot_folder = os.path.join(output_root, f"Plot_{plot_id}")
            ensure_dir(plot_folder)

            valid_hists, valid_daps, debug_images = [], [], []

            geom_orig = row.geometry
            gdf_crs = gdf.crs

            for img_path, dap in usable_images:
                try:
                    with rasterio.open(img_path) as src:
                        # Reproject geometry to raster CRS if needed
                        if gdf_crs and (gdf_crs != src.crs):
                            tmp = gpd.GeoSeries([geom_orig], crs=gdf_crs).to_crs(src.crs)
                            geom_use = [tmp.iloc[0].__geo_interface__]
                        else:
                            geom_use = [geom_orig.__geo_interface__]

                        clipped_img, _ = mask(src, geom_use, crop=True)
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
                    logs.append(f"⚠ Error clipping {img_path}: {e}")
                finally:
                    done_units += 1
                    JOBS[job_id]["percent"] = min(100, int(100 * done_units / total_units))

            if not valid_hists:
                logs.append("No valid histograms; skipping plot.")
                continue

            hue_matrix = np.array(valid_hists)

            # Compute HMI BEFORE smoothing
            hmi_list, g_list, y_list, veg_list, status_list = [], [], [], [], []
            for h in hue_matrix:
                hmi, gpx, ypx, veg = compute_hmi_from_hist(
                    h, HUE_BINS,
                    green_rng=(HUE_GREEN_MIN, HUE_GREEN_MAX),
                    yellow_rng=(HUE_YELL_MIN, HUE_YELL_MAX)
                )
                if veg < MIN_VEG_PIXELS:
                    hmi = np.nan
                hmi_list.append(hmi)
                g_list.append(gpx)
                y_list.append(ypx)
                veg_list.append(veg)
                status_list.append(maturity_status(hmi))

            df_hmi = pd.DataFrame({
                "DAP": valid_daps,
                "HMI": hmi_list,
                "Status": status_list,
                "GreenPixels": g_list,
                "YellowPixels": y_list,
                "VegPixels": veg_list
            })

            # Save Excel
            xls_path = os.path.join(plot_folder, f"HMI_{plot_id}.xlsx")
            with pd.ExcelWriter(xls_path, engine="xlsxwriter") as writer:
                df_hmi.to_excel(writer, index=False, sheet_name="HMI")

            # Shared edges/centers
            edges = np.linspace(0, 120, HUE_BINS + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])

            # HMI trajectory plot
            plt.figure(figsize=(6, 3.2))
            plt.plot(df_hmi["DAP"], df_hmi["HMI"], marker="o", lw=1.8)
            plt.axhline(HMI_T1_TRANSITION, linestyle="--", linewidth=1)
            plt.axhline(HMI_T2_MATURE, linestyle="--", linewidth=1)
            plt.xlabel("Days After Planting (DAP)")
            plt.ylabel("HMI (yellow / (green+yellow))")
            plt.ylim(0, 1)
            plt.title(f"HMI Trajectory – PlotID {plot_id}")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f"HMI_{plot_id}.png"), dpi=300)
            plt.close()

            # Smooth histograms
            hue_matrix_smooth = gaussian_filter1d(hue_matrix, sigma=2, axis=1)

            # === 1) Histograms ===
            fig, axes = plt.subplots(len(valid_daps), 1, figsize=(6, 1.5 * len(valid_daps)), sharex=True)
            if len(valid_daps) == 1:
                axes = [axes]
            colors = plt.cm.viridis(np.linspace(0, 1, len(valid_daps)))
            for ax, hist, dap, c in zip(axes, hue_matrix_smooth, valid_daps, colors):
                hist_rel = hist / hist.sum()
                ax.plot(centers, hist_rel, color=c, lw=1.5, label=f"{dap} DAP")
                ax.set_ylabel("Rel. Freq.")
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            axes[-1].set_xlabel("Hue (degrees)")
            fig.suptitle(f"Hue Histograms - PlotID {plot_id}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f"hue_histograms_{plot_id}.png"), dpi=300)
            plt.close()

            # === 2) 2D Contour ===
            plt.figure(figsize=(8, 6))
            X, Y = np.meshgrid(centers, valid_daps)
            contour = plt.contourf(X, Y, hue_matrix_smooth, levels=50, cmap="plasma")
            cbar = plt.colorbar(contour)
            cbar.set_label("Pixel Count")
            plt.xlabel("Hue (degrees)")
            plt.ylabel("DAP")
            plt.title(f"Hue Intensity 2D Contour - PlotID {plot_id}")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f"hue_contour_{plot_id}.png"), dpi=600)
            plt.close()

            # === 3) 3D Stacked ===
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            max_val = hue_matrix_smooth.max()
            for hist, dap in zip(hue_matrix_smooth, valid_daps):
                xs = centers
                verts = [(xs[0], dap, 0)] + [(x, dap, y / hist.sum()) for x, y in zip(xs, hist)] + [(xs[-1], dap, 0)]
                color = cm.inferno(np.mean(hist) / max_val if max_val > 0 else 0.1)
                ax.add_collection3d(Poly3DCollection([verts], facecolors=color, alpha=0.7))
            ax.set_xlabel("Hue (degrees)")
            ax.set_ylabel("DAP")
            ax.set_zlabel("Rel. Freq.")
            ax.set_title(f"Hue-Time Intensity (Stacked) - PlotID {plot_id}")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f"hue_3D_stacked_{plot_id}.png"), dpi=300)
            plt.close()

            # === 4) Histograms with shaded maturity band ===
            fig, axes = plt.subplots(len(valid_daps), 1, figsize=(6, 1.5 * len(valid_daps)), sharex=True)
            if len(valid_daps) == 1:
                axes = [axes]
            colors = plt.cm.viridis(np.linspace(0, 1, len(valid_daps)))
            for ax, hist, dap, c in zip(axes, hue_matrix_smooth, valid_daps, colors):
                hist_rel = hist / hist.sum()
                ax.axvspan(30, 60, color="orange", alpha=0.25, zorder=0)
                ax.plot(centers, hist_rel, color=c, lw=1.5, label=f"{dap} DAP")
                ax.set_ylabel("Rel. Freq.")
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            axes[-1].set_xlabel("Hue (degrees)")
            fig.suptitle(f"Hue Histograms - PlotID {plot_id}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f"hue_histograms_{plot_id}.png"), dpi=300)
            plt.close()

            # --- Clipped Previews ---
            if SAVE_DEBUG and debug_images:
                fig, axes = plt.subplots(len(debug_images), 1, figsize=(6, 1.5 * len(debug_images)))
                if len(debug_images) == 1:
                    axes = [axes]
                for ax, (img_path, dap) in zip(axes, debug_images):
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    ax.imshow(img); ax.axis("off")
                    ax.text(-10, img.shape[0] // 2, f"{dap} DAP", color="white",
                            fontsize=10, fontweight="bold", va="center", ha="right",
                            bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
                fig.suptitle(f"PlotID {plot_id}", fontsize=14, fontweight="bold", y=0.995)
                plt.subplots_adjust(hspace=0, left=0.18, right=0.98, top=0.97, bottom=0.03)
                plt.savefig(os.path.join(plot_folder, f"clipped_previews_{plot_id}.png"), dpi=300)
                plt.close()

            logs.append(f"✔ Outputs saved → {plot_folder}")

        JOBS[job_id]["percent"] = 100
        JOBS[job_id]["status"] = "done"

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

# ---------------- Web UI (improved styling) ----------------
PAGE_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>HMI Web App</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --border:#e5e7eb; --muted:#6b7280; --bg:#0b1220; --text:#e5e7eb;
      --green-bg:#ecfdf5; --green-br:#a7f3d0;
      --blue-bg:#eff6ff;  --blue-br:#bfdbfe;
    }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .card { max-width: 960px; margin: 0 auto; padding: 20px; border: 1px solid var(--border);
            border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
    .header {
      background: var(--green-bg); border: 1px solid var(--green-br); border-radius: 12px;
      padding: 16px 18px; display: flex; align-items: center; gap: 12px;
    }
    .header h1 { margin: 0; font-size: 22px; line-height: 1.25; }
    .icon { width: 28px; height: 28px; flex: 0 0 28px; }
    .desc {
      background: var(--blue-bg); border: 1px solid var(--blue-br); border-radius: 10px;
      padding: 12px 14px; margin-top: 12px; font-size: 14px;
    }
    label { display:block; margin:16px 0 6px; font-weight:600; }
    input[type=text] { width:100%; padding:12px; border:1px solid #d1d5db; border-radius:10px; }
    .button-row { display:flex; flex-wrap:wrap; gap: 14px; margin-top: 24px; }
    button { padding:12px 18px; border:0; border-radius:12px; background:#111827; color:white; cursor:pointer; }
    button:hover { background:#0b1220; }
    pre { white-space: pre-wrap; background: var(--bg); color:var(--text); padding:16px; border-radius:10px; overflow:auto; }
    .error { color:#b91c1c; font-weight:600; margin-top: 10px; }
    .progress { position: relative; height: 18px; width: 100%; background: #f3f4f6; border-radius: 10px;
                margin-top: 18px; border: 1px solid #e5e7eb; display:none; }
    .bar { height: 100%; width: 0%; border-radius: 10px; transition: width 0.4s ease, background 0.4s ease; }
    .percent-label { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 12px; color:#111827; font-weight:600; }
    footer { margin-top: 18px; font-size: 12px; color: var(--muted); text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <!-- Plant icon (inline SVG) -->
      <svg class="icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M12 22s-2-6 3-11c5-5 7-7 7-7s-2 2-7 3C10 8 8 6 8 6s2 2 1 7c-1 5-5 7-5 7s2-2 7-3"
              stroke="#059669" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <h1>Hue Maturity Index (HMI) – Web App</h1>
    </div>

    <div class="desc">
      The app mirrors your original pipeline, processes <strong>all PlotIDs</strong>, and saves outputs under
      <code>./Output/&lt;layer&gt;/Plot_&lt;PlotID&gt;</code>.
    </div>

    <form id="hmi-form" style="margin-top:18px;">
      <label for="images_dir">Images Directory (root to scan recursively)</label>
      <input type="text" id="images_dir" name="images_dir" placeholder="E.g., D:/UAS_Beans/2025/SVREC" required>

      <label for="vector_path">Vector Path (Shapefile .shp or FileGDB .gdb)</label>
      <input type="text" id="vector_path" name="vector_path" placeholder="E.g., D:/.../New File Geodatabase.gdb OR .../plots.shp" required>

      <label for="layer_name">Layer Name (required for .gdb; ignored for .shp)</label>
      <input type="text" id="layer_name" name="layer_name" placeholder="E.g., SEVREC2502">

      <label for="sowing_date">Sowing Date (MM_DD_YYYY)</label>
      <input type="text" id="sowing_date" name="sowing_date" placeholder="06_03_2025" required>

      <div class="button-row">
        <button type="submit">Run HMI Pipeline</button>
      </div>
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
    function updateBar(p){
      progress.style.display = 'block';
      bar.style.width = p + '%';
      bar.style.background = colorFor(p);
      percent.textContent = p + '%';
    }
    async function poll(jobId){
      let done = false; updateBar(0);
      while(!done){
        const r = await fetch(`/progress/${jobId}`); const data = await r.json(); updateBar(data.percent || 0);
        if(data.status !== 'running'){
          done = true;
          if(data.status === 'done'){
            const lr = await fetch(`/logs/${jobId}`); const ldata = await lr.json();
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
      if(!r.ok){ const d = await r.json().catch(()=>({detail:'Failed to start'}));
        errorBox.style.display = 'block'; errorBox.textContent = d.detail || 'Failed to start'; return; }
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
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"percent": job.get("percent", 0), "status": job.get("status", "running"), "error": job.get("error")}

@app.get("/logs/{job_id}")
def get_logs(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"logs": job.get("logs", [])}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002)
