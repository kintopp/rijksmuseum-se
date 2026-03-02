"""Shared HTML/CSS/JS template for cluster explorer visualizations.

Generates a self-contained HTML page with:
- Plotly scattergl plot (WebGL, handles 20K+ points)
- Custom sidebar legend with cluster groups
- Click cluster → detail panel (bar charts for subjects, types, creators, etc.)
- Click point → open Rijksmuseum website
- Keyboard shortcuts (zoom, pan, labels, noise, help overlay)
- Centroid label annotations

Ported from rijksmuseum-mcp-plus explore-smell-clusters.py, with smell-specific
parts stripped out and generalized for any dimensionality reduction method.
"""

import html as html_mod
import json

import numpy as np
from collections import Counter

# ── Color palette (35 distinct colors) ─────────────────────
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
    "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5",
    "#393B79", "#637939", "#8C6D31", "#843C39", "#7B4173",
    "#BD9E39",
]


def build_cluster_traces(
    labels: np.ndarray,
    coords: np.ndarray,
    art_ids: list[int],
    object_numbers: list[str],
    meta: dict[int, dict],
    hover_texts: list[str],
) -> dict:
    """Build Plotly traces, annotations, legend items, and cluster detail from HDBSCAN labels.

    Returns a dict with keys:
        traces, annotations, legend_items, cluster_detail,
        data_extent, noise_trace_idx, n_clusters
    """
    # ── Pre-compute index arrays once per cluster ─────────
    # Avoids recomputing `labels == cid` and `np.where(mask)` per cluster.
    # Sort by cluster size descending so the sidebar shows largest clusters first.
    cid_counts = Counter(labels[labels != -1])
    sorted_cids = sorted(cid_counts, key=lambda c: cid_counts[c], reverse=True)
    n_clusters = len(sorted_cids)
    noise_indices = np.where(labels == -1)[0]
    cluster_indices: dict[int, np.ndarray] = {}
    for cid in sorted_cids:
        cluster_indices[cid] = np.where(labels == cid)[0]

    # ── Build profiles + traces + annotations in one pass ─
    traces = []
    annotations = []
    legend_items = [{"type": "group", "label": f"CLUSTERS ({n_clusters})"}]
    cluster_detail = {}

    # Noise first (rendered behind, hidden by default)
    noise_trace_idx = -1
    if len(noise_indices) > 0:
        noise_trace_idx = 0
        noise_hover = [
            t + '<br><span style="color:#888">Noise</span>'
            for t in (hover_texts[i] for i in noise_indices)
        ]
        traces.append({
            "x": coords[noise_indices, 0].tolist(),
            "y": coords[noise_indices, 1].tolist(),
            "text": noise_hover,
            "customdata": [object_numbers[i] for i in noise_indices],
            "mode": "markers",
            "type": "scattergl",
            "name": f"Noise ({len(noise_indices):,})",
            "marker": {"color": "#eee", "size": 2, "opacity": 0.2},
            "hovertemplate": "%{text}<extra></extra>",
            "visible": "legendonly",
        })

    trace_idx = 1 if noise_trace_idx == 0 else 0
    for idx, cid in enumerate(sorted_cids):
        indices = cluster_indices[cid]
        size = len(indices)
        cx, cy = float(coords[indices, 0].mean()), float(coords[indices, 1].mean())
        color = COLORS[idx % len(COLORS)]

        # Aggregate metadata
        type_counts = Counter()
        creator_counts = Counter()
        subject_counts = Counter()
        material_counts = Counter()
        technique_counts = Counter()
        for i in indices:
            m = meta[art_ids[i]]
            type_counts.update(m["types"])
            creator_counts.update(m["creators"])
            subject_counts.update(m["subjects"])
            material_counts.update(m["materials"])
            technique_counts.update(m["techniques"])

        # Label: most common type + most common subject
        parts = []
        if type_counts:
            parts.append(type_counts.most_common(1)[0][0])
        if subject_counts:
            parts.append(subject_counts.most_common(1)[0][0])
        label = " \u00b7 ".join(parts) if parts else f"Cluster {cid}"
        short_label = label[:50]

        # Trace — cluster label appended to hover text; <extra></extra> suppresses
        # the secondary colored box that was invisible on light-colored traces.
        cluster_tag = f'<br><span style="color:#888">Cluster {cid}: {html_mod.escape(short_label)}</span>'
        cluster_hover = [hover_texts[i] + cluster_tag for i in indices]
        traces.append({
            "x": coords[indices, 0].tolist(),
            "y": coords[indices, 1].tolist(),
            "text": cluster_hover,
            "customdata": [object_numbers[i] for i in indices],
            "mode": "markers",
            "type": "scattergl",
            "name": f"{cid}: {short_label} ({size:,})",
            "marker": {"color": color, "size": 4, "opacity": 0.6},
            "hovertemplate": "%{text}<extra></extra>",
        })

        # Legend item
        legend_items.append({
            "type": "cluster",
            "cid": int(cid),
            "traceIdx": trace_idx,
            "label": f"{cid}: {short_label}",
            "color": color,
            "size": size,
        })
        trace_idx += 1

        # Annotation (centroid label)
        short = label[:35] + ("..." if len(label) > 35 else "")
        annotations.append({
            "x": cx, "y": cy,
            "text": f"<b>{cid}</b>: {html_mod.escape(short)}",
            "showarrow": False,
            "font": {"size": 11, "color": "#333"},
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "#999",
            "borderwidth": 1,
            "borderpad": 2,
        })

        # Detail panel data
        cluster_detail[int(cid)] = {
            "label": label,
            "size": size,
            "top_types": type_counts.most_common(5),
            "top_creators": creator_counts.most_common(5),
            "top_subjects": subject_counts.most_common(8),
            "top_materials": material_counts.most_common(5),
            "top_techniques": technique_counts.most_common(5),
        }

    # ── Data extent (exclude noise for tighter fit) ───────
    if len(coords) == 0:
        data_extent = {"xMin": -1.0, "xMax": 1.0, "yMin": -1.0, "yMax": 1.0}
    else:
        non_noise = labels != -1
        extent_coords = coords[non_noise] if non_noise.any() else coords
        data_extent = {
            "xMin": float(extent_coords[:, 0].min()),
            "xMax": float(extent_coords[:, 0].max()),
            "yMin": float(extent_coords[:, 1].min()),
            "yMax": float(extent_coords[:, 1].max()),
        }

    return {
        "traces": traces,
        "annotations": annotations,
        "legend_items": legend_items,
        "cluster_detail": cluster_detail,
        "data_extent": data_extent,
        "noise_trace_idx": noise_trace_idx,
        "n_clusters": n_clusters,
    }


def generate_explorer_html(
    *,
    title: str,
    subtitle: str,
    traces: list[dict],
    annotations: list[dict],
    cluster_detail: dict,
    legend_items: list[dict],
    data_extent: dict,
    noise_trace_idx: int,
    n_clusters: int = 0,
    axis_label: str = "Dim",
) -> str:
    """Return a complete self-contained HTML string for the cluster explorer."""

    traces_json = json.dumps(traces)
    cluster_detail_json = json.dumps(cluster_detail)
    legend_items_json = json.dumps(legend_items)
    data_extent_json = json.dumps(data_extent)

    layout = {
        "title": {
            "text": f"Rijksmuseum Collection \u2014 {title}<br><sub>{subtitle}</sub>",
            "font": {"size": 18},
        },
        "xaxis": {"title": f"{axis_label} 1", "showgrid": False, "zeroline": False},
        "yaxis": {"title": f"{axis_label} 2", "showgrid": False, "zeroline": False, "scaleanchor": "x"},
        "hovermode": "closest",
        "showlegend": False,
        "annotations": annotations,
        "paper_bgcolor": "#fafafa",
        "plot_bgcolor": "#fff",
        "margin": {"t": 80, "b": 50, "l": 50, "r": 20},
    }
    layout_json = json.dumps(layout)

    # The HTML template uses {{ }} for literal JS braces inside the f-string
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rijksmuseum \u2014 {title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #fafafa; }}
  #controls {{
    padding: 10px 20px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    background: #fff; border-bottom: 1px solid #e0e0e0;
  }}
  #controls button {{
    padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 12px;
  }}
  #controls button:hover {{ background: #f0f0f0; }}
  #controls .sep {{ width: 1px; height: 20px; background: #ddd; }}
  #controls .info {{ color: #666; font-size: 12px; margin-left: auto; }}
  kbd {{ background: #eee; padding: 1px 5px; border-radius: 3px; border: 1px solid #ccc; font-size: 11px; }}

  /* Layout: sidebar + plot */
  #main {{ display: flex; height: calc(100vh - 44px); }}
  #plot {{ flex: 1; min-width: 0; }}

  /* Custom legend sidebar */
  #sidebar {{
    width: 280px; min-width: 280px; background: #fff;
    border-left: 1px solid #e0e0e0; display: flex; flex-direction: column;
    font-size: 12px; user-select: none;
  }}
  #sidebar-header {{
    padding: 8px 12px; border-bottom: 1px solid #e0e0e0;
    font-weight: 600; font-size: 12px; color: #444;
    display: flex; justify-content: space-between; align-items: center;
  }}
  #sidebar-header button {{
    padding: 2px 8px; border: 1px solid #ccc; border-radius: 3px;
    background: #fff; cursor: pointer; font-size: 10px;
  }}
  #sidebar-header button:hover {{ background: #f0f0f0; }}
  #cluster-list {{
    flex: 1; overflow-y: auto; overflow-x: hidden; padding: 4px 0;
  }}
  .legend-group-title {{
    padding: 6px 12px 2px; font-size: 10px; font-weight: 700; color: #888;
    text-transform: uppercase; letter-spacing: 0.5px; position: sticky;
    top: 0; background: #fff; z-index: 1;
  }}
  .legend-item {{
    display: flex; align-items: center; padding: 3px 12px; cursor: pointer;
    gap: 6px; line-height: 1.3;
  }}
  .legend-item:hover {{ background: #f5f5f5; }}
  .legend-item.dimmed {{ opacity: 0.35; }}
  .legend-swatch {{
    width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
  }}
  .legend-label {{
    flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    font-size: 11px;
  }}
  .legend-count {{
    color: #999; font-size: 10px; flex-shrink: 0;
  }}

  /* Help overlay */
  #help-overlay {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
    z-index: 200; justify-content: center; align-items: center;
  }}
  #help-overlay.visible {{ display: flex; }}
  #help-box {{
    background: #fff; border-radius: 10px; padding: 24px 32px;
    max-width: 460px; box-shadow: 0 8px 30px rgba(0,0,0,0.3);
  }}
  #help-box h3 {{ margin: 0 0 14px; font-size: 16px; }}
  #help-box table {{ border-collapse: collapse; width: 100%; }}
  #help-box td {{ padding: 3px 0; font-size: 13px; }}
  #help-box td:first-child {{ font-family: monospace; font-weight: bold; color: #444; padding-right: 16px; white-space: nowrap; }}
  #help-box .section {{ font-weight: bold; color: #888; padding-top: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}

  /* Zoom toast */
  #zoom-toast {{
    display: none; position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.75); color: #fff; padding: 6px 16px; border-radius: 20px;
    font-size: 13px; z-index: 150; pointer-events: none; transition: opacity 0.3s;
  }}

  /* Detail panel */
  #detail-panel {{
    display: none; position: fixed; right: 296px; top: 56px;
    width: 360px; max-height: calc(100vh - 72px); overflow-y: auto;
    background: #fff; border: 1px solid #ccc; border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15); padding: 16px; z-index: 100;
    font-size: 12px; line-height: 1.5;
  }}
  #detail-panel .close {{ position: absolute; top: 8px; right: 12px; cursor: pointer; font-size: 18px; color: #999; }}
  #detail-panel .close:hover {{ color: #333; }}
  #detail-panel h3 {{ margin-bottom: 6px; font-size: 14px; }}
  #detail-panel .stat {{ margin: 3px 0; }}
  #detail-panel .bar {{
    display: flex; align-items: center; margin: 2px 0; font-size: 11px; gap: 4px;
  }}
  #detail-panel .bar-fill {{
    height: 14px; border-radius: 2px; min-width: 2px;
  }}
  #detail-panel .bar-text {{ white-space: nowrap; color: #555; }}
  .bar-section {{ margin: 10px 0 4px; font-weight: bold; color: #666; font-size: 11px; text-transform: uppercase; }}
</style>
</head>
<body>

<div id="controls">
  <button onclick="toggleLabels()"><kbd>L</kbd> Labels</button>
  <button onclick="toggleNoise()"><kbd>N</kbd> Noise</button>
  <button onclick="showAllClusters()"><kbd>A</kbd> All</button>
  <div class="sep"></div>
  <button onclick="resetZoom()"><kbd>0</kbd> Reset</button>
  <button onclick="toggleHelp()"><kbd>?</kbd> Shortcuts</button>
  <button id="lasso-btn" onclick="toggleLasso()"><kbd>S</kbd> Lasso</button>
  <span class="info">Click point \u2192 Rijksmuseum \u00b7 Double-click cluster \u2192 detail panel</span>
</div>

<div id="main">
  <div id="plot"></div>
  <div id="sidebar">
    <div id="sidebar-header">
      <input type="text" id="filter-input" placeholder="Filter..." style="width:80px;padding:2px 6px;border:1px solid #ccc;border-radius:3px;font-size:10px;">
      <button id="sort-btn" onclick="toggleSort()">Size \u2193</button>
      <button onclick="showAllClusters()">Show all</button>
    </div>
    <div id="cluster-list"></div>
  </div>
</div>
<div id="zoom-toast"></div>

<div id="detail-panel">
  <span class="close" onclick="closePanel()">&times;</span>
  <div id="detail-content"></div>
</div>

<div id="help-overlay" onclick="toggleHelp()">
  <div id="help-box" onclick="event.stopPropagation()">
    <h3>Keyboard Shortcuts</h3>
    <table>
      <tr><td class="section" colspan="2">Zoom</td></tr>
      <tr><td>+  =</td><td>Zoom in</td></tr>
      <tr><td>-</td><td>Zoom out</td></tr>
      <tr><td>0</td><td>Reset zoom (fit all)</td></tr>
      <tr><td>1</td><td>Zoom to 2&times;</td></tr>
      <tr><td>2</td><td>Zoom to 4&times;</td></tr>
      <tr><td>3</td><td>Zoom to 8&times;</td></tr>
      <tr><td>4</td><td>Zoom to 16&times;</td></tr>
      <tr><td class="section" colspan="2">Pan</td></tr>
      <tr><td>&larr; &rarr; &uarr; &darr;</td><td>Pan in direction</td></tr>
      <tr><td>Shift + arrow</td><td>Pan further</td></tr>
      <tr><td class="section" colspan="2">Display</td></tr>
      <tr><td>L</td><td>Toggle cluster labels</td></tr>
      <tr><td>N</td><td>Toggle noise points</td></tr>
      <tr><td>A</td><td>Show all clusters</td></tr>
      <tr><td>S</td><td>Toggle lasso select / zoom</td></tr>
      <tr><td>?  H</td><td>Toggle this help</td></tr>
      <tr><td>Esc</td><td>Close panels / help / lasso</td></tr>
    </table>
  </div>
</div>

<script>
const traces = {traces_json};
const layout = {layout_json};
const clusterDetail = {cluster_detail_json};
const legendItems = {legend_items_json};
const dataExtent = {data_extent_json};

const config = {{
  responsive: true,
  scrollZoom: true,
  modeBarButtonsToRemove: ['select2d'],
  displaylogo: false,
}};

const NOISE_TRACE_IDX = {noise_trace_idx};
let labelsVisible = true;
let noiseVisible = false;
const savedAnnotations = JSON.parse(JSON.stringify(layout.annotations || []));
const traceVisible = traces.map((t, i) => t.visible !== 'legendonly');

Plotly.newPlot('plot', traces, layout, config);

// ── Zoom-dependent hover ──────────────────────────
// At overview zoom, hover is disabled to reduce clutter. Once the user
// zooms past 4×, individual artwork hover becomes useful.
// Uses layout.hovermode (false / 'closest') — authoritative and avoids
// race conditions with per-trace hovertemplate restyles.
let hoverEnabled = false;

function getEffectiveZoom() {{
  const r = getRange();
  const dataW = dataExtent.xMax - dataExtent.xMin;
  const viewW = r.xr[1] - r.xr[0];
  return viewW > 0 ? dataW / viewW : 1;
}}

function setHoverState(enabled) {{
  if (enabled === hoverEnabled) return;
  hoverEnabled = enabled;
  Plotly.relayout('plot', {{ hovermode: enabled ? 'closest' : false }});
}}

// Start with hover disabled
Plotly.relayout('plot', {{ hovermode: false }});

// Re-evaluate on zoom/pan
document.getElementById('plot').on('plotly_relayout', function(ev) {{
  if (!ev) return;
  // Only react to axis range changes (zoom/pan), not our own hovermode changes
  var hasAxisChange = ev['xaxis.range[0]'] !== undefined || ev['xaxis.range'] !== undefined || ev['xaxis.autorange'] !== undefined;
  if (!hasAxisChange) return;
  var zoom = getEffectiveZoom();
  setHoverState(zoom > 4);
}});

// ── Custom sidebar legend ─────────────────────────
(function buildSidebar() {{
  const list = document.getElementById('cluster-list');
  let html = '';
  for (const item of legendItems) {{
    if (item.type === 'group') {{
      html += '<div class="legend-group-title">' + item.label + '</div>';
    }} else {{
      const dimmed = !traceVisible[item.traceIdx] ? ' dimmed' : '';
      html += '<div class="legend-item' + dimmed + '" data-trace="' + item.traceIdx + '" data-cid="' + item.cid + '">'
        + '<span class="legend-swatch" style="background:' + item.color + '"></span>'
        + '<span class="legend-label" title="' + esc(item.label) + '">' + esc(item.label) + '</span>'
        + '<span class="legend-count">' + item.size.toLocaleString() + '</span>'
        + '</div>';
    }}
  }}
  list.innerHTML = html;

  // Click = toggle visibility, double-click = show detail panel
  list.addEventListener('click', function(e) {{
    const el = e.target.closest('.legend-item');
    if (!el) return;
    const idx = parseInt(el.dataset.trace);
    toggleTrace(idx);
    el.classList.toggle('dimmed', !traceVisible[idx]);
  }});

  list.addEventListener('dblclick', function(e) {{
    const el = e.target.closest('.legend-item');
    if (!el) return;
    e.preventDefault();
    const cid = parseInt(el.dataset.cid);
    showClusterDetail(cid);
  }});

  list.addEventListener('contextmenu', function(e) {{
    const el = e.target.closest('.legend-item');
    if (!el) return;
    e.preventDefault();
    const cid = parseInt(el.dataset.cid);
    showClusterDetail(cid);
  }});
}})();

function toggleTrace(idx) {{
  traceVisible[idx] = !traceVisible[idx];
  Plotly.restyle('plot', {{ visible: traceVisible[idx] ? true : 'legendonly' }}, [idx]);
}}

function syncSidebarDimming() {{
  document.querySelectorAll('.legend-item').forEach(el => {{
    const idx = parseInt(el.dataset.trace);
    el.classList.toggle('dimmed', !traceVisible[idx]);
  }});
}}

// ── Sort toggle (Size ↓ / A→Z) ───────────────────
let sortBySize = true;
function toggleSort() {{
  sortBySize = !sortBySize;
  document.getElementById('sort-btn').textContent = sortBySize ? 'Size \u2193' : 'A\u2192Z';
  const list = document.getElementById('cluster-list');
  const items = Array.from(list.querySelectorAll('.legend-item'));
  items.sort(function(a, b) {{
    if (sortBySize) {{
      // Sort by count descending (parse from the .legend-count text)
      const ca = parseInt(a.querySelector('.legend-count').textContent.replace(/,/g, ''));
      const cb = parseInt(b.querySelector('.legend-count').textContent.replace(/,/g, ''));
      return cb - ca;
    }} else {{
      // A→Z by label, stripping the leading "N: " prefix
      const la = a.querySelector('.legend-label').textContent.replace(/^\\d+:\\s*/, '').toLowerCase();
      const lb = b.querySelector('.legend-label').textContent.replace(/^\\d+:\\s*/, '').toLowerCase();
      return la.localeCompare(lb);
    }}
  }});
  items.forEach(function(el) {{ list.appendChild(el); }});
}}

// ── Sidebar filter ────────────────────────────────
(function setupFilter() {{
  const input = document.getElementById('filter-input');
  input.addEventListener('input', function() {{
    const q = this.value.toLowerCase();
    document.querySelectorAll('.legend-item').forEach(function(el) {{
      const label = el.querySelector('.legend-label').textContent.toLowerCase();
      el.style.display = label.includes(q) ? '' : 'none';
    }});
  }});
  input.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') {{
      this.value = '';
      this.dispatchEvent(new Event('input'));
      this.blur();
    }}
  }});
}})();

// ── Click to open artwork ─────────────────────────
document.getElementById('plot').on('plotly_click', function(data) {{
  if (data.points.length > 0) {{
    const objNum = data.points[0].customdata;
    if (objNum) {{
      window.open('https://www.rijksmuseum.nl/nl/collectie/' + objNum, '_blank');
    }}
  }}
}});

// ── Display controls ──────────────────────────────
function showAllClusters() {{
  for (let i = 0; i < traces.length; i++) {{
    if (i === NOISE_TRACE_IDX) {{
      traceVisible[i] = noiseVisible;
      continue;
    }}
    traceVisible[i] = true;
  }}
  const vis = traceVisible.map(v => v ? true : 'legendonly');
  Plotly.restyle('plot', {{ visible: vis }});
  syncSidebarDimming();
  showToast('All clusters shown');
}}

function toggleLabels() {{
  labelsVisible = !labelsVisible;
  Plotly.relayout('plot', {{ 'annotations': labelsVisible ? JSON.parse(JSON.stringify(savedAnnotations)) : [] }});
  showToast(labelsVisible ? 'Labels shown' : 'Labels hidden');
}}

function toggleNoise() {{
  if (NOISE_TRACE_IDX < 0) return;
  noiseVisible = !noiseVisible;
  traceVisible[NOISE_TRACE_IDX] = noiseVisible;
  Plotly.restyle('plot', {{ visible: noiseVisible ? true : 'legendonly' }}, [NOISE_TRACE_IDX]);
  showToast(noiseVisible ? 'Noise shown' : 'Noise hidden');
}}

function resetZoom() {{
  Plotly.relayout('plot', {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
  showToast('Fit all');
}}

// ── Zoom/pan helpers ──────────────────────────────
let toastTimeout;
function showToast(msg) {{
  const el = document.getElementById('zoom-toast');
  el.textContent = msg;
  el.style.display = 'block';
  el.style.opacity = '1';
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(function() {{
    el.style.opacity = '0';
    setTimeout(function() {{ el.style.display = 'none'; }}, 300);
  }}, 1200);
}}

function getRange() {{
  const pl = document.getElementById('plot')._fullLayout;
  return {{
    xr: [Number(pl.xaxis.range[0]), Number(pl.xaxis.range[1])],
    yr: [Number(pl.yaxis.range[0]), Number(pl.yaxis.range[1])]
  }};
}}

function zoomBy(factor) {{
  const r = getRange();
  const xc = (r.xr[0] + r.xr[1]) / 2;
  const yc = (r.yr[0] + r.yr[1]) / 2;
  const xh = (r.xr[1] - r.xr[0]) / 2 * factor;
  const yh = (r.yr[1] - r.yr[0]) / 2 * factor;
  Plotly.relayout('plot', {{
    'xaxis.range': [xc - xh, xc + xh],
    'yaxis.range': [yc - yh, yc + yh]
  }});
  showToast(factor < 1 ? 'Zoom in' : 'Zoom out');
}}

function zoomToLevel(level) {{
  const d = dataExtent;
  const xc = (d.xMin + d.xMax) / 2;
  const yc = (d.yMin + d.yMax) / 2;
  const xh = (d.xMax - d.xMin) / 2 / level * 1.05;
  const yh = (d.yMax - d.yMin) / 2 / level * 1.05;
  Plotly.relayout('plot', {{
    'xaxis.range': [xc - xh, xc + xh],
    'yaxis.range': [yc - yh, yc + yh]
  }});
  showToast(level === 1 ? 'Fit all' : level + '\u00d7 zoom');
}}

function panBy(dx, dy) {{
  const r = getRange();
  const xs = r.xr[1] - r.xr[0];
  const ys = r.yr[1] - r.yr[0];
  Plotly.relayout('plot', {{
    'xaxis.range': [r.xr[0] + xs * dx, r.xr[1] + xs * dx],
    'yaxis.range': [r.yr[0] + ys * dy, r.yr[1] + ys * dy]
  }});
}}

function closePanel() {{
  document.getElementById('detail-panel').style.display = 'none';
}}

function toggleHelp() {{
  document.getElementById('help-overlay').classList.toggle('visible');
}}

// ── HTML escape helper (prevent XSS from vocab labels) ──
function esc(s) {{
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}}

// ── Detail panel ──────────────────────────────────
function showClusterDetail(cid) {{
  const d = clusterDetail[cid];
  if (!d) return;

  function renderBars(items, color) {{
    if (!items || !items.length) return '<em>none</em>';
    const maxVal = items[0][1];
    var html = '';
    for (var i = 0; i < items.length; i++) {{
      var w = Math.max(2, (items[i][1] / maxVal) * 150);
      html += '<div class="bar"><div class="bar-fill" style="width:' + w + 'px;background:' + color + '"></div>'
        + '<span class="bar-text">' + esc(items[i][0]) + ' (' + items[i][1] + ')</span></div>';
    }}
    return html;
  }}

  document.getElementById('detail-content').innerHTML =
    '<h3>Cluster ' + cid + ': ' + esc(d.label) + '</h3>'
    + '<div class="stat"><b>Size:</b> ' + d.size.toLocaleString() + ' artworks</div>'
    + '<div class="bar-section">Top subjects</div>'
    + renderBars(d.top_subjects, '#E63946')
    + '<div class="bar-section">Object types</div>'
    + renderBars(d.top_types, '#457B9D')
    + '<div class="bar-section">Top creators</div>'
    + renderBars(d.top_creators, '#2A9D8F')
    + '<div class="bar-section">Materials</div>'
    + renderBars(d.top_materials, '#E76F51')
    + '<div class="bar-section">Techniques</div>'
    + renderBars(d.top_techniques, '#264653');

  document.getElementById('detail-panel').style.display = 'block';
}}

// ── Lasso selection ───────────────────────────────
let lassoActive = false;
function toggleLasso() {{
  lassoActive = !lassoActive;
  Plotly.relayout('plot', {{ dragmode: lassoActive ? 'lasso' : 'zoom' }});
  document.getElementById('lasso-btn').style.background = lassoActive ? '#e0e7ff' : '';
  showToast(lassoActive ? 'Lasso mode \u2014 draw to select' : 'Zoom mode');
}}

document.getElementById('plot').on('plotly_selected', function(eventData) {{
  if (!eventData || !eventData.points || eventData.points.length === 0) return;
  showSelectionDetail(eventData.points);
}});

function showSelectionDetail(points) {{
  // Parse metadata from hover HTML text
  var typeCounts = {{}};
  var creatorCounts = {{}};
  var subjectCounts = {{}};
  var materialCounts = {{}};
  var techniqueCounts = {{}};

  for (var i = 0; i < points.length; i++) {{
    var html = points[i].text || '';
    // Extract fields via regex on the hover HTML
    var cm = html.match(/Creator:\\s*([^<]+)/);
    var tm = html.match(/Type:\\s*([^<]+)/);
    var sm = html.match(/Subjects:\\s*([^<]+)/);
    var mm = html.match(/Material:\\s*([^<]+)/);
    var tcm = html.match(/Technique:\\s*([^<]+)/);

    if (tm) tm[1].split(',').forEach(function(v) {{ v = v.trim(); if (v) typeCounts[v] = (typeCounts[v] || 0) + 1; }});
    if (cm) cm[1].split(',').forEach(function(v) {{ v = v.trim(); if (v) creatorCounts[v] = (creatorCounts[v] || 0) + 1; }});
    if (sm) sm[1].split(',').forEach(function(v) {{ v = v.trim(); if (v) subjectCounts[v] = (subjectCounts[v] || 0) + 1; }});
    if (mm) mm[1].split(',').forEach(function(v) {{ v = v.trim(); if (v) materialCounts[v] = (materialCounts[v] || 0) + 1; }});
    if (tcm) tcm[1].split(',').forEach(function(v) {{ v = v.trim(); if (v) techniqueCounts[v] = (techniqueCounts[v] || 0) + 1; }});
  }}

  function toSorted(obj, n) {{
    return Object.entries(obj).sort(function(a, b) {{ return b[1] - a[1]; }}).slice(0, n);
  }}

  function renderBars(items, color) {{
    if (!items || !items.length) return '<em>none</em>';
    var maxVal = items[0][1];
    var out = '';
    for (var i = 0; i < items.length; i++) {{
      var w = Math.max(2, (items[i][1] / maxVal) * 150);
      out += '<div class="bar"><div class="bar-fill" style="width:' + w + 'px;background:' + color + '"></div>'
        + '<span class="bar-text">' + esc(items[i][0]) + ' (' + items[i][1] + ')</span></div>';
    }}
    return out;
  }}

  document.getElementById('detail-content').innerHTML =
    '<h3>Selection (' + points.length.toLocaleString() + ' points)</h3>'
    + '<div class="bar-section">Top subjects</div>'
    + renderBars(toSorted(subjectCounts, 8), '#E63946')
    + '<div class="bar-section">Object types</div>'
    + renderBars(toSorted(typeCounts, 5), '#457B9D')
    + '<div class="bar-section">Top creators</div>'
    + renderBars(toSorted(creatorCounts, 5), '#2A9D8F')
    + '<div class="bar-section">Materials</div>'
    + renderBars(toSorted(materialCounts, 5), '#E76F51')
    + '<div class="bar-section">Techniques</div>'
    + renderBars(toSorted(techniqueCounts, 5), '#264653');

  document.getElementById('detail-panel').style.display = 'block';
}}

// ── Keyboard (capture phase \u2014 fires before Plotly) ─
window.addEventListener('keydown', function(e) {{
  var tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (e.metaKey || e.ctrlKey) return;

  var shift = e.shiftKey;
  var step = shift ? 0.3 : 0.1;
  var handled = true;

  switch (e.key) {{
    case '+': case '=': zoomBy(0.6); break;
    case '-': case '_': zoomBy(1.5); break;
    case '0': zoomToLevel(1); break;
    case '1': zoomToLevel(2); break;
    case '2': zoomToLevel(4); break;
    case '3': zoomToLevel(8); break;
    case '4': zoomToLevel(16); break;
    case 'ArrowLeft':  panBy(-step, 0); break;
    case 'ArrowRight': panBy(step, 0); break;
    case 'ArrowUp':    panBy(0, step); break;
    case 'ArrowDown':  panBy(0, -step); break;
    case 'l': case 'L': toggleLabels(); break;
    case 'n': case 'N': toggleNoise(); break;
    case 'a': case 'A': showAllClusters(); break;
    case 's': case 'S': toggleLasso(); break;
    case '?': case 'h': case 'H': toggleHelp(); break;
    case 'Escape':
      document.getElementById('help-overlay').classList.remove('visible');
      closePanel();
      // Exit lasso mode on Escape
      if (lassoActive) toggleLasso();
      break;
    default:
      handled = false;
  }}

  if (handled) {{
    e.preventDefault();
    e.stopPropagation();
  }}
}}, true);
</script>
</body>
</html>"""
