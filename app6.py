"""
=============================================================================
Pharmaceutical Facility Layout Evaluation, Diagnosis & Optimization System
=============================================================================
Capstone Project: Systematic Layout Planning and Development of an Intelligent
Layout Evaluation and Improvement System for a Pharmaceutical Manufacturing Facility

Tech Stack: Python | Streamlit | Pandas | NumPy | Matplotlib | NetworkX | SciPy

Modifications v3  (on top of v2):
  FIX 1 — OVERLAP_K reduced from 800 → 50 so flow cost dominates over
           collision avoidance for sparsely-connected departments.
  FIX 2 — Gravity / betweenness term added to objective: each department
           is pulled toward the flow-weighted centroid of its partners,
           rewarding in-between positioning rather than isolated corners.
  FIX 3 — REL-chart spring constraints added: A-rated pairs accumulate a
           quadratic penalty when their bbox distance exceeds
           ADJACENCY_THRESHOLD, making the hard SLP relationship table
           directly inform the optimizer (not just the flow weights).
  Other — OVERLAP_K, GRAVITY_K, and REL_K are all exposed as sidebar
           sliders so users can tune the balance interactively.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmLayout Intelligence System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}
h1, h2, h3 {
    font-family: 'Share Tech Mono', monospace;
    color: #58a6ff;
    letter-spacing: 0.04em;
}
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 8px 0;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    color: #58a6ff;
    font-weight: bold;
}
.metric-value.good { color: #3fb950; }
.metric-value.warn { color: #d29922; }
.metric-value.bad  { color: #f85149; }

.rec-card {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
}
.rec-card.violation { border-left: 4px solid #f85149; }
.rec-card.warning   { border-left: 4px solid #d29922; }
.rec-card.info      { border-left: 4px solid #58a6ff; }

.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
.stDataFrame { font-size: 0.85rem; }

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    box-shadow: 0 0 16px rgba(88,166,255,0.3);
    transform: translateY(-1px);
}
.cluster-box {
    background: #1c2128;
    border: 1px solid #58a6ff44;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.fix-box {
    background: #1c2128;
    border: 1px solid #3fb95044;
    border-left: 4px solid #3fb950;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #c9d1d9;
}
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEPARTMENT_TYPES = [
    "Raw Material Storage",
    "Process Reactors",
    "Solid Processing",
    "Packaging",
    "Dispatch",
    "Utilities",
    "Effluent Treatment Plant",
    "Hazardous Waste",
]

HAZARD_LEVELS = ["Low", "Medium", "High"]
REL_RATINGS   = ["A", "E", "I", "O", "U", "X"]

PRODUCTION_SEQUENCE = [
    "Raw Material Storage",
    "Process Reactors",
    "Solid Processing",
    "Packaging",
    "Dispatch",
]

SLP_REASON_CODES = {
    1: "Material Flow",
    2: "Personnel Movement",
    3: "Contamination Control",
    4: "Supervision / Communication",
    5: "Shared Equipment / Utilities",
    6: "Safety / Hazard Separation",
}

FLOW_TO_RATING = [
    (0.75, "A"),
    (0.50, "E"),
    (0.25, "I"),
    (0.10, "O"),
    (0.00, "U"),
]

SLP_LINE_SPEC = {
    "A": (4, "#f85149", "solid"),
    "E": (3, "#d29922", "solid"),
    "I": (2, "#3fb950", "solid"),
    "O": (1, "#58a6ff", "solid"),
    "U": (0, "gray",    "dotted"),
    "X": (0, "#ff0000", "solid"),
}

ADJACENCY_THRESHOLD = 10.0
GRID_MIN, GRID_MAX  = 0.0, 100.0

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER CONSTANTS  (v3: tunable via sidebar)
# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — was 800; reduced so flow cost dominates over overlap avoidance
DEFAULT_OVERLAP_K = 50.0

# Fix 2 — weight for the gravity/betweenness pull toward flow-partner centroid
DEFAULT_GRAVITY_K = 0.5

# Fix 3 — spring stiffness for A-rated REL pairs that violate adjacency
DEFAULT_REL_K = 200.0

OVERLAP_GAP = 1.0  # minimum clearance (m) between bounding boxes

# ─────────────────────────────────────────────────────────────────────────────
# DEPT COLORS
# ─────────────────────────────────────────────────────────────────────────────
DEPT_COLORS = {
    "Raw Material Storage":    "#1a4a7a",
    "Process Reactors":        "#7a1a1a",
    "Solid Processing":        "#7a4a1a",
    "Packaging":               "#1a5a2a",
    "Dispatch":                "#1a3a5a",
    "Utilities":               "#3a1a7a",
    "Effluent Treatment Plant":"#1a5a5a",
    "Hazardous Waste":         "#5a1a3a",
}
DEPT_BORDER = {
    "Raw Material Storage":    "#58a6ff",
    "Process Reactors":        "#f85149",
    "Solid Processing":        "#d29922",
    "Packaging":               "#3fb950",
    "Dispatch":                "#a5d6ff",
    "Utilities":               "#bc8cff",
    "Effluent Treatment Plant":"#79c0ff",
    "Hazardous Waste":         "#ff7b72",
}

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SAMPLE DATA
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DEPARTMENTS = pd.DataFrame({
    "Name":   DEPARTMENT_TYPES,
    "X":      [10.0, 50.0, 60.0, 75.0, 90.0, 20.0, 30.0, 45.0],
    "Y":      [80.0, 70.0, 55.0, 40.0, 25.0, 50.0, 20.0, 10.0],
    "Width":  [12.0, 10.0,  9.0,  8.0,  9.0,  7.0,  8.0,  7.0],
    "Height": [10.0,  9.0,  8.0,  7.0,  8.0,  6.0,  7.0,  6.0],
    "Area_m2":[120.,  90.,  72.,  56.,  72.,  42.,  56.,  42.],
    "Hazard": ["Low","High","Medium","Low","Low","Low","Medium","High"],
    "Type":   DEPARTMENT_TYPES,
})

DEFAULT_FLOW = pd.DataFrame({
    "From":   ["Raw Material Storage","Process Reactors","Solid Processing",
               "Packaging","Process Reactors","Solid Processing",
               "Raw Material Storage","Process Reactors","Solid Processing"],
    "To":     ["Process Reactors","Solid Processing","Packaging",
               "Dispatch","Effluent Treatment Plant","Effluent Treatment Plant",
               "Utilities","Hazardous Waste","Hazardous Waste"],
    "Weight": [90, 85, 80, 75, 40, 35, 20, 30, 25],
    "Reason Code": [1, 1, 1, 1, 3, 3, 5, 6, 6],
})

DEFAULT_REL = pd.DataFrame({
    "Dept_A": ["Raw Material Storage","Process Reactors","Solid Processing",
               "Process Reactors","Solid Processing","Process Reactors",
               "Utilities","Utilities"],
    "Dept_B": ["Process Reactors","Solid Processing","Packaging",
               "Effluent Treatment Plant","Hazardous Waste","Utilities",
               "Solid Processing","Process Reactors"],
    "Rating": ["A","A","E","E","E","A","A","A"],
    "Reason Code": [1, 1, 1, 3, 6, 5, 5, 5],
    "Override": [False]*8,
})

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_dept_row(dept_df, name):
    rows = dept_df[dept_df["Name"] == name]
    return rows.iloc[0] if not rows.empty else None


def get_coords(dept_df, name):
    r = get_dept_row(dept_df, name)
    return (float(r["X"]), float(r["Y"])) if r is not None else None


def get_dims(dept_df, name):
    r = get_dept_row(dept_df, name)
    if r is None:
        return (4.0, 4.0)
    return (float(r.get("Width", 8.0)) / 2.0, float(r.get("Height", 8.0)) / 2.0)


def bbox(dept_df, name):
    c = get_coords(dept_df, name)
    if c is None:
        return None
    hw, hh = get_dims(dept_df, name)
    return (c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh)


def bbox_distance(dept_df, name_a, name_b):
    ba = bbox(dept_df, name_a)
    bb = bbox(dept_df, name_b)
    if ba is None or bb is None:
        return None
    dx = max(0.0, max(ba[0], bb[0]) - min(ba[2], bb[2]))
    dy = max(0.0, max(ba[1], bb[1]) - min(ba[3], bb[3]))
    return dx + dy


def score_color(score):
    if score >= 75: return "good"
    if score >= 50: return "warn"
    return "bad"


def nearest_on_rect(rect, px, py):
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = px - cx, py - cy
    if dx == 0 and dy == 0:
        return cx, y0
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    if hw == 0 or hh == 0:
        return np.clip(px, x0, x1), np.clip(py, y0, y1)
    t = min(hw / (abs(dx) + 1e-9), hh / (abs(dy) + 1e-9))
    return cx + dx * t, cy + dy * t


def closest_border_points(ba, bb):
    acx, acy = (ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2
    bcx, bcy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    pa = nearest_on_rect(ba, bcx, bcy)
    pb = nearest_on_rect(bb, acx, acy)
    return pa, pb

# ─────────────────────────────────────────────────────────────────────────────
# AUTO REL CHART SYNTHESIS
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_rel_chart(flow_df, dept_df, manual_overrides_df=None):
    pair_weight = {}
    for _, row in flow_df.iterrows():
        frm, to, w = row["From"], row["To"], row["Weight"]
        pair = tuple(sorted([frm, to]))
        pair_weight[pair] = pair_weight.get(pair, 0) + w

    max_w = max(pair_weight.values()) if pair_weight else 1.0

    records = {}
    for (a, b), w in pair_weight.items():
        norm = w / max_w
        rating = "U"
        for threshold, r in FLOW_TO_RATING:
            if norm >= threshold:
                rating = r
                break
        reason = 1
        if "Reason Code" in flow_df.columns:
            mask = (
                ((flow_df["From"] == a) & (flow_df["To"] == b)) |
                ((flow_df["From"] == b) & (flow_df["To"] == a))
            )
            if mask.any():
                reason = int(flow_df[mask]["Reason Code"].mode().iloc[0])
        records[(a, b)] = {"Dept_A": a, "Dept_B": b,
                            "Rating": rating, "Reason Code": reason,
                            "Override": False, "Source": "Flow-derived"}

    waste_depts = ["Effluent Treatment Plant", "Hazardous Waste"]
    clean_depts = ["Packaging", "Dispatch", "Raw Material Storage"]
    for w in waste_depts:
        for c in clean_depts:
            pair = tuple(sorted([w, c]))
            if pair not in records:
                records[pair] = {"Dept_A": pair[0], "Dept_B": pair[1],
                                 "Rating": "X", "Reason Code": 6,
                                 "Override": False, "Source": "Rule: Hazard Separation"}

    util_pairs = [("Utilities", "Process Reactors"),
                  ("Utilities", "Solid Processing")]
    for (a, b) in util_pairs:
        pair = tuple(sorted([a, b]))
        if pair not in records:
            records[pair] = {"Dept_A": pair[0], "Dept_B": pair[1],
                             "Rating": "A", "Reason Code": 5,
                             "Override": False, "Source": "Rule: Shared Utilities"}

    if manual_overrides_df is not None and not manual_overrides_df.empty:
        for _, row in manual_overrides_df.iterrows():
            if row.get("Override", False):
                pair = tuple(sorted([row["Dept_A"], row["Dept_B"]]))
                records[pair] = {
                    "Dept_A": pair[0], "Dept_B": pair[1],
                    "Rating": row["Rating"],
                    "Reason Code": int(row.get("Reason Code", 1)),
                    "Override": True,
                    "Source": "Manual Override"
                }

    return pd.DataFrame(list(records.values())).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1: TOTAL WEIGHTED TRAVEL DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_travel_distance(dept_df, flow_df):
    rows = []
    for _, row in flow_df.iterrows():
        src, dst, w = row["From"], row["To"], row["Weight"]
        c1 = get_coords(dept_df, src)
        c2 = get_coords(dept_df, dst)
        if c1 is None or c2 is None:
            continue
        d = manhattan(c1[0], c1[1], c2[0], c2[1])
        rows.append({"From": src, "To": dst, "Weight": w,
                     "Distance": round(d, 2),
                     "Weighted Distance": round(w * d, 2)})
    detail = pd.DataFrame(rows)
    total  = detail["Weighted Distance"].sum() if not detail.empty else 0.0
    return total, detail

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2: ADJACENCY COMPLIANCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_adjacency_score(dept_df, rel_df):
    mandatory = rel_df[rel_df["Rating"].isin(["A", "E"])].copy()
    if mandatory.empty:
        return 100.0, [], pd.DataFrame()

    rows = []
    satisfied = 0
    violations = []

    for _, r in mandatory.iterrows():
        d = bbox_distance(dept_df, r["Dept_A"], r["Dept_B"])
        if d is None:
            continue
        d = round(d, 2)
        ok = d <= ADJACENCY_THRESHOLD
        if ok:
            satisfied += 1
        else:
            violations.append({"Pair": f"{r['Dept_A']} ↔ {r['Dept_B']}",
                                "Rating": r["Rating"], "Distance": d})
        rows.append({"Dept A": r["Dept_A"], "Dept B": r["Dept_B"],
                     "Rating": r["Rating"], "Box Distance": d,
                     "Status": "✅ Satisfied" if ok else "❌ Violated"})

    total = len(rows)
    score = (satisfied / total * 100) if total > 0 else 100.0
    return round(score, 2), violations, pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3: ZONING SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_zoning_score(dept_df):
    violations = []

    def bd(a, b):
        return bbox_distance(dept_df, a, b)

    high_haz = dept_df[dept_df["Hazard"] == "High"]["Name"].tolist()
    low_haz  = dept_df[dept_df["Hazard"] == "Low"]["Name"].tolist()
    for h in high_haz:
        for l in low_haz:
            d = bd(h, l)
            if d is not None and d < 10:
                violations.append(
                    f"HAZARD: '{h}' (High) bbox-distance {d:.1f} m from '{l}' (Low) — min 10 m required.")

    d = bd("Utilities", "Dispatch")
    if d is not None and d < 10:
        violations.append(f"ZONING: Utilities ({d:.1f} m) too close to Dispatch.")

    for proc in ["Process Reactors", "Solid Processing"]:
        d = bd("Dispatch", proc)
        if d is not None and d < 15:
            violations.append(
                f"ZONING: Dispatch ({d:.1f} m) too close to '{proc}' — min 15 m required.")

    waste_depts = ["Effluent Treatment Plant", "Hazardous Waste"]
    clean_depts = ["Packaging", "Dispatch", "Raw Material Storage"]
    for w in waste_depts:
        for c in clean_depts:
            d = bd(w, c)
            if d is not None and d < 12:
                violations.append(
                    f"ZONING: '{w}' ({d:.1f} m) too close to '{c}' — min 12 m required.")

    penalty = len(violations) * 10
    score   = max(0.0, 100.0 - penalty)
    return round(score, 2), violations

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 4: FLOW DIRECTIONALITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_directionality_score(dept_df):
    issues = []
    coords = []
    for name in PRODUCTION_SEQUENCE:
        c = get_coords(dept_df, name)
        coords.append((name, c[0] if c else None, c[1] if c else None))

    valid = [(n, x, y) for n, x, y in coords if x is not None]
    if len(valid) < 2:
        return 100.0, []

    dx = valid[-1][1] - valid[0][1]
    dy = valid[-1][2] - valid[0][2]
    use_x = abs(dx) >= abs(dy)

    reversals = 0.0
    checks    = 0

    for i in range(len(valid) - 1):
        n1, x1, y1 = valid[i]
        n2, x2, y2 = valid[i + 1]
        checks += 1
        val1 = x1 if use_x else y1
        val2 = x2 if use_x else y2
        expected_dir = np.sign(dx if use_x else dy)
        actual_dir   = np.sign(val2 - val1)
        if actual_dir != 0 and actual_dir != expected_dir:
            reversals += 1
            issues.append(
                f"DIRECTION REVERSED: '{n1}' → '{n2}' moves against expected "
                f"flow axis ({'X' if use_x else 'Y'}).")

    uc = get_coords(dept_df, "Utilities")
    if uc:
        for target in ["Process Reactors", "Solid Processing"]:
            tc = get_coords(dept_df, target)
            if tc:
                d = manhattan(uc[0], uc[1], tc[0], tc[1])
                if d > 25:
                    issues.append(
                        f"UTILITIES: 'Utilities' is {d:.1f} m from '{target}' — should be ≤ 25 m.")
                    reversals += 0.5

    for ws in ["Process Reactors", "Solid Processing"]:
        wsc = get_coords(dept_df, ws)
        for wk in ["Effluent Treatment Plant", "Hazardous Waste"]:
            wkc = get_coords(dept_df, wk)
            if wsc and wkc:
                d = manhattan(wsc[0], wsc[1], wkc[0], wkc[1])
                if d > 30:
                    issues.append(
                        f"WASTE ROUTING: '{ws}' → '{wk}' distance is {d:.1f} m — should be ≤ 30 m.")
                    reversals += 0.25

    score = max(0.0, 100.0 - (reversals / max(checks, 1)) * 100.0)
    return round(score, 2), issues

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 5: OVERALL PERFORMANCE INDEX
# ─────────────────────────────────────────────────────────────────────────────

def compute_overall_index(travel_total, adj_score, zone_score, dir_score,
                           baseline_travel=None,
                           w_travel=25.0, w_adj=25.0, w_zone=25.0, w_dir=25.0):
    baseline     = baseline_travel if baseline_travel else 10000.0
    travel_score = max(0.0, 100.0 * (1.0 - travel_total / baseline))
    total_w      = w_travel + w_adj + w_zone + w_dir
    if total_w == 0:
        total_w = 100.0
    opi = (w_travel * travel_score + w_adj * adj_score +
           w_zone * zone_score + w_dir * dir_score) / total_w
    return round(opi, 2), round(travel_score, 2)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH-BASED CLUSTERING (NetworkX)
# ─────────────────────────────────────────────────────────────────────────────

def build_flow_graph(dept_df, flow_df):
    G = nx.DiGraph()
    for _, row in dept_df.iterrows():
        G.add_node(row["Name"], x=row["X"], y=row["Y"],
                   hazard=row["Hazard"], dept_type=row["Type"])
    for _, row in flow_df.iterrows():
        if row["From"] in G.nodes and row["To"] in G.nodes:
            G.add_edge(row["From"], row["To"], weight=row["Weight"])
    return G


def suggest_clusters(G, flow_df, dept_df, top_n=3):
    pair_flows = {}
    for _, row in flow_df.iterrows():
        pair = tuple(sorted([row["From"], row["To"]]))
        pair_flows[pair] = pair_flows.get(pair, 0) + row["Weight"]

    scored = []
    for (a, b), w in pair_flows.items():
        ca = get_coords(dept_df, a)
        cb = get_coords(dept_df, b)
        if ca and cb:
            d = manhattan(ca[0], ca[1], cb[0], cb[1])
            scored.append({"Dept A": a, "Dept B": b,
                            "Combined Flow": w,
                            "Current Distance": round(d, 2),
                            "Urgency Score": round(w * d, 2)})

    scored.sort(key=lambda x: x["Urgency Score"], reverse=True)
    centrality = nx.degree_centrality(G)
    return scored[:top_n], centrality


def plot_flow_graph(G, dept_df, ax):
    hazard_color = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}
    pos = {r["Name"]: (r["X"], r["Y"])
           for _, r in dept_df.iterrows() if r["Name"] in G.nodes}
    node_colors = [hazard_color.get(G.nodes[n].get("hazard", "Low"), "#58a6ff")
                   for n in G.nodes]
    weights = [G[u][v]["weight"] for u, v in G.edges]
    max_w   = max(weights) if weights else 1
    widths  = [1.0 + 4.0 * w / max_w for w in weights]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=900, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=6, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#58a6ff",
                           alpha=0.7, arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=6, font_color="#8b949e", ax=ax)
    patches = [mpatches.Patch(color=c, label=f"{k} Hazard")
               for k, c in hazard_color.items()]
    ax.legend(handles=patches, loc="upper right",
              fontsize=7, facecolor="#161b22", labelcolor="white")
    ax.set_facecolor("#0d1117")
    ax.set_title("Facility Flow Graph", color="#58a6ff", fontsize=11, pad=10)
    ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT OPTIMIZATION  (v3 — three-fix refactor)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_layout(dept_df, flow_df, rel_df,
                    overlap_k=DEFAULT_OVERLAP_K,
                    gravity_k=DEFAULT_GRAVITY_K,
                    rel_k=DEFAULT_REL_K):
    """
    Minimize a composite objective:

      f = FLOW_COST + OVERLAP_PENALTY + GRAVITY_TERM + REL_SPRING

    ── Fix 1 — OVERLAP_K reduced (default 50, was 800) ──────────────────────
      Collision avoidance no longer drowns out flow signals.
      OVERLAP_PENALTY = OVERLAP_K × Σ (overlap_x² + overlap_y²)
        activated only when bounding boxes actually collide.

    ── Fix 2 — GRAVITY_TERM (betweenness pull) ──────────────────────────────
      Each department is attracted toward the flow-weighted centroid of all
      its neighbours.  This rewards in-between positioning instead of letting
      sparsely-connected departments drift to empty corners.

        target_x_i = Σ(w_ij × x_j) / Σ(w_ij)   for all flow partners j
        GRAVITY_TERM += GRAVITY_K × Σ_i w_total_i × [(cx_i−target_x_i)²
                                                     + (cy_i−target_y_i)²]

    ── Fix 3 — REL_SPRING (A-rated REL pairs) ────────────────────────────────
      For every A-rated department pair in the REL chart whose bbox distance
      exceeds ADJACENCY_THRESHOLD, apply a quadratic spring:

        REL_SPRING += REL_K × (bbox_dist − threshold)²

      This makes the hard SLP relationship table a direct optimizer input,
      not just a post-hoc diagnostic.

    Parameters
    ----------
    dept_df   : DataFrame with Name, X, Y, Width, Height columns
    flow_df   : DataFrame with From, To, Weight columns
    rel_df    : DataFrame with Dept_A, Dept_B, Rating columns
    overlap_k : penalty multiplier for bounding-box collisions  (Fix 1)
    gravity_k : multiplier for gravity/betweenness pull          (Fix 2)
    rel_k     : spring stiffness for A-rated REL violations      (Fix 3)

    Returns
    -------
    opt_df, new_total, pct_improve, success_flag, objective_breakdown
    """
    names = dept_df["Name"].tolist()
    n     = len(names)
    idx   = {name: i for i, name in enumerate(names)}

    # Half-dimension arrays (for bbox collision math)
    hw_arr = np.array([
        float(dept_df.loc[dept_df["Name"] == nm, "Width"].iloc[0]) / 2.0
        for nm in names
    ])
    hh_arr = np.array([
        float(dept_df.loc[dept_df["Name"] == nm, "Height"].iloc[0]) / 2.0
        for nm in names
    ])

    x0 = dept_df[["X", "Y"]].values.flatten().astype(float)

    # Build flow pair list
    flow_pairs = []
    for _, row in flow_df.iterrows():
        fi, ti = idx.get(row["From"]), idx.get(row["To"])
        if fi is not None and ti is not None:
            flow_pairs.append((fi, ti, float(row["Weight"])))

    # ── Fix 2: precompute per-department total flow weight and partner list ──
    # partner_data[i] = list of (j, weight)
    partner_data = [[] for _ in range(n)]
    for fi, ti, w in flow_pairs:
        partner_data[fi].append((ti, w))
        partner_data[ti].append((fi, w))  # undirected gravity

    # ── Fix 3: precompute A-rated REL pairs as index tuples ─────────────────
    a_pairs = []
    if rel_df is not None and not rel_df.empty:
        for _, row in rel_df[rel_df["Rating"] == "A"].iterrows():
            ai = idx.get(row["Dept_A"])
            bi = idx.get(row["Dept_B"])
            if ai is not None and bi is not None:
                a_pairs.append((ai, bi))

    def objective(coords):
        xy = coords.reshape(n, 2)
        cost = 0.0

        # ── Flow cost: weighted centroid Manhattan distance ───────────────────
        for fi, ti, w in flow_pairs:
            cost += w * (abs(xy[fi, 0] - xy[ti, 0]) + abs(xy[fi, 1] - xy[ti, 1]))

        # ── Fix 1: Bounding-box overlap penalty (K = overlap_k, was 800) ─────
        for a in range(n):
            for b in range(a + 1, n):
                dx = abs(xy[a, 0] - xy[b, 0])
                dy = abs(xy[a, 1] - xy[b, 1])
                req_x = hw_arr[a] + hw_arr[b] + OVERLAP_GAP
                req_y = hh_arr[a] + hh_arr[b] + OVERLAP_GAP
                overlap_x = max(0.0, req_x - dx)
                overlap_y = max(0.0, req_y - dy)
                if overlap_x > 0 and overlap_y > 0:
                    cost += overlap_k * (overlap_x ** 2 + overlap_y ** 2)

        # ── Fix 2: Gravity / betweenness pull toward flow-partner centroid ───
        for i, partners in enumerate(partner_data):
            if not partners:
                continue
            total_w = sum(w for _, w in partners)
            if total_w < 1e-9:
                continue
            target_x = sum(xy[j, 0] * w for j, w in partners) / total_w
            target_y = sum(xy[j, 1] * w for j, w in partners) / total_w
            cost += gravity_k * total_w * (
                (xy[i, 0] - target_x) ** 2 + (xy[i, 1] - target_y) ** 2
            )

        # ── Fix 3: REL-chart spring for A-rated pairs beyond threshold ───────
        for ai, bi in a_pairs:
            dx = abs(xy[ai, 0] - xy[bi, 0])
            dy = abs(xy[ai, 1] - xy[bi, 1])
            # bbox gap (same formula as bbox_distance but inline for speed)
            req_x = hw_arr[ai] + hw_arr[bi]
            req_y = hh_arr[ai] + hh_arr[bi]
            gap_x = max(0.0, dx - req_x)
            gap_y = max(0.0, dy - req_y)
            bbox_dist = gap_x + gap_y
            if bbox_dist > ADJACENCY_THRESHOLD:
                excess = bbox_dist - ADJACENCY_THRESHOLD
                cost += rel_k * (excess ** 2)

        return cost

    bounds = [(GRID_MIN, GRID_MAX)] * (2 * n)

    result = minimize(
        objective, x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-11, "gtol": 1e-9}
    )

    opt_coords = result.x.reshape(n, 2)
    opt_df = dept_df.copy()
    opt_df["X"] = np.round(opt_coords[:, 0], 2)
    opt_df["Y"] = np.round(opt_coords[:, 1], 2)

    new_total,  _ = compute_travel_distance(opt_df, flow_df)
    orig_total, _ = compute_travel_distance(dept_df, flow_df)
    pct_improve   = ((orig_total - new_total) / orig_total * 100) if orig_total > 0 else 0.0

    # Breakdown for transparency panel
    breakdown = {
        "overlap_k_used": overlap_k,
        "gravity_k_used": gravity_k,
        "rel_k_used": rel_k,
        "a_pairs_count": len(a_pairs),
        "flow_pairs_count": len(flow_pairs),
    }

    return opt_df, round(new_total, 2), round(pct_improve, 2), result.success, breakdown

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(travel_detail, adj_violations, zone_violations,
                              dir_issues, cluster_suggestions, dept_df, flow_df):
    recs = []

    if not travel_detail.empty:
        total_travel = travel_detail["Weighted Distance"].sum()
        for _, row in travel_detail.nlargest(3, "Weighted Distance").iterrows():
            pct = row["Weighted Distance"] / total_travel * 100 if total_travel else 0
            msg = (f"'{row['From']}' → '{row['To']}' contributes {pct:.1f}% of total "
                   f"travel distance (WD={row['Weighted Distance']:.0f}, D={row['Distance']:.1f} m). "
                   f"Reduce separation.")
            recs.append(("violation" if pct > 20 else "warning", msg))

    for v in adj_violations:
        recs.append(("violation",
            f"ADJACENCY: '{v['Pair']}' rated '{v['Rating']}' — bbox distance {v['Distance']:.1f} m "
            f"exceeds threshold {ADJACENCY_THRESHOLD} m. Move blocks closer."))

    for v in zone_violations:
        recs.append(("violation", v))

    for issue in dir_issues:
        recs.append(("warning", issue))

    for c in cluster_suggestions:
        recs.append(("info",
            f"CLUSTER: '{c['Dept A']}' ↔ '{c['Dept B']}' — flow {c['Combined Flow']}, "
            f"distance {c['Current Distance']} m, urgency {c['Urgency Score']:.0f}. "
            f"Group spatially."))

    waste_pairs = [("Process Reactors",  "Effluent Treatment Plant"),
                   ("Solid Processing",  "Effluent Treatment Plant"),
                   ("Process Reactors",  "Hazardous Waste"),
                   ("Solid Processing",  "Hazardous Waste")]
    for src, dst in waste_pairs:
        cs, cd = get_coords(dept_df, src), get_coords(dept_df, dst)
        if cs and cd:
            d = manhattan(cs[0], cs[1], cd[0], cd[1])
            if d > 30:
                recs.append(("warning",
                    f"WASTE ROUTING: '{src}' → '{dst}' is {d:.1f} m — target ≤ 30 m."))

    if not recs:
        recs.append(("info", "No significant issues detected. Layout appears well-configured."))
    return recs

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK LAYOUT VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_block_layout(dept_df, rel_df, ax, title="Block Layout"):
    ax.set_facecolor("#0d1117")

    all_x, all_y = [], []
    for _, row in dept_df.iterrows():
        cx, cy = row["X"], row["Y"]
        hw = float(row.get("Width",  8)) / 2
        hh = float(row.get("Height", 8)) / 2
        all_x += [cx - hw, cx + hw]
        all_y += [cy - hh, cy + hh]

    pad = 8
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    drawn_pairs = set()
    for _, r in rel_df.iterrows():
        a, b, rating = r["Dept_A"], r["Dept_B"], r["Rating"]
        pair = tuple(sorted([a, b]))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)

        n_lines, color, lstyle = SLP_LINE_SPEC.get(rating, (0, "gray", "solid"))
        ba  = bbox(dept_df, a)
        bb_ = bbox(dept_df, b)
        if ba is None or bb_ is None:
            continue

        if rating == "X":
            ca = get_coords(dept_df, a)
            cb = get_coords(dept_df, b)
            if ca and cb:
                mx, my = (ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2
                ax.plot(mx, my, marker="x", color="#f85149",
                        markersize=14, markeredgewidth=2.5, zorder=4)
                ax.plot([ca[0], cb[0]], [ca[1], cb[1]],
                        color="#f85149", lw=0.8, alpha=0.35,
                        linestyle="dashed", zorder=3)
            continue

        if n_lines == 0:
            continue

        pa, pb = closest_border_points(ba, bb_)
        vx = pb[0] - pa[0]
        vy = pb[1] - pa[1]
        length = np.hypot(vx, vy)
        if length < 1e-6:
            continue
        perp    = np.array([-vy / length, vx / length])
        spacing = 0.5
        offsets = np.linspace(-(n_lines - 1) / 2, (n_lines - 1) / 2, n_lines) * spacing

        for off in offsets:
            p1 = np.array(pa) + perp * off
            p2 = np.array(pb) + perp * off
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, lw=1.6, alpha=0.85,
                    linestyle=lstyle, zorder=3)

    for _, row in dept_df.iterrows():
        cx   = float(row["X"])
        cy   = float(row["Y"])
        w    = float(row.get("Width",  8))
        h    = float(row.get("Height", 8))
        name = row["Name"]
        fill = DEPT_COLORS.get(row["Type"], "#1c2128")
        edge = DEPT_BORDER.get(row["Type"], "#58a6ff")

        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.2",
            linewidth=2.0, edgecolor=edge,
            facecolor=fill, alpha=0.88, zorder=5
        )
        ax.add_patch(rect)

        words  = name.split()
        label  = "\n".join(words[:2]) + ("\n..." if len(words) > 2 else "")
        area   = float(row.get("Area_m2", w * h))
        hazard = row.get("Hazard", "")
        haz_c  = {"High": "#f85149", "Medium": "#d29922", "Low": "#3fb950"}.get(hazard, "white")

        ax.text(cx, cy + 0.6, label, ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold", zorder=7)
        ax.text(cx, cy - h / 2 + 1.2, f"{area:.0f} m²",
                ha="center", va="bottom", fontsize=5.0, color="#8b949e", zorder=7)
        ax.text(cx, cy + h / 2 - 1.2, hazard,
                ha="center", va="top", fontsize=5.0, color=haz_c, fontweight="bold", zorder=7)

    legend_elements = []
    for rating, (n_l, color, ls) in SLP_LINE_SPEC.items():
        if rating == "U":
            continue
        label = {"A": "A — Absolutely Necessary (4 lines)",
                 "E": "E — Especially Important (3 lines)",
                 "I": "I — Important (2 lines)",
                 "O": "O — Ordinary (1 line)",
                 "X": "X — Undesirable (✕)"}.get(rating, rating)
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor=color,
                           alpha=0.75, label=label))

    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=6, facecolor="#161b22", labelcolor="white",
              framealpha=0.9, edgecolor="#30363d")
    ax.set_title(title, color="#58a6ff", fontsize=11, pad=10, fontfamily="monospace")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.set_xlabel("X (m)", color="#8b949e", fontsize=8)
    ax.set_ylabel("Y (m)", color="#8b949e", fontsize=8)
    ax.grid(True, color="#21262d", linewidth=0.4, linestyle="--", zorder=0)


def plot_layout(dept_df, flow_df, ax, title="Layout", show_flow=True):
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-5, GRID_MAX + 10)
    ax.set_ylim(-5, GRID_MAX + 10)

    if show_flow:
        max_w = flow_df["Weight"].max() if not flow_df.empty else 1
        for _, row in flow_df.iterrows():
            c1 = get_coords(dept_df, row["From"])
            c2 = get_coords(dept_df, row["To"])
            if c1 and c2:
                alpha = 0.25 + 0.55 * row["Weight"] / max_w
                ax.annotate("", xy=(c2[0], c2[1]), xytext=(c1[0], c1[1]),
                    arrowprops=dict(arrowstyle="->", color="#58a6ff",
                                   lw=0.8 + 2.0 * row["Weight"] / max_w,
                                   alpha=alpha))

    haz_c = {"High": "#f85149", "Medium": "#d29922", "Low": "#3fb950"}
    for _, row in dept_df.iterrows():
        color = DEPT_BORDER.get(row["Type"], "#8b949e")
        ax.scatter(row["X"], row["Y"], s=320, c=color,
                   zorder=5, edgecolors="white", linewidths=0.8, alpha=0.92)
        short = "\n".join(row["Name"].split()[:2])
        ax.text(row["X"], row["Y"] + 4.5, short,
                ha="center", va="bottom", fontsize=6.5,
                color="white", fontweight="bold", zorder=6)
        ax.text(row["X"], row["Y"] - 5, row["Hazard"],
                ha="center", va="top", fontsize=5.5,
                color=haz_c.get(row["Hazard"], "gray"), zorder=6)

    for i in range(len(PRODUCTION_SEQUENCE) - 1):
        c1 = get_coords(dept_df, PRODUCTION_SEQUENCE[i])
        c2 = get_coords(dept_df, PRODUCTION_SEQUENCE[i + 1])
        if c1 and c2:
            ax.annotate("", xy=(c2[0], c2[1]), xytext=(c1[0], c1[1]),
                arrowprops=dict(arrowstyle="-|>", color="#3fb950",
                               lw=1.8, alpha=0.6, linestyle="dashed"))

    ax.set_title(title, color="#58a6ff", fontsize=11, pad=10, fontfamily="monospace")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.set_xlabel("X (m)", color="#8b949e", fontsize=8)
    ax.set_ylabel("Y (m)", color="#8b949e", fontsize=8)
    ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global ADJACENCY_THRESHOLD, GRID_MAX

    st.markdown("""
    <div style="padding:24px 0 12px 0;">
        <h1 style="font-size:1.8rem;margin:0;">🏭 PharmLayout Intelligence System</h1>
        <p style="color:#8b949e;font-family:'Share Tech Mono',monospace;font-size:0.8rem;margin-top:6px;">
            PHARMACEUTICAL FACILITY LAYOUT · EVALUATION · BLOCK PLANNING · OPTIMIZATION
        </p>
        <p style="color:#3fb950;font-family:'Share Tech Mono',monospace;font-size:0.75rem;margin-top:2px;">
            v3 — Three-Fix Optimizer Refactor: Balanced Overlap Penalty · Gravity/Betweenness Pull · REL-Chart Springs
        </p>
    </div><hr>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📋 Data Input")
        data_source = st.radio("Data Source",
                               ["Use Sample Data", "Manual Entry"], index=0)

        st.markdown("---")
        st.markdown("**Layout Parameters**")
        adj_threshold_input = st.slider("Adjacency Threshold (m)", 5, 30,
                                        int(ADJACENCY_THRESHOLD))
        grid_max_input = st.slider("Grid Size (m)", 50, 200, int(GRID_MAX))

        st.markdown("---")
        # ── OPI WEIGHT SLIDERS ────────────────────────────────────────────────
        st.markdown("**⚖️ OPI Weight Distribution (%)**")
        st.caption("Adjust priority weights — total should equal 100 %")
        w_travel = st.slider("Travel Distance Weight", 0, 100, 25, step=5)
        w_adj    = st.slider("Adjacency Compliance Weight", 0, 100, 25, step=5)
        w_zone   = st.slider("Zoning Weight", 0, 100, 25, step=5)
        w_dir    = st.slider("Directionality Weight", 0, 100, 25, step=5)
        total_w  = w_travel + w_adj + w_zone + w_dir
        if total_w != 100:
            st.warning(f"⚠️ Weights sum to {total_w} % (should be 100 %)")
        else:
            st.success("✅ Weights sum to 100 %")

        st.markdown("---")
        # ── v3: OPTIMIZER TUNING SLIDERS ─────────────────────────────────────
        st.markdown("**🔧 v3 Optimizer Tuning**")
        st.caption("Balance the three objective components")

        overlap_k = st.slider(
            "Fix 1 — Overlap Penalty (K)",
            min_value=5, max_value=400, value=int(DEFAULT_OVERLAP_K), step=5,
            help="Lower = flow cost dominates; Higher = overlap avoidance dominates. "
                 "Was 800 in v2 (too aggressive). Default 50."
        )
        gravity_k = st.slider(
            "Fix 2 — Gravity / Betweenness Pull (K)",
            min_value=0.0, max_value=5.0, value=DEFAULT_GRAVITY_K, step=0.1,
            help="Pulls each dept toward the flow-weighted centroid of its "
                 "neighbours. Higher = stronger in-between positioning."
        )
        rel_k = st.slider(
            "Fix 3 — REL Spring Stiffness (K)",
            min_value=0, max_value=500, value=int(DEFAULT_REL_K), step=10,
            help="Spring penalty for A-rated REL pairs beyond adjacency threshold. "
                 "Higher = stronger pull from the SLP relationship chart."
        )

        st.markdown("---")
        run_btn = st.button("▶  RUN ANALYSIS", use_container_width=True)

    ADJACENCY_THRESHOLD = float(adj_threshold_input)
    GRID_MAX            = float(grid_max_input)

    # ── DATA ENTRY TABS ──────────────────────────────────────────────────────
    tab_dept, tab_flow, tab_rel, tab_rel_auto = st.tabs([
        "Departments", "Flow Matrix", "REL Chart (Manual)", "Auto REL Synthesis"
    ])

    dept_col_cfg = {
        "Hazard": st.column_config.SelectboxColumn("Hazard", options=HAZARD_LEVELS),
        "Type":   st.column_config.SelectboxColumn("Type",   options=DEPARTMENT_TYPES),
    }

    if data_source == "Use Sample Data":
        dept_df = DEFAULT_DEPARTMENTS.copy()
        flow_df = DEFAULT_FLOW.copy()
        rel_df  = DEFAULT_REL.copy()

        with tab_dept:
            st.markdown('<div class="section-header">Department Definitions (with Dimensions)</div>',
                        unsafe_allow_html=True)
            st.caption("X, Y = centroid position (m).  Width & Height = footprint in metres.")
            dept_df = st.data_editor(dept_df, num_rows="dynamic",
                                     column_config=dept_col_cfg,
                                     use_container_width=True, key="dept_edit")

        with tab_flow:
            st.markdown('<div class="section-header">Flow Data — with SLP Reason Codes</div>',
                        unsafe_allow_html=True)
            flow_rc_cfg = {
                "Reason Code": st.column_config.SelectboxColumn(
                    "Reason Code",
                    options=list(SLP_REASON_CODES.keys()),
                    help="\n".join(f"{k}: {v}" for k, v in SLP_REASON_CODES.items())
                )
            }
            flow_df = st.data_editor(flow_df, num_rows="dynamic",
                                     column_config=flow_rc_cfg,
                                     use_container_width=True, key="flow_edit")
            with st.expander("SLP Reason Code Reference"):
                st.table(pd.DataFrame(list(SLP_REASON_CODES.items()),
                                      columns=["Code", "Description"]))

        with tab_rel:
            st.markdown('<div class="section-header">REL Chart — Manual Entry / Override</div>',
                        unsafe_allow_html=True)
            st.caption("Set Override=True to lock a rating against auto-synthesis.")
            rel_col_cfg = {
                "Rating": st.column_config.SelectboxColumn("Rating", options=REL_RATINGS),
                "Override": st.column_config.CheckboxColumn("Override"),
            }
            rel_df = st.data_editor(rel_df, num_rows="dynamic",
                                    column_config=rel_col_cfg,
                                    use_container_width=True, key="rel_edit")

    else:
        with tab_dept:
            st.markdown('<div class="section-header">Department Definitions</div>',
                        unsafe_allow_html=True)
            dept_template = pd.DataFrame({
                "Name":   DEPARTMENT_TYPES,
                "X":      [0.0]*8, "Y": [0.0]*8,
                "Width":  [10.0]*8, "Height": [8.0]*8,
                "Area_m2":[80.0]*8,
                "Hazard": ["Low"]*8,
                "Type":   DEPARTMENT_TYPES,
            })
            dept_df = st.data_editor(dept_template, num_rows="dynamic",
                                     column_config=dept_col_cfg,
                                     use_container_width=True, key="dept_manual")

        with tab_flow:
            flow_template = pd.DataFrame({
                "From": [""], "To": [""], "Weight": [0], "Reason Code": [1]})
            flow_df = st.data_editor(flow_template, num_rows="dynamic",
                                     use_container_width=True, key="flow_manual")

        with tab_rel:
            rel_template = pd.DataFrame({
                "Dept_A": [""], "Dept_B": [""], "Rating": ["A"],
                "Reason Code": [1], "Override": [False]})
            rel_df = st.data_editor(rel_template, num_rows="dynamic",
                                    use_container_width=True, key="rel_manual")

    with tab_rel_auto:
        st.markdown('<div class="section-header">🤖 Automatic REL Chart Synthesis</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        **Algorithm:**
        1. Aggregate all flow pairs and normalise weights to [0, 1].
        2. Map normalised weight → SLP rating using thresholds:
           `≥ 0.75 → A`, `≥ 0.50 → E`, `≥ 0.25 → I`, `≥ 0.10 → O`, `< 0.10 → U`
        3. Assign SLP reason code from the flow row's Reason Code column.
        4. Inject domain rules (waste separation = X/6, utilities = A/5).
        5. Apply any rows marked **Override = True** from the Manual REL tab.
        """)
        if st.button("🔄 Generate Auto REL Chart"):
            try:
                auto_rel = synthesize_rel_chart(flow_df, dept_df, rel_df)
                st.success(f"Generated {len(auto_rel)} relationship pairs.")
                st.dataframe(auto_rel, use_container_width=True)
                st.session_state["auto_rel"] = auto_rel
            except Exception as e:
                st.error(f"Synthesis error: {e}")

        if "auto_rel" in st.session_state:
            use_auto = st.checkbox("✅ Use Auto-Synthesised REL Chart for analysis", value=True)
            if use_auto:
                rel_df = st.session_state["auto_rel"]
                st.info("Auto-synthesised REL chart will be used for the analysis run.")

    # ── RUN ANALYSIS ─────────────────────────────────────────────────────────
    if run_btn or (data_source == "Use Sample Data" and "auto_rel" not in st.session_state):

        if dept_df is None or dept_df.empty or flow_df is None or flow_df.empty:
            st.warning("Please enter department and flow data before running analysis.")
            st.stop()

        dept_df = dept_df.dropna(subset=["Name", "X", "Y"]).reset_index(drop=True)
        flow_df = flow_df.dropna(subset=["From", "To", "Weight"]).reset_index(drop=True)
        for col, default in [("Width", 8.0), ("Height", 8.0), ("Area_m2", 64.0)]:
            if col not in dept_df.columns:
                dept_df[col] = default

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # ── COMPUTE ORIGINAL METRICS ─────────────────────────────────────────
        travel_total, travel_detail = compute_travel_distance(dept_df, flow_df)
        adj_score, adj_violations, adj_detail = compute_adjacency_score(dept_df, rel_df)
        zone_score, zone_violations = compute_zoning_score(dept_df)
        dir_score,  dir_issues      = compute_directionality_score(dept_df)

        baseline_travel = max(travel_total * 1.5, 1000.0)
        opi, travel_score = compute_overall_index(
            travel_total, adj_score, zone_score, dir_score,
            baseline_travel, w_travel, w_adj, w_zone, w_dir
        )

        # ── METRICS ROW ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Original Layout Metrics</div>',
                    unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)

        def metric_card(col, label, value, unit="", color_class=None):
            cls = color_class or score_color(50)
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">{value}<span style="font-size:0.9rem">{unit}</span></div>
            </div>""", unsafe_allow_html=True)

        with col1: metric_card(col1, "Total Travel Distance", f"{travel_total:,.0f}", " m", "warn")
        with col2: metric_card(col2, "Adjacency Compliance",  f"{adj_score:.1f}", "%",
                               score_color(adj_score))
        with col3: metric_card(col3, "Zoning Score",          f"{zone_score:.1f}", "%",
                               score_color(zone_score))
        with col4: metric_card(col4, "Directionality Score",  f"{dir_score:.1f}", "%",
                               score_color(dir_score))
        with col5: metric_card(col5, "OPI (Weighted)",        f"{opi:.1f}", "%",
                               score_color(opi))

        st.caption(f"OPI weights — Travel: {w_travel}% | Adjacency: {w_adj}% | "
                   f"Zoning: {w_zone}% | Directionality: {w_dir}%")

        # ── BLOCK LAYOUT ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">True Block Layout (SLP Closeness Lines)</div>',
                    unsafe_allow_html=True)
        fig_b, ax_b = plt.subplots(figsize=(14, 8))
        fig_b.patch.set_facecolor("#0d1117")
        plot_block_layout(dept_df, rel_df, ax_b, title="Facility Block Layout — Original")
        plt.tight_layout(pad=2.0)
        st.pyplot(fig_b)
        plt.close(fig_b)

        # ── FLOW GRAPH ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Flow Graph (NetworkX)</div>',
                    unsafe_allow_html=True)
        G = build_flow_graph(dept_df, flow_df)
        fig_g, ax_g = plt.subplots(figsize=(12, 6))
        fig_g.patch.set_facecolor("#0d1117")
        plot_flow_graph(G, dept_df, ax_g)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig_g)
        plt.close(fig_g)

        # ── BREAKDOWNS ───────────────────────────────────────────────────────
        with st.expander("📦 Travel Distance Breakdown", expanded=False):
            if not travel_detail.empty:
                td = travel_detail.sort_values("Weighted Distance", ascending=False).copy()
                td["% of Total"] = (td["Weighted Distance"] / td["Weighted Distance"].sum() * 100).round(1)
                st.dataframe(td.reset_index(drop=True), use_container_width=True)

        with st.expander("📐 Adjacency Compliance Detail", expanded=False):
            if not adj_detail.empty:
                st.dataframe(adj_detail, use_container_width=True)
            for v in adj_violations:
                st.markdown(
                    f'<div class="rec-card violation">❌ {v["Pair"]} | '
                    f'Rating: {v["Rating"]} | Box Distance: {v["Distance"]} m</div>',
                    unsafe_allow_html=True)

        with st.expander("📋 Active REL Chart", expanded=False):
            st.dataframe(rel_df, use_container_width=True)

        # ── OPTIMIZATION MODULE ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## ⚙️ Optimization Module")

        # ── v3: FIX SUMMARY PANEL ─────────────────────────────────────────────
        st.markdown('<div class="section-header">v3 Active Fixes</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="fix-box">
            <b>Fix 1 — Overlap Penalty K = {overlap_k}</b> (v2 was 800)<br>
            Flow cost now dominates. Collision avoidance activates only on actual bbox overlap,
            not as a field that repels distant departments.
        </div>
        <div class="fix-box">
            <b>Fix 2 — Gravity / Betweenness Pull K = {gravity_k:.1f}</b><br>
            Each department is pulled toward the flow-weighted centroid of its neighbours.
            Sparsely-connected depts (like Neutralisation Tank) can no longer drift to empty corners —
            they're gravitationally anchored between their partners.
        </div>
        <div class="fix-box">
            <b>Fix 3 — REL-Chart Spring K = {rel_k}</b><br>
            A-rated REL pairs that violate the adjacency threshold accumulate a quadratic spring penalty.
            The SLP relationship table is now a direct optimizer input, not just a post-hoc diagnostic.
        </div>
        """, unsafe_allow_html=True)

        # ── A) GRAPH CLUSTERING ───────────────────────────────────────────────
        st.markdown('<div class="section-header">A) Graph-Based Cluster Suggestions (NetworkX)</div>',
                    unsafe_allow_html=True)
        cluster_suggestions, centrality = suggest_clusters(G, flow_df, dept_df, top_n=3)
        c1c, c2c = st.columns([3, 2])

        with c1c:
            st.markdown("**Top Department Pairs to Group Spatially**")
            for i, c in enumerate(cluster_suggestions, 1):
                st.markdown(f"""
                <div class="cluster-box">
                    <b style="color:#58a6ff">#{i} — {c['Dept A']} ↔ {c['Dept B']}</b><br>
                    <small style="color:#8b949e">
                        Flow: <b style="color:#3fb950">{c['Combined Flow']}</b> &nbsp;|&nbsp;
                        Distance: <b style="color:#d29922">{c['Current Distance']} m</b> &nbsp;|&nbsp;
                        Urgency: <b style="color:#f85149">{c['Urgency Score']:.0f}</b>
                    </small>
                </div>""", unsafe_allow_html=True)

        with c2c:
            st.markdown("**Degree Centrality**")
            cent_df = pd.DataFrame([{"Department": k, "Centrality": round(v, 4)}
                                    for k, v in sorted(centrality.items(), key=lambda x: -x[1])])
            st.dataframe(cent_df, use_container_width=True, height=220)

        # ── B) SCIPY OPTIMIZATION ────────────────────────────────────────────
        st.markdown('<div class="section-header">B) Layout Refinement via SciPy (L-BFGS-B) — v3</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        **v3 Composite Objective:**
        ```
        f = Σ(Flow_ij × Manhattan(cᵢ,cⱼ))          ← flow cost
          + {overlap_k} × Σ(overlap_x² + overlap_y²)   ← Fix 1: balanced overlap penalty
          + {gravity_k} × Σ w_total_i × ‖cᵢ − centroid_i‖²  ← Fix 2: gravity pull
          + {rel_k} × Σ_A-pairs max(0, bbox_dist − threshold)²  ← Fix 3: REL springs
        ```
        Bounds: all centroids within `[0, {int(GRID_MAX)}] m`
        """)

        with st.spinner("Running v3 SciPy L-BFGS-B optimisation (three-fix objective)…"):
            opt_df, opt_travel, pct_improve, opt_success, breakdown = optimize_layout(
                dept_df, flow_df, rel_df,
                overlap_k=float(overlap_k),
                gravity_k=float(gravity_k),
                rel_k=float(rel_k)
            )

        st.markdown(
            f"**Status:** {'✅ Converged' if opt_success else '⚠️ Sub-optimal convergence (result usable)'}  "
            f"| Original: `{travel_total:,.1f} m` → Optimised: `{opt_travel:,.1f} m` "
            f"| **Improvement: `{pct_improve:.1f}%`**  "
            f"| A-pairs constrained: `{breakdown['a_pairs_count']}`  "
            f"| Flow pairs: `{breakdown['flow_pairs_count']}`"
        )

        with st.expander("📍 Optimised Department Coordinates", expanded=False):
            coord_cmp = dept_df[["Name","X","Y","Width","Height","Area_m2"]].copy()
            coord_cmp.columns = ["Department","Orig X","Orig Y","W","H","Area m²"]
            coord_cmp["Opt X"] = opt_df["X"].values
            coord_cmp["Opt Y"] = opt_df["Y"].values
            coord_cmp["ΔX"]    = (coord_cmp["Opt X"] - coord_cmp["Orig X"]).round(2)
            coord_cmp["ΔY"]    = (coord_cmp["Opt Y"] - coord_cmp["Orig Y"]).round(2)
            st.dataframe(coord_cmp, use_container_width=True)

        # ── COMPUTE OPTIMISED METRICS ─────────────────────────────────────────
        opt_adj_score,  opt_adj_viol, _ = compute_adjacency_score(opt_df, rel_df)
        opt_zone_score, _               = compute_zoning_score(opt_df)
        opt_dir_score,  _               = compute_directionality_score(opt_df)
        opt_opi, _                      = compute_overall_index(
            opt_travel, opt_adj_score, opt_zone_score, opt_dir_score,
            baseline_travel, w_travel, w_adj, w_zone, w_dir
        )

        # ── COMPARISON TABLE ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Metrics Comparison: Original vs Optimised</div>',
                    unsafe_allow_html=True)

        def pct_chg(orig, new):
            if orig == 0: return "—"
            chg = (new - orig) / orig * 100
            return f"{'+'if chg>=0 else ''}{chg:.1f}%"

        comparison = pd.DataFrame({
            "Metric": ["Total Weighted Travel Distance (m)",
                       "Adjacency Compliance (%)",
                       "Zoning Score (%)",
                       "Directionality Score (%)",
                       f"OPI — w({w_travel}/{w_adj}/{w_zone}/{w_dir}) (%)"],
            "Original":  [f"{travel_total:,.1f}", f"{adj_score:.1f}",
                          f"{zone_score:.1f}", f"{dir_score:.1f}", f"{opi:.1f}"],
            "Optimised": [f"{opt_travel:,.1f}", f"{opt_adj_score:.1f}",
                          f"{opt_zone_score:.1f}", f"{opt_dir_score:.1f}", f"{opt_opi:.1f}"],
            "Change":    [
                f"{pct_improve:.1f}% reduction" if pct_improve >= 0 else f"{abs(pct_improve):.1f}% increase",
                pct_chg(adj_score, opt_adj_score),
                pct_chg(zone_score, opt_zone_score),
                pct_chg(dir_score, opt_dir_score),
                pct_chg(opi, opt_opi),
            ]
        })
        st.dataframe(comparison.set_index("Metric"), use_container_width=True)

        # ── BLOCK LAYOUT COMPARISON ───────────────────────────────────────────
        st.markdown('<div class="section-header">Block Layout Comparison</div>',
                    unsafe_allow_html=True)

        fig3, axes3 = plt.subplots(1, 2, figsize=(18, 8))
        fig3.patch.set_facecolor("#0d1117")
        plot_block_layout(dept_df, rel_df, axes3[0],
                          title=f"Original Block Layout  |  Travel: {travel_total:,.0f} m")
        plot_block_layout(opt_df, rel_df, axes3[1],
                          title=f"Optimised Block Layout (v3)  |  Travel: {opt_travel:,.0f} m  "
                                f"({pct_improve:.1f}% ↓)")
        plt.tight_layout(pad=2.5)
        st.pyplot(fig3)
        plt.close(fig3)

        with st.expander("🔵 Centroid-View Scatter Comparison", expanded=False):
            fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
            fig4.patch.set_facecolor("#0d1117")
            plot_layout(dept_df, flow_df, axes4[0],
                        title=f"Original  |  Travel: {travel_total:,.0f} m")
            plot_layout(opt_df,  flow_df, axes4[1],
                        title=f"Optimised v3 |  Travel: {opt_travel:,.0f} m ({pct_improve:.1f}% ↓)")
            plt.tight_layout(pad=2.0)
            st.pyplot(fig4)
            plt.close(fig4)

        # ── RECOMMENDATIONS ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 💡 Recommendations")

        recs  = generate_recommendations(travel_detail, adj_violations, zone_violations,
                                         dir_issues, cluster_suggestions, dept_df, flow_df)
        icons = {"violation": "🔴", "warning": "🟡", "info": "🔵"}
        for sev, msg in recs:
            st.markdown(f'<div class="rec-card {sev}">{icons[sev]} {msg}</div>',
                        unsafe_allow_html=True)

        # ── FORMULA REFERENCE ─────────────────────────────────────────────────
        with st.expander("📐 Formula Reference & Methodology (v3)", expanded=False):
            st.markdown(f"""
**v3 Optimization Objective — Composite (L-BFGS-B)**
```
minimize f =

  [Flow Cost]
  Σ(Flow_ij × (|cx_i−cx_j| + |cy_i−cy_j|))

+ [Fix 1 — Overlap Penalty  K={overlap_k}]
  K × Σ_{{pairs}} (overlap_x² + overlap_y²)
  activated only when bbox_i and bbox_j physically collide
  (was K=800 in v2 — too large, caused isolated-corner local minima)

+ [Fix 2 — Gravity / Betweenness Pull  K={gravity_k}]
  K × Σ_i  w_total_i × [(cx_i − target_x_i)² + (cy_i − target_y_i)²]
  where  target_x_i = Σ_j(w_ij × cx_j) / Σ_j(w_ij)
  → pulls each dept toward the flow-weighted centroid of its neighbours
  → sparsely-connected depts can no longer drift to empty corners

+ [Fix 3 — REL-Chart Spring  K={rel_k}]
  K × Σ_{{A-pairs}} max(0, bbox_dist(i,j) − {ADJACENCY_THRESHOLD})²
  → quadratic spring for every A-rated REL pair beyond adjacency threshold
  → makes the SLP relationship chart a direct optimizer constraint

Bounds: 0 ≤ cx_i, cy_i ≤ {int(GRID_MAX)} m   |   maxiter=10000
```

**Other metrics (unchanged from v2)**
```
Travel Distance : T = Σ (Flow_ij × Manhattan(cᵢ,cⱼ))
Adjacency Score : bbox_dist ≤ {ADJACENCY_THRESHOLD} m → satisfied
Zoning Score    : 100 − 10 × violations
Directionality  : 100 − (reversals/checks) × 100
OPI             : Σ (weight_k × score_k) / Σ weight_k
```
""")

        # ── FOOTER ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <p style="text-align:center;color:#484f58;font-family:'Share Tech Mono',monospace;font-size:0.75rem;">
        PharmLayout Intelligence System v3 · Three-Fix Optimizer Refactor ·
        Streamlit + NetworkX + SciPy + Matplotlib
        </p>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()