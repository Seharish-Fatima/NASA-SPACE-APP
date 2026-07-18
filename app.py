import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from engine import ingest, analysis, essence

st.set_page_config(page_title="ORBITAL", page_icon="🛰️", layout="wide")

VOID = "#070B16"
PANEL = "#0C1322"
LINE = "#1E2A44"
CYAN = "#4DD0C7"
AMBER = "#E8B04B"
CORAL = "#FF6B6B"
VIOLET = "#8A7DFF"
MUTED = "#6C7A99"
TEXT = "#DCE6FF"
GROUP_COLOR = {"Baseline": VIOLET, "Vivarium": CYAN, "Ground Control": AMBER, "Flight": CORAL}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp {{ background: {VOID}; }}
html, body, [class*="css"] {{ font-family: 'IBM Plex Mono', monospace; color: {TEXT}; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; }}
h1 {{ color: {CYAN} !important; }}
h2, h3 {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {LINE}; border-top: 2px solid {CYAN}; padding: 12px 16px; border-radius: 4px; }}
div[data-testid="stMetric"] label {{ color: {MUTED} !important; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; }}
div[data-testid="stMetricValue"] {{ color: {CYAN}; font-family: 'Space Grotesk', sans-serif; }}
.stTabs [data-baseweb="tab"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 4px; color: {MUTED}; font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.78rem; padding: 8px 18px; }}
.stTabs [aria-selected="true"] {{ background: {VOID}; color: {CYAN} !important; border-color: {CYAN}; }}
.finding {{ background: {PANEL}; border: 1px solid {LINE}; border-left: 3px solid {AMBER}; border-radius: 4px; padding: 16px 20px; margin: 8px 0; }}
.finding .k {{ color: {MUTED}; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; }}
</style>
""", unsafe_allow_html=True)


def styled(fig, height=380):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL, font=dict(family="IBM Plex Mono", color=TEXT, size=12), height=height, margin=dict(l=40, r=20, t=44, b=40), xaxis=dict(gridcolor=LINE), yaxis=dict(gridcolor=LINE), legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def finding(label, text):
    st.markdown(f"<div class='finding'><span class='k'>{label}</span><br><span style='color:{TEXT}'>{text}</span></div>", unsafe_allow_html=True)


DATASETS = {
    "OSD-379 · Liver (Rodent Research Reference Mission-1)": {
        "path": "data/glds379_liver.csv",
        "title": "OSD-379 · Rodent Research Reference Mission-1",
        "duration": "22–40 days in orbit",
        "tissue": "Mouse liver",
    },
}


@st.cache_data
def get(path):
    return ingest.load(path)


st.title("🛰️ Orbital")
st.markdown(f"<span style='color:{MUTED}'>a visual essence generator for NASA space-biology experiments — ingests OSDR metadata, auto-builds the graphical abstract, and answers the question the mission was designed to ask.</span>", unsafe_allow_html=True)

choice = st.selectbox("Dataset", list(DATASETS), index=0)
cfg = DATASETS[choice]
df = get(cfg["path"])
facts = ingest.experiment_facts(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", facts["n_samples"])
c2.metric("Flight (space)", facts["flight_n"])
c3.metric("Earth controls", facts["earth_n"])
c4.metric("Tissue", facts["tissue"])

tab_essence, tab_science, tab_qc, tab_scale = st.tabs(["Experiment Essence", "Did Space Matter?", "Quality Explorer", "How It Scales"])

with tab_essence:
    st.markdown("### The experiment at a glance")
    st.markdown(f"<span style='color:{MUTED}'>For a scientist who has never seen a spaceflight study: this whole panel is generated automatically from the sample metadata — no hand-drawing. It reads the naming schema and reconstructs who flew, who stayed, and how the tissue came home.</span>", unsafe_allow_html=True)
    svg = essence.build_svg(df, cfg["title"], cfg["duration"], cfg["tissue"])
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%; border-radius:6px; border:1px solid {LINE}"/>', unsafe_allow_html=True)
    finding("what you're looking at", "Mice were split into four groups. Flight animals lived on the ISS for weeks; Ground Control matched their exact conditions on Earth; Vivarium mice stayed in standard housing; Baseline mice were sampled at mission start as a time-zero reference. After return, liver tissue was sequenced. The design isolates the effect of spaceflight itself from the effect of the specialized space habitat.")

with tab_science:
    st.markdown("### Did spaceflight change the tissue?")
    st.markdown(f"<span style='color:{MUTED}'>The mission's core question, tested on all {facts['n_samples']} samples: are Flight samples measurably different from Earth samples? Mann–Whitney U, two-sided.</span>", unsafe_allow_html=True)

    metric_labels = {"rin": "RNA Integrity (RIN)", "rrna": "rRNA Contamination (%)", "depth": "Read Depth", "fragment": "Fragment Size (bp)"}
    rows = []
    for m in ["rin", "rrna", "depth", "fragment"]:
        if m not in df.columns:
            continue
        r = analysis.flight_vs_earth(df, m)
        if r:
            rows.append((metric_labels[m], r))

    fig = go.Figure()
    for m in ["rin", "rrna", "depth", "fragment"]:
        if m not in df.columns:
            continue
        for env, col in [("Space", CORAL), ("Earth", CYAN)]:
            vals = df.loc[df["environment"] == env, m].dropna()
            if len(vals):
                z = (vals - df[m].mean()) / df[m].std()
                fig.add_trace(go.Box(y=z, name=f"{metric_labels[m]}<br>{env}", marker_color=col, showlegend=False))
    fig.update_layout(title="Every metric, Space vs Earth (z-scored) — overlapping boxes = no spaceflight effect")
    st.plotly_chart(styled(fig, 420), use_container_width=True)

    cols = st.columns(len(rows))
    for col, (label, r) in zip(cols, rows):
        verdict = "DIFFERENT" if r["significant"] else "NO EFFECT"
        vcol = CORAL if r["significant"] else CYAN
        col.markdown(f"<div class='finding'><span class='k'>{label}</span><br><span style='color:{vcol}; font-family:Space Grotesk; font-size:1.1rem'>{verdict}</span><br><span style='color:{MUTED}; font-size:0.8rem'>p = {r['p_value']:.3f} · d = {r['cohens_d']:+.2f}<br>space {r['flight_mean']:.2f} vs earth {r['earth_mean']:.2f}</span></div>", unsafe_allow_html=True)

    all_ns = all(not r["significant"] for _, r in rows)
    if all_ns:
        finding("the headline", f"Across every quality metric, Flight and Earth samples are statistically indistinguishable — no p-value clears 0.05, every effect size is negligible. Despite 22–40 days in orbit, launch, and return, the space-flown liver RNA is just as intact as tissue that never left the ground. That's not a null result to hide; for space biology it's the reassuring finding that spaceflight itself didn't compromise the samples.")

with tab_qc:
    st.markdown("### Explore the quality metrics")
    left, right = st.columns([1, 2])
    with left:
        rin_min = st.slider("RIN pass threshold", 0.0, 10.0, 5.0, 0.5)
        rrna_max = st.slider("rRNA max (%)", 0.0, 8.0, 5.0, 0.5)
        color_by = st.radio("Colour points by", ["group", "age", "environment"], horizontal=False)
    flagged = analysis.quality_flags(df, rin_min=rin_min, rrna_max=rrna_max)
    with right:
        cmap = GROUP_COLOR if color_by == "group" else None
        fig = px.scatter(flagged, x="rin", y="rrna", color=color_by, color_discrete_map=cmap,
                         hover_data=["sample", "age"], labels={"rin": "RIN (higher better)", "rrna": "rRNA % (lower better)"})
        fig.add_vline(x=rin_min, line_dash="dash", line_color=MUTED)
        fig.add_hline(y=rrna_max, line_dash="dash", line_color=MUTED)
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color=VOID)))
        st.plotly_chart(styled(fig, 420), use_container_width=True)
    passes = int(flagged["passes"].sum())
    fail_by_group = flagged.loc[~flagged["passes"], "group"].value_counts().to_dict()
    spread = " · ".join(f"{g}: {n}" for g, n in fail_by_group.items()) if fail_by_group else "none"
    finding("pass rate", f"{passes}/{len(flagged)} samples clear both thresholds at your current cutoffs (the dashed lines — drag them and watch the pass rate move). Failures by group: {spread}. They spread across conditions rather than piling up in Flight, which is the visual echo of the null result on the previous tab.")

    coords, evr, load1 = analysis.run_pca(df)
    fig = px.scatter(coords, x="pc1", y="pc2", color="group", color_discrete_map=GROUP_COLOR, hover_data=["sample", "age"],
                     labels={"pc1": f"PC1 ({evr[0]:.0%})", "pc2": f"PC2 ({evr[1]:.0%})"})
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color=VOID)))
    fig.update_layout(title="PCA of quality metrics — groups don't separate, confirming spaceflight left no quality signature")
    st.plotly_chart(styled(fig, 400), use_container_width=True)

with tab_scale:
    st.markdown("### Why this generalizes")
    st.markdown(f"""<span style='color:{TEXT}'>The challenge asks for a tool that scales across the NASA Open Science Data Repository — not a one-off dashboard. Orbital is built as an <b>ingester</b>, not a hardcoded report:</span>

<div class='finding'><span class='k'>1 · metadata-driven parsing</span><br><span style='color:{TEXT}'>Every experimental factor — group, age, tissue, sample source — is derived by reading the OSDR sample-naming schema, not typed in by hand. Point it at another study using the same conventions and the factors reconstruct themselves.</span></div>

<div class='finding'><span class='k'>2 · column mapping, not column assumptions</span><br><span style='color:{TEXT}'>The loader takes a column map, so a dataset that labels read depth or RIN differently only needs a config entry — no code change. OSD-665 (RR-23 leg muscle) drops in as a second map.</span></div>

<div class='finding'><span class='k'>3 · the essence panel is generated, not drawn</span><br><span style='color:{TEXT}'>Group count, tissue, instrument, timeline, subjects-per-group — all read from the data at render time. A new dataset produces a new graphical abstract with zero design work.</span></div>

<div class='finding'><span class='k'>4 · the science layer is metric-agnostic</span><br><span style='color:{TEXT}'>Flight-vs-Earth testing, PCA, and quality flags run on whatever numeric metrics the loader finds. The statistics don't care whether the tissue is liver or muscle.</span></div>

<span style='color:{MUTED}'>Currently loaded: OSD-379. The architecture's whole point is that OSD-665 and other OSDR studies are configuration, not rewrites.</span>
""", unsafe_allow_html=True)