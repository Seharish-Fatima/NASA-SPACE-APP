import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

METRICS = {
    "rin": ("RNA Integrity (RIN)", "higher is better", 1),
    "rrna": ("rRNA Contamination (%)", "lower is better", -1),
    "depth": ("Read Depth", "higher is better", 1),
    "fragment": ("Fragment Size (bp)", "context", 0),
}


def flight_vs_earth(df, metric):
    flight = df.loc[df["group"] == "Flight", metric].dropna()
    earth = df.loc[df["group"] != "Flight", metric].dropna()
    if len(flight) < 3 or len(earth) < 3:
        return None
    u, p = stats.mannwhitneyu(flight, earth, alternative="two-sided")
    pooled_sd = np.sqrt((flight.var(ddof=1) + earth.var(ddof=1)) / 2)
    d = (flight.mean() - earth.mean()) / pooled_sd if pooled_sd > 0 else 0.0
    return {
        "flight_mean": float(flight.mean()),
        "earth_mean": float(earth.mean()),
        "flight_n": len(flight),
        "earth_n": len(earth),
        "p_value": float(p),
        "cohens_d": float(d),
        "significant": bool(p < 0.05),
    }


def group_summary(df, metric):
    g = df.groupby("group")[metric].agg(["mean", "std", "count"]).round(3)
    return g.reindex([x for x in ["Baseline", "Vivarium", "Ground Control", "Flight"] if x in g.index])


def anova_across_groups(df, metric):
    groups = [df.loc[df["group"] == g, metric].dropna() for g in df["group"].unique()]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        return None
    f, p = stats.f_oneway(*groups)
    return {"f_stat": float(f), "p_value": float(p), "significant": bool(p < 0.05)}


def run_pca(df, metrics=("rin", "rrna", "depth", "fragment")):
    cols = [m for m in metrics if m in df.columns]
    sub = df.dropna(subset=cols)
    X = StandardScaler().fit_transform(sub[cols])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    out = sub[["sample", "group", "age", "environment"]].copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    return out, pca.explained_variance_ratio_, dict(zip(cols, pca.components_[0]))


def quality_flags(df, rin_min=5.0, rrna_max=5.0, depth_min=20e6):
    d = df.copy()
    d["flag_rin"] = d["rin"] < rin_min
    d["flag_rrna"] = d["rrna"] > rrna_max
    d["flag_depth"] = d["depth"] < depth_min
    d["passes"] = ~(d["flag_rin"] | d["flag_rrna"] | d["flag_depth"])
    return d