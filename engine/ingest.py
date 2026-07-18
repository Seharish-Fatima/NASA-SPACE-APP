import re
import pandas as pd

GROUP_CODES = {
    "FLT": ("Flight", "Flown on the ISS"),
    "GC": ("Ground Control", "Earth, ISS-matched environment"),
    "VIV": ("Vivarium", "Earth, standard housing"),
    "BSL": ("Baseline", "Euthanized at mission start"),
}
AGE_CODES = {"OLD": "Old", "YNG": "Young"}
TISSUE_CODES = {"LVR": "Liver", "EDL": "Leg muscle (EDL)", "SOL": "Soleus muscle", "QUA": "Quadriceps"}
SOURCE_CODES = {"ISS-T": "ISS-terminal", "LAR": "Live animal return"}

COLMAP = {
    "sample": "Sample Name",
    "rin": "Parameter Value[QA Score]",
    "rrna": "Parameter Value[rRNA Contamination]",
    "depth": "Parameter Value[Read Depth]",
    "fragment": "Parameter Value[Fragment Size]",
    "readlen": "Parameter Value[Read Length]",
    "prep_date": "Comment[Library Prep Date]",
    "instrument": "Parameter Value[sequencing instrument]",
    "library": "Parameter Value[library selection]",
    "layout": "Parameter Value[library layout]",
}


def _first_code(tokens, table):
    for t in tokens:
        if t in table:
            return t
    return None


def parse_sample(name):
    tokens = name.split("_")
    group = _first_code(tokens, GROUP_CODES)
    age = _first_code(tokens, AGE_CODES)
    tissue = _first_code(tokens, TISSUE_CODES)
    source = _first_code(tokens, SOURCE_CODES)
    animal = tokens[-1] if tokens else None
    return {
        "group_code": group,
        "group": GROUP_CODES.get(group, ("Unknown", ""))[0],
        "age": AGE_CODES.get(age, "Unspecified"),
        "tissue": TISSUE_CODES.get(tissue, "Unspecified"),
        "source": SOURCE_CODES.get(source, "Unspecified"),
        "animal_id": animal,
    }


def load(path, colmap=None):
    cm = colmap or COLMAP
    raw = pd.read_csv(path)
    df = pd.DataFrame()
    df["sample"] = raw[cm["sample"]]
    for key in ("rin", "rrna", "depth", "fragment", "readlen"):
        if cm[key] in raw.columns:
            df[key] = pd.to_numeric(raw[cm[key]], errors="coerce")
    for key in ("prep_date", "instrument", "library", "layout"):
        if cm.get(key) in raw.columns:
            df[key] = raw[cm[key]]
    factors = df["sample"].map(parse_sample).apply(pd.Series)
    df = pd.concat([df, factors], axis=1)
    df["is_flight"] = df["group"] == "Flight"
    df["environment"] = df["group"].map(lambda g: "Space" if g == "Flight" else "Earth")
    return df


GROUP_ORDER = ["Baseline", "Vivarium", "Ground Control", "Flight"]


def group_counts(df):
    tab = df.groupby(["group", "age"]).size().unstack(fill_value=0)
    tab = tab.reindex([g for g in GROUP_ORDER if g in tab.index])
    return tab


def experiment_facts(df):
    prep = df["prep_date"].dropna() if "prep_date" in df else pd.Series(dtype=str)
    return {
        "n_samples": len(df),
        "n_groups": df["group"].nunique(),
        "tissue": df["tissue"].mode().iat[0] if not df["tissue"].mode().empty else "Unspecified",
        "instrument": df["instrument"].mode().iat[0] if "instrument" in df and not df["instrument"].mode().empty else "Unknown",
        "flight_n": int((df["group"] == "Flight").sum()),
        "earth_n": int((df["group"] != "Flight").sum()),
        "ages": sorted(a for a in df["age"].unique() if a != "Unspecified"),
        "groups": [g for g in GROUP_ORDER if g in df["group"].unique()],
    }