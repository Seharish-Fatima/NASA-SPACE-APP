# 🛰️ Orbital

NASA sent mice to the International Space Station for up to 40 days, brought them home, and sequenced their livers. The obvious question — *did being in space wreck the tissue?* — has an answer sitting in the data, and it's not the one you'd guess.

> Built for the 2024 NASA Space Apps Challenge — [Visualize Space Science](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/visualize-space-science/). The brief: build a scalable tool that ingests NASA space-biology metadata and auto-generates compelling visualizations, for an audience with little to no spaceflight-experiment background.

🔗 **Live app:** [orbital-osdr.streamlit.app](https://orbital-osdr.streamlit.app/)

---

## 🚀 Overview

Orbital reads NASA's open space-biology metadata, auto-builds a graphical abstract of the experiment for people who've never seen one, and then actually answers the mission's question across all 141 samples.

The target reader is a scientist with zero spaceflight-experiment experience — so the tool explains itself.

## 🧠 Why It Exists

Most space-biology datasets sit behind a wall of sample-naming conventions and repository jargon that only makes sense if you already know the study. The challenge asked for something that scales across NASA's whole Open Science Data Repository, not a one-off chart for one dataset. So Orbital doesn't hardcode a study — it reads the metadata schema and builds the explanation from that.

---

## 📊 What It Actually Does (no cap)

Four tabs, from "what even is this experiment" to "here's the science":

**Experiment Essence** — the centrepiece: a graphical abstract **generated from the metadata, not drawn by hand.** It reads the sample-naming schema and reconstructs the whole study — launch → ISS → return → sequence, the four experimental groups, subjects per group, tissue, instrument — as one infographic. Point it at a differently-shaped OSDR dataset and it draws a different abstract with no design work.

**Did Space Matter?** — the mission's actual question, tested on all 141 samples with Mann–Whitney U. The headline: across **every** quality metric, Flight and Earth samples are statistically indistinguishable (no p-value below 0.05, every effect size negligible). Despite launch, 22–40 days in orbit, and return, the space-flown RNA is as intact as tissue that never left the ground. For space biology, a clean null is the reassuring result.

**Quality Explorer** — drag the RIN and rRNA thresholds, recolour by group/age/environment, watch which samples fail. The failures scatter across all conditions rather than clustering in Flight — which is *why* the space-vs-earth test comes back null. Plus PCA that refuses to separate the groups, confirming spaceflight left no quality signature.

**How It Scales** — the challenge asked for a tool that generalizes across the NASA Open Science Data Repository, not a one-off. This tab shows the architecture: metadata-driven parsing, column-mapping instead of column-assumptions, a generated-not-drawn essence panel, and metric-agnostic statistics. OSD-665 (the RR-23 leg-muscle study) drops in as configuration, not a rewrite.

Every tab ends with a plain-language finding — because the reader might be a biologist who's never touched a launch manifest.

---

## 🛠 Project Structure

```
orbital/
├── app.py                    # the actual app (Streamlit UI)
├── engine/
│   ├── ingest.py             # metadata parser — sample schema → experimental factors
│   ├── analysis.py           # flight-vs-earth stats, PCA, quality flags
│   └── essence.py            # auto-generated SVG graphical abstract
├── data/glds379_liver.csv    # OSD-379 liver RNA-seq metadata, bundled
└── requirements.txt
```

`engine/` doesn't know Streamlit exists. The parser turns `RR8_LVR_FLT_ISS-T_OLD_BI3` into structured factors (liver, flight, ISS-terminal, old) by reading the OSDR naming convention — that decoding is the whole reason the tool scales instead of being hardcoded to one study. Every function was tested against all 141 real samples before the UI existed.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data is bundled — it just works. Or skip the setup entirely — it's already live: **[orbital-osdr.streamlit.app](https://orbital-osdr.streamlit.app/)**

---

## 📂 Dataset

NASA Open Science Data Repository, [OSD-379](https://osdr.nasa.gov/bio/repo/data/studies/OSD-379) — Rodent Research Reference Mission-1, mouse liver RNA-seq, mice flown on the ISS for 22–40 days.

## ⚠️ Important Notes

* A null result is still a result. "No significant difference" isn't the tool failing to find something — it's the actual finding, and it's reported as such rather than dressed up.
* Statistical significance ≠ biological importance, and vice versa. Effect sizes are reported alongside p-values on purpose.
* Built and validated against OSD-379. The architecture is designed to extend to other OSDR studies (e.g. OSD-665), but that extension is config, not a guarantee — it hasn't been run against every dataset the repository holds.

## 🔮 Future Improvements

* Wire in OSD-665 (RR-23, leg muscle) as a second live dataset
* Direct OSDR API pull instead of a bundled CSV
* Additional quality metrics beyond RIN/rRNA
* Downloadable graphical abstract (PNG/SVG export)

## 👩‍💻 Author

Built by **Seharish Fatima** for the 2024 NASA Space Apps Challenge — because a dataset that only makes sense to the person who collected it isn't actually open.
