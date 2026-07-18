from .ingest import GROUP_CODES, experiment_facts, group_counts

GROUP_COLOR = {"Baseline": "#8A7DFF", "Vivarium": "#4DD0C7", "Ground Control": "#E8B04B", "Flight": "#FF6B6B"}


def build_svg(df, mission_title, duration, tissue_label):
    facts = experiment_facts(df)
    counts = group_counts(df)
    groups = facts["groups"]
    W, H = 900, 520
    cx = W // 2

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Space Grotesk, sans-serif">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#070B16"/>')
    for i in range(60):
        import random
        random.seed(i)
        sx, sy, r = random.randint(0, W), random.randint(0, H), random.choice([0.5, 0.8, 1.1])
        parts.append(f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="#2A3550" opacity="0.6"/>')

    parts.append(f'<text x="{cx}" y="44" fill="#DCE6FF" font-size="26" font-weight="700" text-anchor="middle">{mission_title}</text>')
    parts.append(f'<text x="{cx}" y="70" fill="#6C7A99" font-size="14" text-anchor="middle">{facts["n_samples"]} samples · {tissue_label} · {facts["instrument"]}</text>')

    y0 = 120
    parts.append(f'<circle cx="150" cy="{y0}" r="30" fill="#0E1626" stroke="#4DD0C7" stroke-width="2"/>')
    parts.append(f'<circle cx="150" cy="{y0}" r="15" fill="#2A6E8F"/>')
    parts.append(f'<text x="150" y="{y0+58}" fill="#8FA0C0" font-size="13" text-anchor="middle">Launch from Earth</text>')
    parts.append(f'<line x1="192" y1="{y0}" x2="{cx-95}" y2="{y0}" stroke="#33415E" stroke-width="2" stroke-dasharray="5,5"/>')
    parts.append(f'<rect x="{cx-95}" y="{y0-26}" width="190" height="52" rx="8" fill="#0E1626" stroke="#FF6B6B" stroke-width="2"/>')
    parts.append(f'<rect x="{cx-70}" y="{y0-9}" width="26" height="18" rx="3" fill="none" stroke="#FF8A8A" stroke-width="2"/>')
    parts.append(f'<line x1="{cx-78}" y1="{y0}" x2="{cx-70}" y2="{y0}" stroke="#FF8A8A" stroke-width="2"/>')
    parts.append(f'<line x1="{cx-44}" y1="{y0}" x2="{cx-36}" y2="{y0}" stroke="#FF8A8A" stroke-width="2"/>')
    parts.append(f'<text x="{cx+20}" y="{y0-2}" fill="#FF8A8A" font-size="17" font-weight="700" text-anchor="middle">ISS</text>')
    parts.append(f'<text x="{cx+20}" y="{y0+16}" fill="#8FA0C0" font-size="11" text-anchor="middle">{duration}</text>')
    parts.append(f'<line x1="{cx+95}" y1="{y0}" x2="{W-192}" y2="{y0}" stroke="#33415E" stroke-width="2" stroke-dasharray="5,5"/>')
    parts.append(f'<circle cx="{W-150}" cy="{y0}" r="30" fill="#0E1626" stroke="#4DD0C7" stroke-width="2"/>')
    parts.append(f'<circle cx="{W-150}" cy="{y0}" r="7" fill="none" stroke="#4DD0C7" stroke-width="2"/>')
    parts.append(f'<line x1="{W-145}" y1="{y0+5}" x2="{W-138}" y2="{y0+12}" stroke="#4DD0C7" stroke-width="2"/>')
    parts.append(f'<text x="{W-150}" y="{y0+58}" fill="#8FA0C0" font-size="13" text-anchor="middle">Return &amp; sequence</text>')

    parts.append(f'<text x="{cx}" y="238" fill="#DCE6FF" font-size="16" font-weight="600" text-anchor="middle">Experimental groups</text>')
    n = len(groups)
    card_w, gap = 180, 24
    total = n * card_w + (n - 1) * gap
    startx = (W - total) // 2
    for i, g in enumerate(groups):
        x = startx + i * (card_w + gap)
        col = GROUP_COLOR.get(g, "#8892A6")
        total_n = int(counts.loc[g].sum()) if g in counts.index else 0
        desc = GROUP_CODES.get({v[0]: k for k, v in GROUP_CODES.items()}.get(g, ""), ("", ""))[1]
        parts.append(f'<rect x="{x}" y="260" width="{card_w}" height="160" rx="10" fill="#0C1322" stroke="{col}" stroke-width="1.5"/>')
        parts.append(f'<rect x="{x}" y="260" width="{card_w}" height="6" rx="3" fill="{col}"/>')
        parts.append(f'<text x="{x+card_w//2}" y="294" fill="{col}" font-size="16" font-weight="700" text-anchor="middle">{g}</text>')
        parts.append(f'<text x="{x+card_w//2}" y="332" fill="#DCE6FF" font-size="32" font-weight="700" text-anchor="middle">{total_n}</text>')
        parts.append(f'<text x="{x+card_w//2}" y="350" fill="#6C7A99" font-size="11" text-anchor="middle">mice</text>')
        for j, line in enumerate(_wrap(desc, 26)):
            parts.append(f'<text x="{x+card_w//2}" y="{372+j*14}" fill="#8FA0C0" font-size="10.5" text-anchor="middle">{line}</text>')

    if facts["ages"]:
        parts.append(f'<text x="{cx}" y="452" fill="#6C7A99" font-size="13" text-anchor="middle">Each group split by age: {" · ".join(facts["ages"])}</text>')
    parts.append(f'<text x="{cx}" y="486" fill="#4A5570" font-size="11" text-anchor="middle">Auto-generated from sample metadata — scales to any OSDR dataset with the same naming schema</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:3]