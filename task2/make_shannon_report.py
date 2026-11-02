#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_shannon_report.py —— 纯标准库版（无 pandas/numpy/matplotlib）
读 position_comparison.csv，生成 summary/top_human_wins/enriched 和 5 张 SVG 图。

用法：
    python3 make_shannon_report.py --csv position_comparison.csv --outdir report
"""

import argparse
import csv
import math
from pathlib import Path
from collections import Counter


# ---------- CSV 工具 ----------
def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------- SVG 基础 ----------
def svg_open(w, h):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" '
        f'font-family="PingFang SC, Microsoft YaHei, Noto Sans CJK SC, sans-serif">'
        f'<rect width="{w}" height="{h}" fill="white"/>'
    )


def svg_close():
    return "</svg>"


def text(x, y, s, size=12, anchor="middle", fill="#333", weight="normal"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, color="#333", width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{d}/>'
    )


def rect(x, y, w, h, fill="#4C78A8"):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"/>'
    )


def circle(cx, cy, r, fill="#1f77b4", opacity=0.8):
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
        f'fill="{fill}" fill-opacity="{opacity}"/>'
    )


def polyline(points, color="#1f77b4", width=1.5):
    if not points:
        return ""
    pstr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pstr}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def lin(v, vmin, vmax, pmin, pmax):
    if vmax == vmin:
        return (pmin + pmax) / 2
    return pmin + (v - vmin) / (vmax - vmin) * (pmax - pmin)


def draw_axes(svg, w, h, xmin, xmax, ymin, ymax, xlabel, ylabel, title,
              ml=80, mr=40, mt=55, mb=70, x_ticks=None, y_ticks=None):
    x0, x1 = ml, w - mr
    y0, y1 = mt, h - mb
    svg.append(line(x0, y1, x1, y1, "#333", 1.2))
    svg.append(line(x0, y0, x0, y1, "#333", 1.2))

    if x_ticks is None:
        x_ticks = [xmin + (xmax - xmin) * i / 5 for i in range(6)]
    if y_ticks is None:
        y_ticks = [ymin + (ymax - ymin) * i / 5 for i in range(6)]

    for t in x_ticks:
        xt = lin(t, xmin, xmax, x0, x1)
        svg.append(line(xt, y1, xt, y0, "#e6e6e6", 1))
        svg.append(text(xt, y1 + 18, f"{t:g}", 11, "middle"))
    for t in y_ticks:
        yt = lin(t, ymin, ymax, y1, y0)
        svg.append(line(x0, yt, x1, yt, "#e6e6e6", 1))
        svg.append(text(x0 - 8, yt + 4, f"{t:g}", 11, "end"))

    svg.append(text((x0 + x1) / 2, h - 20, xlabel, 13, "middle"))
    cy = (y0 + y1) / 2
    svg.append(
        f'<text x="22" y="{cy:.1f}" font-size="13" fill="#333" '
        f'text-anchor="middle" transform="rotate(-90 22 {cy:.1f})">{esc(ylabel)}</text>'
    )
    svg.append(text((x0 + x1) / 2, mt - 15, title, 15, "middle", "#222", "bold"))

    return (
        lambda v: lin(v, xmin, xmax, x0, x1),
        lambda v: lin(v, ymin, ymax, y1, y0),
    )


# ---------- 5 张图 ----------
def fig_guess_hist(data, path):
    def cat(g):
        if g <= 1: return "1次"
        if g == 2: return "2次"
        if g == 3: return "3次"
        if g >= 20: return "20次"
        return "4-19次"

    order = ["1次", "2次", "3次", "4-19次", "20次"]
    cnt = Counter(cat(d["人类猜测次数"]) for d in data)
    vals = [cnt.get(k, 0) for k in order]

    w, h = 720, 460
    svg = [svg_open(w, h)]
    ymax = max(vals + [1]) * 1.15
    xmap, ymap = draw_axes(svg, w, h, 0, len(order), 0, ymax, "", "位置数",
                           "人类猜中所需次数分布", x_ticks=[])
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    for i, (k, v) in enumerate(zip(order, vals)):
        xa = xmap(i) + 6
        xb = xmap(i + 1) - 6
        yt = ymap(v)
        svg.append(rect(xa, yt, xb - xa, ymap(0) - yt, colors[i]))
        svg.append(text((xa + xb) / 2, yt - 5, str(v), 12, "middle"))
        svg.append(text((xa + xb) / 2, ymap(0) + 18, k, 12, "middle"))
    svg.append(svg_close())
    path.write_text("".join(svg), encoding="utf-8")


def fig_bits_by_position(data, wins, path):
    w, h = 1000, 480
    xs = [d["位置"] for d in data]
    xmin, xmax = min(xs), max(xs)
    ymax_data = max(max(d["N=2_bits"], d["N=3_bits"]) for d in data)
    ymax = max(11, ymax_data * 1.05)
    svg = [svg_open(w, h)]
    xmap, ymap = draw_axes(svg, w, h, xmin, xmax, 0, ymax,
                           "位置", "bits/字", "逐位置信息量：人类 vs N=2 vs N=3")
    for d in wins[:15]:
        xa = xmap(d["位置"] - 0.5)
        xb = xmap(d["位置"] + 0.5)
        svg.append(
            f'<rect x="{xa:.1f}" y="55" width="{xb-xa:.1f}" '
            f'height="{h-70-55:.1f}" fill="red" fill-opacity="0.08"/>'
        )
    for key, color, lw in [("N=3_bits", "#2ca02c", 1.2),
                           ("N=2_bits", "#ff7f0e", 1.2),
                           ("人类信息量", "#1f77b4", 1.8)]:
        pts = [(xmap(d["位置"]), ymap(d[key])) for d in data if d[key] == d[key]]
        svg.append(polyline(pts, color, lw))
    lx, ly = w - 250, 60
    for i, (lab, c) in enumerate([("人类", "#1f77b4"), ("N=2", "#ff7f0e"), ("N=3", "#2ca02c")]):
        svg.append(line(lx, ly + i * 20, lx + 24, ly + i * 20, c, 3))
        svg.append(text(lx + 32, ly + i * 20 + 4, lab, 12, "start"))
    svg.append(svg_close())
    path.write_text("".join(svg), encoding="utf-8")


def fig_delta(data, path):
    w, h = 1000, 480
    xs = [d["位置"] for d in data]
    xmin, xmax = min(xs), max(xs)
    d2 = [d["delta_N2"] for d in data]
    d3 = [d["delta_N3"] for d in data]
    ymin = min(0, min(d2 + d3) * 1.1)
    ymax = max(d2 + d3) * 1.05
    svg = [svg_open(w, h)]
    xmap, ymap = draw_axes(svg, w, h, xmin, xmax, ymin, ymax,
                           "位置", "bits/字 差值",
                           "模型比人类多出的信息量（越大越卡壳）")
    svg.append(line(xmap(xmin), ymap(0), xmap(xmax), ymap(0), "#333", 1, dash="4,3"))
    for key, color in [("delta_N3", "#2ca02c"), ("delta_N2", "#ff7f0e")]:
        pts = [(xmap(d["位置"]), ymap(d[key])) for d in data if d[key] == d[key]]
        svg.append(polyline(pts, color, 1.2))
    lx, ly = w - 230, 60
    for i, (lab, c) in enumerate([("N=2 - 人类", "#ff7f0e"), ("N=3 - 人类", "#2ca02c")]):
        svg.append(line(lx, ly + i * 20, lx + 24, ly + i * 20, c, 3))
        svg.append(text(lx + 32, ly + i * 20 + 4, lab, 12, "start"))
    svg.append(svg_close())
    path.write_text("".join(svg), encoding="utf-8")


def fig_scatter(data, path):
    w = h = 620
    xs = [d["人类信息量"] for d in data]
    ys = [d["N=2_bits"] for d in data]
    m = max(max(xs), max(ys)) * 1.1
    svg = [svg_open(w, h)]
    xmap, ymap = draw_axes(svg, w, h, 0, m, 0, m,
                           "人类信息量 (bits/字)", "N=2 信息量 (bits/字)",
                           "人类 vs N=2（对角线以上模型更差）",
                           ml=75, mr=40, mt=55, mb=65)
    svg.append(line(xmap(0), ymap(0), xmap(m), ymap(m), "#888", 1, dash="5,4"))
    for d in data:
        svg.append(circle(xmap(d["人类信息量"]), ymap(d["N=2_bits"]), 3, "#1f77b4", 0.55))
    svg.append(text(xmap(m * 0.75), ymap(m * 0.75) - 8, "y=x", 12, "middle", "#888"))
    svg.append(svg_close())
    path.write_text("".join(svg), encoding="utf-8")


def fig_summary(data, path):
    w, h = 640, 440
    labels = ["人类", "N=2", "N=3"]
    vals = [
        sum(d["人类信息量"] for d in data) / len(data),
        sum(d["N=2_bits"] for d in data) / len(data),
        sum(d["N=3_bits"] for d in data) / len(data),
    ]
    ymax = max(vals) * 1.25
    svg = [svg_open(w, h)]
    xmap, ymap = draw_axes(svg, w, h, 0, 3, 0, ymax, "", "bits/字",
                           "平均信息量/交叉熵对比（越低越好）", x_ticks=[])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (lab, v, c) in enumerate(zip(labels, vals, colors)):
        xa = xmap(i) + 20
        xb = xmap(i + 1) - 20
        yt = ymap(v)
        svg.append(rect(xa, yt, xb - xa, ymap(0) - yt, c))
        svg.append(text((xa + xb) / 2, yt - 6, f"{v:.2f}", 13, "middle"))
        svg.append(text((xa + xb) / 2, ymap(0) + 18, lab, 13, "middle"))
    svg.append(svg_close())
    path.write_text("".join(svg), encoding="utf-8")


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="position_comparison.csv")
    ap.add_argument("--outdir", default="report")
    ap.add_argument("--max-guess", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.csv)
    print("列：", list(rows[0].keys()) if rows else "(空)")

    data = []
    for r in rows:
        pos = int(fnum(r.get("位置")))
        guess = max(1.0, fnum(r.get("人类猜测次数")))
        capped = min(guess, args.max_guess)
        hb = math.log2(capped)
        n2 = fnum(r.get("N=2_bits"))
        n3 = fnum(r.get("N=3_bits"))
        if n2 != n2 or n3 != n3:
            continue
        data.append({
            "位置": pos,
            "前文": (r.get("前文") or "").strip(),
            "真实字": (r.get("真实字") or "").strip(),
            "人类猜测次数": guess,
            "猜测次数_上限": capped,
            "人类信息量": hb,
            "N=2_bits": n2,
            "N=3_bits": n3,
            "delta_N2": n2 - hb,
            "delta_N3": n3 - hb,
            "首猜中": 1 if guess == 1 else 0,
            "人胜N2候选": 1 if (guess <= 2 and n2 >= 9) else 0,
        })
    data.sort(key=lambda d: d["位置"])
    n = len(data)

    mean_guess = sum(d["人类猜测次数"] for d in data) / n
    mean_hb = sum(d["人类信息量"] for d in data) / n
    log2_mean = math.log2(mean_guess)
    mean_n2 = sum(d["N=2_bits"] for d in data) / n
    mean_n3 = sum(d["N=3_bits"] for d in data) / n
    first = sum(d["首猜中"] for d in data) / n
    wins = sorted([d for d in data if d["人胜N2候选"]],
                  key=lambda d: (-d["delta_N2"], -d["delta_N3"]))

    summary_rows = [
        ["位置数", n, "测试字数"],
        ["平均猜测次数", f"{mean_guess:.4f}", f"放弃猜测按{args.max_guess}次计"],
        ["首猜率", f"{first:.4f}", "1次猜中占比"],
        ["人类平均信息量_逐位置平均", f"{mean_hb:.4f}", "主指标 bits/字"],
        ["人类_log2平均猜测次数", f"{log2_mean:.4f}", "非逐位置平均口径"],
        ["人类等效困惑度", f"{2**mean_hb:.4f}", "2^人类平均信息量"],
        ["N2平均交叉熵", f"{mean_n2:.4f}", "bits/字"],
        ["N3平均交叉熵", f"{mean_n3:.4f}", "bits/字"],
        ["N2等效困惑度", f"{2**mean_n2:.4f}", ""],
        ["N3等效困惑度", f"{2**mean_n3:.4f}", ""],
        ["N2减人类平均差值", f"{mean_n2-mean_hb:.4f}", "越大模型越差"],
        ["N3减人类平均差值", f"{mean_n3-mean_hb:.4f}", "越大模型越差"],
        ["人胜N2候选位置数", len(wins), "人类<=2次且N=2 bits>=9"],
    ]
    write_csv(outdir / "summary.csv", ["指标", "值", "说明"], summary_rows)

    keys = ["位置", "前文", "真实字", "人类猜测次数", "猜测次数_上限", "人类信息量",
            "N=2_bits", "N=3_bits", "delta_N2", "delta_N3", "首猜中", "人胜N2候选"]
    write_csv(outdir / "position_enriched.csv", keys,
              [[d[k] for k in keys] for d in data])

    keys2 = ["位置", "前文", "真实字", "人类猜测次数", "人类信息量",
             "N=2_bits", "N=3_bits", "delta_N2", "delta_N3"]
    write_csv(outdir / "top_human_wins.csv", keys2,
              [[d[k] for k in keys2] for d in wins])

    fig_guess_hist(data, outdir / "fig1_guess_hist.svg")
    fig_bits_by_position(data, wins, outdir / "fig2_bits_by_position.svg")
    fig_delta(data, outdir / "fig3_delta.svg")
    fig_scatter(data, outdir / "fig4_scatter.svg")
    fig_summary(data, outdir / "fig5_summary_bar.svg")

    print(f"\n位置数：{n}")
    print(f"平均猜测次数：{mean_guess:.2f}")
    print(f"首猜率：{first:.3f}")
    print(f"人类逐位置平均信息量：{mean_hb:.2f} bits/字")
    print(f"人类 log2(平均猜测次数)：{log2_mean:.2f} bits/字")
    print(f"N=2 平均：{mean_n2:.2f} bits/字")
    print(f"N=3 平均：{mean_n3:.2f} bits/字")
    print(f"人胜候选位置数：{len(wins)}")
    print("\nTop 人胜位置：")
    for d in wins[:15]:
        print(f"  位置{d['位置']:>4} ...{d['前文'][-10:]:>12} -> {d['真实字']}  "
              f"人类{d['人类猜测次数']:.0f}次/{d['人类信息量']:.2f}bits  "
              f"N2={d['N=2_bits']:.2f} N3={d['N=3_bits']:.2f}  Δ2={d['delta_N2']:.2f}")
    print(f"\n输出目录：{outdir.resolve()}")
    print("SVG 可用浏览器或 macOS 预览打开；CSV 可用 Excel / Numbers 打开。")


if __name__ == "__main__":
    main()