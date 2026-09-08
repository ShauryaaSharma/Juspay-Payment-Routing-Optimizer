"""Minimal SVG chart writer -- no matplotlib, no runtime dependencies.

Charts are part of the deliverable, so they should not sit behind a plotting
stack that may or may not be installed. Everything here emits plain SVG that
opens in any browser.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import numpy as np

PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#72B7B2",
    "#EECA3B",
]

_W, _H = 960, 440
_ML, _MR, _MT, _MB = 78, 190, 46, 58


def _downsample(y: np.ndarray, n: int = 900) -> np.ndarray:
    """Block-mean to at most ``n`` points so the SVG stays small and smooth."""
    if y.size <= n:
        return y.astype(float)
    edges = np.linspace(0, y.size, n + 1).astype(int)
    return np.array([y[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _frame(title: str, xlabel: str, ylabel: str, x0: float, x1: float,
           y0: float, y1: float, xticks: int = 7, yticks: int = 5) -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'width="{_W}" height="{_H}" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{_W}" height="{_H}" fill="#ffffff"/>',
        f'<text x="{_ML}" y="26" font-size="17" font-weight="600" fill="#1a1a1a">'
        f"{html.escape(title)}</text>",
    ]
    px0, px1 = _ML, _W - _MR
    py0, py1 = _H - _MB, _MT

    for i in range(yticks + 1):
        frac = i / yticks
        val = y0 + frac * (y1 - y0)
        y = py0 + frac * (py1 - py0)
        parts.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" '
                     f'stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{px0 - 10}" y="{y + 4:.1f}" font-size="11" '
                     f'fill="#666" text-anchor="end">{_fmt(val)}</text>')

    for i in range(xticks + 1):
        frac = i / xticks
        val = x0 + frac * (x1 - x0)
        x = px0 + frac * (px1 - px0)
        parts.append(f'<text x="{x:.1f}" y="{py0 + 20:.1f}" font-size="11" '
                     f'fill="#666" text-anchor="middle">{_fmt(val)}</text>')

    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}" stroke="#999"/>')
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}" stroke="#999"/>')
    parts.append(f'<text x="{(px0 + px1) / 2:.0f}" y="{_H - 14}" font-size="12" '
                 f'fill="#444" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text x="20" y="{(py0 + py1) / 2:.0f}" font-size="12" fill="#444" '
                 f'text-anchor="middle" transform="rotate(-90 20 {(py0 + py1) / 2:.0f})">'
                 f"{html.escape(ylabel)}</text>")
    return parts


def line_chart(
    path: str,
    series: dict[str, np.ndarray],
    title: str,
    xlabel: str = "transaction",
    ylabel: str = "value",
    x_max: float | None = None,
    y_range: tuple[float, float] | None = None,
    shaded: Sequence[tuple[float, float, str]] = (),
    xticks: int = 7,
) -> str:
    """Multi-series line chart. ``shaded`` marks x-ranges (e.g. outages)."""
    data = {k: _downsample(np.asarray(v, dtype=float)) for k, v in series.items()}
    npts = max(len(v) for v in data.values())
    x_max = x_max if x_max is not None else npts

    if y_range is None:
        lo = min(float(np.nanmin(v)) for v in data.values())
        hi = max(float(np.nanmax(v)) for v in data.values())
        pad = (hi - lo) * 0.08 or 0.01
        y_range = (lo - pad, hi + pad)
    y0, y1 = y_range

    parts = _frame(title, xlabel, ylabel, 0, x_max, y0, y1, xticks=xticks)
    px0, px1, py0, py1 = _ML, _W - _MR, _H - _MB, _MT

    def sx(i: int, n: int) -> float:
        return px0 + (i / max(n - 1, 1)) * (px1 - px0)

    def sy(v: float) -> float:
        return py0 + ((v - y0) / (y1 - y0 or 1)) * (py1 - py0)

    for lo, hi, label in shaded:
        xa = px0 + (lo / x_max) * (px1 - px0)
        xb = px0 + (hi / x_max) * (px1 - px0)
        parts.append(f'<rect x="{xa:.1f}" y="{py1}" width="{max(xb - xa, 1):.1f}" '
                     f'height="{py0 - py1}" fill="#E45756" opacity="0.07"/>')
        parts.append(f'<text x="{(xa + xb) / 2:.1f}" y="{py1 - 6}" font-size="10" '
                     f'fill="#B0413E" text-anchor="middle">{html.escape(label)}</text>')

    for idx, (label, values) in enumerate(data.items()):
        n = len(values)
        pts = " ".join(f"{sx(i, n):.1f},{sy(v):.1f}" for i, v in enumerate(values))
        colour = PALETTE[idx % len(PALETTE)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.8" stroke-linejoin="round"/>')
        ly = _MT + 8 + idx * 22
        parts.append(f'<line x1="{px1 + 18}" y1="{ly}" x2="{px1 + 42}" y2="{ly}" '
                     f'stroke="{colour}" stroke-width="3"/>')
        parts.append(f'<text x="{px1 + 48}" y="{ly + 4}" font-size="12" fill="#333">'
                     f"{html.escape(label)}</text>")

    parts.append("</svg>")
    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path


def stacked_area(
    path: str, alloc: np.ndarray, names: list[str], title: str,
    xlabel: str = "transaction", smooth: int = 400,
) -> str:
    """Traffic allocation over time -- shows *how* a router redistributes."""
    k, n = alloc.shape
    cols = min(600, n)
    edges = np.linspace(0, n, cols + 1).astype(int)
    binned = np.array([[alloc[i, a:b].mean() if b > a else 0.0
                        for a, b in zip(edges[:-1], edges[1:])] for i in range(k)])
    binned = binned / np.maximum(binned.sum(axis=0, keepdims=True), 1e-9)

    parts = _frame(title, xlabel, "traffic share", 0, n, 0.0, 1.0)
    px0, px1, py0, py1 = _ML, _W - _MR, _H - _MB, _MT

    def sx(i: int) -> float:
        return px0 + (i / max(cols - 1, 1)) * (px1 - px0)

    def sy(v: float) -> float:
        return py0 + v * (py1 - py0)

    cum = np.zeros(cols)
    for i in range(k):
        lower = cum.copy()
        cum = cum + binned[i]
        top = " ".join(f"{sx(j):.1f},{sy(cum[j]):.1f}" for j in range(cols))
        bot = " ".join(f"{sx(j):.1f},{sy(lower[j]):.1f}" for j in range(cols - 1, -1, -1))
        colour = PALETTE[i % len(PALETTE)]
        parts.append(f'<polygon points="{top} {bot}" fill="{colour}" opacity="0.85"/>')
        ly = _MT + 8 + i * 22
        parts.append(f'<rect x="{px1 + 18}" y="{ly - 7}" width="22" height="11" fill="{colour}"/>')
        parts.append(f'<text x="{px1 + 48}" y="{ly + 3}" font-size="12" fill="#333">'
                     f"{html.escape(names[i])}</text>")

    parts.append("</svg>")
    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path
