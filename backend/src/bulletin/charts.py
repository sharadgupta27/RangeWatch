"""Static, print-quality figures for the PDF bulletin (server-rendered with matplotlib)."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Wedge  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from src.modeling.severity_index import COMPONENTS  # noqa: E402

INK = "#1f2933"
MUTED = "#6b7785"
NATIVE = "#2a9d8f"
INTRODUCED = "#e76f51"
UNKNOWN = "#8d99ae"
ZONE_COLORS = ["#e9ecef", "#2a9d8f", "#f4a261", "#d62828"]  # zones 0..3
COMPONENT_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _to_data_uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _extent(arr: np.ndarray, transform: Affine) -> tuple[float, float, float, float]:
    h, w = arr.shape
    return (transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f)


def _plot_polygon(ax: plt.Axes, geojson: dict[str, Any], **kw: Any) -> None:
    geom = shape(geojson)
    for poly in getattr(geom, "geoms", [geom]):
        x, y = poly.exterior.xy
        ax.plot(x, y, **kw)


def native_range_map(
    suit: np.ndarray,
    transform: Affine,
    threshold: float,
    occ: pd.DataFrame,
    native_geojson: dict[str, Any],
) -> str:
    """Map panel 1 — native-range occurrences over suitability with the threshold contour."""
    minx, miny, maxx, maxy = shape(native_geojson).bounds
    padx, pady = max(5, (maxx - minx) * 0.25), max(5, (maxy - miny) * 0.25)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ext = _extent(suit, transform)
    land = np.where(np.isnan(suit), np.nan, 0)
    ax.imshow(land, extent=ext, cmap=ListedColormap(["#f1f3f5"]), interpolation="nearest")
    im = ax.imshow(suit, extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ys = np.linspace(ext[3], ext[2], suit.shape[0])
    xs = np.linspace(ext[0], ext[1], suit.shape[1])
    ax.contour(
        xs, ys, np.nan_to_num(suit, nan=0), levels=[threshold], colors="white", linewidths=0.6
    )
    _plot_polygon(ax, native_geojson, color=INK, lw=1.0, ls="--")
    for label, color in (("native", NATIVE), ("introduced", INTRODUCED), ("unknown", UNKNOWN)):
        sub = occ[occ["effective_label"] == label]
        if len(sub):
            ax.scatter(
                sub["longitude"],
                sub["latitude"],
                s=6,
                c=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=0.9,
                label=f"{label} ({len(sub):,})",
            )
    ax.set_xlim(max(-180, minx - padx), min(180, maxx + padx))
    ax.set_ylim(max(-90, miny - pady), min(90, maxy + pady))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=6, frameon=True, markerscale=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Suitability (cloglog)")
    return _to_data_uri(fig)


def global_projection_map(
    zones: np.ndarray, mess_arr: np.ndarray, transform: Affine, native_geojson: dict[str, Any]
) -> str:
    """Map panel 2 — global zones with MESS extrapolation hatched (always shown together)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ext = _extent(zones, transform)
    z = np.where(zones == 255, np.nan, zones.astype("float32"))
    ax.set_facecolor("#dbe9f4")
    ax.imshow(
        z,
        extent=ext,
        cmap=ListedColormap(ZONE_COLORS),
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest",
    )
    ys = np.linspace(ext[3], ext[2], mess_arr.shape[0])
    xs = np.linspace(ext[0], ext[1], mess_arr.shape[1])
    extrap = np.where(np.isnan(mess_arr), 0, (mess_arr < 0).astype(float))
    ax.contourf(xs, ys, extrap, levels=[0.5, 1.5], colors="none", hatches=["////"])
    _plot_polygon(ax, native_geojson, color=INK, lw=0.8, ls="--")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Patch(color=ZONE_COLORS[1], label="Suitable – native range"),
        Patch(color=ZONE_COLORS[2], label="Suitable – established outside native range"),
        Patch(color=ZONE_COLORS[3], label="Candidate invasion/expansion zone"),
        Patch(facecolor="white", edgecolor=INK, hatch="////", label="MESS < 0 (extrapolation)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6, frameon=True)
    return _to_data_uri(fig)


CONSENSUS_COLORS = ["#4393c3", "#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]


def consensus_map(consensus: np.ndarray, transform: Affine, native_geojson: dict[str, Any]) -> str:
    """How many of the five extrapolation diagnostics flag each land cell (0–5)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ext = _extent(consensus, transform)
    c = np.where(consensus == 255, np.nan, consensus.astype("float32"))
    ax.set_facecolor("#dbe9f4")
    ax.imshow(
        c,
        extent=ext,
        cmap=ListedColormap(CONSENSUS_COLORS),
        vmin=-0.5,
        vmax=5.5,
        interpolation="nearest",
    )
    _plot_polygon(ax, native_geojson, color=INK, lw=0.8, ls="--")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Patch(color=col, label="none" if n == 0 else f"{n} of 5")
        for n, col in enumerate(CONSENSUS_COLORS)
    ]
    ax.legend(
        handles=handles,
        title="Diagnostics flagging extrapolation",
        title_fontsize=6,
        loc="lower left",
        fontsize=6,
        frameon=True,
        ncol=2,
    )
    return _to_data_uri(fig)


def severity_gauge(severity: dict[str, Any]) -> str:
    """Radial gauge: the arc is split into weighted component contributions."""
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.28, color="#e9ecef"))
    start = 180.0
    for comp, color in zip(COMPONENTS, COMPONENT_COLORS, strict=True):
        sweep = severity["contributions"][comp] * 180
        ax.add_patch(Wedge((0, 0), 1.0, start - sweep, start, width=0.28, color=color))
        start -= sweep
    ax.text(
        0,
        0.12,
        f"{severity['score_0_100']:.0f}",
        ha="center",
        va="center",
        fontsize=22,
        color=INK,
        weight="bold",
    )
    ax.text(
        0, -0.12, severity["category"].upper(), ha="center", va="center", fontsize=8, color=MUTED
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.25, 1.05)
    handles = [
        Patch(
            color=c,
            label=f"{k.replace('_', ' ')}: {severity['components'][k]:.2f} "
            f"× w={severity['weights'][k]:.2f}",
        )
        for k, c in zip(COMPONENTS, COMPONENT_COLORS, strict=True)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6,
        frameon=False,
        ncol=1,
    )
    return _to_data_uri(fig)


def variable_importance_chart(importance: dict[str, float]) -> str:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(3.6, 0.25 * len(items) + 0.6))
    ax.barh([k.upper() for k, _ in items], [v for _, v in items], color="#2a9d8f")
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=6, color=MUTED)
    ax.set_xlabel("Permutation importance (% of total)")
    return _to_data_uri(fig)


def timeline_chart(timeline: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 1.8))
    if not timeline.empty:
        piv = timeline.pivot_table(
            index="year", columns="effective_label", values="n", aggfunc="sum", fill_value=0
        )
        piv = piv[piv.index >= piv.index.max() - 40]
        bottom = np.zeros(len(piv))
        for label, color in (("native", NATIVE), ("introduced", INTRODUCED)):
            if label in piv:
                ax.bar(piv.index, piv[label], bottom=bottom, color=color, label=label, width=0.8)
                bottom += piv[label].to_numpy()
        ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylabel("Records / year")
    return _to_data_uri(fig)
