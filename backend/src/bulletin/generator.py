"""Automated per-species bulletin: Jinja2 → HTML → WeasyPrint PDF (server-rendered)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rasterio.enums import Resampling

from src.bulletin import charts
from src.config import Settings
from src.domain import BIOCLIM_DESCRIPTIONS, ModelVersionRecord, NativeRange, SpeciesRecord
from src.features.raster_sampler import read_raster_overview
from src.modeling.maxent_trainer import TRANSFERABILITY_CAVEAT
from src.modeling.native_range import effective_range_labels
from src.modeling.severity_index import COMPONENTS, methods_footnote
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository

log = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).parent / "templates"

REFERENCES = [
    "Phillips, S.J., Anderson, R.P. & Schapire, R.E. (2006). Maximum entropy modeling of "
    "species geographic distributions. Ecological Modelling 190: 231–259.",
    "Anderson, C.B. (2023). elapid: Species distribution modeling tools for Python. Journal "
    "of Open Source Software 8(84): 4930.",
    "Fick, S.E. & Hijmans, R.J. (2017). WorldClim 2: new 1-km spatial resolution climate "
    "surfaces for global land areas. International Journal of Climatology 37: 4302–4315.",
    "Elith, J., Kearney, M. & Phillips, S. (2010). The art of modelling range-shifting "
    "species. Methods in Ecology and Evolution 1: 330–342 (MESS).",
    "Mesgaran, M.B., Cousens, R.D. & Webber, B.L. (2014). Here be dragons: a tool for "
    "quantifying novelty due to covariate range and correlation change when projecting species "
    "distribution models. Diversity and Distributions 20: 1147–1159 (exDet).",
    "Owens, H.L. et al. (2013). Constraints on interpretation of ecological niche models by "
    "limited environmental ranges on calibration areas. Ecological Modelling 263: 10–18 (MOP).",
    "Cobos, M.E., Owens, H.L., Soberón, J. & Peterson, A.T. (2024). Detailed multivariate "
    "comparisons of environments with mobility oriented parity. Frontiers of Biogeography 17: "
    "e132916 (mop).",
    "Velazco, S.J.E., Rose, M.B., De Marco Jr., P., Regan, H.M. & Franklin, J. (2024). How far "
    "can I extrapolate my species distribution model? Exploring Shape, a novel method. "
    "Ecography 2024: e06992 (Shape).",
    "Meyer, H. & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of "
    "applicability of spatial prediction models. Methods in Ecology and Evolution 12: "
    "1620–1633 (AOA).",
    "Hirzel, A.H. et al. (2006). Evaluating the ability of habitat suitability models to "
    "predict species presences. Ecological Modelling 199: 142–152 (Continuous Boyce Index).",
    "Broennimann, O. et al. (2015). Ecological niche transferability using invasive species "
    "as a case study. PLoS ONE 10(3): e0119891.",
    "Liu, C., Wolter, C., Xian, W. & Jeschke, J.M. (2020/2022). How well do species "
    "distribution models predict occurrences in exotic ranges? Global Ecology and "
    "Biogeography.",
    "Manzoor, S.A., Griffiths, G. & Lukac, M. (2018). Species distribution model "
    "transferability and model grain size. Scientific Reports 8: 7168.",
    "Beaumont, L.J. et al. (2009/2013). Improving transferability of introduced species' "
    "distribution models (combined native + invaded range calibration).",
    "Kass, J.M. et al. (2023). wallace 2: a shiny app for modeling species niches and "
    "distributions redesigned to facilitate expansion via module contributions. Ecography.",
    "GEO BON — BON in a Box, Species Distribution Models indicator pipeline.",
]


class BulletinGenerator:
    def __init__(self, repo: Repository, artifacts: ArtifactStore, settings: Settings) -> None:
        self.repo = repo
        self.artifacts = artifacts
        self.settings = settings
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "j2"]),
        )
        self.env.filters["num"] = lambda v, d=0: f"{v:,.{d}f}" if v is not None else "–"

    # ------------------------------------------------------------------ public
    def render_html(self, taxon_key: int, model_version: int) -> str:
        return self.env.get_template("bulletin.html.j2").render(
            **self.build_context(taxon_key, model_version)
        )

    def generate(self, taxon_key: int, model_version: int) -> dict[str, str]:
        html = self.render_html(taxon_key, model_version)
        vdir_uri = str(self.artifacts.version_dir(taxon_key, model_version))
        vdir = self.artifacts.local_path(vdir_uri)
        vdir.mkdir(parents=True, exist_ok=True)
        html_path = vdir / "bulletin.html"
        html_path.write_text(html, encoding="utf-8")
        out = {"html": f"{vdir_uri}/bulletin.html"}
        from weasyprint import HTML  # imported lazily: needs system Pango/Cairo libraries

        HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(vdir / "bulletin.pdf")
        out["pdf"] = f"{vdir_uri}/bulletin.pdf"
        return out

    def build_context(self, taxon_key: int, model_version: int) -> dict[str, Any]:
        species = self.repo.get_species(taxon_key)
        mv = self.repo.get_model_version(taxon_key, model_version)
        nr = self.repo.get_native_range(taxon_key)
        if species is None or mv is None or nr is None:
            raise ValueError(f"Missing species/model/native range for {taxon_key} v{model_version}")
        m = mv.metrics
        sev = mv.severity or {}

        occ = self.repo.load_occurrences(taxon_key)
        occ = occ.assign(effective_label=effective_range_labels(occ, nr.geometry))
        years = pd.to_datetime(occ["event_date"], errors="coerce").dt.year
        timeline = (
            occ.assign(year=years)
            .dropna(subset=["year"])
            .groupby(["year", "effective_label"])
            .size()
            .reset_index(name="n")
        )

        suit, t_suit = read_raster_overview(
            self.artifacts.local_path(mv.artifacts["suitability"]), max_width=1800
        )
        zones, t_zones = read_raster_overview(
            self.artifacts.local_path(mv.artifacts["zones"]),
            max_width=1800,
            mask_fn=lambda a: a,
            resampling=Resampling.nearest,
        )
        mess_arr, _ = read_raster_overview(
            self.artifacts.local_path(mv.artifacts["mess"]), max_width=1800
        )
        consensus_fig = None
        if "consensus" in mv.artifacts:
            consensus, t_cons = read_raster_overview(
                self.artifacts.local_path(mv.artifacts["consensus"]),
                max_width=1800,
                mask_fn=lambda a: a,
                resampling=Resampling.nearest,
            )
            consensus_fig = charts.consensus_map(consensus, t_cons, nr.geometry)
        figures = {
            "native_map": charts.native_range_map(suit, t_suit, m["threshold"], occ, nr.geometry),
            "global_map": charts.global_projection_map(zones, mess_arr, t_zones, nr.geometry),
            "consensus_map": consensus_fig,
            "gauge": charts.severity_gauge(sev) if sev else None,
            "importance": charts.variable_importance_chart(m["variable_importance"]),
            "timeline": charts.timeline_chart(timeline),
        }
        return {
            "species": species,
            "mv": mv,
            "m": m,
            "sev": sev,
            "nr": nr,
            "native_status": native_range_status_text(nr),
            "repro": m["reproducibility"],
            "figures": figures,
            "summary": self.executive_summary(species, mv),
            "outlook": self.expansion_outlook(species, mv, timeline),
            "extrapolation": extrapolation_context(m["projection"].get("extrapolation")),
            "caveat": TRANSFERABILITY_CAVEAT,
            "severity_footnote": methods_footnote(sev) if sev else "",
            "severity_components": COMPONENTS,
            "bioclim_names": BIOCLIM_DESCRIPTIONS,
            "references": REFERENCES,
            "generated": datetime.now(UTC).strftime("%d %B %Y, %H:%M UTC"),
        }

    # --------------------------------------------------------------- narrative
    @staticmethod
    def _model_phrase(mv: ModelVersionRecord) -> str:
        if mv.model_type == "B":
            return "combined native + invaded-range model (Model B)"
        return "native-range-only model (Model A, lower confidence)"

    def executive_summary(self, species: SpeciesRecord, mv: ModelVersionRecord) -> str:
        m, sev = mv.metrics, mv.severity or {}
        p = m["projection"]
        top = sorted(m["variable_importance"].items(), key=lambda kv: -kv[1])[:3]
        top_txt = ", ".join(
            f"{BIOCLIM_DESCRIPTIONS.get(k, k).lower()} ({k.upper()}, {v:.0f}%)" for k, v in top
        )
        conf = sev.get("confidence", {})
        return (
            f"{species.scientific_name} scores {sev.get('score_0_100', float('nan')):.0f}/100 on "
            f"the invasion severity index ({sev.get('category', 'n/a')}; confidence: "
            f"{conf.get('level', 'n/a')}). The {self._model_phrase(mv)} identifies "
            f"{p['candidate_area_km2']:,.0f} km² of climatically suitable land outside the "
            f"confirmed native range that is not yet known to be occupied (candidate "
            f"invasion/expansion zones), plus {p['established_outside_area_km2']:,.0f} km² "
            f"suitable and already occupied outside the native range. "
            f"{100 * p['candidate_mess_ok_fraction']:.0f}% of the candidate area lies within the "
            f"training climate space (MESS ≥ 0)"
            f"{self._consensus_phrase(p.get('extrapolation'))}. "
            f"Top contributing predictors: {top_txt}."
        )

    @staticmethod
    def _consensus_phrase(ext: dict[str, Any] | None) -> str:
        if not ext:
            return ""
        return (
            f"; {100 * ext['candidate_consensus_ok_fraction']:.0f}% is flagged by none of the "
            f"five extrapolation diagnostics (MESS, exDet, MOP, Shape, AOA) and "
            f"{100 * ext['candidate_consensus_majority_fraction']:.0f}% by at least three"
        )

    def expansion_outlook(
        self, species: SpeciesRecord, mv: ModelVersionRecord, timeline: pd.DataFrame
    ) -> dict[str, Any]:
        p = mv.metrics["projection"]
        intro = timeline[timeline["effective_label"] == "introduced"]
        now = datetime.now(UTC).year
        recent = int(intro.loc[intro["year"] > now - 5, "n"].sum())
        prior = int(intro.loc[(intro["year"] > now - 10) & (intro["year"] <= now - 5), "n"].sum())
        if prior == 0 and recent == 0:
            trend = "No dated records outside the native range in the last ten years."
        elif prior == 0:
            trend = (
                f"{recent:,} records outside the native range in the last five years "
                "(none in the five years before)."
            )
        else:
            change = (recent - prior) / prior * 100
            trend = (
                f"{recent:,} records outside the native range in the last five years vs. "
                f"{prior:,} in the previous five ({change:+.0f}%). Record counts also "
                "reflect growing observer effort, not only spread."
            )
        return {
            "trend": trend,
            "regions": p.get("top_candidate_regions", []),
            "limiting": [
                {**d, "name": BIOCLIM_DESCRIPTIONS.get(d["variable"], d["variable"])}
                for d in p.get("top_limiting_variables", [])
            ],
            "spread_rate": (mv.severity or {}).get("inputs", {}).get("new_cells_per_year"),
        }


def extrapolation_context(ext: dict[str, Any] | None) -> dict[str, Any] | None:
    """The advanced-diagnostics section: per-method verdicts plus the variables driving novel
    combinations (None for models trained before the diagnostics existed)."""
    if not ext:
        return None
    return {
        **ext,
        "combinatorial": [
            {**d, "name": BIOCLIM_DESCRIPTIONS.get(d["variable"], d["variable"])}
            for d in ext.get("top_combinatorial_variables", [])
        ],
    }


def native_range_status_text(nr: NativeRange) -> str:
    ts = nr.confirmed_ts.strftime("%Y-%m-%d") if nr.confirmed_ts else "not confirmed"
    return f"{nr.status} ({nr.source}; confirmed {ts})"
