"""Model cards: plain-language summaries of one model version (improvements.md U13).

Written for non-modellers. The confidence label and the transferability caveat come first,
then what the map shows, how good the model is, where it extrapolates, what data and choices
went into it, and its limitations. Every sentence is templated from what the version already
stores (metrics, reproducibility block, artifacts); nothing is re-derived. The card is served
as JSON (rendered by the frontend) and as Markdown (Jinja2, like the bulletin).

This module must stay free of the modelling stack: the API process imports it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.domain import BIOCLIM_DESCRIPTIONS, ModelVersionRecord, SpeciesRecord

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
ALGORITHM_NAMES = {
    "maxent": "MaxEnt (presence–background)",
    "ensemble": "a weighted ensemble of MaxEnt, GLM, GAM, random forest and boosted trees",
    "integrated": "an integrated model combining presence-only and survey data (experimental)",
    "ppm": "a point-process model with an explicit observer-effort term",
}


def _pct(v: Any, digits: int = 0) -> str:
    return "n/a" if v is None else f"{float(v) * 100:.{digits}f}%"


def _num(v: Any, digits: int = 2) -> str:
    return "n/a" if v is None else f"{float(v):.{digits}f}"


def _auc_words(v: float | None) -> str:
    if v is None:
        return "could not be measured"
    if v >= 0.9:
        return "excellent"
    if v >= 0.8:
        return "good"
    if v >= 0.7:
        return "fair"
    return "poor"


def _cbi_words(v: float | None) -> str:
    # Hirzel et al. 2006: CBI > 0 means predictions track the presences better than chance.
    if v is None:
        return "could not be measured"
    if v >= 0.7:
        return "strong"
    if v >= 0.4:
        return "moderate"
    if v > 0:
        return "weak"
    return "no better than chance"


def _variable(name: str) -> str:
    desc = BIOCLIM_DESCRIPTIONS.get(name)
    return f"{desc.lower()} ({name})" if desc else name


def model_card(sp: SpeciesRecord, mv: ModelVersionRecord) -> dict[str, Any]:
    m = mv.metrics
    repro = m.get("reproducibility") or {}
    proj = m.get("projection") or {}
    ext = proj.get("extrapolation") or {}
    algo = m.get("algorithm", "maxent")
    name = sp.canonical_name or sp.scientific_name
    model_b = mv.model_type == "B"
    label = m.get("confidence_label", "lower")
    auc_v, cbi_v = m.get("auc_mean"), m.get("cbi_mean")
    labels = m.get("n_records_by_effective_label") or {}
    land = proj.get("land_area_km2") or 0.0
    extrapolated_land = (proj.get("extrapolation_area_km2") or 0.0) / land if land else None

    caveats = [m.get("transferability_caveat", "")]
    if not model_b:
        caveats.append(
            "This model was trained on the native range only (Model A): no introduced-range "
            "records were available. Treat projections outside the native range as lower "
            "confidence."
        )
    caveats.append(
        "Suitability is about climate only. It does not say the species is present, or that "
        "it will arrive: dispersal, land use, competitors and control measures are not modelled."
    )
    if algo in ("integrated", "ppm"):
        caveats.append(
            f"The modelling method ({ALGORITHM_NAMES[algo]}) is experimental in RangeWatch."
        )

    headline = (
        f"{name}: {'standard' if label == 'standard' else 'lower'}-confidence climate "
        f"suitability model, version {mv.model_version}"
        + (" (trained on native and introduced records)" if model_b else " (native range only)")
    )

    importance = sorted(
        (m.get("variable_importance") or {}).items(), key=lambda kv: kv[1], reverse=True
    )[:3]
    sections: list[dict[str, Any]] = []
    cand = proj.get("candidate_area_km2")
    sections.append(
        {
            "id": "what",
            "title": "What the map shows",
            "paragraphs": [
                f"The map scores every land cell from 0 to 1 for how similar its climate is to "
                f"where {name} has been recorded. Cells at or above "
                f"{_num(m.get('threshold'), 3)} (rule: {m.get('threshold_rule', 'n/a')}) are "
                "called suitable.",
                (
                    "Suitable cells outside the native range and away from known records form the "
                    f"candidate expansion zone: about {cand:,.0f} km²."
                    if cand is not None
                    else "No candidate expansion zone was computed."
                ),
            ],
            "facts": [
                {"label": "Model version", "value": str(mv.model_version)},
                {"label": "Trained", "value": mv.trained_ts.strftime("%Y-%m-%d")},
                {"label": "Method", "value": ALGORITHM_NAMES.get(algo, algo)},
            ],
        }
    )
    sections.append(
        {
            "id": "quality",
            "title": "How good is the model?",
            "paragraphs": [
                "The model was tested on regions it did not see during fitting (spatial "
                f"block cross-validation). Its ability to rank recorded places above random "
                f"places is {_auc_words(auc_v)} (AUC {_num(auc_v)}), and how well higher "
                f"scores match more records is {_cbi_words(cbi_v)} (Boyce index {_num(cbi_v)}).",
                *(
                    [
                        "The climate variables that matter most are "
                        + ", ".join(f"{_variable(k)} ({v:.0f}%)" for k, v in importance)
                        + "."
                    ]
                    if importance
                    else []
                ),
            ],
            "facts": [
                {"label": "AUC (spatial CV)", "value": _num(auc_v)},
                {"label": "Boyce index (spatial CV)", "value": _num(cbi_v)},
                {"label": "TSS (spatial CV)", "value": _num(m.get("tss_mean"))},
            ],
        }
    )
    independent = mv.artifacts.get("independent_evaluation") or {}
    if independent:
        lines = []
        for entry in independent.values():
            met = entry.get("metrics") or {}
            parts = [f"{met.get('n_presence', 0)} presences, {met.get('n_absence', 0)} absences"]
            if met.get("sensitivity") is not None:
                parts.append(f"{_pct(met['sensitivity'])} of presences called suitable")
            if met.get("auc_presence_absence") is not None:
                parts.append(f"AUC {_num(met['auc_presence_absence'])}")
            if met.get("cbi") is not None:
                parts.append(f"Boyce index {_num(met['cbi'])}")
            lines.append(f"{entry.get('name')}: " + "; ".join(parts) + ".")
        sections.append(
            {
                "id": "independent",
                "title": "Checked against independent data",
                "paragraphs": lines,
                "facts": [],
            }
        )
    severity = mv.severity or {}
    conf = severity.get("confidence") or {}
    sections.append(
        {
            "id": "extrapolation",
            "title": "Where the map is guesswork",
            "paragraphs": [
                f"In about {_pct(extrapolated_land)} of land the climate lies outside what the "
                "model was trained on (MESS < 0). Scores there are extrapolations: use the "
                "extrapolation overlay on the map before reading anything into them.",
                f"In the candidate expansion zone, {_pct(proj.get('candidate_mess_ok_fraction'))} "
                "of the area is within the training climate"
                + (
                    f", and {_pct(ext.get('candidate_consensus_ok_fraction'))} is flagged by "
                    "none of the five extrapolation diagnostics."
                    if ext.get("candidate_consensus_ok_fraction") is not None
                    else "."
                ),
            ],
            "facts": [
                {"label": "Land extrapolated (MESS)", "value": _pct(extrapolated_land)},
                {"label": "Severity confidence", "value": str(conf.get("level", "n/a"))},
            ],
        }
    )
    effort = m.get("sampling_effort") or {}
    data_paragraphs = [
        f"{m.get('n_presence', 'n/a')} presences (after cleaning and spatial thinning) from "
        f"{sum(int(v) for v in labels.values()):,} usable records: "
        + ", ".join(f"{v:,} {k}" for k, v in labels.items())
        + ".",
    ]
    quality = repro.get("data_quality") or {}
    if quality:
        data_paragraphs.append(
            f"{quality.get('n_excluded', 0):,} records failed data-quality tests and were left "
            "out (they stay in the record store)."
        )
    if effort.get("status") == "available":
        data_paragraphs.append(
            f"{_pct(effort.get('candidate_unsampled_fraction'))} of the candidate zone has no "
            "records of related species at all: there, a missing record says nothing about "
            "whether the species could live there."
        )
    region = repro.get("study_region")
    if region:
        data_paragraphs.append(
            f"The model covers a study region only ({', '.join(f'{v:g}' for v in region['bbox'])}"
            f", resolution {region['resolution']})."
        )
    sections.append(
        {
            "id": "data",
            "title": "What data went in",
            "paragraphs": data_paragraphs,
            "facts": [
                {"label": "Climate data", "value": str(repro.get("bioclim_version_used", "n/a"))},
                {"label": "GBIF download", "value": repro.get("gbif_download_doi") or "n/a"},
                {"label": "Background", "value": str(m.get("background_method", "n/a"))},
            ],
        }
    )
    if severity:
        sections.append(
            {
                "id": "severity",
                "title": "Severity index",
                "paragraphs": [
                    f"Severity {severity.get('score_0_100', 'n/a')} / 100 "
                    f"({severity.get('category', 'n/a')}). It is a weighted sum of four "
                    "documented parts — "
                    + ", ".join(
                        f"{k.replace('_', ' ')} {_pct(v)}"
                        for k, v in (severity.get("weights") or {}).items()
                    )
                    + " — not a prediction of damage.",
                    *([conf["note"]] if conf.get("note") else []),
                ],
                "facts": [],
            }
        )
    citation = repro.get("gbif_citation", "")
    return {
        "taxon_key": sp.taxon_key,
        "scientific_name": sp.scientific_name,
        "model_version": mv.model_version,
        "generated_ts": datetime.now(UTC).isoformat(),
        "confidence_label": label,
        "model_type": mv.model_type,
        "headline": headline,
        "transferability_caveat": m.get("transferability_caveat", ""),
        "caveats": [c for c in caveats if c],
        "sections": sections,
        "citation": citation,
        "source": "Generated from the metrics and lineage stored with this model version.",
    }


def model_card_markdown(card: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(enabled_extensions=("html",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("model_card.md.j2").render(card=card)
