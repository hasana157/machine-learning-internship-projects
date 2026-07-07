"""
persona_generator.py
----------------------
Translates KMeans cluster statistics into business-readable customer
personas (e.g. "Platinum", "At-Risk") with recommended marketing actions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jinja2 import Template

TEMPLATE_STR = """Cluster {{ cluster.id }}: {{ cluster.persona_name }}
=====================
Size: {{ cluster.count }} customers ({{ "%.1f"|format(cluster.pct) }}%)

RFM Profile:
- Recency: {{ cluster.r_rank }} (avg {{ "%.1f"|format(cluster.r_mean) }} days)
- Frequency: {{ cluster.f_rank }} (avg {{ "%.1f"|format(cluster.f_mean) }} purchases)
- Monetary: {{ cluster.m_rank }} (avg {{ "%.2f"|format(cluster.m_mean) }})

Key Insight: {{ cluster.insight }}

Recommended Actions:
{% for action in cluster.actions %}- {{ action }}
{% endfor %}"""


def _rank(value: float, q1: float, q2: float) -> str:
    """Rank a value against the 33rd/66th percentile split."""
    if value <= q1:
        return "low"
    if value <= q2:
        return "medium"
    return "high"


def _persona_name(r_rank: str, f_rank: str, m_rank: str) -> str:
    # Lower Recency rank ("low") = MORE recent, since a low day-count is good.
    if f_rank == "high" and m_rank == "high":
        return "Platinum"
    if f_rank in ("medium", "high") and m_rank in ("medium", "high") and r_rank != "high":
        return "Gold"
    if r_rank == "high" or f_rank == "low":
        return "At-Risk"
    return "Silver"


def _recommended_actions(persona_name: str) -> list:
    mapping = {
        "Platinum": [
            "Enroll in VIP loyalty tier with premium support",
            "Offer early access to new products and limited drops",
            "Assign a dedicated account contact for high-touch service",
        ],
        "Gold": [
            "Send targeted loyalty rewards and bundle offers",
            "Upsell complementary products based on purchase history",
            "Invite to a mid-tier loyalty program",
        ],
        "Silver": [
            "Run engagement nurture emails to build purchase frequency",
            "Offer a small discount to encourage a second purchase",
            "Monitor for early drift into the At-Risk segment",
        ],
        "At-Risk": [
            "Launch a win-back campaign with a time-limited incentive",
            "Send a short exit survey to understand disengagement",
            "Deprioritize paid acquisition spend on this segment",
        ],
    }
    return mapping.get(persona_name, ["Review manually; no rule matched."])


def _insight(persona_name: str, r_mean: float, f_mean: float, m_mean: float) -> str:
    insights = {
        "Platinum": (
            f"These customers purchase frequently (~{f_mean:.0f}x) and spend the most "
            f"(~{m_mean:,.0f}), and bought recently (~{r_mean:.0f} days ago) -- your highest-value segment."
        ),
        "Gold": (
            f"Solid, engaged customers with moderate frequency (~{f_mean:.0f}x) and spend "
            f"(~{m_mean:,.0f}); good candidates for loyalty upsells."
        ),
        "Silver": (
            f"Occasional buyers (~{f_mean:.0f}x/period, ~{m_mean:,.0f} spend) who could grow "
            f"into Gold with the right nudge."
        ),
        "At-Risk": (
            f"Haven't purchased in ~{r_mean:.0f} days with low frequency (~{f_mean:.0f}x) -- "
            f"at risk of churning without intervention."
        ),
    }
    return insights.get(persona_name, "No specific insight rule matched this cluster.")


class PersonaGenerator:
    """Compute per-cluster statistics and generate narrative personas."""

    def __init__(self, template: Optional[str] = None):
        self.template = Template(template or TEMPLATE_STR)
        self.personas_: Optional[list] = None

    def fit(self, rfm: pd.DataFrame, labels: np.ndarray) -> "PersonaGenerator":
        """
        Parameters
        ----------
        rfm : DataFrame with columns CustomerID, Recency, Frequency, Monetary
        labels : cluster label per row of `rfm` (same order/length)
        """
        df = rfm.copy()
        df["cluster"] = labels
        total = len(df)

        # Global quantile thresholds per dimension for Low/Medium/High ranking
        r_q1, r_q2 = df["Recency"].quantile([0.33, 0.66])
        f_q1, f_q2 = df["Frequency"].quantile([0.33, 0.66])
        m_q1, m_q2 = df["Monetary"].quantile([0.33, 0.66])

        personas = []
        for cluster_id, group in df.groupby("cluster"):
            r_mean, f_mean, m_mean = group["Recency"].mean(), group["Frequency"].mean(), group["Monetary"].mean()
            r_std, f_std, m_std = group["Recency"].std(ddof=0), group["Frequency"].std(ddof=0), group["Monetary"].std(ddof=0)

            r_rank = _rank(r_mean, r_q1, r_q2)
            f_rank = _rank(f_mean, f_q1, f_q2)
            m_rank = _rank(m_mean, m_q1, m_q2)

            name = _persona_name(r_rank, f_rank, m_rank)
            actions = _recommended_actions(name)
            insight = _insight(name, r_mean, f_mean, m_mean)

            personas.append({
                "id": int(cluster_id),
                "persona_name": name,
                "count": int(len(group)),
                "pct": float(len(group) / total * 100),
                "r_mean": float(r_mean), "f_mean": float(f_mean), "m_mean": float(m_mean),
                "r_std": float(r_std), "f_std": float(f_std), "m_std": float(m_std),
                "r_rank": r_rank, "f_rank": f_rank, "m_rank": m_rank,
                "actions": actions,
                "insight": insight,
            })

        # Sort by descending monetary value for a sensible display order
        personas.sort(key=lambda p: p["m_mean"], reverse=True)
        self.personas_ = personas
        return self

    def generate_personas(self) -> list:
        if self.personas_ is None:
            raise RuntimeError("Call fit() before generate_personas().")
        return self.personas_

    def to_json(self, path: str | Path) -> None:
        if self.personas_ is None:
            raise RuntimeError("Call fit() before to_json().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.personas_, f, indent=2)

    def to_markdown(self) -> str:
        if self.personas_ is None:
            raise RuntimeError("Call fit() before to_markdown().")
        parts = []
        for p in self.personas_:
            class _Obj:
                pass
            obj = _Obj()
            for k, v in p.items():
                setattr(obj, k, v)
            parts.append(self.template.render(cluster=obj))
        return "\n\n".join(parts)
