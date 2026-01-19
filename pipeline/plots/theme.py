# Ten plik ustawia wspolne parametry estetyczne dla Plotly w pipeline.
from __future__ import annotations

from typing import Any, Dict

import plotly.graph_objects as go


def apply_plotly_theme(fig: go.Figure) -> None:
    """Apply shared Plotly styling for pipeline figures."""
    layout: Dict[str, Any] = {
        "template": "plotly_white",
        "font": {"family": "Arial", "size": 13},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "legend": {"orientation": "h"},
    }
    fig.update_layout(**layout)
