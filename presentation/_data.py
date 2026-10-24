"""Recorded aggregate results used by the public LecTrace presentation."""

INTERNAL_RESULTS = [
    {"model": "B28", "accuracy": 0.8622, "auroc": 0.9444},
    {"model": "B64 plain", "accuracy": 0.9054, "auroc": 0.9751},
]

EXTERNAL_RESULTS = [
    {"dataset": "RSNA", "model": "B64 plain", "accuracy": 0.8057, "auroc": 0.8720},
    {"dataset": "RSNA", "model": "B64 AC-JSD", "accuracy": 0.8093, "auroc": 0.9012},
    {"dataset": "CheXpert", "model": "B64 plain", "accuracy": 0.6224, "auroc": 0.7212},
    {"dataset": "CheXpert", "model": "B64 AC-JSD", "accuracy": 0.8392, "auroc": 0.9310},
]

XAI_RESULTS = [
    {"measure": "In-box Grad-CAM energy", "AC-JSD result": "10.65%", "95% interval": "9.42%–11.92%"},
    {"measure": "Relative energy gain", "AC-JSD result": "+19.6%", "95% interval": "12.8%–27.5%"},
    {"measure": "Pointing-hit rate", "AC-JSD result": "8.67%", "95% interval": "5.61%–11.86%"},
    {"measure": "Mean opacity-box area", "AC-JSD result": "12.26%", "95% interval": "11.19%–13.35%"},
]


def metric_chart(title, values, x, color):
    """Return a compact grouped-bar Vega-Lite specification."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": 520,
        "height": 250,
        "data": {"values": values},
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": x, "type": "nominal", "axis": {"labelAngle": 0}},
            "xOffset": {"field": color},
            "y": {"field": "value", "type": "quantitative", "scale": {"domain": [0.5, 1.0]}},
            "color": {"field": color, "type": "nominal"},
            "tooltip": [
                {"field": x, "type": "nominal"},
                {"field": color, "type": "nominal"},
                {"field": "value", "type": "quantitative", "format": ".4f"},
            ],
        },
    }
