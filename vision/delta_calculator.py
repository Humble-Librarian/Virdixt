"""Deterministic math module for chart-to-text conversion.

Parses DePlot linearized table output, computes percentage deltas,
and generates concessive conjunction sentences for FinBERT injection.
Pure Python, zero ML dependencies, microsecond execution.
"""


def parse_deplot_table(raw_string: str) -> list[dict]:
    """Parses DePlot's linearized output into structured data.

    DePlot outputs tables as pipe-separated values with <0x0A> line breaks.
    Example input: "Metric | Q1 | Q2<0x0A>Revenue | 45.0 | 48.2"
    """
    data = []
    lines = raw_string.strip().split("<0x0A>")
    if not lines or len(lines) < 2:
        return data

    headers = [h.strip() for h in lines[0].split("|")]
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) == len(headers):
            row = dict(zip(headers, cells))
            data.append(row)
    return data


def compute_deltas(data: list[dict]) -> list[dict]:
    """Computes percentage changes between first and last periods for each metric."""
    if not data or len(data) < 2:
        return []

    headers = list(data[0].keys())
    if len(headers) < 2:
        return []

    deltas = []
    first_row = data[0]
    last_row = data[-1]

    for col in headers[1:]:
        try:
            val_first = float(
                first_row[col].replace(',', '').replace('$', '').replace('%', '')
            )
            val_last = float(
                last_row[col].replace(',', '').replace('$', '').replace('%', '')
            )
            if val_first != 0:
                pct_change = ((val_last - val_first) / abs(val_first)) * 100
                deltas.append({
                    "metric": col,
                    "delta": pct_change,
                    "first": val_first,
                    "last": val_last,
                })
        except ValueError:
            pass

    return deltas


def format_concessive_sentence(deltas: list[dict]) -> str:
    """Converts computed deltas into concessive conjunction sentences.

    If two metrics diverge (one up, one down), uses "Although X grew, Y declined".
    If both trend the same direction, uses "Both X and Y grew/declined".
    """
    if len(deltas) < 2:
        if len(deltas) == 1:
            d = deltas[0]
            direction = "grew" if d["delta"] > 0 else "declined"
            return f"{d['metric']} {direction} {abs(d['delta']):.1f}%."
        return ""

    d1 = deltas[0]
    d2 = deltas[1]

    dir1 = "grew" if d1["delta"] > 0 else "declined"
    dir2 = "grew" if d2["delta"] > 0 else "declined"

    if (d1["delta"] > 0) != (d2["delta"] > 0):
        return (
            f"Although {d1['metric']} {dir1} {abs(d1['delta']):.1f}%, "
            f"{d2['metric']} {dir2} {abs(d2['delta']):.1f}%."
        )
    else:
        return (
            f"Both {d1['metric']} and {d2['metric']} {dir1} by "
            f"{abs(d1['delta']):.1f}% and {abs(d2['delta']):.1f}% respectively."
        )
