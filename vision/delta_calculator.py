"""Deterministic math module for chart-to-text conversion.

Parses DePlot linearized table output, computes percentage deltas,
and generates concessive conjunction sentences for FinBERT injection.
Pure Python, zero ML dependencies, microsecond execution.
"""
import re


def parse_deplot_table(raw_string: str) -> list[dict]:
    """Parses DePlot's linearized output into structured data.

    DePlot outputs tables as pipe-separated values with <0x0A> line breaks.
    Example input: "Metric | Q1 | Q2<0x0A>Revenue | $45M | $48M<0x0A>Margin | 18% | 11%"
    """
    data = []
    lines = [line.strip() for line in raw_string.replace("\n", "<0x0A>").split("<0x0A>") if line.strip()]
    if not lines or len(lines) < 2:
        return data

    headers = [h.strip() for h in lines[0].split("|")]
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) == len(headers):
            row = dict(zip(headers, cells))
            data.append(row)
    return data


def _clean_numeric(val_str: str) -> float | None:
    """Extracts floating point number from financial strings like '$45.5M', '18%', '4,200'."""
    if not val_str:
        return None
    cleaned = val_str.replace(',', '').replace('$', '').replace('%', '').strip()
    # Handle multipliers like M, B, K if needed
    multiplier = 1.0
    if cleaned.upper().endswith('B'):
        multiplier = 1e9
        cleaned = cleaned[:-1]
    elif cleaned.upper().endswith('M'):
        multiplier = 1e6
        cleaned = cleaned[:-1]
    elif cleaned.upper().endswith('K'):
        multiplier = 1e3
        cleaned = cleaned[:-1]
    
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if match:
        try:
            return float(match.group(0)) * (multiplier if multiplier != 1.0 else 1.0)
        except ValueError:
            return None
    return None


def compute_deltas(data: list[dict]) -> list[dict]:
    """Computes percentage changes between first and last periods for each metric.

    Supports both formats:
    1. Metric-as-row (standard DePlot format):
       Metric | Q1 | Q2
       Revenue | $45M | $48M
       Margin | 18% | 11%
    2. Period-as-row (transposed table):
       Period | Revenue | Margin
       Q1 | 45 | 18
       Q2 | 48 | 11
    """
    if not data:
        return []

    headers = list(data[0].keys())
    if len(headers) < 2:
        return []

    first_col_header = headers[0].lower()
    deltas = []

    # Check if table is Period-as-row
    is_period_rows = any(kw in first_col_header for kw in ["period", "quarter", "year", "date", "q1", "q2", "q3", "q4", "202"])
    
    if is_period_rows and len(data) >= 2:
        # Transposed table: each column in headers[1:] is a metric, row 0 is period 1, row -1 is period N
        first_row = data[0]
        last_row = data[-1]
        for col in headers[1:]:
            val_first = _clean_numeric(first_row.get(col, ""))
            val_last = _clean_numeric(last_row.get(col, ""))
            if val_first is not None and val_last is not None and val_first != 0:
                pct_change = ((val_last - val_first) / abs(val_first)) * 100.0
                deltas.append({
                    "metric": col,
                    "delta": pct_change,
                    "first": val_first,
                    "last": val_last,
                })
    else:
        # Standard DePlot format: each row is a metric, headers[1] is start period, headers[-1] is end period
        start_col = headers[1]
        end_col = headers[-1]
        for row in data:
            metric_name = row.get(headers[0], "").strip()
            val_first = _clean_numeric(row.get(start_col, ""))
            val_last = _clean_numeric(row.get(end_col, ""))
            if val_first is not None and val_last is not None and val_first != 0:
                pct_change = ((val_last - val_first) / abs(val_first)) * 100.0
                deltas.append({
                    "metric": metric_name,
                    "delta": pct_change,
                    "first": val_first,
                    "last": val_last,
                })

    return deltas


def format_concessive_sentence(deltas: list[dict]) -> str:
    """Converts computed deltas into concessive conjunction sentences.

    If two metrics diverge (one up, one down), uses "Although X grew Y%, Z declined W%".
    If both trend the same direction, uses "Both X and Y grew/declined by ...".
    """
    if not deltas:
        return ""

    if len(deltas) == 1:
        d = deltas[0]
        direction = "grew" if d["delta"] >= 0 else "declined"
        return f"{d['metric']} {direction} {abs(d['delta']):.1f}%."

    d1 = deltas[0]
    d2 = deltas[1]

    dir1 = "grew" if d1["delta"] >= 0 else "declined"
    dir2 = "grew" if d2["delta"] >= 0 else "declined"

    if (d1["delta"] >= 0) != (d2["delta"] >= 0):
        return (
            f"Although {d1['metric']} {dir1} {abs(d1['delta']):.1f}%, "
            f"{d2['metric']} {dir2} {abs(d2['delta']):.1f}%."
        )
    else:
        return (
            f"Both {d1['metric']} and {d2['metric']} {dir1} by "
            f"{abs(d1['delta']):.1f}% and {abs(d2['delta']):.1f}% respectively."
        )
