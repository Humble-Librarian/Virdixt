"""Deterministic math module for chart-to-text and spreadsheet-to-text conversion.

Parses DePlot linearized tables and tabular spreadsheet data, computes percentage deltas,
detects metric polarity (direct vs inverted risk metrics), sign flips, and variance,
and generates concessive conjunction sentences for FinBERT tokenization.
Pure Python, zero external ML dependencies, microsecond execution.
"""
import re
from typing import List, Dict, Any, Optional

# Metrics where an increase represents growth/positive performance
DIRECT_METRICS = {
    "revenue", "sales", "net income", "net profit", "operating income", "operating profit",
    "ebitda", "gross profit", "free cash flow", "cash flow", "cash", "cash and cash equivalents",
    "arr", "mrr", "operating cash flow", "margin", "gross margin", "operating margin",
    "net margin", "eps", "earnings per share", "dividends", "equity", "bookings", "headcount",
    "customer count", "arpu", "retention"
}

# Metrics where an increase represents risk, debt burden, distress, or cost pressure
INVERTED_METRICS = {
    "debt", "total debt", "long-term debt", "short-term debt", "liabilities", "total liabilities",
    "expenses", "operating expenses", "opex", "sg&a", "sga", "cost of goods sold", "cogs",
    "loss", "net loss", "operating loss", "impairments", "restructuring costs", "churn",
    "churn rate", "debt-to-equity", "leverage", "burn rate", "provision for bad debts",
    "interest expense", "default rate", "delinquency", "overhead", "capex"
}


def _clean_numeric(val: Any) -> Optional[float]:
    """Extracts floating point number from financial strings like '$45.5M', '(12.0)', '18%', '4,200'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if not val_str:
        return None
    
    # Handle accounting negative format: (12.5) -> -12.5
    is_negative = False
    if val_str.startswith("(") and val_str.endswith(")"):
        is_negative = True
        val_str = val_str[1:-1].strip()
    
    cleaned = val_str.replace(',', '').replace('$', '').replace('%', '').replace('€', '').replace('£', '').strip()
    if cleaned.startswith("-"):
        is_negative = True
        cleaned = cleaned[1:].strip()
    elif cleaned.startswith("+"):
        cleaned = cleaned[1:].strip()

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
            num = float(match.group(0)) * multiplier
            return -num if is_negative else num
        except ValueError:
            return None
    return None


def _format_currency(val: float) -> str:
    """Formats numeric value into concise financial notation ($12.5M, $450K, $1.2B) to minimize BERT tokens."""
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.1f}K"
    else:
        return f"{sign}${abs_val:,.1f}"


def classify_metric_polarity(metric_name: str) -> str:
    """Classifies metric as 'inverted' (debt/loss/expense) or 'direct' (revenue/profit/cash)."""
    norm = metric_name.lower().replace("_", " ").replace("-", " ").strip()
    for inv in INVERTED_METRICS:
        if inv in norm:
            return "inverted"
    for dir_m in DIRECT_METRICS:
        if dir_m in norm:
            return "direct"
    # Default to direct if unknown
    return "direct"


def format_delta_clause(metric: str, first_val: float, last_val: float, delta_pct: float) -> tuple[str, bool]:
    """Formats a single metric's movement into an accounting statement clause.
    
    Returns (clause_text, is_positive_development).
    """
    polarity = classify_metric_polarity(metric)
    f_str = _format_currency(first_val)
    l_str = _format_currency(last_val)
    
    # Check for sign flip: Loss to Profit or Profit to Loss
    if first_val < 0 and last_val > 0:
        clause = f"{metric} reversed from an operating loss of {_format_currency(abs(first_val))} into a profit of {l_str}"
        return (clause, True)
    elif first_val > 0 and last_val < 0:
        clause = f"{metric} swung from a profit of {f_str} into a severe loss of {_format_currency(abs(last_val))}"
        return (clause, False)
    
    abs_pct = abs(delta_pct)
    
    if polarity == "inverted":
        if delta_pct > 5.0:
            verb = "surged" if delta_pct > 25.0 else "increased"
            clause = f"{metric} {verb} {abs_pct:.1f}% (from {f_str} to {l_str})"
            return (clause, False)  # Higher debt/expense is a negative development
        elif delta_pct < -5.0:
            verb = "contracted" if delta_pct < -25.0 else "decreased"
            clause = f"{metric} {verb} {abs_pct:.1f}% (reduced to {l_str})"
            return (clause, True)   # Reduced debt/expense is a positive development
        else:
            clause = f"{metric} remained flat at {l_str}"
            return (clause, True)
    else:
        # Direct metric
        if delta_pct > 5.0:
            verb = "grew" if delta_pct > 25.0 else "increased"
            clause = f"{metric} {verb} {abs_pct:.1f}% (to {l_str})"
            return (clause, True)   # Higher revenue/profit is a positive development
        elif delta_pct < -5.0:
            verb = "contracted" if delta_pct < -25.0 else "declined"
            clause = f"{metric} {verb} {abs_pct:.1f}% (declined from {f_str} to {l_str})"
            return (clause, False)  # Lower revenue/profit is a negative development
        else:
            clause = f"{metric} remained essentially flat at {l_str}"
            return (clause, True)



def parse_deplot_table(raw_string: str) -> list[dict]:
    """Parses DePlot's linearized output into structured data."""
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


def compute_deltas(data: list[dict]) -> list[dict]:
    """Computes percentage changes between first and last periods for each metric."""
    if not data:
        return []

    headers = list(data[0].keys())
    if len(headers) < 2:
        return []

    first_col_header = headers[0].lower()
    deltas = []

    # Check if table is Period-as-row
    is_period_rows = any(kw in first_col_header for kw in ["period", "quarter", "year", "date", "q1", "q2", "q3", "q4", "202", "month"])
    
    if is_period_rows and len(data) >= 2:
        first_row = data[0]
        last_row = data[-1]
        for col in headers[1:]:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["note", "comment", "audit", "status", "risk", "remark"]):
                continue
            val_first = _clean_numeric(first_row.get(col, ""))
            val_last = _clean_numeric(last_row.get(col, ""))
            if val_first is not None and val_last is not None:
                if val_first != 0:
                    pct_change = ((val_last - val_first) / abs(val_first)) * 100.0
                else:
                    pct_change = 100.0 if val_last > 0 else (-100.0 if val_last < 0 else 0.0)
                deltas.append({
                    "metric": col,
                    "delta": pct_change,
                    "first": val_first,
                    "last": val_last,
                })
    else:
        # Metric-as-row: find candidate numeric period columns (exclude notes, status, and variance/diff columns)
        non_metric_cols = headers[1:]
        candidate_cols = []
        for c in non_metric_cols:
            c_lower = c.lower()
            if any(kw in c_lower for kw in ["note", "comment", "audit", "status", "risk", "remark", "description", "variance", "change", "diff", "%", "pct"]):
                continue
            # Check if column has numbers
            has_num = any(_clean_numeric(r.get(c)) is not None for r in data)
            if has_num:
                candidate_cols.append(c)

        if len(candidate_cols) >= 2:
            start_col = candidate_cols[0]
            end_col = candidate_cols[-1]
            for row in data:
                metric_name = str(row.get(headers[0], "")).strip()
                if not metric_name:
                    continue
                val_first = _clean_numeric(row.get(start_col, ""))
                val_last = _clean_numeric(row.get(end_col, ""))
                if val_first is not None and val_last is not None:
                    if val_first != 0:
                        pct_change = ((val_last - val_first) / abs(val_first)) * 100.0
                    else:
                        pct_change = 100.0 if val_last > 0 else (-100.0 if val_last < 0 else 0.0)
                    deltas.append({
                        "metric": metric_name,
                        "delta": pct_change,
                        "first": val_first,
                        "last": val_last,
                    })

    return deltas


def format_concessive_sentence(deltas: list[dict]) -> str:
    """Converts computed deltas into concessive conjunction sentences with polarity awareness."""
    if not deltas:
        return ""

    positives = []
    distresses = []

    for d in deltas:
        clause, is_pos = format_delta_clause(d["metric"], d["first"], d["last"], d["delta"])
        if is_pos:
            positives.append(clause)
        else:
            distresses.append(clause)

    if positives and distresses:
        pos_text = ", and ".join(positives) if len(positives) <= 2 else "; ".join(positives)
        dist_text = ", and ".join(distresses) if len(distresses) <= 2 else "; ".join(distresses)
        return f"Although {pos_text}, {dist_text}."
    elif distresses:
        dist_text = ", and ".join(distresses) if len(distresses) <= 2 else "; ".join(distresses)
        return f"Financial distress signals detected: {dist_text}."
    elif positives:
        pos_text = ", and ".join(positives) if len(positives) <= 2 else "; ".join(positives)
        return f"Operational expansion observed: {pos_text}."
    return ""


def synthesize_table_narrative(rows: list[dict], table_name: str = "Financial Table") -> str:
    """Fully sentencifies a tabular matrix (CSV or Excel sheet).
    
    Extracts numerical movements, variance against budget/targets, and any qualitative commentary columns.
    """
    if not rows:
        return ""

    # 1. Identify narrative text columns vs numerical columns
    headers = list(rows[0].keys())
    text_cols = []
    
    # Check column types
    for h in headers:
        h_lower = h.lower()
        if any(kw in h_lower for kw in ["note", "comment", "narrative", "remark", "description", "audit", "status", "risk", "opinion"]):
            text_cols.append(h)

    # 2. Check for Target vs Actual variance structure
    has_target = any("target" in h.lower() or "budget" in h.lower() or "forecast" in h.lower() for h in headers)
    has_actual = any("actual" in h.lower() or "realized" in h.lower() or "current" in h.lower() for h in headers)
    
    variance_clauses = []
    if has_target and has_actual:
        target_col = next(h for h in headers if "target" in h.lower() or "budget" in h.lower() or "forecast" in h.lower())
        actual_col = next(h for h in headers if "actual" in h.lower() or "realized" in h.lower() or "current" in h.lower())
        
        for r in rows:
            metric = str(r.get(headers[0], "")).strip()
            if not metric:
                continue
            t_val = _clean_numeric(r.get(target_col))
            a_val = _clean_numeric(r.get(actual_col))
            if t_val is not None and a_val is not None:
                polarity = classify_metric_polarity(metric)
                var_pct = ((a_val - t_val) / abs(t_val)) * 100.0 if t_val != 0 else 0.0
                
                if polarity == "inverted":
                    if a_val > t_val:
                        variance_clauses.append(f"{metric} exceeded budget target by {abs(var_pct):.1f}% (${a_val:,.1f} vs target ${t_val:,.1f})")
                    else:
                        variance_clauses.append(f"{metric} stayed within budget by {abs(var_pct):.1f}% (${a_val:,.1f} vs target ${t_val:,.1f})")
                else:
                    if a_val >= t_val:
                        variance_clauses.append(f"{metric} beat target expectations by {abs(var_pct):.1f}% (${a_val:,.1f} vs target ${t_val:,.1f})")
                    else:
                        variance_clauses.append(f"{metric} missed target expectations by {abs(var_pct):.1f}% (${a_val:,.1f} vs target ${t_val:,.1f})")

    # 3. Compute period-over-period or metric deltas
    deltas = compute_deltas(rows)
    concessive_sentence = format_concessive_sentence(deltas)

    # 4. Extract qualitative commentary / audit notes
    qualitative_sentences = []
    for r in rows:
        metric = str(r.get(headers[0], "")).strip()
        for c in text_cols:
            note = str(r.get(c, "")).strip()
            if note and note.lower() not in ["none", "n/a", "-", "null", "", "pass"]:
                if metric:
                    qualitative_sentences.append(f"Regarding {metric}: {note}")
                else:
                    qualitative_sentences.append(f"{c}: {note}")

    # 5. Assemble final coherent financial narrative
    sections = []
    if concessive_sentence:
        sections.append(concessive_sentence)
    if variance_clauses:
        sections.append("Budget variance analysis: " + "; ".join(variance_clauses) + ".")
    if qualitative_sentences:
        sections.append("Auditor & Narrative Notes: " + " ".join(qualitative_sentences))

    if not sections:
        # Fallback raw tabular serialization
        raw_items = []
        for r in rows:
            item_str = ", ".join([f"{k}: {v}" for k, v in r.items() if v is not None and str(v).strip()])
            if item_str:
                raw_items.append(item_str)
        return f"{table_name}: " + ". ".join(raw_items)

    return f"[{table_name}] " + " ".join(sections)


