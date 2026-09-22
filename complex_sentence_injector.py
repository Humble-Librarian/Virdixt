"""
Complex Sentence Injector for Financial Sentiment Analysis.
Generates multi-clause, contrasting, concessive, and forward-looking guidance statements.
Teaches the model financial priority and nuanced discourse reasoning.
"""

import random

COMPANIES = [
    "The enterprise", "The corporation", "The industrial conglomerate", "The technology giant",
    "The pharmaceutical firm", "The automotive group", "The commercial bank", "The retailer"
]


# Complex Multi-clause Patterns
# Class 1: Complex Negatives (Looks positive in subordinate clause, but main clause is negative)
COMPLEX_NEGATIVE_PATTERNS = [
    "Although consolidated revenue expanded by {rev_pct} YoY, severe gross margin compression and surging SG&A expenses caused net operating profit to collapse {drop_pct}.",
    "Despite reporting record quarterly revenues of {amount}, management drastically reduced full-year fiscal earnings guidance citing macroeconomic headwinds and order cancellations.",
    "While the company successfully secured a {amount} emergency liquidity facility, the onerous {rate} coupon and restrictive covenants severely constrain future capital deployment.",
    "Even though third-quarter shipments rose {rev_pct}, an unexpected {amount} non-cash goodwill impairment charge pushed bottom-line earnings deep into negative territory.",
    "Despite settling the protracted patent litigation for {amount}, lingering operational disruptions and lost exclusive market rights continue to depress forward cash flows.",
    "While domestic sales demonstrated resilient growth of {rev_pct}, severe international currency devaluation and export tariffs wiped out operating gains.",
    "Although customer acquisition climbed {rev_pct}, average revenue per user deteriorated by {drop_pct}, signaling unsustainable promotional discounting.",
    "Notwithstanding top-line expansion across cloud divisions, mounting hardware depreciation and high debt servicing costs caused cash reserves to decline {drop_pct}."
]

# Class 2: Complex Positives (Looks negative in subordinate clause, but core operational reality is positive)
COMPLEX_POSITIVE_PATTERNS = [
    "Operating cash flow reached positive territory at {amount} for the first time in eight quarters, notwithstanding a non-recurring {amount} litigation settlement expense.",
    "Although reported headline revenue contracted {drop_pct} due to strong foreign exchange headwinds, constant-currency organic sales expanded by a robust {rev_pct}.",
    "Despite incurring heavy upfront restructuring charges of {amount} related to facility closures, normalized operating margin expanded by {rev_pct} to a 5-year high.",
    "While fourth-quarter deliveries temporarily declined {drop_pct} from supply constraints, firm order backlogs reached a record {amount}, ensuring multi-year revenue visibility.",
    "Even though higher research & development spending reduced short-term net margin by {drop_pct}, the company successfully commercialized two high-margin SaaS platforms.",
    "Notwithstanding widespread sector-wide demand softness, the firm gained {rev_pct} market share and raised its annual dividend distribution by {rev_pct}.",
    "Although legacy hardware sales dropped {drop_pct}, high-margin recurring subscription ARR surged {rev_pct}, driving total gross profit to record levels.",
    "Despite absorbing a temporary {amount} inventory write-down, adjusted EBITDA grew {rev_pct} YoY backed by unprecedented operational discipline."
]

# Class 3: Complex Neutrals (Balanced corporate restructuring, asset swaps, mixed non-directional filings)
COMPLEX_NEUTRAL_PATTERNS = [
    "The board approved the planned divestiture of its European packaging division for {amount}, with proceeds split evenly between debt retirement and general working capital.",
    "While full-year revenues rose {rev_pct} in North America, European sales declined by an identical {rev_pct}, leaving global consolidated turnover virtually flat.",
    "The corporation completed a scheduled {amount} debt swap, replacing maturing 2026 senior notes with 2031 notes at comparable coupon yields.",
    "Although the enterprise added 450 engineering personnel to support software development, it reduced operational administrative staff by a corresponding number.",
    "The company restructured its dual-class share architecture into a single unified voting structure following approval by an extraordinary shareholder quorum.",
    "While gross profit margins fluctuated across quarterly cycles, full-year blended operating margin remained steady at exactly {rate}.",
    "The firm announced the standard transition of external auditor mandates from PwC to Deloitte in compliance with statutory rotation governance requirements."
]


def generate_complex_samples(samples_per_class: int = 500) -> list[dict]:
    """Generates nuanced multi-clause financial sentences."""
    rev_pcts = ["12.4%", "15.8%", "18.2%", "21.5%", "26.0%", "31.4%"]
    drop_pcts = ["8.5%", "14.2%", "19.8%", "25.0%", "32.6%", "41.0%"]
    amounts = ["$35 million", "$65 million", "$110 million", "$240 million", "$550 million", "$1.1 billion"]
    rates = ["9.5%", "11.2%", "12.8%", "14.0%"]

    data = []

    # 1. Complex Negatives
    for _ in range(samples_per_class):
        pattern = random.choice(COMPLEX_NEGATIVE_PATTERNS)
        text = pattern.format(
            company=random.choice(COMPANIES),
            rev_pct=random.choice(rev_pcts),
            drop_pct=random.choice(drop_pcts),
            amount=random.choice(amounts),
            rate=random.choice(rates)
        )
        data.append({"text": text, "label": "negative"})

    # 2. Complex Positives
    for _ in range(samples_per_class):
        pattern = random.choice(COMPLEX_POSITIVE_PATTERNS)
        text = pattern.format(
            company=random.choice(COMPANIES),
            rev_pct=random.choice(rev_pcts),
            drop_pct=random.choice(drop_pcts),
            amount=random.choice(amounts),
            rate=random.choice(rates)
        )
        data.append({"text": text, "label": "positive"})

    # 3. Complex Neutrals
    for _ in range(samples_per_class):
        pattern = random.choice(COMPLEX_NEUTRAL_PATTERNS)
        text = pattern.format(
            company=random.choice(COMPANIES),
            rev_pct=random.choice(rev_pcts),
            drop_pct=random.choice(drop_pcts),
            amount=random.choice(amounts),
            rate=random.choice(rates)
        )
        data.append({"text": text, "label": "neutral"})

    return data
