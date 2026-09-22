"""
Synthetic Financial Data Builder
Generates realistic, domain-grounded financial sentences using authentic corporate finance structures.
"""

import random

COMPANIES = [
    "The enterprise", "The corporation", "The automotive group", "The semiconductor manufacturer",
    "The retail conglomerate", "The biotechnology firm", "The commercial bank", "The software provider",
    "The industrial supplier", "The energy utility", "The telecom operator", "The aerospace contractor"
]

METRICS = [
    "Consolidated revenue", "Operating EBITDA", "Free cash flow", "Gross profit margin",
    "Net operating income", "Recurring ARR", "Same-store sales volume", "Return on equity"
]

PERCENTAGES = ["6.5%", "9.2%", "14.8%", "18.3%", "22.0%", "27.5%", "34.0%", "42.1%"]
AMOUNTS = ["$45 million", "$85 million", "$120 million", "$250 million", "$420 million", "$1.2 billion"]


# High-grade authentic corporate finance patterns
NEGATIVE_TEMPLATES = [
    "{company} reported that {metric} contracted {pct} year-over-year due to persistent customer churn.",
    "{company} recognized a pre-tax impairment charge of {amount} related to goodwill from legacy acquisitions.",
    "{company} faced severe margin compression as raw material and freight logistics expenses surged {pct}.",
    "S&P downgraded {company}'s credit rating to junk status following mounting refinancing vulnerabilities.",
    "{company} announced a comprehensive workforce restructuring plan involving the termination of 1,800 positions.",
    "{company} reported a net loss of {amount} as supply chain bottlenecks delayed key product deliveries.",
    "{company} breached senior debt leverage covenants, triggering emergency credit renegotiations with lenders.",
    "{company} suspended quarterly common share dividend payments to preserve deteriorating liquidity reserves.",
    "{company} was ordered to pay {amount} in regulatory penalties following an antitrust compliance investigation.",
    "Order cancellations increased {pct} across core segments, forcing {company} to reduce operational capacity."
]

POSITIVE_TEMPLATES = [
    "{company} reported {metric} growth of {pct}, comfortably beating Wall Street consensus projections.",
    "{company} secured a high-margin multi-year commercial contract valued at {amount} with an anchor client.",
    "{company} completed the successful refinancing of {amount} in senior debt at significantly reduced interest rates.",
    "{company} raised full-year fiscal earnings guidance following record quarterly operating efficiency.",
    "{company} announced an accelerated {amount} share repurchase program backed by expanding free cash flow.",
    "Gross margins expanded by {pct} as {company} realized substantial economies of scale in manufacturing.",
    "{company} achieved record net profit of {amount}, driven by strong international market penetration.",
    "{company} reached sustained GAAP operating profitability for the first time in company history.",
    "Recurring subscription revenue surged {pct} YoY, demonstrating strong product-market fit for {company}.",
    "{company} received full regulatory marketing clearance for its flagship commercial platform."
]

NEUTRAL_TEMPLATES = [
    "{company} scheduled its annual meeting of stockholders for October 24 at corporate headquarters.",
    "{company} announced the appointment of a new Senior Vice President of Human Resources.",
    "{company} submitted its standard quarterly Form 10-Q filing with the Securities and Exchange Commission.",
    "{company} completed the planned divestment of an unutilized real estate asset for {amount}.",
    "{company} concluded scheduled biennial maintenance across its European manufacturing facilities.",
    "{company} shares traded flat in modest volume following the close of regular market hours.",
    "{company} confirmed its presentation schedule for the upcoming global technology investors symposium.",
    "{company} completed the routine rollover of its existing {amount} revolving credit facility.",
    "{company} announced that its registered corporate office will relocate to Chicago, Illinois.",
    "The board of directors of {company} ratified the appointment of independent external auditors for fiscal 2026."
]


def generate_synthetic_samples(samples_per_class: int = 1000) -> list[dict]:
    """Generates balanced, non-trivial financial statements."""
    data = []

    # 1. Negatives
    for _ in range(samples_per_class):
        tmpl = random.choice(NEGATIVE_TEMPLATES)
        text = tmpl.format(
            company=random.choice(COMPANIES),
            metric=random.choice(METRICS),
            pct=random.choice(PERCENTAGES),
            amount=random.choice(AMOUNTS)
        )
        data.append({"text": text, "label": "negative"})

    # 2. Positives
    for _ in range(samples_per_class):
        tmpl = random.choice(POSITIVE_TEMPLATES)
        text = tmpl.format(
            company=random.choice(COMPANIES),
            metric=random.choice(METRICS),
            pct=random.choice(PERCENTAGES),
            amount=random.choice(AMOUNTS)
        )
        data.append({"text": text, "label": "positive"})

    # 3. Neutrals
    for _ in range(samples_per_class):
        tmpl = random.choice(NEUTRAL_TEMPLATES)
        text = tmpl.format(
            company=random.choice(COMPANIES),
            metric=random.choice(METRICS),
            pct=random.choice(PERCENTAGES),
            amount=random.choice(AMOUNTS)
        )
        data.append({"text": text, "label": "neutral"})

    return data
