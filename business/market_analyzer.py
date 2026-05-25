"""
Market analysis: identify niches, evaluate opportunities, score ideas.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Pre-validated low-competition niches for German micro-market
STARTER_NICHES = [
    {
        "name": "Tierpension & Tierpflege",
        "pain_points": ["Buchungsverwaltung", "Kundenkommunikation", "Rechnungen"],
        "products": ["Buchungsformular-Template", "E-Mail-Vorlagen-Set", "Preislisten-Template"],
        "avg_price_eur": 12.0,
        "competition": "low",
    },
    {
        "name": "Kleinunternehmer Excel/PDF",
        "pain_points": ["Angebote erstellen", "Rechnungen", "Stundenerfassung"],
        "products": ["Angebots-Template", "Rechnungs-Excel-Vorlage", "Stundennachweise"],
        "avg_price_eur": 9.0,
        "competition": "medium",
    },
    {
        "name": "Vereine & Clubs",
        "pain_points": ["Mitgliederverwaltung", "Satzungen", "Einladungen", "Protokolle"],
        "products": ["Satzungs-Template", "Protokoll-Vorlage", "Mitgliederbrief-Set"],
        "avg_price_eur": 8.0,
        "competition": "low",
    },
    {
        "name": "Handwerker & Dienstleister",
        "pain_points": ["Angebote", "AGB", "Kundenkommunikation"],
        "products": ["AGB-Template", "Angebots-Vorlage Handwerk", "Auftragsbestätigung"],
        "avg_price_eur": 15.0,
        "competition": "medium",
    },
    {
        "name": "Hobby-Verkäufer & Etsy-Creator",
        "pain_points": ["Produktbeschreibungen", "Social Media", "Preiskalkulation"],
        "products": ["Produktbeschreibungs-Templates", "Preiskalkulations-Sheet", "Social-Media-Posts"],
        "avg_price_eur": 7.0,
        "competition": "high",
    },
]


def get_starter_niches() -> list[dict]:
    return STARTER_NICHES


def score_niche(niche: dict) -> float:
    comp_score = {"low": 1.0, "medium": 0.6, "high": 0.3}.get(niche.get("competition", "medium"), 0.5)
    price_score = min(niche.get("avg_price_eur", 5) / 20.0, 1.0)
    product_count = len(niche.get("products", []))
    return round((comp_score * 0.5 + price_score * 0.3 + min(product_count / 5, 1.0) * 0.2), 3)


def get_prioritized_niches() -> list[dict]:
    scored = [(score_niche(n), n) for n in STARTER_NICHES]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, **n} for s, n in scored]


def estimate_monthly_revenue(niche: dict, sales_per_month: int = 3) -> float:
    return niche.get("avg_price_eur", 5.0) * sales_per_month


def quick_product_ideas(niche_name: str) -> list[str]:
    niche_map = {n["name"]: n for n in STARTER_NICHES}
    niche = niche_map.get(niche_name, {})
    return niche.get("products", [])
