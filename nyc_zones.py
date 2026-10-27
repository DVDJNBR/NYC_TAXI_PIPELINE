"""Référentiel des zones TLC, lu depuis le CSV officiel.

Les 265 zones étaient jusqu'ici recopiées à la main dans un dictionnaire de
150 lignes au milieu du dashboard. Deux noms y étaient dupliqués (« East
Chelsea » et « Gramercy » apparaissaient deux fois, sur des identifiants
différents) et l'arrondissement n'y figurait pas, ce qui interdisait toute
lecture par borough.

Le fichier vient de la même origine que les Parquet de courses — le bucket
public de la TLC — et il est versionné avec le projet : une table de
correspondance de 12 ko n'a pas besoin d'être téléchargée au démarrage.

    data/reference/taxi_zone_lookup.csv
    LocationID,Borough,Zone,service_zone
"""

import functools
import pathlib

import pandas as pd

_CSV = pathlib.Path(__file__).resolve().parent / "data" / "reference" / "taxi_zone_lookup.csv"

# Les cinq arrondissements, plus les deux valeurs de service que la TLC range
# au même niveau. L'ordre est celui du volume de courses, pour que les Sankey
# et les légendes sortent dans un ordre stable et lisible.
BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR"]

# Une couleur par arrondissement. Manhattan prend l'accent de la marque —
# c'est 92 % des prises en charge, autant que ce soit lui qui porte le doré.
# Les autres restent dans des neutres chauds pour ne pas fabriquer six
# accents concurrents.
BOROUGH_COLORS = {
    "Manhattan": "#d9a441",
    "Queens": "#7f9bb5",
    "Brooklyn": "#b58b6a",
    "Bronx": "#8f8aa6",
    "Staten Island": "#6f8f7a",
    "EWR": "#9a9aa4",
    "Unknown": "#6b6773",
}

# Identifiants des aéroports dans le référentiel TLC.
JFK, LGA, EWR = 132, 138, 1


# Colonnes du référentiel, dans l'ordre du CSV de la TLC.
_COLUMNS = ["LOCATIONID", "BOROUGH", "ZONE", "SERVICE_ZONE"]


@functools.lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    """Le référentiel entier, colonnes en majuscules comme les agrégats.

    Rend une table vide si le fichier manque : les zones s'affichent alors
    par leur identifiant au lieu de faire tomber la page entière. Le fichier
    est versionné, ce cas ne devrait se produire que sur un vieux clone.
    """
    if not _CSV.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(_CSV)
    df.columns = [c.upper() for c in df.columns]
    return df


@functools.lru_cache(maxsize=1)
def names() -> dict:
    """location_id -> nom de zone."""
    t = table()
    return dict(zip(t["LOCATIONID"], t["ZONE"]))


@functools.lru_cache(maxsize=1)
def boroughs() -> dict:
    """location_id -> arrondissement."""
    t = table()
    return dict(zip(t["LOCATIONID"], t["BOROUGH"]))


def label(zone_id) -> str:
    """Nom de zone, ou l'identifiant brut si la TLC ne le connaît pas."""
    return names().get(zone_id, str(zone_id))


def color(borough: str) -> str:
    return BOROUGH_COLORS.get(borough, BOROUGH_COLORS["Unknown"])
