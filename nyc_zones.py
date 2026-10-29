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
import json
import pathlib

import pandas as pd

_REF = pathlib.Path(__file__).resolve().parent / "data" / "reference"
_CSV = _REF / "taxi_zone_lookup.csv"
_GEO = _REF / "taxi_zones.json"

# Les cinq arrondissements, plus les deux valeurs de service que la TLC range
# au même niveau. L'ordre est celui du volume de courses, pour que les Sankey
# et les légendes sortent dans un ordre stable et lisible.
BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR"]

# Une couleur par arrondissement. Manhattan prend l'accent de la marque —
# c'est 92 % des prises en charge, autant que ce soit lui qui porte le doré.
# Les autres restent dans des neutres chauds pour ne pas fabriquer six
# accents concurrents.
BOROUGH_COLORS = {
    "Manhattan": "#7e7a82",
    "Queens": "#7f9bb5",
    "Brooklyn": "#b58b6a",
    "Bronx": "#8f8aa6",
    "Staten Island": "#6f8f7a",
    "EWR": "#9a9aa4",
    "Unknown": "#6b6773",
}

# Identifiants des aéroports dans le référentiel TLC.
JFK, LGA, EWR = 132, 138, 1


# ---------------------------------------------------------------------------
# Secteurs : l'arrondissement, sauf Manhattan, coupé en quatre
# ---------------------------------------------------------------------------
# Manhattan pèse 83 % des courses, et presque toutes restent dedans. À la
# maille de l'arrondissement, un trajet Harlem - Wall Street (11 km) et un
# trajet Midtown - Midtown East (900 m) sont le même déplacement : « Manhattan
# vers Manhattan ». Le diagramme des flux n'avait alors qu'une seule chose à
# montrer, et c'était une boucle.
#
# La coupe se fait sur la latitude du centroïde, et non sur une liste de
# quartiers écrite à la main : les bornes sont des rues que tout le monde
# connaît, elles se vérifient, et une republication du référentiel par la TLC
# ne les périme pas.
#
# Les noms de secteurs restent en anglais, comme les 263 noms de zones de la
# TLC et comme les arrondissements : ce sont des toponymes, pas de l'interface.
# « Midtown » ne se traduit pas, et traduire la moitié d'une liste en laissant
# l'autre ferait deux référentiels pour un seul territoire.
#
#     14th St     40.7375    Downtown        <- Financial District, Village
#     59th St     40.7685    Midtown         <- Times Sq, Grand Central, Chelsea
#     110th St    40.8000    Uptown          <- Upper East / West Side, Central Park
#                            Harlem & North  <- Harlem, Washington Heights, Inwood
MANHATTAN_BANDS = [
    (40.7375, "Downtown"),
    (40.7685, "Midtown"),
    (40.8000, "Uptown"),
    (float("inf"), "Harlem & North"),
]

# Les trois aéroports forment leur propre secteur. Dans le référentiel TLC
# ce sont des zones ordinaires : JFK et LaGuardia tombent dans le Queens,
# Newark dans sa propre valeur de service. Fondus dans leur arrondissement,
# ils disparaissaient d'un diagramme où ils pèsent 7 % des courses — et ces
# 7 % ne ressemblent à rien d'autre. Un vol n'est pas un déplacement urbain :
# il part d'un point unique, à heure fixe, vers toute la ville.
AIRPORT_AREA = "Airports"
AIRPORT_IDS = {132: "Airports", 138: "Airports", 1: "Airports"}

# Ordre des secteurs : Manhattan du sud au nord, puis les autres
# arrondissements par volume, et les aéroports en dernier. C'est lui qui
# range les nœuds du diagramme de flux, où une suite géographique se suit
# mieux qu'un classement.
AREAS = [band for _, band in MANHATTAN_BANDS] + [
    "Queens", "Brooklyn", "Bronx", "Staten Island", AIRPORT_AREA,
]

# Les quatre bandes de Manhattan prennent la même matière — l'or de la marque —
# en quatre valeurs, du sud au nord. Quatre teintes étrangères en feraient
# quatre arrondissements de plus ; une rampe dit qu'il s'agit d'un seul
# territoire, découpé.
# Manhattan tient les quatre cinquièmes des flux : en or, ses quatre bandes
# faisaient virer tout le diagramme au jaune, et le jaune appartient aux
# taxis. En brique, elles criaient. Elles prennent donc la matière des tours
# elles-mêmes — granit sombre au sud, calcaire pâle au nord — en quatre
# valeurs d'une seule famille, pour qu'on lise encore un arrondissement et
# non quatre. Les couleurs vives restent aux arrondissements voisins, qui
# sont petits : c'est là qu'elles servent à distinguer.
#
# Les aéroports prennent un bleu-vert franc, la seule teinte saturée du jeu :
# ils sont 7 % des courses et doivent se repérer d'un coup d'œil, sur le
# ruban comme sur la carte, où ce sont eux qui portent les repères.
AREA_COLORS = {
    "Downtown": "#585560",
    "Midtown": "#7e7a82",
    "Uptown": "#a8a29f",
    "Harlem & North": "#cfc8be",
    AIRPORT_AREA: "#2f7d8c",
    **{b: c for b, c in BOROUGH_COLORS.items()
       if b not in ("Manhattan", "EWR")},
}


@functools.lru_cache(maxsize=1)
def _centroids() -> dict:
    """location_id -> latitude du centroïde, depuis la géométrie TLC.

    Rend un dictionnaire vide si le fichier manque : les zones de Manhattan
    retombent alors sur « Midtown », ce qui est faux mais ne fait pas tomber
    la page. Le fichier est produit par scripts/H_build_geo.py.
    """
    if not _GEO.exists():
        return {}
    return {z["id"]: z["lat"] for z in json.loads(_GEO.read_text())}


def _band(lat: float) -> str:
    for top, name in MANHATTAN_BANDS:
        if lat < top:
            return name
    return MANHATTAN_BANDS[-1][1]


@functools.lru_cache(maxsize=1)
def areas() -> dict:
    """location_id -> secteur : l'arrondissement, ou la bande pour Manhattan."""
    lat = _centroids()
    out = {}
    for zone_id, borough in boroughs().items():
        if zone_id in AIRPORT_IDS:
            out[zone_id] = AIRPORT_IDS[zone_id]
        elif borough == "Manhattan":
            out[zone_id] = _band(lat.get(zone_id, 40.75))
        else:
            out[zone_id] = borough
    return out


def area_color(area: str) -> str:
    return AREA_COLORS.get(area, BOROUGH_COLORS["Unknown"])


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
