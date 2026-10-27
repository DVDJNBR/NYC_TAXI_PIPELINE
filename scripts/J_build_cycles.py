"""Profils cycliques : la journée, la semaine et l'année types.

Le cockpit ne rejoue pas une chronologie, il tourne en boucle sur un cycle.
Trois cycles, exclusifs l'un de l'autre :

    hour    24 pas, minuit -> minuit, tous jours confondus
    dow      7 pas, lundi -> dimanche, toutes heures confondues
    month   12 pas, janvier -> décembre, les deux années confondues

Toutes les valeurs sont ramenées à une **occurrence** du pas, jamais à un
total brut. Sans cela le cycle mensuel mentirait : la période couvre janvier
2024 à septembre 2025, donc janvier est observé deux fois et novembre une
seule. Un total ferait chuter la fin d'année de moitié, pour une raison de
comptage et non de trafic. Le dénominateur est le nombre de jours
réellement observés pour chaque pas, lu dans daily.parquet.

Deux tables en sortie, toutes deux lues par les composants du cockpit :

    data/aggregates/zone_cycle.parquet   zone x 43 pas x 3 mesures
    data/aggregates/od_cycle.parquet     flux entre secteurs x 43 pas

La distance est stockée en somme et non en moyenne : une moyenne de moyennes
donnerait le même poids à une zone de trois courses et à Midtown. La division
par le nombre de courses est faite à la lecture, ce qui la rend pondérée.

    .venv/bin/python -I scripts/J_build_cycles.py
"""

import pathlib
import sys
import time

import duckdb
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import nyc_zones  # noqa: E402  (après l'ajout du dépôt au chemin)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "yellow_taxi" / "*.parquet"
OUT = ROOT / "data" / "aggregates"
LOOKUP = ROOT / "data" / "reference" / "taxi_zone_lookup.csv"
DAILY = OUT / "daily.parquet"

START, END = "2024-01-01", "2025-10-01"
CLEAN = f"""
    trip_distance > 0 AND trip_distance <= 100
    AND total_amount > 0
    AND tpep_pickup_datetime < tpep_dropoff_datetime
    AND tpep_pickup_datetime >= TIMESTAMP '{START}'
    AND tpep_pickup_datetime <  TIMESTAMP '{END}'
"""

# Expression du pas pour chaque cycle. dayofweek de DuckDB rend 0 le
# dimanche ; le décalage met le lundi en tête, comme partout ailleurs dans
# l'application.
STEPS = {
    "hour": "hour(tpep_pickup_datetime)",
    "dow": "(dayofweek(tpep_pickup_datetime) + 6) % 7",
    "month": "month(tpep_pickup_datetime) - 1",
}


def denominators() -> pd.DataFrame:
    """Nombre de jours observés derrière chaque pas de chaque cycle.

    Lu dans daily.parquet plutôt que dans les Parquet bruts : la liste des
    journées y est déjà, et la relire coûterait un scan complet de plus.
    """
    if not DAILY.exists():
        sys.exit("data/aggregates/daily.parquet manquant : lancer d'abord G.")
    d = pd.read_parquet(DAILY, columns=["PICKUP_DATE"])
    dates = pd.to_datetime(d["PICKUP_DATE"])
    dates = dates[(dates >= START) & (dates < END)]

    rows = [("hour", h, len(dates)) for h in range(24)]
    dow = dates.dt.dayofweek.value_counts()
    rows += [("dow", int(k), int(v)) for k, v in dow.items()]
    month = dates.dt.month.value_counts()
    rows += [("month", int(k) - 1, int(v)) for k, v in month.items()]
    return pd.DataFrame(rows, columns=["SCOPE", "STEP", "DAYS"])


def zone_cycle(con) -> pd.DataFrame:
    """Une ligne par zone de prise en charge, par cycle et par pas."""
    parts = []
    for scope, expr in STEPS.items():
        t0 = time.time()
        df = con.execute(f"""
            SELECT
                '{scope}'          AS SCOPE,
                {expr}             AS STEP,
                PULocationID::INT  AS ZONE_ID,
                count(*)           AS TOTAL_TRIPS,
                sum(total_amount)  AS TOTAL_REVENUE,
                sum(trip_distance) AS SUM_DISTANCE
            FROM read_parquet('{RAW}', union_by_name = true)
            WHERE {CLEAN}
            GROUP BY 1, 2, 3
        """).fetchdf()
        print(f"  zone/{scope:<6} {len(df):>6} lignes  {time.time() - t0:.1f} s")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def od_cycle(con) -> pd.DataFrame:
    """Flux entre secteurs, par cycle et par pas.

    Secteur et non arrondissement : Manhattan est coupé en quatre bandes par
    nyc_zones, sans quoi 83 % des courses tiendraient dans une seule boucle
    Manhattan - Manhattan et le diagramme n'aurait rien à montrer.

    Les trois cycles sortent d'une seule lecture : la jointure sur le
    référentiel coûte plus cher que le regroupement, autant ne la faire
    qu'une fois et laisser DuckDB produire les trois agrégats ensemble.
    """
    t0 = time.time()
    # La correspondance zone -> secteur est calculée en Python, depuis la
    # géométrie. Elle est passée à DuckDB comme une table, ce qui évite un
    # CASE de 263 branches dans le SQL.
    sectors = pd.DataFrame(sorted(nyc_zones.areas().items()),
                           columns=["LocationID", "Area"])
    sectors = sectors[sectors["Area"].isin(nyc_zones.AREAS)]
    con.register("sectors", sectors)

    selects = []
    for scope, expr in STEPS.items():
        selects.append(f"""
            SELECT '{scope}' AS SCOPE, {expr} AS STEP,
                   pu.Area AS PU_AREA, dz.Area AS DO_AREA,
                   count(*) AS TOTAL_TRIPS, sum(total_amount) AS TOTAL_REVENUE
            FROM read_parquet('{RAW}', union_by_name = true) t
            JOIN sectors pu ON pu.LocationID = t.PULocationID
            JOIN sectors dz ON dz.LocationID = t.DOLocationID
            WHERE {CLEAN}
            GROUP BY 1, 2, 3, 4
        """)
    df = con.execute(" UNION ALL ".join(selects)).fetchdf()
    print(f"  flux         {len(df):>6} lignes  {time.time() - t0:.1f} s")
    return df


def main() -> None:
    if not list((ROOT / "data" / "yellow_taxi").glob("*.parquet")):
        sys.exit("Aucun Parquet brut dans data/yellow_taxi/.")
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    days = denominators()

    zones = zone_cycle(con).merge(days, on=["SCOPE", "STEP"])
    # Par occurrence du pas : « courses une heure moyenne », « courses un
    # lundi moyen », « courses un jour de janvier ».
    zones["TOTAL_TRIPS"] = zones["TOTAL_TRIPS"] / zones["DAYS"]
    zones["TOTAL_REVENUE"] = zones["TOTAL_REVENUE"] / zones["DAYS"]
    zones["AVG_DISTANCE"] = zones["SUM_DISTANCE"] / zones["TOTAL_TRIPS"] / zones["DAYS"]
    zones = zones.drop(columns=["SUM_DISTANCE", "DAYS"])
    zones.to_parquet(OUT / "zone_cycle.parquet", index=False)

    od = od_cycle(con).merge(days, on=["SCOPE", "STEP"])
    od["TOTAL_TRIPS"] = od["TOTAL_TRIPS"] / od["DAYS"]
    od["TOTAL_REVENUE"] = od["TOTAL_REVENUE"] / od["DAYS"]
    od = od.drop(columns=["DAYS"])
    od.to_parquet(OUT / "od_cycle.parquet", index=False)

    for name in ("zone_cycle", "od_cycle"):
        size = (OUT / f"{name}.parquet").stat().st_size / 1024
        print(f"✓ {name:<12} {size:>6.0f} ko")


if __name__ == "__main__":
    main()
