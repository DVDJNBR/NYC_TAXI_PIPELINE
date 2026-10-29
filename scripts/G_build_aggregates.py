"""Tables croisées pour le dashboard — une passe DuckDB par table.

Les agrégats historiques (daily, hourly, zones…) sont déjà écrasés sur une
seule dimension : `hourly` ne connaît pas le jour de la semaine, `zones` ne
connaît ni la destination ni le temps. Impossible d'y lire autre chose qu'un
classement, et impossible d'y brancher un filtrage croisé.

Ce script reprend les Parquet bruts et produit les croisements manquants. Les
filtres sont ceux de `clean_trips` côté dbt : distance dans ]0, 100] mi,
montant positif, prise en charge avant dépose. Les tables sortent dans
data/aggregates/ à côté des autres, et le dashboard les lit comme les autres.

Chaque requête relit les 74 M de lignes — environ 15 s par table sur un
portable. C'est un travail hors ligne, lancé à la main quand les Parquet
changent, pas au démarrage de l'application.

    .venv/bin/python -I scripts/G_build_aggregates.py
"""

import pathlib
import sys
import time
import urllib.request

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "yellow_taxi" / "*.parquet"
OUT = ROOT / "data" / "aggregates"
REF = ROOT / "data" / "reference"

# Table de correspondance officielle de la TLC : LocationID -> arrondissement,
# nom de zone, type de desserte. 265 lignes, publiée à côté des Parquet.
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
LOOKUP = REF / "taxi_zone_lookup.csv"

# Fenêtre d'étude : les Parquet débordent des deux côtés (une poignée de
# courses datées 2002 ou 2026 dans la source), on coupe aux mois complets.
START, END = "2024-01-01", "2025-10-01"

# Prédicat commun, celui du modèle STAGING.
CLEAN = f"""
    trip_distance > 0 AND trip_distance <= 100
    AND total_amount > 0
    AND tpep_pickup_datetime < tpep_dropoff_datetime
    AND tpep_pickup_datetime >= TIMESTAMP '{START}'
    AND tpep_pickup_datetime <  TIMESTAMP '{END}'
"""

# Durée en minutes, bornée : quelques courses portent une dépose des heures
# après la prise en charge, ce qui ruine toute moyenne de vitesse.
MINUTES = "date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0"

QUERIES = {
    # Heure x jour de semaine — 168 cellules. La vue canonique du jeu : elle
    # contient les deux profils actuels, qui en deviennent les marges.
    "hour_weekday": f"""
        SELECT
            dayofweek(tpep_pickup_datetime)            AS DOW,
            hour(tpep_pickup_datetime)                 AS PICKUP_HOUR,
            count(*)                                   AS TOTAL_TRIPS,
            sum(total_amount)                          AS TOTAL_REVENUE,
            avg(fare_amount)                           AS AVG_FARE,
            avg(trip_distance)                         AS AVG_DISTANCE,
            avg(100 * tip_amount / nullif(fare_amount, 0)) AS AVG_TIP_PCT,
            avg(trip_distance / nullif({MINUTES}, 0) * 60) AS AVG_SPEED_MPH
        FROM read_parquet('{RAW}', union_by_name = true)
        WHERE {CLEAN} AND {MINUTES} BETWEEN 1 AND 180
        GROUP BY 1, 2
        ORDER BY 1, 2
    """,

    # Distribution des distances. La moyenne de 3,44 mi cache une masse de
    # courses sous 2 mi et une bosse aéroport autour de 16-18 mi : le palier
    # de tarif forfaitaire JFK ne se voit que là.
    "distance_bins": f"""
        SELECT
            least(floor(trip_distance * 2) / 2, 25)    AS DISTANCE_BIN,
            count(*)                                   AS TOTAL_TRIPS,
            avg(total_amount)                          AS AVG_TOTAL,
            avg(fare_amount)                           AS AVG_FARE,
            avg({MINUTES})                             AS AVG_MINUTES,
            avg(100 * tip_amount / nullif(fare_amount, 0)) AS AVG_TIP_PCT
        FROM read_parquet('{RAW}', union_by_name = true)
        WHERE {CLEAN} AND {MINUTES} BETWEEN 1 AND 180
        GROUP BY 1
        ORDER BY 1
    """,

    # Flux entre arrondissements. Sept origines, sept destinations : de quoi
    # dire où vont les courses, et non plus seulement d'où elles partent.
    "od_borough": f"""
        SELECT
            pu.Borough                                 AS PU_BOROUGH,
            dz.Borough                                 AS DO_BOROUGH,
            count(*)                                   AS TOTAL_TRIPS,
            sum(total_amount)                          AS TOTAL_REVENUE,
            avg(trip_distance)                         AS AVG_DISTANCE
        FROM read_parquet('{RAW}', union_by_name = true) t
        JOIN read_csv('{LOOKUP}') pu ON pu.LocationID = t.PULocationID
        JOIN read_csv('{LOOKUP}') dz ON dz.LocationID = t.DOLocationID
        WHERE {CLEAN} AND pu.Borough <> 'Unknown' AND dz.Borough <> 'Unknown'
        GROUP BY 1, 2
        ORDER BY 3 DESC
    """,

    # Zones d'arrivée, pendant de `zones` qui ne tient que les départs.
    "dropoff_zones": f"""
        SELECT
            t.DOLocationID                             AS ZONE_ID,
            any_value(z.Zone)                          AS ZONE_NAME,
            any_value(z.Borough)                       AS BOROUGH,
            count(*)                                   AS TOTAL_TRIPS,
            sum(total_amount)                          AS TOTAL_REVENUE,
            avg(fare_amount)                           AS AVG_FARE,
            avg(trip_distance)                         AS AVG_DISTANCE
        FROM read_parquet('{RAW}', union_by_name = true) t
        JOIN read_csv('{LOOKUP}') z ON z.LocationID = t.DOLocationID
        WHERE {CLEAN}
        GROUP BY 1
        ORDER BY 4 DESC
        LIMIT 150
    """,

    # Desserte aéroportuaire mois par mois, dans les deux sens. JFK et LGA
    # pèsent 11,5 % des courses d'après `profile` ; reste à savoir quand.
    "airport_monthly": f"""
        SELECT
            date_trunc('month', tpep_pickup_datetime)  AS MONTH,
            CASE WHEN pu.service_zone = 'Airports' THEN pu.Zone ELSE dz.Zone END AS AIRPORT,
            CASE WHEN pu.service_zone = 'Airports' THEN 'Départ' ELSE 'Arrivée' END AS DIRECTION,
            count(*)                                   AS TOTAL_TRIPS,
            avg(total_amount)                          AS AVG_TOTAL,
            avg(trip_distance)                         AS AVG_DISTANCE
        FROM read_parquet('{RAW}', union_by_name = true) t
        JOIN read_csv('{LOOKUP}') pu ON pu.LocationID = t.PULocationID
        JOIN read_csv('{LOOKUP}') dz ON dz.LocationID = t.DOLocationID
        WHERE {CLEAN}
          AND (pu.service_zone = 'Airports' OR dz.service_zone = 'Airports')
        GROUP BY 1, 2, 3
        ORDER BY 1, 2
    """,

    # Nombre de passagers déclaré. Sert surtout à montrer ce que la source
    # contient de douteux : des courses à zéro passager.
    "passengers": f"""
        SELECT
            coalesce(passenger_count, -1)              AS PASSENGERS,
            count(*)                                   AS TOTAL_TRIPS,
            avg(total_amount)                          AS AVG_TOTAL,
            avg(trip_distance)                         AS AVG_DISTANCE
        FROM read_parquet('{RAW}', union_by_name = true)
        WHERE {CLEAN}
        GROUP BY 1
        ORDER BY 1
    """,
}


def fetch_lookup() -> None:
    if LOOKUP.exists():
        return
    REF.mkdir(parents=True, exist_ok=True)
    print(f"→ téléchargement de {LOOKUP.name}")
    urllib.request.urlretrieve(LOOKUP_URL, LOOKUP)


def main(only: str | None = None) -> None:
    if not list((ROOT / "data" / "yellow_taxi").glob("*.parquet")):
        sys.exit("Aucun Parquet brut dans data/yellow_taxi/.")
    fetch_lookup()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    for name, query in QUERIES.items():
        if only and only != name:
            continue
        target = OUT / f"{name}.parquet"
        t0 = time.time()
        con.sql(f"COPY ({query}) TO '{target}' (FORMAT PARQUET)")
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0]
        print(f"✓ {name:<16} {rows:>6} lignes  {time.time() - t0:5.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
