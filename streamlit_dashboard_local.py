"""
================================================================================
VERSION LOCALE (SANS SNOWFLAKE) - DUCKDB
================================================================================
Ce script est une alternative isolée pour exécuter le dashboard NYC Yellow Taxi 
sur un VPS ou une machine locale sans nécessiter de compte Snowflake actif.
Il interroge directement les fichiers Parquet téléchargés via DuckDB en local.

Prérequis : 
1. Avoir les données en local via `python scripts/B_load_local_parquet.py`.
2. Lancer l'appli avec : `streamlit run streamlit_dashboard_local.py`.
================================================================================
NYC Yellow Taxi — Dashboard analytique
"""


from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

import nyc_dashboard

st.set_page_config(**nyc_dashboard.PAGE_CONFIG)

# ---------------------------------------------------------------------------
# Connexion DuckDB (Fichiers Parquet locaux)
# ---------------------------------------------------------------------------
_TS_PICKUP = "tpep_pickup_datetime"
_DATE_FILTER = (
    f"AND {_TS_PICKUP} >= '2023-01-01'::DATE "
    f"AND {_TS_PICKUP} <  '2025-11-01'::DATE"
)

# union_by_name : le schéma TLC bouge d'une année sur l'autre.
# cbd_congestion_fee n'apparaît qu'avec le péage urbain (2025) ; sans cette
# option DuckDB échoue sur
#   Binder Error: Referenced column "cbd_congestion_fee" not found
_BILLING_FILTER = (
    "TRIP_DISTANCE > 0 AND TRIP_DISTANCE <= 100 AND TOTAL_AMOUNT > 0 "
    f"AND {_TS_PICKUP} >= '2023-01-01'::DATE "
    f"AND {_TS_PICKUP} <  '2025-11-01'::DATE"
)

SOURCE_TABLE = "read_parquet('data/yellow_taxi/*.parquet', union_by_name=true)"

@st.cache_resource
def get_connection():
    conn = duckdb.connect()
    # Un seul scan complet du Parquet, matérialisé en table temporaire :
    # les requêtes de load_data() agrègent ensuite dessus au lieu de
    # relire les 1,3 Go de fichiers sources une fois par requête.
    #
    # Volontairement étroite : à 75 M lignes, chaque colonne gardée ici coûte
    # de la mémoire en permanence. Les agrégats qui ont besoin des colonnes de
    # facturation (péage, taxes, péages routiers) interrogent directement les
    # Parquet plus bas — DuckDB n'y lit alors que les colonnes citées.
    conn.execute(f"""
        CREATE TEMP TABLE trips AS
        SELECT
            CAST({_TS_PICKUP} AS DATE)        AS pickup_date,
            date_part('hour', {_TS_PICKUP})   AS pickup_hour,
            TRIP_DISTANCE,
            TOTAL_AMOUNT,
            FARE_AMOUNT,
            TIP_AMOUNT,
            PULOCATIONID,
            DOLOCATIONID,
            PAYMENT_TYPE,
            PASSENGER_COUNT,
            -- Durée en secondes : tpep_dropoff_datetime n'était pas exploité.
            -- Elle donne la vitesse, seule mesure de la congestion réelle.
            date_diff('second', {_TS_PICKUP}, tpep_dropoff_datetime) AS trip_seconds
        FROM {SOURCE_TABLE}
        WHERE TRIP_DISTANCE > 0
          -- Plafond anti-aberrations : la source TLC contient des courses
          -- jusqu'à ~398 000 mi (erreurs GPS). Moins de 0,01 % des lignes,
          -- mais elles doublaient la distance moyenne (3,4 -> 7,1 mi en 2025)
          -- et fabriquaient un faux +42 % d'une année sur l'autre.
          AND TRIP_DISTANCE <= 100
          {_DATE_FILTER}
    """)
    return conn

@st.cache_data(ttl=3600)
def query(sql) -> pd.DataFrame:
    conn = get_connection()
    df = conn.execute(sql).fetchdf()
    df.columns = [col.upper() for col in df.columns]
    return df

_AGG_DIR = Path("data/aggregates")
_AGG_NAMES = ("daily", "hourly", "zones", "profile",
              "congestion", "revenue_split", "payment")

def _read_aggregates():
    if not all((_AGG_DIR / f"{name}.parquet").exists() for name in _AGG_NAMES):
        return None
    return tuple(pd.read_parquet(_AGG_DIR / f"{name}.parquet") for name in _AGG_NAMES)

def _write_aggregates(tables):
    _AGG_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in zip(_AGG_NAMES, tables):
        df.to_parquet(_AGG_DIR / f"{name}.parquet")

@st.cache_data(ttl=3600)
def load_data():
    # Agrégats déjà calculés sur disque (persistent entre redémarrages du
    # conteneur, contrairement à st.cache_data qui ne vit qu'en RAM) : on les
    # relit directement sans rescanner les 1,3 Go de Parquet bruts.
    cached = _read_aggregates()
    if cached is not None:
        return cached

    daily = query("""
        SELECT
            pickup_date,
            COUNT(*)                                         AS total_trips,
            SUM(TOTAL_AMOUNT)                                AS total_revenue,
            AVG(TRIP_DISTANCE)                                AS avg_distance,
            AVG(TOTAL_AMOUNT)                                 AS avg_fare,
            AVG(TIP_AMOUNT / NULLIF(FARE_AMOUNT, 0) * 100)   AS avg_tip_pct
        FROM trips
        WHERE TOTAL_AMOUNT > 0
        GROUP BY 1
        ORDER BY 1
    """)

    hourly = query("""
        SELECT
            pickup_hour,
            COUNT(*)          AS total_trips,
            SUM(TOTAL_AMOUNT) AS total_revenue,
            AVG(TOTAL_AMOUNT) AS avg_fare,
            AVG(TIP_AMOUNT / NULLIF(FARE_AMOUNT, 0) * 100) AS avg_tip_pct,
            AVG(TRIP_DISTANCE) AS avg_distance,
            -- Bornes 60 s - 3 h : écarte compteurs oubliés et courses nulles.
            AVG(CASE WHEN trip_seconds BETWEEN 60 AND 10800
                     THEN TRIP_DISTANCE / (trip_seconds / 3600.0) END) AS avg_speed_mph,
            CASE
                WHEN pickup_hour BETWEEN 0  AND 5  THEN 'Nuit (0h-6h)'
                WHEN pickup_hour BETWEEN 6  AND 9  THEN 'Matin (6h-10h)'
                WHEN pickup_hour BETWEEN 10 AND 16 THEN 'Journée (10h-17h)'
                WHEN pickup_hour BETWEEN 17 AND 20 THEN 'Soir (17h-21h)'
                ELSE                                     'Soirée (21h-0h)'
            END             AS tranche
        FROM trips
        -- Groupement par nom et non par position : ajouter une colonne
        -- dans le SELECT décalait l'index et faisait pointer le GROUP BY
        -- sur un agrégat.
        GROUP BY pickup_hour, tranche
        ORDER BY pickup_hour
    """)

    zones = query("""
        SELECT
            PULOCATIONID                             AS zone_id,
            COUNT(*)                                 AS total_trips,
            SUM(TOTAL_AMOUNT)                        AS total_revenue,
            AVG(TOTAL_AMOUNT)                        AS avg_fare,
            AVG(TRIP_DISTANCE)                       AS avg_distance,
            AVG(TIP_AMOUNT / NULLIF(FARE_AMOUNT, 0) * 100) AS avg_tip_pct
        FROM trips
        GROUP BY 1
        ORDER BY total_trips DESC
        LIMIT 100
    """)

    profile = query("""
        SELECT
            COUNT(*)                                                     AS total_trips,
            AVG(TRIP_DISTANCE)                                           AS avg_distance,
            AVG(TOTAL_AMOUNT)                                            AS avg_fare,
            AVG(TIP_AMOUNT / NULLIF(FARE_AMOUNT, 0) * 100)              AS avg_tip_pct,
            SUM(CASE WHEN TIP_AMOUNT > 0 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*)                                               AS pct_avec_pourboire,
            SUM(CASE WHEN PAYMENT_TYPE = 1 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*)                                               AS pct_carte,
            SUM(CASE WHEN PULOCATIONID IN (132, 138)
                       OR DOLOCATIONID IN (132, 138) THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*)                                       AS pct_aeroport_jfk,
            SUM(CASE WHEN PULOCATIONID = 137
                       OR DOLOCATIONID = 137 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*)                                       AS pct_aeroport_lga,
            AVG(PASSENGER_COUNT)                                         AS avg_passagers
        FROM trips
        WHERE TOTAL_AMOUNT > 0
    """)

    # --- Péage urbain de Manhattan -------------------------------------
    # Entré en vigueur le 5 janvier 2025. La série quotidienne montre la
    # bascule au jour près : 0 course taxée le 3, 399 le 4, 50 944 le 5.
    congestion = query(f"""
        SELECT
            CAST({_TS_PICKUP} AS DATE)                                AS pickup_date,
            COUNT(*)                                                  AS total_trips,
            SUM(CASE WHEN CBD_CONGESTION_FEE > 0 THEN 1 ELSE 0 END)   AS cbd_trips,
            COALESCE(SUM(CBD_CONGESTION_FEE), 0)                      AS cbd_collected
        FROM {SOURCE_TABLE}
        WHERE {_BILLING_FILTER}
        GROUP BY 1
        ORDER BY 1
    """)

    # --- Décomposition de ce que paie le passager ----------------------
    # Le prix moyen n'est pas un bloc : course, pourboire, péages, taxes
    # et surcharges se mesurent séparément.
    revenue_split = query(f"""
        SELECT
            COUNT(*)                     AS total_trips,
            SUM(FARE_AMOUNT)             AS fare,
            SUM(TIP_AMOUNT)              AS tip,
            SUM(TOLLS_AMOUNT)            AS tolls,
            SUM(CONGESTION_SURCHARGE)    AS congestion_surcharge,
            COALESCE(SUM(CBD_CONGESTION_FEE), 0) AS cbd_fee,
            SUM(AIRPORT_FEE)             AS airport_fee,
            SUM(MTA_TAX)                 AS mta_tax,
            SUM(EXTRA)                   AS extra,
            SUM(IMPROVEMENT_SURCHARGE)   AS improvement,
            SUM(TOTAL_AMOUNT)            AS total
        FROM {SOURCE_TABLE}
        WHERE {_BILLING_FILTER}
    """)

    # --- Pourboire par moyen de paiement -------------------------------
    # Les pourboires en espèces ne sont jamais saisis par le compteur :
    # 0,01 % des courses en espèces en portent un, contre 93,9 % par carte.
    # Sans cette lecture, le taux global passe pour un comportement.
    payment = query(f"""
        SELECT
            PAYMENT_TYPE                                              AS payment_type,
            COUNT(*)                                                  AS total_trips,
            SUM(CASE WHEN TIP_AMOUNT > 0 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*)                                            AS pct_avec_pourboire,
            AVG(TIP_AMOUNT / NULLIF(FARE_AMOUNT, 0) * 100)            AS avg_tip_pct,
            AVG(TOTAL_AMOUNT)                                         AS avg_total
        FROM {SOURCE_TABLE}
        WHERE {_BILLING_FILTER}
        GROUP BY 1
        ORDER BY total_trips DESC
    """)

    tables = (daily, hourly, zones, profile, congestion, revenue_split, payment)
    _write_aggregates(tables)
    return tables

if __name__ == "__main__":
    nyc_dashboard.run(load_data)
