"""Profils cycliques : la journée, la semaine et l'année types.

Le cockpit ne rejoue pas une chronologie, il tourne en boucle sur un cycle.
Trois cycles, exclusifs l'un de l'autre :

    hour     24 pas, minuit -> minuit, tous jours confondus
    dow     168 pas, lundi 0 h -> dimanche 23 h
    month    12 pas, janvier -> décembre, heures confondues

Seule la semaine porte l'heure dans son pas. Un pas d'une journée entière y
écrasait le cycle jour/nuit, alors que ses sept journées ont des formes
vraiment différentes. L'année, elle, garde un pas par mois : ses douze
profils de 24 h ne s'écartent que de 1,5 à 4,4 % de leur forme moyenne, donc
l'heure n'y aurait répété douze fois la même vague. Voir le commentaire de
STEPS pour les mesures.

Toutes les valeurs sont ramenées à une **heure moyenne** du pas, jamais à un
total brut : le dénominateur est le nombre d'occurrences du pas fois le
nombre d'heures qu'il couvre. Voir le commentaire de HOURS. Sans cela le cycle mensuel mentirait : la période couvre janvier
2024 à septembre 2025, donc janvier est observé deux fois et novembre une
seule. Un total ferait chuter la fin d'année de moitié, pour une raison de
comptage et non de trafic. Le dénominateur est le nombre de jours
réellement observés pour chaque pas, lu dans daily.parquet.

Trois tables en sortie, toutes lues par les composants du cockpit :

    data/aggregates/zone_cycle.parquet   zone x 204 pas x 15 colonnes
    data/aggregates/od_cycle.parquet     flux entre secteurs x 204 pas
    data/aggregates/dur_cycle.parquet    204 pas x 31 bacs de duree

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

# Les bacs de la distribution des durées : trente-et-un bacs de deux
# minutes, le dernier recueillant tout ce qui dépasse une heure. Deux
# minutes parce que le sommet de la distribution tient dans 6 à 10 minutes :
# des bacs de cinq l'écraseraient en deux barres, des bacs d'une minute
# donneraient 61 colonnes pour une case de 232 px de large.
DUR_BINS, DUR_WIDTH = 31, 120

# Expression du pas pour chaque cycle. dayofweek de DuckDB rend 0 le
# dimanche ; le décalage met le lundi en tête, comme partout ailleurs dans
# l'application.
#
# dow multiplie son rang par 24 et y ajoute l'heure. Sans cela un pas
# moyennait les vingt-quatre heures d'une journée et le cycle jour/nuit
# disparaissait de la semaine.
# Mesuré sur les trente-sept zones qui dépassent cinquante courses par pas :
# le volume varie d'un facteur 24,4 en médiane entre l'heure de pointe et
# l'heure creuse de la zone, et jusqu'à 171,2 pour la zone 138 — une course
# à 4 h contre deux cent soixante-sept à 14 h. La zone 48, elle, ne varie
# que d'un facteur 6,4. Leurs moyennes journalières sont voisines, 141 et
# 119 : la carte les dessinait donc comme des jumelles, et montrait un état
# de ville qui n'existe à aucune heure.
STEPS = {
    "hour": "hour(tpep_pickup_datetime)",
    "dow": "((dayofweek(tpep_pickup_datetime) + 6) % 7) * 24 "
           "+ hour(tpep_pickup_datetime)",
    "month": "month(tpep_pickup_datetime) - 1",
}

# Combien d'heures un pas couvre. Un pas d'heure en couvre une, un pas de dow
# aussi — « mardi 14 h » — mais un pas de month couvre le mois entier, toutes
# heures confondues.
#
# Ce diviseur met les trois cycles dans la même unité : la course par heure.
# Sans lui, une colonne du cycle de l'année valait vingt-quatre fois la même
# colonne du cycle de la journée, pour la seule raison qu'un mois a vingt-
# quatre heures par jour. La carte, qui empile un bloc par vingt courses à
# l'heure, montait alors à 304 blocs sur l'année contre 33 sur la semaine, et
# le ruban annonçait 127 000 courses là où la ville en fait 5 300.
HOURS = {"hour": 1, "dow": 1, "month": 24}


def denominators() -> pd.DataFrame:
    """Nombre de jours observés derrière chaque pas de chaque cycle.

    Lu dans daily.parquet plutôt que dans les Parquet bruts : la liste des
    journées y est déjà, et la relire coûterait un scan complet de plus.

    Les pas de dow portent l'heure, mais le dénominateur ne la regarde pas :
    les vingt-quatre heures d'un même lundi partagent les quatre-vingt-onze
    lundis de la période. Le rapport reste donc « courses à 14 h un lundi
    moyen ».

    Deux heures par an échappent à cette règle, aux changements d'heure : le
    2 h du dimanche de mars n'existe pas, et celui d'octobre compte double.
    Sur quatre-vingt-onze dimanches, ces deux pas sont faux de 2 % environ.
    Corriger coûterait un calendrier de fuseau pour deux cases sur 204.
    """
    if not DAILY.exists():
        sys.exit("data/aggregates/daily.parquet manquant : lancer d'abord G.")
    d = pd.read_parquet(DAILY, columns=["PICKUP_DATE"])
    dates = pd.to_datetime(d["PICKUP_DATE"])
    dates = dates[(dates >= START) & (dates < END)]

    rows = [("hour", h, len(dates)) for h in range(24)]
    # dayofweek de pandas met déjà le lundi à 0, comme l'expression SQL.
    dow = dates.dt.dayofweek.value_counts()
    rows += [("dow", int(k) * 24 + h, int(v))
             for k, v in dow.items() for h in range(24)]
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
                sum(trip_distance) AS SUM_DISTANCE,
                -- Le dénominateur de la répartition des passagers, et non
                -- TOTAL_TRIPS : passenger_count est absent sur 15,7 % des
                -- courses, et son taux d'absence suit l'heure — 9,9 % à
                -- 16 h, 39,6 % à 4 h. Laissé au dénominateur, il ferait
                -- raconter ses trous à la case plutôt que les passagers.
                count(*) FILTER (WHERE passenger_count >= 1) AS TRIPS_PAX,
                count(*) FILTER (WHERE passenger_count = 1)  AS PAX_SOLO,
                count(*) FILTER (WHERE passenger_count = 2)  AS PAX_DUO,
                count(*) FILTER (WHERE passenger_count >= 3) AS PAX_TRIO,
                -- Le moyen de paiement, en trois parts, sur les seules
                -- courses qui en déclarent un. payment_type = 0 ne figure
                -- pas au dictionnaire de la TLC, qui va de 1 à 6 : c'est
                -- 15,66 % des courses, et ce sont exactement les mêmes
                -- lignes que celles sans passenger_count (15,66 % de
                -- recouvrement, 0,00 % d'un côté sans l'autre). Les deux
                -- champs tombent donc ensemble.
                --
                -- Ce n'est pas un vendeur en particulier : les trois y
                -- passent — 16,9 % chez le 2, 11,4 % chez le 1, 100 % chez
                -- le 6 qui ne pèse que 12 000 courses. C'est une dérive de
                -- déclaration dans le temps : 4,1 % en janvier 2024,
                -- 27,6 % en juin 2025. Les deux cases de répartition
                -- reposent donc sur une base qui se réduit au fil de la
                -- période, et le cycle du mois en est le plus affecté
                -- puisqu'il moyenne deux années de janvier à septembre.
                --
                -- Le pourboire n'est relevé que sur la carte ; un
                -- pourboire en espèces n'entre jamais dans les données.
                count(*) FILTER (WHERE payment_type IN (1, 2)) AS TRIPS_PAY,
                count(*) FILTER (WHERE payment_type = 1
                                   AND tip_amount > 0)         AS PAY_TIP,
                count(*) FILTER (WHERE payment_type = 1
                                   AND tip_amount <= 0)        AS PAY_CARD,
                count(*) FILTER (WHERE payment_type = 2)       AS PAY_CASH,
                sum(date_diff('second', tpep_pickup_datetime,
                              tpep_dropoff_datetime)) / 60.0 AS SUM_MINUTES
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


def dur_cycle(con) -> pd.DataFrame:
    """Distribution des durées de course, par cycle et par pas.

    Bacs de deux minutes jusqu'à une heure, puis un bac de débordement : la
    case du cockpit trace cette distribution et la compare à la moyenne du
    cycle. Un seul nombre n'y suffisait pas — la médiane ne passe que de 11 à
    15 minutes sur la journée, alors que la part au-delà d'une demi-heure est
    multipliée par 5,3 et celle au-delà d'une heure par 46. C'est la queue
    qui porte l'information, et il faut la forme entière pour la montrer.

    Sans dénominateur par occurrence, contrairement aux deux autres tables :
    la case ne lit que des parts à l'intérieur d'un pas, et tous les bacs
    d'un même pas partagent le même diviseur. Le ramener à la journée
    moyenne diviserait numérateur et dénominateur par le même nombre.

    Ville entière et non par zone : 43 pas x 31 bacs font 1 333 lignes, là où
    la même grille par zone en ferait 300 000 pour une case qui somme de
    toute façon la ville.
    """
    t0 = time.time()
    selects = []
    for scope, expr in STEPS.items():
        selects.append(f"""
            SELECT '{scope}' AS SCOPE, {expr} AS STEP,
                   least({DUR_BINS - 1},
                         cast(floor(date_diff('second', tpep_pickup_datetime,
                                              tpep_dropoff_datetime)
                                    / {DUR_WIDTH}.0) AS integer)) AS BIN,
                   count(*) AS TOTAL_TRIPS
            FROM read_parquet('{RAW}', union_by_name = true)
            WHERE {CLEAN}
            GROUP BY 1, 2, 3
        """)
    df = con.execute(" UNION ALL ".join(selects)).fetchdf()
    print(f"  durees       {len(df):>6} lignes  {time.time() - t0:.1f} s")
    return df


def main() -> None:
    if not list((ROOT / "data" / "yellow_taxi").glob("*.parquet")):
        sys.exit("Aucun Parquet brut dans data/yellow_taxi/.")
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    days = denominators()

    zones = zone_cycle(con).merge(days, on=["SCOPE", "STEP"])
    zones["HOURS"] = zones["SCOPE"].map(HOURS)
    # La distance moyenne se prend sur les totaux bruts, avant toute
    # normalisation : c'est un rapport de deux sommes du même pas, donc tout
    # dénominateur commun s'y annulerait de toute façon.
    zones["AVG_DISTANCE"] = zones["SUM_DISTANCE"] / zones["TOTAL_TRIPS"]
    # Tout le reste est un numérateur, et passe par le même dénominateur :
    # le nombre d'occurrences du pas, fois le nombre d'heures qu'il couvre.
    # L'unité de sortie est donc « par heure moyenne » sur les trois cycles.
    # Les garder en brut les rendrait incomparables d'un pas à l'autre, pour
    # la seule raison que janvier est observé deux fois et dure 744 heures.
    per_hour = zones["DAYS"] * zones["HOURS"]
    for col in ("TOTAL_TRIPS", "TOTAL_REVENUE",
                "TRIPS_PAX", "PAX_SOLO", "PAX_DUO", "PAX_TRIO",
                "TRIPS_PAY", "PAY_TIP", "PAY_CARD", "PAY_CASH",
                "SUM_MINUTES"):
        zones[col] = zones[col] / per_hour
    zones = zones.drop(columns=["SUM_DISTANCE", "DAYS", "HOURS"])
    zones.to_parquet(OUT / "zone_cycle.parquet", index=False)

    dur_cycle(con).to_parquet(OUT / "dur_cycle.parquet", index=False)

    od = od_cycle(con).merge(days, on=["SCOPE", "STEP"])
    od["HOURS"] = od["SCOPE"].map(HOURS)
    per_hour = od["DAYS"] * od["HOURS"]
    od["TOTAL_TRIPS"] = od["TOTAL_TRIPS"] / per_hour
    od["TOTAL_REVENUE"] = od["TOTAL_REVENUE"] / per_hour
    od = od.drop(columns=["DAYS", "HOURS"])
    od.to_parquet(OUT / "od_cycle.parquet", index=False)

    for name in ("zone_cycle", "od_cycle", "dur_cycle"):
        size = (OUT / f"{name}.parquet").stat().st_size / 1024
        print(f"✓ {name:<12} {size:>6.0f} ko")


if __name__ == "__main__":
    main()
