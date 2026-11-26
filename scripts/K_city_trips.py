"""Les matrices de flux de la maquette, et les deux aéroports qu'elle dessine.

`data/reference/nyc_city.json` porte la maquette : les emprises au sol, les
2 601 bâtiments, un vivier de points par secteur, et les flux entre secteurs.
La géométrie ne change que si la TLC republie ses contours ; les flux, eux,
changent à chaque rechargement des Parquet. Ce script ne réécrit donc que la
partie qui dépend des courses, et laisse la maquette intacte.

Deux corrections sont faites ici, et c'est pourquoi le script existe :

1. `trips` portait les vingt-quatre heures de la journée moyenne, et rien
   d'autre. Sur le cycle de la semaine ou de l'année, le composant n'avait
   que leur moyenne — un seul nombre, le même aux sept ou douze pas. Le
   nombre de taxis à l'écran ne bougeait donc pas d'un lundi à un samedi.
   `trips` devient un tableau par cycle.

   L'unité est la course par heure, sur les trois cycles. C'est désormais
   J_build_cycles.py qui s'en charge, par son dictionnaire HOURS : le pas de
   month y est divisé par vingt-quatre à l'écriture d'od_cycle.parquet. Ce
   script lit donc les flux tels quels. Il a porté cette division lui-même
   un temps, et la carte à blocs ne l'avait pas : elle montait à 304 étages
   sur l'année contre 33 sur la semaine. La division appartient au builder,
   où tous les consommateurs en profitent.

2. Les aéroports étaient invisibles. `od_cycle` les sépare du Queens, mais la maquette n'avait de vivier de points qu'à Newark,
   où les taxis jaunes ne chargent jamais (3 courses sur la période, contre
   5 119 à JFK et 3 382 à LaGuardia). Les 7 à 10 % de courses qui touchent
   un aéroport se posaient donc à Newark ou nulle part. Ce script ajoute un
   vivier et un nœud pour JFK et pour LaGuardia, qui sont chacun un secteur
   des agrégats.

3. La carte ne dessine plus les flux secteur à secteur mais un champ de
   courants, calculé dans le navigateur à partir de tous les trajets zone à
   zone. Ce script les écrit dans `data/reference/nyc_flows.json` : pour
   chaque couple (départ, arrivée) de zones distinctes, les courses par
   heure en moyenne sur la période et la durée médiane ; pour chaque zone,
   son centroïde et son secteur. Ce sont les Parquet bruts qui sont lus, les
   agrégats de J s'arrêtant au secteur.

    PYTHONPATH=. .venv/bin/python scripts/K_city_trips.py
"""

import json
import pathlib
import random

import duckdb

import nyc_zones
import J_build_cycles as J

ROOT = pathlib.Path(__file__).resolve().parent.parent
OD = ROOT / "data" / "aggregates" / "od_cycle.parquet"
ZONES = ROOT / "data" / "reference" / "taxi_zones.json"
CITY = ROOT / "data" / "reference" / "nyc_city.json"
FLOWS = ROOT / "data" / "reference" / "nyc_flows.json"

# Le nombre de pas par cycle, à tenir d'accord avec STEPS de
# J_build_cycles.py et CYCLES de nyc_live.py.
CYCLES = {"hour": 24, "dow": 7, "month": 12}

# Les deux aéroports que la maquette dessine, dans l'ordre où le composant
# les reçoit. Newark n'y est pas : les taxis jaunes n'y chargent pas, et son
# vivier existe déjà comme secteur 8 de la maquette.
AIR = [(132, "JFK"), (138, "LGA")]
POOL_N = 2600          # autant que les viviers de secteurs déjà en place
SEED = 20240101        # un tirage reproductible : le fichier est versionné


def _rings(geometry: dict) -> list:
    """Tous les anneaux extérieurs d'une géométrie, Polygon ou MultiPolygon."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    return [poly[0] for poly in geometry["coordinates"]]


def _area(ring: list) -> float:
    """Aire du polygone, par la formule du lacet. Sert à répartir le tirage
    entre les anneaux d'un MultiPolygon."""
    a = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % len(ring)][0], ring[(i + 1) % len(ring)][1]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def _inside(ring: list, x: float, y: float) -> bool:
    """Lancer de rayon : le point est dedans si le bord est croisé un nombre
    impair de fois à sa droite."""
    hit = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                hit = not hit
    return hit


def sample(geometry: dict, n: int, rng: random.Random) -> list:
    """Un vivier de n points tirés dans la zone.

    Tirage par rejet dans la boîte englobante, anneau par anneau, au prorata
    de la surface : une piste d'aéroport est un polygone concave, et un
    tirage uniforme dans la boîte aurait posé des taxis dans la baie.
    """
    rings = _rings(geometry)
    areas = [_area(r) for r in rings]
    total = sum(areas) or 1.0
    out = []
    for ring, area in zip(rings, areas):
        want = round(n * area / total)
        if want <= 0:
            continue
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        w, e, s, nn = min(xs), max(xs), min(ys), max(ys)
        got, tries, cap = 0, 0, want * 400
        while got < want and tries < cap:
            tries += 1
            x = rng.uniform(w, e)
            y = rng.uniform(s, nn)
            if _inside(ring, x, y):
                out.append((x, y))
                got += 1
    if not out:
        raise RuntimeError("vivier vide : la géométrie n'a pas d'intérieur")
    # Le rejet rend parfois un point de moins qu'attendu, sur un anneau très
    # découpé. On complète en recyclant : le composant pioche au hasard dans
    # le vivier, et un doublon sur 2 600 ne se voit pas.
    while len(out) < n:
        out.append(out[rng.randrange(len(out))])
    return out[:n]


def matrices(con) -> dict:
    """Les flux secteur à secteur, un tableau par cycle, en courses/heure."""
    areas = list(nyc_zones.AREAS)
    idx = {a: i for i, a in enumerate(areas)}
    out = {}
    for scope, steps in CYCLES.items():
        grid = [[[0.0] * len(areas) for _ in areas] for _ in range(steps)]
        rows = con.execute(f"""
            SELECT STEP, PU_AREA, DO_AREA, sum(TOTAL_TRIPS)
            FROM read_parquet('{OD.as_posix()}')
            WHERE SCOPE = '{scope}'
            GROUP BY 1, 2, 3
        """).fetchall()
        for step, pu, do, v in rows:
            if pu not in idx or do not in idx or step >= steps:
                continue
            grid[step][idx[pu]][idx[do]] += float(v)
        out[scope] = [[[round(v, 2) for v in row] for row in m] for m in grid]
    return out


def zone_flows(con, zones: dict) -> dict:
    """Tous les trajets zone à zone, pour le champ de courants de la carte.

    Courses par heure : le compte de la paire divisé par les heures de la
    période, comme les cycles de J. Durée : la médiane des courses de la
    paire, en minutes. Les trajets internes à une zone sont gardés à part du
    calcul côté navigateur, qui les ignore faute de direction.
    """
    days = J.denominators()
    hours = int(days[(days.SCOPE == "hour") & (days.STEP == 0)].DAYS.iloc[0]) * 24
    rows = con.execute(f"""
        SELECT PULocationID::INT, DOLocationID::INT, count(*),
               median(date_diff('second', tpep_pickup_datetime,
                                tpep_dropoff_datetime)) / 60.0
        FROM read_parquet('{J.RAW.as_posix()}', union_by_name = true)
        WHERE {J.CLEAN}
          AND PULocationID BETWEEN 1 AND 263
          AND DOLocationID BETWEEN 1 AND 263
          AND PULocationID <> DOLocationID
        GROUP BY 1, 2
    """).fetchall()
    area = nyc_zones.areas()
    zc = {str(z): [round(float(zones[z]["lon"]), 4), round(float(zones[z]["lat"]), 4),
                   area.get(z, "EWR")]
          for z in zones}
    od = [[o, d, round(n / hours, 4), round(m, 1)]
          for o, d, n, m in rows if str(o) in zc and str(d) in zc]
    return {"zc": zc, "od": od}


def main() -> None:
    city = json.loads(CITY.read_text())
    zones = {z["id"]: z for z in json.loads(ZONES.read_text())}
    con = duckdb.connect()
    rng = random.Random(SEED)

    city["trips"] = matrices(con)

    # Les viviers des aéroports prennent la suite des neuf secteurs, dans le
    # même emballage : une origine, une échelle, des entiers. Le composant
    # pioche dedans par indice, sans savoir qu'il s'agit d'aéroports.
    b = city["pools"]["b"]
    sc = city["pools"]["sc"]
    # Les viviers sont tronqués au nombre de secteurs avant d'ajouter ceux des
    # aéroports : sans cela, deux exécutions du script en empilaient deux jeux
    # et les indices écrits dans city["air"] désignaient les derniers, laissant
    # les premiers dans le fichier sans que rien ne les lise. Le script doit
    # pouvoir tourner deux fois de suite et rendre le même fichier.
    city["pools"]["pt"] = city["pools"]["pt"][:len(city["order"])]
    air = []
    for zid, code in AIR:
        z = zones[zid]
        pts = sample(z["geometry"], POOL_N, rng)
        flat = []
        for x, y in pts:
            flat.append(int(round((x - b[0]) * sc)))
            flat.append(int(round((y - b[1]) * sc)))
        city["pools"]["pt"].append(flat)
        air.append({"code": code, "lonlat": [z["lon"], z["lat"]],
                    "pool": len(city["pools"]["pt"]) - 1})
    # Chaque nœud porte le nom de son secteur dans les agrégats : la carte
    # le relie aux flux par ce nom, comme les neuf secteurs de la maquette.
    names = dict(zip((z for z, _ in AIR), nyc_zones.AIRPORT_AREAS))
    for n, (zid, _) in zip(air, AIR):
        n["area"] = names[zid]
    city["air"] = {"node": air}

    CITY.write_text(json.dumps(city, separators=(",", ":")))
    fl = zone_flows(con, zones)
    FLOWS.write_text(json.dumps(fl, separators=(",", ":")))
    print(f"✓ {FLOWS.name:<18} {FLOWS.stat().st_size / 1024:.0f} ko  "
          f"{len(fl['od'])} trajets zone à zone")
    size = CITY.stat().st_size / 1024
    print(f"✓ {CITY.name:<18} {size:.0f} ko")
    for scope, steps in CYCLES.items():
        tot = [sum(sum(r) for r in m) for m in city["trips"][scope]]
        print(f"  {scope:<6} {steps:>2} pas   "
              f"{min(tot):>8,.0f} … {max(tot):>8,.0f} courses/heure")
    for n in air:
        print(f"  {n['code']}  vivier {n['pool']}  "
              f"{len(city['pools']['pt'][n['pool']]) // 2} points  "
              f"{n['lonlat']}")


if __name__ == "__main__":
    main()
