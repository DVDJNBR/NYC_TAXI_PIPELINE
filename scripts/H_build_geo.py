"""Géométrie des zones TLC et série quotidienne par zone, pour la carte.

Deux sorties, toutes deux lues par le cockpit :

    data/reference/taxi_zones.json   contours et centroïdes des 263 zones

Les séries temporelles par zone sont produites à part, par
scripts/J_build_cycles.py : elles dépendent des Parquet de courses, pas de
la géométrie, et n'ont pas à être recalculées quand la TLC republie ses
contours.

La géométrie est publiée par la TLC en shapefile, projeté en EPSG:2263
(State Plane New York Long Island, en pieds). Le navigateur veut du WGS 84 :
la conversion est faite ici, une fois, pas à chaque affichage.

Pourquoi DuckDB et non geopandas : l'extension `spatial` lit le shapefile,
porte PROJ pour la reprojection et simplifie les contours, le tout sans
ajouter une dépendance au projet. DuckDB est déjà là pour les agrégats.

Les contours sont simplifiés à 30 m avant export. À l'échelle d'une carte de
ville, personne ne voit la différence entre un littoral à 1 m et à 30 m, et
le fichier passe de plusieurs mégaoctets à quelques centaines de kilooctets —
ce qui compte, puisqu'il voyage jusqu'au navigateur.

    .venv/bin/python -I scripts/H_build_geo.py
"""

import json
import pathlib
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "yellow_taxi" / "*.parquet"
REF = ROOT / "data" / "reference"
OUT = ROOT / "data" / "aggregates"

SHP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
GEO_JSON = REF / "taxi_zones.json"

# Même fenêtre et mêmes filtres que scripts/G_build_aggregates.py.
START, END = "2024-01-01", "2025-10-01"
CLEAN = f"""
    trip_distance > 0 AND trip_distance <= 100
    AND total_amount > 0
    AND tpep_pickup_datetime < tpep_dropoff_datetime
    AND tpep_pickup_datetime >= TIMESTAMP '{START}'
    AND tpep_pickup_datetime <  TIMESTAMP '{END}'
"""

# Tolérance de simplification, en pieds : l'unité du système projeté d'origine.
# 100 pieds valent 30 m.
SIMPLIFY_FT = 100


def _shapefile(workdir: pathlib.Path) -> pathlib.Path:
    """Télécharge et dézippe le shapefile officiel, rend le chemin du .shp."""
    archive = workdir / "taxi_zones.zip"
    print(f"→ {SHP_URL}")
    urllib.request.urlretrieve(SHP_URL, archive)
    with zipfile.ZipFile(archive) as z:
        # Extraction nominative : une archive décrit ses propres chemins, et
        # un chemin absolu ou remontant écrirait hors du dossier de travail.
        for member in z.namelist():
            name = pathlib.PurePosixPath(member).name
            if name.startswith("taxi_zones."):
                with z.open(member) as src, open(workdir / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    shp = workdir / "taxi_zones.shp"
    if not shp.exists():
        sys.exit("Archive TLC sans taxi_zones.shp.")
    return shp


def build_geometry(con) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        shp = _shapefile(pathlib.Path(tmp))
        rows = con.execute(f"""
            WITH z AS (
                SELECT
                    LocationID::INT                        AS id,
                    zone                                   AS name,
                    borough,
                    ST_Transform(ST_Simplify(geom, {SIMPLIFY_FT}),
                                 'EPSG:2263', 'EPSG:4326', true) AS g
                FROM ST_Read('{shp}')
            )
            SELECT id, name, borough,
                   ST_X(ST_Centroid(g)) AS lon,
                   ST_Y(ST_Centroid(g)) AS lat,
                   ST_AsGeoJSON(g)      AS geojson
            FROM z
            ORDER BY id
        """).fetchall()

    # Un objet par zone plutôt qu'une FeatureCollection : le composant veut
    # une table indexée par identifiant, pas un format d'échange.
    zones = [
        {"id": i, "name": n, "borough": b,
         "lon": round(lon, 5), "lat": round(lat, 5),
         "geometry": json.loads(gj)}
        for i, n, b, lon, lat, gj in rows
    ]
    REF.mkdir(parents=True, exist_ok=True)
    GEO_JSON.write_text(json.dumps(zones, separators=(",", ":")))
    size = GEO_JSON.stat().st_size / 1024
    print(f"✓ {GEO_JSON.name:<22} {len(zones):>4} zones  {size:.0f} ko")


def main() -> None:
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    build_geometry(con)
    print("Séries par zone : lancer scripts/J_build_cycles.py.")


if __name__ == "__main__":
    main()
