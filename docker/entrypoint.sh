#!/bin/sh
set -e

if [ -z "$(ls -A /app/data/yellow_taxi/*.parquet 2>/dev/null)" ]; then
    echo "Aucune donnée trouvée dans /app/data/yellow_taxi — téléchargement initial (peut prendre plusieurs minutes)..."
    python scripts/B_load_local_parquet.py
fi

# Les sept tables que _read_aggregates() exige, nommées ici une par une.
# La garde portait sur « au moins un .parquet dans le dossier », ce qui
# était vrai dès que les agrégats du cockpit ou une version précédente du
# dashboard existaient : le pré-calcul était sauté, _read_aggregates()
# renvoyait None faute des trois dernières tables, et chaque session
# rescannait les 1,3 Go de Parquet bruts sans aboutir — un chargement en
# boucle, sur une page déployée.
for table in daily hourly zones profile congestion revenue_split payment; do
    if [ ! -f "/app/data/aggregates/$table.parquet" ]; then
        echo "Pré-calcul des agrégats du dashboard ($table manquant, une seule fois)..."
        python -c "from streamlit_dashboard_local import load_data; load_data()"
        break
    fi
done

# Les agrégats du cockpit sont d'une autre fabrication que ceux du
# dashboard : ils croisent la zone et le pas du cycle, et c'est J qui les
# écrit, après G. Sans eux le cockpit n'affiche que son message d'absence,
# ce qui en production revient à une page vide.
#
# La garde porte aussi sur le contenu de zone_cycle, et pas seulement sur
# l'existence du fichier : le bloc des nombres lit onze colonnes que J
# n'écrit que depuis la version 10 du payload (répartition des passagers,
# moyens de paiement, minutes, milles, recette). Un fichier laissé par une
# version antérieure existe, donc passait la garde, et nyc_live._city()
# s'arrêtait net au premier rendu. Le schéma se lit sans charger les données.
#
# Elle compte aussi les pas du cycle dow, qui sont passés de 7 à 168 en
# version 11 : les pas portent désormais l'heure. Une table d'avant a les
# bonnes colonnes et le mauvais nombre de pas — elle passait donc la garde
# du schéma, et la carte montrait une semaine de moyennes journalières, où
# le cycle jour/nuit est écrasé. Seules deux colonnes sont lues.
#
# dur_cycle.parquet est la troisième table, écrite par le même script depuis
# la version 10 : la case des durées en lit la distribution complète.
needs_cycles=0
if [ ! -f /app/data/aggregates/zone_cycle.parquet ] \
   || [ ! -f /app/data/aggregates/od_cycle.parquet ] \
   || [ ! -f /app/data/aggregates/dur_cycle.parquet ]; then
    needs_cycles=1
elif ! python -c "
import sys
import pyarrow.parquet as pq
import pyarrow.compute as pc
P = '/app/data/aggregates/zone_cycle.parquet'
cols = set(pq.read_schema(P).names)
need = {'TOTAL_REVENUE', 'TRIPS_PAX', 'PAX_SOLO', 'PAX_DUO', 'PAX_TRIO',
        'TRIPS_PAY', 'PAY_TIP', 'PAY_CARD', 'PAY_CASH', 'SUM_MINUTES'}
if not need <= cols:
    sys.exit(1)
t = pq.read_table(P, columns=['SCOPE', 'STEP'])
steps = pc.unique(t['STEP'].filter(pc.equal(t['SCOPE'], 'dow')))
sys.exit(0 if len(steps) == 168 else 1)
" 2>/dev/null; then
    echo "zone_cycle.parquet est d'une version antérieure — reconstruction."
    needs_cycles=1
fi
if [ "$needs_cycles" = 1 ]; then
    echo "Pré-calcul des cycles du cockpit (une seule fois, ~2min)..."
    python scripts/G_build_aggregates.py
    python scripts/J_build_cycles.py
    # Les matrices de flux de la maquette sortent des mêmes agrégats : les
    # reconstruire ensemble, sinon la carte montrerait la version livrée
    # avec l'image pendant que le reste du cockpit lirait la neuve.
    python scripts/K_city_trips.py
fi

# La maquette de ville, posée avant le premier visiteur. _serve_city() la
# copie aussi, mais depuis Python, donc seulement quand un rendu du cockpit
# appelle meta() : jusque-là static/ est vide et la carte n'a pas ses tours.
# Le conteneur recrée static/ à chaque démarrage, d'où la copie ici plutôt
# qu'une vérification d'existence. cp -f et non cp -n : la source versionnée
# fait foi, une copie laissée par un conteneur précédent peut être périmée.
mkdir -p /app/static
cp -f /app/data/reference/nyc_city.json /app/static/nyc_city.json 2>/dev/null || true

# Pré-calcul du cockpit avant d'accepter des visiteurs. Sans lui, le premier
# visiteur après chaque démarrage attendait que meta() écrive static/
# (une trentaine de secondes sur le VPS, page grise pendant ce temps). Le
# tampon que meta() pose dans static/ évite ensuite à la session Streamlit de
# refaire le calcul. Un échec ici ne doit pas empêcher le démarrage : la
# session le refera, comme avant.
python -c "import nyc_live; nyc_live.meta()" >/dev/null 2>&1 || true

exec streamlit run streamlit_dashboard_local.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
