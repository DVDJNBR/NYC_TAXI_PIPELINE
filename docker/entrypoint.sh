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
if [ ! -f /app/data/aggregates/zone_cycle.parquet ] \
   || [ ! -f /app/data/aggregates/od_cycle.parquet ]; then
    echo "Pré-calcul des cycles du cockpit (une seule fois, ~2min)..."
    python scripts/G_build_aggregates.py
    python scripts/J_build_cycles.py
fi

exec streamlit run streamlit_dashboard_local.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
