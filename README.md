# NYC Taxi Pipeline

Mode d'emploi pour reproduire le brief NYC Taxi avec Snowflake et dbt.

## Prérequis

- Python 3.11+
- Compte Snowflake (essai gratuit)
- Git

## Installation

```bash
git clone <repository>
cd nyc_taxi_pipeline
uv sync
```

## Configuration

Créer le fichier `.env` avec vos credentials Snowflake :

```env
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username  
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE_PASSWORD=your_role_password
```

## Étapes du Brief

### 1. Configuration Snowflake (Étape 1.1)

```bash
inv test-connection
```

Crée l'infrastructure :
- Warehouse `NYC_TAXI_WH`
- Database `NYC_TAXI_DB` 
- Schémas `RAW`, `STAGING`, `FINAL`
- Rôle `NYCTRANSFORM`

### 2. Chargement des Données (Étape 1.2)

```bash
inv load-data
```

Charge les données 2024-2025 :
- Télécharge les fichiers Parquet depuis NYC Open Data
- Upload vers Snowflake RAW.yellow_taxi_trips
- ~77M lignes chargées

### 3. Analyse et Nettoyage (Étape 1.3)

```bash
inv data-analysis
```

Analyse la qualité des données :
- Détecte les valeurs manquantes (16.24%)
- Identifie les montants négatifs (3.67%)
- Génère le rapport `reports/raw_data_quality_report.md`

### 4. Transformations (Étape 1.4)

```bash
inv transformations
```

Crée les tables STAGING et FINAL :
- `STAGING.clean_trips` : Données nettoyées
- `FINAL.daily_summary` : Résumés quotidiens
- `FINAL.zone_analysis` : Analyse par zone
- `FINAL.hourly_patterns` : Patterns horaires

### 5. Option Avancée : dbt Core

```bash
inv dbt-transformations
```

Utilise dbt pour les transformations :
- Modèles dans `nyc_taxi_pipeline/models/`
- Tests de qualité automatiques
- Documentation auto-générée

## Analyses et Rapports

### Générer les graphiques

```bash
inv generate-report
```

Crée les visualisations matplotlib dans `reports/`.

### Dashboard interactif

```bash
inv dashboard
```

Lance Streamlit sur http://localhost:8501

### Analyse des données RAW

```bash
inv raw-analysis  
```

Lance Jupyter avec l'analyse de qualité.

## Pipeline Complet

Pour exécuter toutes les étapes d'un coup :

```bash
inv full-pipeline
```

## Résultats

- **77M lignes** de données NYC Taxi (2024-2025)
- **Architecture 3 couches** RAW → STAGING → FINAL
- **5 tables analytiques** dans FINAL
- **Dashboard Streamlit** interactif
- **Tests dbt** automatiques

## Structure

```
scripts/
├── A_snowflake_config.py    # Étape 1.1 : Infrastructure
├── B_load_data.py           # Étape 1.2 : Chargement données
├── B_load_local_parquet.py  # Alternative : DL Parquet strict (sans Snowflake)
├── C_data_analysis.py       # Étape 1.3 : Analyse qualité
├── D_transformations.py     # Étape 1.4 : Transformations
├── E_generate_report.py     # Graphiques matplotlib
└── F_dbt_transformations.py # Option dbt Core

SQL/
├── Snowflake/              # Requêtes infrastructure
└── dbt/                    # Modèles dbt

nyc_taxi_pipeline/          # Projet dbt Core
reports/                    # Analyses et graphiques
streamlit_dashboard_local.py# Dashboard web (DuckDB sur les Parquet)
nyc_dashboard.py            # Rendu du dashboard, partagé
nyc_content.py              # Onglets À propos et Pipeline de données
nyc_theme.py                # Thème visuel et composants
tasks.py                    # Commandes Invoke
```

---

## ⚡ Alternative In-Memory (VPS / Sans Snowflake)

Dans le cas où le compte gratuit Snowflake a expiré ou pour faire tourner le Dashboard sur un VPS restreint en ressources, deux scripts alternatifs "locaux" sont fournis :

1. **Obtention des données :**
   Exécutez `python scripts/B_load_local_parquet.py`. Les fichiers Parquets seront téléchargés en mode pur et stockés dans `/data/yellow_taxi/`.
   
2. **Dashboard Local (DuckDB) :**
   Exécutez `streamlit run streamlit_dashboard_local.py`. Cette application lit directement le dossier `/data/yellow_taxi/*.parquet` ultra-rapidement sans nécessiter de base distante.

*Veillez à supprimer ou ignorer les gros fichiers `.parquet` si vous poussez sur Github.*