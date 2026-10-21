"""
Tasks Invoke pour le projet NYC Taxi Pipeline
Automatise l'exécution des scripts Snowflake et DBT
"""

from invoke.tasks import task
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Charger les variables d'environnement
load_dotenv()

# Console Rich
console = Console()

# Configuration des chemins
SCRIPTS_DIR = Path("scripts")
SQL_DIR = Path("SQL/Snowflake")
DATA_DIR = Path("data/yellow_taxi")

@task
def create_env_template(c):
    """Créer un template .env avec les variables nécessaires"""
    console.print("📝 Création du template .env...", style="blue")
    c.run("python scripts/setup_env_template.py", pty=True)

@task
def setup_env(c):
    """Vérifier et configurer l'environnement"""
    console.print("🔧 Vérification de l'environnement...", style="blue")
    
    # Vérifier les variables d'environnement nécessaires pour ton script
    required_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER", 
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ROLE_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        console.print(f"❌ Variables manquantes dans .env: {', '.join(missing_vars)}", style="red")
        return False
    
    console.print("✅ Variables d'environnement OK", style="green")
    return True

@task
def test_connection(c):
    """Tester la connexion Snowflake (création infrastructure)"""
    console.print("🔌 Test de connexion Snowflake...", style="blue")
    c.run("python scripts/A_snowflake_config.py", pty=True)

@task
def fix_permissions(c):
    """Corriger les permissions manquantes"""
    console.print("🔧 Correction des permissions...", style="blue")
    c.run("python scripts/fix_permissions.py", pty=True)

@task
def test_infrastructure(c):
    """Tester que l'infrastructure Snowflake fonctionne"""
    console.print("🧪 Test de l'infrastructure créée...", style="blue")
    c.run("python scripts/test_snowflake_setup.py", pty=True)



@task
def create_infrastructure(c):
    """Créer l'infrastructure Snowflake (warehouse, database, schemas)"""
    print("🏗️ Création de l'infrastructure Snowflake...")
    
    sql_files = [
        "create_infrastructure.sql",
        "create_role.sql", 
        "grant_permissions.sql"
    ]
    
    for sql_file in sql_files:
        sql_path = SQL_DIR / sql_file
        if sql_path.exists():
            print(f"📄 Exécution de {sql_file}...")
            # TODO: Ajouter l'exécution SQL via snowflake-connector
        else:
            print(f"⚠️ Fichier manquant: {sql_path}")

@task
def create_tables(c):
    """Créer les tables dans Snowflake"""
    print("📊 Création des tables...")
    
    sql_files = [
        "create_taxi_trips_table.sql",
        "infer_parquet_schema.sql"
    ]
    
    for sql_file in sql_files:
        sql_path = SQL_DIR / sql_file
        if sql_path.exists():
            print(f"📄 Exécution de {sql_file}...")
            # TODO: Ajouter l'exécution SQL
        else:
            print(f"⚠️ Fichier manquant: {sql_path}")

@task
def load_data(c):
    """Étape 1.2 : Chargement des données 2024-2025"""
    console.print("📥 Étape 1.2 : Chargement des données...", style="blue")
    c.run("python scripts/B_load_data.py", pty=True)

@task
def data_analysis(c):
    """Étape 1.3 : Analyse et nettoyage des données"""
    console.print("🔍 Étape 1.3 : Analyse des données...", style="blue")
    c.run("python scripts/C_data_analysis.py", pty=True)

@task
def transformations(c):
    """Étape 1.4 : Transformations de base"""
    console.print("🔄 Étape 1.4 : Transformations...", style="blue")
    c.run("python scripts/D_transformations.py", pty=True)

@task
def generate_report(c):
    """Générer un rapport HTML visuel"""
    console.print("📊 Génération du rapport HTML...", style="blue")
    c.run("python scripts/E_generate_report.py", pty=True)

@task
def dbt_transformations(c):
    """Transformations avec dbt Core"""
    console.print("🔄 Transformations dbt Core...", style="blue")
    c.run("python scripts/F_dbt_transformations.py", pty=True)

@task
def raw_analysis(c):
    """Lancer l'analyse des données RAW"""
    console.print("🔍 Analyse des données RAW...", style="blue")
    console.print("📊 Ouvrir: reports/raw_data_analysis.ipynb", style="cyan")
    c.run("jupyter lab reports/raw_data_analysis.ipynb", pty=True)

@task
def dashboard(c):
    """Lancer le dashboard Streamlit interactif"""
    console.print("🚀 Dashboard Streamlit interactif...", style="blue")
    console.print("🌐 Ouverture automatique: http://localhost:8501", style="cyan")
    c.run("streamlit run streamlit_dashboard_local.py", pty=True)

@task
def jupyter_dashboard(c):
    """Lancer le dashboard Jupyter (alternatif)"""
    console.print("📊 Dashboard Jupyter...", style="blue")
    console.print("📊 Ouvrir: reports/nyc_taxi_dashboard.ipynb", style="cyan")
    c.run("jupyter lab reports/nyc_taxi_dashboard.ipynb", pty=True)

@task
def run_transformations(c):
    """Exécuter les transformations SQL (STAGING → FINAL)"""
    print("🔄 Exécution des transformations...")
    # TODO: Ajouter les scripts de transformation

@task
def run_tests(c):
    """Exécuter les tests de qualité des données"""
    print("🧪 Tests de qualité des données...")
    # TODO: Ajouter les tests de validation

@task
def full_pipeline(c):
    """Exécuter le pipeline complet selon le brief"""
    console.print(Panel.fit("🚀 Pipeline NYC Taxi - Tronc Commun", style="bold green"))
    
    # Étapes exactes du brief
    steps = [
        ("setup_env", "1.1 Vérification environnement"),
        ("test_connection", "1.1 Configuration Snowflake"),
        ("load_data", "1.2 Chargement des données"),
        ("data_analysis", "1.3 Analyse et nettoyage"),
        ("transformations", "1.4 Transformations de base")
    ]
    
    for step_func, step_desc in steps:
        console.print(f"\n📋 [blue]Étape:[/blue] {step_desc}")
        try:
            globals()[step_func](c)
            console.print(f"✅ [green]{step_desc} - Terminé[/green]")
        except Exception as e:
            console.print(f"❌ [red]{step_desc} - Erreur: {e}[/red]")
            break
    
    console.print(Panel.fit("🎉 Tronc commun terminé!", style="bold green"))

@task
def clean_logs(c):
    """Nettoyer les fichiers de logs"""
    print("🧹 Nettoyage des logs...")
    c.run("rm -f logs/*.log", warn=True)
    print("✅ Logs nettoyés")

@task
def status(c):
    """Afficher le statut du projet"""
    # Créer un panel principal
    console.print(Panel.fit("📊 NYC Taxi Pipeline - Statut", style="bold blue"))
    
    # Créer un tableau pour les statistiques
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Ressource", style="cyan")
    table.add_column("Quantité", justify="right", style="green")
    table.add_column("Statut", justify="center")
    
    # Vérifier les fichiers de données
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    table.add_row("📁 Fichiers Parquet", str(len(parquet_files)), "✅" if parquet_files else "❌")
    
    # Vérifier les scripts SQL
    sql_files = list(SQL_DIR.glob("*.sql"))
    table.add_row("📄 Scripts SQL", str(len(sql_files)), "✅" if sql_files else "❌")
    
    # Vérifier les logs
    log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
    table.add_row("📋 Fichiers logs", str(len(log_files)), "✅" if log_files else "⚪")
    
    console.print(table)
    
    # Panel des commandes disponibles
    commands_text = """[cyan]inv setup-env[/cyan]          - Vérifier l'environnement
[cyan]inv test-connection[/cyan]    - Tester Snowflake  
[cyan]inv create-infrastructure[/cyan] - Créer l'infrastructure
[cyan]inv load-data[/cyan]          - Charger les données
[cyan]inv full-pipeline[/cyan]      - Pipeline complet
[cyan]inv status[/cyan]             - Afficher ce statut"""
    
    console.print(Panel(commands_text, title="🔧 Commandes disponibles", style="yellow"))