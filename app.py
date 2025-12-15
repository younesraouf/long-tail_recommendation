import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- IMPORTS DES MODULES DU PROJET ---
# On suppose que le dossier 'src' est au même niveau que app.py
from src.data_loader import DataLoader
from src.cf_model import ItemBasedCF
from src.optimizer import MORSOptimizer

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MORS Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 MORS: Multi-Objective Recommendation")
st.markdown("""
**Système de recommandation optimisant la Longue Traîne.**  
Ce dashboard permet de visualiser le compromis entre la **Précision** (Items populaires) et la **Nouveauté** (Items de niche/Long Tail).
""")

# --- FONCTIONS UTILITAIRES ---

@st.cache_resource
def load_system(dataset_name):
    """
    Charge les données, entraîne le modèle CF et prépare les stats.
    Mis en cache pour la performance.
    """
    status_text = st.empty()
    bar = st.progress(0)
    
    # 1. Chargement Données
    status_text.text(f"Chargement du dataset {dataset_name}...")
    # CORRECTION DU CHEMIN ICI : On force le chemin relatif depuis la racine
    loader = DataLoader(active_dataset=dataset_name, processed_data_dir="data/processed")
    df = loader.load_active_dataset()
    titles = loader.load_item_titles()
    bar.progress(25)
    
    # 2. Split & Stats
    status_text.text("Calcul des statistiques (Moyenne/Variance)...")
    train_df, _ = loader.get_train_test_split(df)
    item_stats = loader.get_item_statistics(train_df)
    bar.progress(50)
    
    # 3. Matrice & CF
    status_text.text("Construction de la matrice User-Item et calcul de similarité...")
    train_matrix = loader.get_user_item_matrix(train_df)
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    bar.progress(100)
    
    status_text.empty()
    bar.empty()
    
    return loader, df, item_stats, cf, titles

def split_pareto_solutions(solutions):
    """
    Sépare les solutions en deux listes : 
    1. Le Front de Pareto (Non-dominées)
    2. Les solutions dominées (Le nuage gris)
    """
    pareto = []
    dominated = []
    
    for sol_a in solutions:
        is_dominated = False
        for sol_b in solutions:
            # Si sol_b est meilleure ou égale partout, et strictement meilleure sur au moins un point
            if (sol_b['accuracy'] >= sol_a['accuracy'] and 
                sol_b['novelty'] >= sol_a['novelty'] and 
                (sol_b['accuracy'] > sol_a['accuracy'] or sol_b['novelty'] > sol_a['novelty'])):
                is_dominated = True
                break
        
        if is_dominated:
            dominated.append(sol_a)
        else:
            pareto.append(sol_a)
            
    # Tri du front pour tracer une ligne propre
    pareto.sort(key=lambda x: x['accuracy'])
    return pareto, dominated

def plot_pareto_advanced(pareto_sols, dominated_sols):
    """Génère le graphique avancé avec distinction Front/Nuage."""
    
    par_acc = [s['accuracy'] for s in pareto_sols]
    par_nov = [s['novelty'] for s in pareto_sols]
    dom_acc = [s['accuracy'] for s in dominated_sols]
    dom_nov = [s['novelty'] for s in dominated_sols]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Nuage gris (Solutions explorées mais rejetées)
    ax.scatter(dom_acc, dom_nov, c='gray', alpha=0.3, s=30, label='Solutions explorées', zorder=1)
    
    # 2. Ligne du Front
    ax.plot(par_acc, par_nov, c='#1f77b4', linewidth=2, alpha=0.8, zorder=2)
    
    # 3. Points du Front (Optimaux)
    ax.scatter(par_acc, par_nov, c='#1f77b4', s=60, edgecolors='white', label='Front de Pareto', zorder=3)
    
    # 4. Extrêmes (Étoiles)
    if len(pareto_sols) > 0:
        # Max Précision (Dernier de la liste triée par accuracy)
        ax.scatter(par_acc[-1], par_nov[-1], c='crimson', s=180, marker='*', label='Max Précision', zorder=4)
        
        # Max Nouveauté (Celui qui a le score novelty le plus haut)
        idx_max_nov = max(range(len(par_nov)), key=par_nov.__getitem__)
        ax.scatter(par_acc[idx_max_nov], par_nov[idx_max_nov], c='limegreen', s=180, marker='*', label='Max Nouveauté', zorder=4)

    ax.set_title("Espace de recherche & Front de Pareto", fontsize=14)
    ax.set_xlabel("Précision (Somme des notes prédites)")
    ax.set_ylabel("Nouveauté (Score Long Tail)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    
    return fig

def show_list_with_highlight(sol, label, titles_map, highlight_items=None):
    """Affiche une liste de films en ajoutant des émojis pour les nouveautés."""
    if highlight_items is None: highlight_items = set()
    
    data = []
    for item_id in sol['items']:
        title = titles_map.get(item_id, f"Item {item_id}")
        if item_id in highlight_items:
            title = f"✨ {title}" # Marqueur visuel
        data.append(title)
        
    df_res = pd.DataFrame({"Films Recommandés": data})
    st.markdown(f"**{label}**")
    st.caption(f"Précision: {sol['accuracy']:.2f} | Nouveauté: {sol['novelty']:.2f}")
    st.dataframe(df_res, height=300, use_container_width=True)

# --- INTERFACE SIDEBAR ---
st.sidebar.header("⚙️ Configuration")

ds_choice = st.sidebar.selectbox("Choisir le Dataset", ["movielens", "jester", "netflix"])
user_id = st.sidebar.number_input("ID Utilisateur Cible", min_value=1, value=1)

st.sidebar.subheader("Paramètres Algorithme (MORS)")
k_items = st.sidebar.slider("Longueur de la liste (K)", 5, 20, 10)
n_gen = st.sidebar.slider("Générations", 10, 300, 100) # Augmenté un peu par défaut
pop_size = st.sidebar.slider("Taille Population", 10, 100, 50)

# --- CORPS PRINCIPAL ---

# 1. Chargement initial
try:
    loader, df, item_stats, cf, titles = load_system(ds_choice)
    st.success(f"Système chargé : {ds_choice.capitalize()} ({len(df)} notes).")
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# 2. Onglets
tab1, tab2 = st.tabs(["📊 Analyse Dataset", "🚀 Recommandation MORS"])

with tab1:
    st.subheader("Distribution Longue Traîne")
    item_counts = df.groupby('item_id').size().sort_values(ascending=False).values
    
    fig_tail, ax_tail = plt.subplots(figsize=(12, 5))
    ax_tail.plot(item_counts, color='blue')
    ax_tail.fill_between(range(len(item_counts)), item_counts, color='blue', alpha=0.1)
    
    cutoff = int(len(item_counts) * 0.2)
    ax_tail.axvline(x=cutoff, color='red', linestyle='--')
    ax_tail.text(cutoff*1.1, max(item_counts)*0.8, 'Frontière 20/80', color='red')
    
    ax_tail.set_title(f"Distribution des notes - {ds_choice}")
    ax_tail.set_ylabel("Nombre de notes")
    ax_tail.set_xlabel("Items (triés par popularité)")
    st.pyplot(fig_tail)
    
    col_a, col_b = st.columns(2)
    col_a.metric("Nombre total d'items", len(item_counts))
    col_b.metric("Items en Long Tail (>80%)", int(len(item_counts)*0.8))

with tab2:
    if st.button("Lancer l'Optimisation", type="primary"):
        
        # A. Phase 1 : Candidats CF
        with st.spinner("Phase 1 : Génération des candidats (Item-Based CF)..."):
            if user_id not in cf.train_matrix.index:
                st.error(f"L'utilisateur {user_id} n'existe pas dans le Train Set.")
                candidates = []
            else:
                candidates = cf.get_top_k_candidates(user_id, k=k_items*5)
        
        if len(candidates) > 0:
            # B. Phase 2 : Optimisation MORS
            with st.spinner(f"Phase 2 : Évolution génétique ({n_gen} générations)..."):
                optimizer = MORSOptimizer(
                    candidates, 
                    item_stats, 
                    list_length=k_items, 
                    population_size=pop_size
                )
                solutions = optimizer.run(generations=n_gen)
            
            # C. Traitement des résultats (Séparation Front vs Nuage)
            pareto_sols, dominated_sols = split_pareto_solutions(solutions)
            
            st.divider()
            col_graph, col_list = st.columns([1.5, 1])
            
            with col_graph:
                st.subheader("Visualisation des Solutions")
                fig_pareto = plot_pareto_advanced(pareto_sols, dominated_sols)
                st.pyplot(fig_pareto)
                
                st.info(f"Solutions générées : {len(solutions)} | Optimales (Pareto) : {len(pareto_sols)}")
            
            with col_list:
                st.subheader("Comparaison des Listes")
                
                # Sélection des meilleures solutions
                # Le front est trié par accuracy croissante -> Le dernier est Max Acc
                sol_acc = pareto_sols[-1]
                # Le max novelty peut être n'importe où, on le cherche
                sol_nov = max(pareto_sols, key=lambda x: x['novelty'])
                
                # Identification des items "Découverte" (présents dans Nov mais pas dans Acc)
                items_in_acc = set(sol_acc['items'])
                items_in_nov = set(sol_nov['items'])
                discoveries = items_in_nov - items_in_acc

                sub_tab1, sub_tab2 = st.tabs(["🎯 Max Précision", "🌟 Max Nouveauté"])
                
                with sub_tab1:
                    show_list_with_highlight(sol_acc, "Focus : Qualité Prédite", titles)
                
                with sub_tab2:
                    show_list_with_highlight(sol_nov, "Focus : Découverte (Long Tail)", titles, highlight_items=discoveries)
                    if len(discoveries) > 0:
                        st.success(f"L'algo a introduit {len(discoveries)} items originaux (marqués par ✨).")
                    else:
                        st.warning("Les listes sont identiques (Compromis difficile à trouver pour cet utilisateur).")
                    
        elif user_id in cf.train_matrix.index:
            st.warning("Aucun candidat trouvé (Cold Start ou pas assez de données).")