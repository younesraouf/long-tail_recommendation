import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import re

# --- IMPORTS DES MODULES DU PROJET ---
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
**Implémentation du framework MORS.**  
Explorez le compromis entre la **Précision** (Mainstream) et la **Nouveauté** (Long Tail).
""")

# --- FONCTIONS UTILITAIRES ---

@st.cache_resource
def load_system(dataset_name):
    """Charge les données et le modèle."""
    status_text = st.empty()
    bar = st.progress(0)
    
    # 1. Chargement
    status_text.text(f"Chargement du dataset {dataset_name}...")
    loader = DataLoader(active_dataset=dataset_name, processed_data_dir="data/processed")
    df = loader.load_active_dataset()
    titles = loader.load_item_titles()
    bar.progress(25)
    
    # 2. Stats
    status_text.text("Calcul des statistiques...")
    train_df, _ = loader.get_train_test_split(df)
    item_stats = loader.get_item_statistics(train_df)
    bar.progress(50)
    
    # 3. Modèle CF
    status_text.text("Entraînement du Filtrage Collaboratif...")
    train_matrix = loader.get_user_item_matrix(train_df)
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    bar.progress(100)
    
    status_text.empty()
    bar.empty()
    return loader, df, item_stats, cf, titles

def split_pareto_solutions(solutions):
    """Sépare le front de Pareto des solutions dominées."""
    pareto = []
    dominated = []
    for sol_a in solutions:
        is_dominated = False
        for sol_b in solutions:
            if (sol_b['accuracy'] >= sol_a['accuracy'] and 
                sol_b['novelty'] >= sol_a['novelty'] and 
                (sol_b['accuracy'] > sol_a['accuracy'] or sol_b['novelty'] > sol_a['novelty'])):
                is_dominated = True
                break
        if is_dominated: dominated.append(sol_a)
        else: pareto.append(sol_a)
    pareto.sort(key=lambda x: x['accuracy'])
    return pareto, dominated

def plot_pareto_advanced(pareto_sols, dominated_sols, selected_idx=None):
    """Trace le graphique Accuracy vs Novelty."""
    par_acc = [s['accuracy'] for s in pareto_sols]
    par_nov = [s['novelty'] for s in pareto_sols]
    dom_acc = [s['accuracy'] for s in dominated_sols]
    dom_nov = [s['novelty'] for s in dominated_sols]
    
    fig, ax = plt.subplots(figsize=(8, 5)) # Taille ajustée pour le layout côte à côte
    
    # Nuage gris
    ax.scatter(dom_acc, dom_nov, c='gray', alpha=0.3, s=30, label='Solutions explorées', zorder=1)
    # Ligne bleue
    ax.plot(par_acc, par_nov, c='#1f77b4', linewidth=2, alpha=0.8, zorder=2)
    # Points bleus
    ax.scatter(par_acc, par_nov, c='#1f77b4', s=60, edgecolors='white', label='Front de Pareto', zorder=3)
    
    # Surbrillance sélection
    if selected_idx is not None and 0 <= selected_idx < len(pareto_sols):
        sel = pareto_sols[selected_idx]
        ax.scatter(sel['accuracy'], sel['novelty'], c='orange', s=250, marker='X', edgecolors='black', label='Sélection', zorder=5)

    # Étoiles extrêmes
    if len(pareto_sols) > 0:
        ax.scatter(par_acc[-1], par_nov[-1], c='crimson', s=180, marker='*', label='Max Précision', zorder=4)
        idx_max_nov = max(range(len(par_nov)), key=par_nov.__getitem__)
        ax.scatter(par_acc[idx_max_nov], par_nov[idx_max_nov], c='limegreen', s=180, marker='*', label='Max Nouveauté', zorder=4)

    ax.set_title("1. Position sur le Front de Pareto")
    ax.set_xlabel("Précision Globale")
    ax.set_ylabel("Nouveauté Globale")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower left', fontsize='small')
    return fig

# --- FONCTION D'AFFICHAGE (Moyenne + Variance) ---
def show_list_with_highlight(sol, titles_map, item_stats):
    """
    Affiche la liste des films avec Popularité, Moyenne et Variance.
    """
    titles_list = []
    ratings_count_list = []
    mu_list = []
    sigma_list = []

    for item_id in sol['items']:
        # Titre
        title = titles_map.get(item_id, f"Item {item_id}")
        titles_list.append(title)
        
        # Stats
        try:
            stats = item_stats.loc[item_id]
            count = int(stats['popularity'])
            mu = float(stats['mu'])
            sigma = float(stats['sigma'])
        except (KeyError, ValueError):
            count = 0
            mu = 0.0
            sigma = 0.0
            
        ratings_count_list.append(count)
        mu_list.append(mu)
        sigma_list.append(sigma)
        
    df_res = pd.DataFrame({
        "Films Recommandés": titles_list,
        "Popularité": ratings_count_list,
        "Moyenne (μ)": mu_list,
        "Variance (σ)": sigma_list
    })
    
    # Affichage avec configuration des colonnes pour faire joli
    st.dataframe(
        df_res, 
        use_container_width=True,
        column_config={
            "Films Recommandés": st.column_config.TextColumn("Titre du Film"),
            "Popularité": st.column_config.NumberColumn("Nb Votes", format="%d ⭐"),
            "Moyenne (μ)": st.column_config.NumberColumn("Moyenne", format="%.2f"),
            "Variance (σ)": st.column_config.NumberColumn("Variance", format="%.2f"),
        }
    )

# --- GRAPHIQUE CARTOGRAPHIE ---
def plot_item_positioning(item_list, item_stats, titles_map, pred_map):
    """
    Scatter Plot: X = Score de Nouveauté, Y = Note Prédite
    """
    data = []
    for item_id in item_list:
        try:
            stats = item_stats.loc[item_id]
            mu = stats['mu']
            sigma = stats['sigma']
            novelty_score = 1.0 / (mu * (sigma + 1.0)**2 + 0.001)
            user_pred = pred_map.get(item_id, 0)
            
            data.append({
                'Titre': titles_map.get(item_id, str(item_id)),
                'Novelty': novelty_score,
                'Predicted': user_pred,
                'Popularity': int(stats['popularity'])
            })
        except KeyError: continue
        
    if not data: return None
    
    df_chart = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(8, 5)) # Taille ajustée
    
    # Nuage de points
    scatter = ax.scatter(
        df_chart['Novelty'], 
        df_chart['Predicted'], 
        c=df_chart['Predicted'], 
        cmap='plasma', 
        s=150, 
        edgecolors='black',
        alpha=0.9
    )
    
    # Annotations
    for i, row in df_chart.iterrows():
        short_title = (row['Titre'][:12] + '..') if len(row['Titre']) > 12 else row['Titre']
        ax.annotate(short_title, (row['Novelty'], row['Predicted']), 
                   xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8, weight='bold')
    
    # Lignes médianes
    med_x = df_chart['Novelty'].median()
    med_y = df_chart['Predicted'].median()
    ax.axvline(x=med_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=med_y, color='gray', linestyle=':', alpha=0.5)
    
    # Labels Quadrants
    ax.text(df_chart['Novelty'].max(), df_chart['Predicted'].max() + 0.1, "💎 PÉPITES", 
            color='purple', ha='right', weight='bold')
    ax.text(df_chart['Novelty'].min(), df_chart['Predicted'].max() + 0.1, "✅ VALEURS SÛRES", 
            color='orange', ha='left', weight='bold')

    ax.set_xlabel("Degré de Nouveauté (Score MORS) →")
    ax.set_ylabel("Note Prédite pour VOUS (1-5) ↑")
    ax.set_title("2. Cartographie : Mainstream vs Pépites")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig

# --- INTERFACE SIDEBAR ---

with st.sidebar:
    st.header("🎛️ Configuration")
    
    st.subheader("1. Données")
    ds_choice = st.selectbox("Dataset", ["movielens", "jester", "netflix"])
    user_id = st.number_input("ID Utilisateur", min_value=1, value=1)
    
    st.divider()
    
    st.subheader("2. Paramètres MORS")
    k_list_len = st.slider("Longueur Liste Finale (k)", 5, 20, 10)
    n_gen = st.slider("Générations (gen)", 10, 500, 200)
    pop_size = st.slider("Population (pop)", 20, 200, 100)
    
    st.markdown("") 
    
    with st.expander("🛠️ Paramètres Avancés (Table 1)"):
        K_candidates = st.number_input("Taille CF-K (K)", value=50)
        n_neighbor = st.number_input("Taille Voisinage (ns)", value=10)
        prob_cross = st.slider("Proba Crossover (pc)", 0.0, 1.0, 0.9)
        prob_mut = st.slider("Proba Mutation (pm)", 0.0, 1.0, 0.1)
        update_sz = st.number_input("Update Size (us)", value=3)


# --- CORPS PRINCIPAL ---

# 1. Chargement
try:
    loader, df, item_stats, cf, titles = load_system(ds_choice)
    st.success(f"Système chargé : {ds_choice.capitalize()} ({len(df)} notes).")
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# Gestion de l'état (Session State)
if 'pareto_sols' not in st.session_state:
    st.session_state.pareto_sols = []
if 'dominated_sols' not in st.session_state:
    st.session_state.dominated_sols = []
if 'has_run' not in st.session_state:
    st.session_state.has_run = False
if 'pred_map' not in st.session_state:
    st.session_state.pred_map = {}

# 2. Onglets
tab1, tab2, tab3 = st.tabs(["📊 Analyse Dataset", "🚀 Lancer MORS", "🔍 Explorateur de Solutions"])

# --- TAB 1 : ANALYSE ---
with tab1:
    st.subheader("Distribution de la Longue Traîne")
    item_counts = df.groupby('item_id').size().sort_values(ascending=False).values
    fig_tail, ax_tail = plt.subplots(figsize=(12, 5))
    ax_tail.plot(item_counts, color='blue', linewidth=1.5)
    ax_tail.fill_between(range(len(item_counts)), item_counts, color='blue', alpha=0.1)
    
    cutoff = int(len(item_counts) * 0.2)
    ax_tail.axvline(x=cutoff, color='red', linestyle='--', alpha=0.8)
    ax_tail.text(cutoff*1.05, max(item_counts)*0.8, 'Tête (Popularité)', color='red')
    
    ax_tail.set_title(f"Distribution des notes - {ds_choice}")
    st.pyplot(fig_tail)
    
    st.divider()
    col_a, col_b = st.columns(2)
    col_a.metric("Nombre total d'items", len(item_counts))
    col_b.metric("Items en Long Tail (>80%)", int(len(item_counts)*0.8))

# --- TAB 2 : LANCEMENT ---
with tab2:
    st.write("Cliquez ci-dessous pour lancer l'algorithme génétique.")
    
    if st.button("Lancer l'Optimisation", type="primary", use_container_width=True):
        
        # A. Phase 1
        with st.spinner(f"Phase 1 : Génération de {K_candidates} candidats (CF)..."):
            if user_id not in cf.train_matrix.index:
                st.error(f"L'utilisateur {user_id} n'existe pas dans le Train Set.")
                candidates = []
            else:
                candidates = cf.get_top_k_candidates(user_id, k=K_candidates)
                st.session_state.pred_map = {item: score for item, score in candidates}
        
        if len(candidates) > 0:
            # B. Phase 2
            with st.spinner(f"Phase 2 : Optimisation ({n_gen} générations)..."):
                optimizer = MORSOptimizer(
                    candidates, 
                    item_stats, 
                    list_length=k_list_len, 
                    population_size=pop_size,
                    mutation_rate=prob_mut,      
                    crossover_rate=prob_cross,   
                    neighbor_size=n_neighbor,    
                    update_size=update_sz        
                )
                solutions = optimizer.run(generations=n_gen)
            
            # Calcul diversité
            for sol in solutions:
                sol['diversity'] = cf.calculate_list_diversity(sol['items'])

            # C. Stockage Session
            pareto, dominated = split_pareto_solutions(solutions)
            st.session_state.pareto_sols = pareto
            st.session_state.dominated_sols = dominated
            st.session_state.has_run = True
            
        elif user_id in cf.train_matrix.index:
            st.warning("Aucun candidat trouvé (Cold Start ou pas assez de données).")

    # AFFICHAGE AUTOMATIQUE DES RÉSULTATS
    if st.session_state.has_run and len(st.session_state.pareto_sols) > 0:
        st.divider()
        col_graph, col_list = st.columns([1.5, 1])
        
        pareto_sols = st.session_state.pareto_sols
        dominated_sols = st.session_state.dominated_sols
        
        with col_graph:
            st.subheader("Front de Pareto Global")
            fig_pareto = plot_pareto_advanced(pareto_sols, dominated_sols)
            st.pyplot(fig_pareto)
        
        with col_list:
            st.subheader("Comparaison Rapide")
            sol_acc = pareto_sols[-1]
            sol_nov = max(pareto_sols, key=lambda x: x['novelty'])
            
            sub_tab1, sub_tab2 = st.tabs(["🎯 Max Précision", "🌟 Max Nouveauté"])
            with sub_tab1: show_list_with_highlight(sol_acc, titles, item_stats)
            with sub_tab2: show_list_with_highlight(sol_nov, titles, item_stats)

# --- TAB 3 : EXPLORATEUR (Mise en page Structurée) ---
with tab3:
    if not st.session_state.has_run:
        st.info("👋 Veuillez d'abord lancer l'optimisation dans l'onglet 'Lancer MORS'.")
    else:
        pareto_sols = st.session_state.pareto_sols
        dominated_sols = st.session_state.dominated_sols
        pred_map = st.session_state.pred_map
        
        if len(pareto_sols) == 0:
            st.warning("Aucune solution optimale disponible.")
        else:
            # EN-TÊTE : Contrôles et KPIs
            st.markdown("### 🔍 Explorateur Interactif")
            
            nb_sols = len(pareto_sols)
            col_sel, col_kpi = st.columns([2, 1])
            with col_sel:
                selected_idx = st.slider(
                    "Choisir le compromis (0 = Nouveauté Max / Fin = Précision Max)", 
                    0, nb_sols - 1, int(nb_sols/2)
                )
            
            selected_sol = pareto_sols[selected_idx]
            
            with col_kpi:
                st.markdown("**Performances :**")
                k1, k2, k3 = st.columns(3)
                k1.metric("Précision", f"{selected_sol['accuracy']:.1f}")
                k2.metric("Nouveauté", f"{selected_sol['novelty']:.1f}")
                div = selected_sol.get('diversity', 0)
                k3.metric("Diversité", f"{div:.3f}")

            st.divider()

            # LIGNE 1 : Les deux Graphiques côte à côte
            c1, c2 = st.columns(2)
            
            with c1:
                # Graphique 1: Pareto
                fig_pareto = plot_pareto_advanced(pareto_sols, dominated_sols, selected_idx=selected_idx)
                st.pyplot(fig_pareto)

            with c2:
                # Graphique 2: Cartographie
                fig_analysis = plot_item_positioning(selected_sol['items'], item_stats, titles, pred_map)
                if fig_analysis:
                    st.pyplot(fig_analysis)
            
            st.divider()

            # LIGNE 2 : Le Tableau détaillé (Pleine largeur)
            st.markdown(f"#### 3. Détails de la Solution #{selected_idx}")
            show_list_with_highlight(selected_sol, titles, item_stats)