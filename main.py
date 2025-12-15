import time
import sys
import os

# Imports depuis le dossier src
from src.data_loader import DataLoader
from src.cf_model import ItemBasedCF
from src.optimizer import MORSOptimizer
from src.utils import plot_pareto_front

def main():
    print("==================================================")
    print("      PROJET MORS : Long Tail Recommendation      ")
    print("==================================================")

    # ---------------------------------------------------------
    # ÉTAPE 1 : Chargement
    # ---------------------------------------------------------
    # Correction du chemin: "data/processed" car on lance depuis la racine
    loader = DataLoader(processed_data_dir="data/processed", active_dataset='movielens')
    
    loader.display_table_2_summary()
    df = loader.load_active_dataset()
    titles = loader.load_item_titles()
    
    # Split et Stats
    train_df, _ = loader.get_train_test_split(df)
    item_stats = loader.get_item_statistics(train_df)

    # ---------------------------------------------------------
    # ÉTAPE 2 : Filtrage Collaboratif
    # ---------------------------------------------------------
    print("\n[PHASE 1] Entraînement du modèle CF...")
    train_matrix = loader.get_user_item_matrix(train_df)
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    
    target_user = 1
    print(f"\n[PHASE 1] Génération des candidats pour User {target_user}...")
    candidates = cf.get_top_k_candidates(target_user, k=100)

    # ---------------------------------------------------------
    # ÉTAPE 3 : Optimisation (MORS)
    # ---------------------------------------------------------
    print(f"\n[PHASE 2] Démarrage de l'Optimiseur MORS...")
    optimizer = MORSOptimizer(
        candidates=candidates, 
        item_stats=item_stats, 
        list_length=10, 
        population_size=50
    )
    
    start_time = time.time()
    solutions = optimizer.run(generations=50) 
    elapsed = time.time() - start_time
    print(f"Optimisation terminée en {elapsed:.2f}s. ({len(solutions)} solutions)")

    # ---------------------------------------------------------
    # ÉTAPE 4 : Calcul Diversité
    # ---------------------------------------------------------
    print("\n[EVALUATION] Calcul de la Diversité...")
    for sol in solutions:
        # Appel de la méthode corrigée dans cf_model.py
        sol['diversity'] = cf.calculate_list_diversity(sol['items'])

    # ---------------------------------------------------------
    # ÉTAPE 5 : Résultats
    # ---------------------------------------------------------
    # Tri par Précision
    solutions.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n--- RÉSULTATS COMPARATIFS ---")
    
    if len(solutions) > 0:
        # Solution A: Précision
        best_acc = solutions[0]
        print(f"\n[Solution A : Max Précision]")
        print(f"   Accuracy : {best_acc['accuracy']:.2f}")
        print(f"   Novelty  : {best_acc['novelty']:.2f}")
        print(f"   Diversity: {best_acc['diversity']:.4f}")
        print(f"   Films: {[titles.get(i, i) for i in best_acc['items'][:3]]} ...")

        # Solution B: Nouveauté (On retrie la liste)
        solutions.sort(key=lambda x: x['novelty'], reverse=True)
        best_nov = solutions[0]
        print(f"\n[Solution B : Max Nouveauté]")
        print(f"   Accuracy : {best_nov['accuracy']:.2f}")
        print(f"   Novelty  : {best_nov['novelty']:.2f}")
        print(f"   Diversity: {best_nov['diversity']:.4f}")
        print(f"   Films: {[titles.get(i, i) for i in best_nov['items'][:3]]} ...")

    # ---------------------------------------------------------
    # ÉTAPE 6 : Graphique
    # ---------------------------------------------------------
    print("\n[PHASE 3] Génération du graphique...")
    plot_pareto_front(solutions, target_user, save_path="pareto_front_user1.png")
    
    print("\n=== FIN DU PROGRAMME ===")

if __name__ == "__main__":
    main()