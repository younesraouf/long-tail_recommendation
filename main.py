from src.data_loader import DataLoader
from src.cf_model import ItemBasedCF
from src.optimizer import MORSOptimizer
from src.utils import plot_pareto_front

def main():
    print("=== MORS Project: Multi-Objective Recommendation System ===")
    
    # --- PHASE 1: Chargement des données ---
    print("\n[PHASE 1] Loading Data")
    loader = DataLoader('data/raw/ml-100k')
    df = loader.load_ratings()
    
    if df is None: return

    # On garde le split pour la bonne forme, mais on entraîne sur le Train Set
    train_df, _ = loader.get_train_test_split(df)
    train_matrix = loader.get_user_item_matrix(train_df)
    item_stats = loader.get_item_statistics(train_df)
    
    # --- PHASE 2: Filtrage Collaboratif (Génération du Pool) ---
    print("\n[PHASE 2] Generating Candidate Pool (CF)")
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    
    USER_ID = 1
    
    # Stratégie "Injection de Chaos" pour garantir la diversité
    # On prend 800 films, on garde les 150 meilleurs + 150 de la "Long Tail"
    large_pool = cf.get_top_k_candidates(user_id=USER_ID, k=800)
    
    safe_candidates = large_pool[:150]    # Les "Best-Sellers"
    risky_candidates = large_pool[-150:]  # La "Long Tail"
    
    candidates = safe_candidates + risky_candidates
    print(f"\n[INFO] Optimization Pool ready: {len(candidates)} items (Mixed Strategy).")

    # --- PHASE 3 & 4: Optimisation MORS (Algorithme Génétique) ---
    print("\n[PHASE 3] Running Evolutionary Algorithm (MORS)...")
    
    optimizer = MORSOptimizer(
        candidates=candidates, 
        item_stats=item_stats, 
        list_length=10, 
        population_size=50,
        mutation_rate=0.3
    )
    
    pareto_solutions = optimizer.run(generations=50)
    print(f"\n[Done] Found {len(pareto_solutions)} Pareto-optimal solutions.")
    
    # --- PHASE 5: Visualisation & Résultats ---
    print("\n[PHASE 5] Results & Visualization")
    
    # 1. Sauvegarder le graphique (C'est la preuve scientifique du papier)
    plot_pareto_front(pareto_solutions, USER_ID)
    
    # 2. Afficher les titres pour l'analyse humaine
    titles_map = loader.load_movie_titles()
    
    def display_solution(sol, label):
        print(f"\n👉 {label}")
        print(f"   Optimization Score -> Accuracy (F1): {sol['accuracy']:.2f} | Novelty (F2): {sol['novelty']:.4f}")
        print(f"   Recommended Movies:")
        for item_id in sol['items']:
            title = titles_map.get(item_id, "Unknown Title")
            # On affiche l'ID pour vérifier si c'est un film rare
            print(f"     - [{item_id}] {title}")

    # Afficher les deux extrêmes du Front de Pareto
    best_acc = max(pareto_solutions, key=lambda x: x['accuracy'])
    display_solution(best_acc, "Solution: MAX ACCURACY (Conservative)")
    
    best_nov = max(pareto_solutions, key=lambda x: x['novelty'])
    display_solution(best_nov, "Solution: MAX NOVELTY (Discovery / Long Tail)")

    print("\n✅ Project execution complete. Check 'pareto_front.png'.")

if __name__ == "__main__":
    main()