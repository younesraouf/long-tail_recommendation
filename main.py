from src.data_loader import DataLoader
from src.cf_model import ItemBasedCF
from src.optimizer import MORSOptimizer

def main():
    print("=== MORS Project: Long Tail Recommendation System ===")
    
    # --- PHASE 1: Data Loading ---
    print("\n[PHASE 1] Loading and Preprocessing Data")
    loader = DataLoader('data/raw/ml-100k')
    df = loader.load_ratings()
    
    if df is None: return

    train_df, _ = loader.get_train_test_split(df)
    train_matrix = loader.get_user_item_matrix(train_df)
    # Important: Retrieve item statistics for Phase 3
    item_stats = loader.get_item_statistics(train_df)
    
    # --- PHASE 2: Collaborative Filtering ---
    print("\n[PHASE 2] Initializing Collaborative Filtering Model")
    cf = ItemBasedCF(train_matrix)
    cf.compute_similarity()
    
    USER_ID = 1
    # PARAMÈTRE OPTIMISÉ : 300 candidats pour plus de diversité
    K_POOL_SIZE = 300 
    
    candidates = cf.get_top_k_candidates(user_id=USER_ID, k=K_POOL_SIZE)
    
    print(f"\n[INFO] Generated pool of {len(candidates)} candidates for User {USER_ID}.")
    print(f"Top 3 candidates (by accuracy): {candidates[:3]}")

    # --- PHASE 3: Evolutionary Optimization ---
    print("\n[PHASE 3] Starting Evolutionary Optimization (MORS)")
    
    # PARAMÈTRE OPTIMISÉ : mutation_rate = 0.3
    optimizer = MORSOptimizer(
        candidates=candidates, 
        item_stats=item_stats, 
        list_length=10, 
        population_size=50,
        mutation_rate=0.3 
    )
    
    # Run for 50 generations
    pareto_solutions = optimizer.run(generations=50)
    
    print(f"\n[Done] Optimization complete. Found {len(pareto_solutions)} unique solutions.")
    
    # --- Display Results with Titles ---
    print("\n[RESULTS] Analysis of Recommendations")
    
    # Chargement des titres pour l'affichage
    titles_map = loader.load_movie_titles()

    def print_solution(solution, label):
        print(f"\n{label}")
        print(f"   Metrics: Accuracy={solution['accuracy']:.2f} | Novelty={solution['novelty']:.4f}")
        print(f"   Movies:")
        for item_id in solution['items']:
            title = titles_map.get(item_id, "Unknown Title")
            print(f"     - [{item_id}] {title}")

    # 1. Solution Max Précision
    best_acc_sol = max(pareto_solutions, key=lambda x: x['accuracy'])
    print_solution(best_acc_sol, "Max Accuracy Solution (Safe bets / Blockbusters)")
    
    # 2. Solution Max Nouveauté
    best_nov_sol = max(pareto_solutions, key=lambda x: x['novelty'])
    print_solution(best_nov_sol, "Max Novelty Solution (Hidden Gems / Long Tail)")

if __name__ == "__main__":
    main()