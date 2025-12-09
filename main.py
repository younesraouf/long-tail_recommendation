from src.data_loader import DataLoader
from src.cf_model import ItemBasedCF

def main():
    
    # --- PHASE 1: Data Loading ---
    print("--- Loading Data ---")
    loader = DataLoader('data/raw/ml-100k')
    df = loader.load_ratings()
    
    if df is None:
        return

    train_df, _ = loader.get_train_test_split(df)
    train_matrix = loader.get_user_item_matrix(train_df)
    
    # --- PHASE 2: Collaborative Filtering ---
    print("\n--- Initializing Collaborative Filtering Model ---")
    
    # Instantiate
    cf = ItemBasedCF(train_matrix)
    
    # Train (Compute Similarities)
    cf.compute_similarity()
    
    # Test on a specific user
    USER_ID = 1
    K_CANDIDATES = 10
    
    candidates = cf.get_top_k_candidates(user_id=USER_ID, k=K_CANDIDATES)
    
    print(f"\nTop {K_CANDIDATES} Raw Recommendations for User {USER_ID}:")
    print("-" * 40)
    print(f"{'Item ID':<10} | {'Predicted Rating':<15}")
    print("-" * 40)
    for film, note in candidates:
        print(f"{film:<10} | {note:.4f}")

if __name__ == "__main__":
    main()