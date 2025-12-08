from src.data_loader import DataLoader
# Les imports suivants seront utilisés dans les phases 2 et 3
# from src.cf_model import ItemBasedCF
# from src.optimizer import MORSOptimizer

def main():
    print("=== Projet MORS: Long Tail Recommendation (Phase 1) ===")
    
    # 1. Configuration du chemin
    # Assurez-vous que le dossier ml-100k est bien dans data/raw/
    data_path = 'data/raw/ml-100k' 
    loader = DataLoader(data_path)
    
    # 2. Chargement des données
    df = loader.load_ratings()
    
    if df is not None:
        # 3. Séparation Train / Test
        train_df, test_df = loader.get_train_test_split(df)
        
        # 4. Création de la matrice d'entraînement (pour le Filtrage Collaboratif)
        train_matrix = loader.get_user_item_matrix(train_df)
        print(f"✅ Matrice d'entraînement créée : {train_matrix.shape} (Users x Items)")
        
        # 5. Calcul des statistiques des items (pour l'Optimisation Multi-Objectifs)
        item_stats = loader.get_item_statistics(train_df)
        print("\n--- Exemple de statistiques (Top 5 films les plus notés) ---")
        print(item_stats.sort_values(by='popularity', ascending=False).head())

        print("\n🎉 Phase 1 terminée ! Les données sont prêtes pour le modèle.")

if __name__ == "__main__":
    main()