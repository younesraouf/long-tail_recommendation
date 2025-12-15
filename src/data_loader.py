import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Gestionnaire de données pour le framework MORS.
    
    Responsabilités :
    1. Afficher les statistiques comparatives (Table 2).
    2. Charger le dataset actif pour l'entraînement.
    3. Séparer Train/Test selon la méthodologie de l'article.
    """

    def __init__(self, processed_data_dir="../data/processed", active_dataset='movielens'):
        """
        Args:
            processed_data_dir (str): Chemin vers le dossier contenant les .csv générés par preprocess.py
            active_dataset (str): Le dataset à charger pour l'entraînement ('movielens', 'jester', 'netflix').
        """
        self.data_dir = processed_data_dir
        self.active_dataset = active_dataset.lower()
        
        # Mapping des noms de fichiers
        self.files_map = {
            'movielens': 'movielens_processed.csv',
            'jester': 'jester_processed.csv',
            'netflix': 'netflix_subset_processed.csv'
        }

    def display_table_2_summary(self):
        """
        Lit tous les datasets disponibles et affiche le tableau comparatif (Table 2 de l'article).
        """
        print("\n" + "="*70)
        print("Table 2: Statistics of datasets used in this work")
        print("="*70)
        
        stats_list = []
        
        # Noms d'affichage jolis pour le tableau
        display_names = {
            'movielens': 'MovieLens',
            'jester': 'Jester',
            'netflix': 'Netflix'
        }

        for key, filename in self.files_map.items():
            file_path = os.path.join(self.data_dir, filename)
            dataset_name = display_names.get(key, key)
            
            if os.path.exists(file_path):
                try:
                    # Lecture rapide pour stats
                    df = pd.read_csv(file_path)
                    n_users = df['user_id'].nunique()
                    n_items = df['item_id'].nunique()
                    n_ratings = len(df)
                    
                    stats_list.append({
                        'Data sets': dataset_name,
                        'No. of users': n_users,
                        'No. of items': n_items,
                        'No. of ratings': n_ratings
                    })
                except Exception as e:
                    stats_list.append({
                        'Data sets': dataset_name,
                        'No. of users': 'Error', 'No. of items': 'Error', 'No. of ratings': str(e)
                    })
            else:
                stats_list.append({
                    'Data sets': dataset_name,
                    'No. of users': 'N/A (Not found)', 'No. of items': '-', 'No. of ratings': '-'
                })

        # Affichage avec Pandas
        summary_df = pd.DataFrame(stats_list)
        # Forcer l'ordre des colonnes
        summary_df = summary_df[['Data sets', 'No. of users', 'No. of items', 'No. of ratings']]
        
        print(summary_df.to_string(index=False))
        print("="*70 + "\n")

    def load_active_dataset(self):
        """
        Charge le dataset spécifié dans __init__ (self.active_dataset).
        Retourne un DataFrame.
        """
        filename = self.files_map.get(self.active_dataset)
        if not filename:
            raise ValueError(f"Dataset inconnu: {self.active_dataset}")
            
        file_path = os.path.join(self.data_dir, filename)
        print(f"[INFO] Chargement du dataset actif : {self.active_dataset.capitalize()}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}. Avez-vous lancé preprocess.py ?")
            
        df = pd.read_csv(file_path)
        print(f"[INFO] Chargé avec succès : {len(df)} lignes.")
        return df

    def get_train_test_split(self, df, test_size=0.2):
        """
        Sépare en Train (80%) et Test (20%).
        """
        print(f"[INFO] Séparation Train/Test (Test size = {test_size})...")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        print(f"   -> Train: {len(train_df)} ratings")
        print(f"   -> Test:  {len(test_df)} ratings")
        return train_df, test_df

    def get_user_item_matrix(self, df):
        """
        Convertit le DataFrame en matrice pivot (Users x Items).
        """
        # Utilisation de pivot_table qui gère les doublons potentiels (aggfunc='mean')
        matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', aggfunc='mean').fillna(0)
        return matrix
    
    # --- C'EST CETTE MÉTHODE QUI MANQUAIT ---
    def get_item_statistics(self, df):
        """
        Calcule les statistiques (Popularité, Moyenne, Variance) pour chaque item.
        Nécessaire pour l'Eq. 9 (Nouveauté) dans optimizer.py.
        """
        print("[INFO] Calcul des statistiques des items (Mean/Var)...")
        
        # On groupe par item et on calcule les stats sur la colonne 'rating'
        item_stats = df.groupby('item_id')['rating'].agg(['count', 'mean', 'var'])
        
        # Renommage des colonnes pour coller au papier
        item_stats.columns = ['popularity', 'mu', 'sigma']
        
        # Si un item n'a qu'une seule note, la variance est NaN. On remplace par 0.
        item_stats['sigma'] = item_stats['sigma'].fillna(0)
        
        return item_stats
    
    def load_item_titles(self):
        """
        Charge le fichier u.item pour associer les IDs aux vrais Titres.
        Retourne un dictionnaire {item_id: "Titre du film"}
        """
        # Chemin vers u.item (dans le dossier raw/ml-100k)
        # Note: on suppose que la structure est respectée
        raw_path = os.path.join(self.data_dir, '..', 'raw', 'ml-100k', 'u.item')
        
        # Si le chemin relatif est complexe, on essaie de le construire proprement
        if not os.path.exists(raw_path):
            # Fallback si on est dans src/
            raw_path = "../data/raw/ml-100k/u.item"
            
        print(f"[INFO] Chargement des titres depuis : {raw_path}")
        
        titles = {}
        try:
            # u.item est séparé par des '|' et encodé en 'latin-1'
            with open(raw_path, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        item_id = int(parts[0])
                        title = parts[1]
                        titles[item_id] = title
            return titles
        except FileNotFoundError:
            print("[WARNING] Fichier u.item introuvable. Les titres ne seront pas affichés.")
            return {}

# --- EXECUTION DIRECTE ---
if __name__ == "__main__":
    # Instanciation
    loader = DataLoader(active_dataset='movielens')
    
    # 1. Afficher le tableau complet des 3 datasets
    loader.display_table_2_summary()
    
    # 2. Charger le dataset actif (MovieLens par défaut)
    df = loader.load_active_dataset()
    
    # 3. Test de split
    train, test = loader.get_train_test_split(df)