import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Classe responsable du chargement et du prétraitement des données MovieLens.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        # Noms des colonnes spécifiques à MovieLens 100k
        self.column_names = ['user_id', 'item_id', 'rating', 'timestamp']

    def load_ratings(self):
        """
        Charge le fichier u.data et retourne un DataFrame Pandas.
        """
        file_path = f"{self.data_path}/u.data"
        print(f"📥 Chargement des données depuis : {file_path}")
        
        try:
            # u.data est séparé par des tabulations (\t)
            df = pd.read_csv(file_path, sep='\t', names=self.column_names)
            print(f"✅ Données chargées : {len(df)} notes trouvées.")
            return df
        except FileNotFoundError:
            print(f"❌ Erreur critique : Le fichier n'a pas été trouvé ici : {file_path}")
            print("👉 Vérifiez que vous avez bien dézippé ml-100k.zip dans data/raw/")
            return None

    def get_train_test_split(self, df, test_size=0.2):
        """
        Divise les données en ensembles d'entraînement et de test.
        """
        print(f"✂️  Division des données (Test size = {test_size})...")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        print(f"   Train set : {len(train_data)} notes")
        print(f"   Test set  : {len(test_data)} notes")
        return train_data, test_data

    def get_user_item_matrix(self, df):
        """
        Transforme le DataFrame en une Matrice (Pivot Table).
        Lignes = Utilisateurs
        Colonnes = Films
        Valeurs = Notes
        """
        print("📊 Création de la matrice Utilisateur-Item...")
        # Remplir les notes manquantes par 0
        matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        return matrix

    def get_item_statistics(self, df):
        """
        Calcule les statistiques par item nécessaires pour l'objectif de Nouveauté (Phase 3).
        Retourne un DataFrame avec : popularity (count), mu (mean), sigma (variance).
        """
        print("📈 Calcul des statistiques des items (pour l'objectif Nouveauté)...")
        
        # On groupe par ID de film et on calcule Count, Mean, Variance
        item_stats = df.groupby('item_id')['rating'].agg(['count', 'mean', 'var'])
        
        # Si un film n'a qu'une seule note, la variance est NaN. On remplace par 0.
        item_stats['var'] = item_stats['var'].fillna(0)
        
        # Renommer les colonnes pour correspondre aux formules du papier
        item_stats.columns = ['popularity', 'mu', 'sigma']
        
        return item_stats