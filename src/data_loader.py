import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Handles loading and preprocessing of MovieLens datasets.

    This class is responsible for reading raw data files, splitting them
    into training/testing sets, and computing necessary item statistics
    (mean, variance) required for the MORS framework.
    """

    def __init__(self, data_path):
        """
        Initialize the DataLoader.

        Args:
            data_path (str): Path to the directory containing the dataset (e.g., 'data/raw/ml-100k').
        """
        self.data_path = data_path
        # Column names specific to the MovieLens 100k u.data file
        self.column_names = ['user_id', 'item_id', 'rating', 'timestamp']

    def load_ratings(self):
        """
        Loads the 'u.data' file into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing user ratings, or None if file not found.
        """
        file_path = os.path.join(self.data_path, "u.data")
        print(f"Loading data from: {file_path}")
        
        try:
            # u.data is tab-separated
            df = pd.read_csv(file_path, sep='\t', names=self.column_names)
            print(f"Data loaded successfully. Total ratings: {len(df)}")
            return df
        except FileNotFoundError:
            print(f"Critical: File not found at {file_path}")
            return None

    def get_train_test_split(self, df, test_size=0.2):
        """
        Splits the dataset into training and testing sets.

        Args:
            df (pd.DataFrame): The full dataset.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: (train_df, test_df)
        """
        print(f"Splitting data (Test size = {test_size})...")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        print(f"   - Training set: {len(train_data)} ratings")
        print(f"   - Testing set:  {len(test_data)} ratings")
        
        return train_data, test_data

    def get_user_item_matrix(self, df):
        """
        Converts the rating DataFrame into a User-Item Matrix (Pivot Table).
        
        Rows represent Users, Columns represent Items, values represent Ratings.
        Missing values are filled with 0.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: The User-Item matrix.
        """
        print("Creating User-Item Matrix...")
        matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        return matrix

    def get_item_statistics(self, df):
        """
        Computes item statistics required for the Novelty objective (Phase 3).
        
        Calculates:
        - Popularity (count)
        - Mean rating (mu)
        - Variance of ratings (sigma)

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame indexed by item_id with columns ['popularity', 'mu', 'sigma'].
        """
        print("Computing item statistics (Mean/Variance)...")
        
        # Group by item_id and calculate metrics
        item_stats = df.groupby('item_id')['rating'].agg(['count', 'mean', 'var'])
        
        # If an item has only 1 rating, variance is NaN. Fill with 0.
        item_stats['var'] = item_stats['var'].fillna(0)
        
        # Rename columns to match the paper's mathematical notation
        item_stats.columns = ['popularity', 'mu', 'sigma']
        
        return item_stats
        
    def load_movie_titles(self):
        """
        Charge le fichier u.item pour associer ID -> Titre.
        """
        # u.item est séparé par '|' et peut avoir des problèmes d'encodage (latin-1)
        item_path = os.path.join(self.data_path, "u.item")
        print(f"[INFO] Loading movie titles from: {item_path}")
        try:
            # On ne charge que les 2 premières colonnes : ID et Titre
            movies = pd.read_csv(item_path, sep='|', header=None, encoding='latin-1', usecols=[0, 1])
            movies.columns = ['item_id', 'title']
            # On transforme en dictionnaire {id: 'Titre'} pour aller vite
            return movies.set_index('item_id')['title'].to_dict()
        except Exception as e:
            print(f"[WARNING] Could not load titles: {e}")
            return {}