STDIN
import pandas as pd
import os

class DataLoader:
    """
    Classe responsable du chargement et du prétraitement des données MovieLens.
    """
    def __init__(self, data_path):
        self.data_path = data_path

    def load_ratings(self):
        """Charge le fichier u.data et retourne un DataFrame."""
        print(f"Chargement des données depuis {self.data_path}...")
        # TODO: Implémenter le chargement avec pandas
        pass

    def get_train_test_split(self, test_size=0.2):
        """Divise les données en ensembles d'entraînement et de test."""
        # TODO: Utiliser train_test_split
        pass

    def get_item_statistics(self, df):
        """Calcule la moyenne et la variance pour l'objectif de Nouveauté."""
        # TODO: Calculer mean et var par item
        pass
