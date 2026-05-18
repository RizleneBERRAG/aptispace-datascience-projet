import pandas as pd
import numpy as np
from typing import List

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Charge le jeu de données brut depuis le chemin spécifié.
    
    Parameters:
    -----------
    file_path : str
        Le chemin d'accès au fichier CSV brut.
        
    Returns:
    --------
    pd.DataFrame
        Le DataFrame chargé.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Données chargées avec succès. Dimensions : {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        raise e

def clean_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Exemple minimal fonctionnel : convertit la colonne de dates au type datetime standard.
    Les étudiants devront l'enrichir pour gérer les fuseaux horaires ou formats locaux hétérogènes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée.
    date_col : str
        Le nom de la colonne contenant les dates.
        
    Returns:
    --------
    pd.DataFrame
        Le DataFrame avec la colonne de dates convertie.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    print(f"📅 Dates nettoyées dans '{date_col}'. Incohérences converties en NaNs : {df[date_col].isna().sum()}")
    return df

def impute_missing_values(df: pd.DataFrame, columns: List[str], method: str = 'interpolate') -> pd.DataFrame:
    """
    Exemple minimal fonctionnel : impute les valeurs manquantes à l'aide d'une interpolation 
    linéaire ou d'une imputation par médiane sur les colonnes ciblées.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée.
    columns : List[str]
        Les colonnes sur lesquelles appliquer l'imputation.
    method : str, default 'interpolate'
        Méthode d'imputation : 'interpolate' (linéaire) ou 'median' (remplacement par la médiane).
        
    Returns:
    --------
    pd.DataFrame
        Le DataFrame avec valeurs manquantes imputées.
    """
    df = df.copy()
    for col in columns:
        if method == 'interpolate':
            df[col] = df[col].interpolate(method='linear')
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].ffill()
    print(f"🔧 Valeurs manquantes imputées avec la méthode '{method}' pour les colonnes : {columns}")
    return df

def handle_outliers(df: pd.DataFrame, columns: List[str], min_val: float, max_val: float) -> pd.DataFrame:
    """
    Exemple minimal fonctionnel : remplace toutes les valeurs en dehors des seuils [min_val, max_val]
    par NaN afin qu'elles puissent être imputées/interpolées par la suite.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée.
    columns : List[str]
        Les colonnes numériques à auditer.
    min_val : float
        Le seuil minimum admissible.
    max_val : float
        Le seuil maximum admissible.
        
    Returns:
    --------
    pd.DataFrame
        Le DataFrame traité.
    """
    df = df.copy()
    for col in columns:
        outlier_mask = (df[col] < min_val) | (df[col] > max_val)
        outliers_count = outlier_mask.sum()
        df.loc[outlier_mask, col] = np.nan
        print(f"🚨 Anomalies (outliers) détectées dans '{col}' : {outliers_count} remplacées par NaN")
    return df

def feature_engineering(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Exemple minimal fonctionnel : extrait les attributs de date basiques (heure, jour de la semaine).
    Les étudiants pourront ajouter des encodages trigonométriques (sin/cos) ou des moyennes mobiles.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le DataFrame d'entrée.
    date_col : str
        Le nom de la colonne temporelle de type datetime.
        
    Returns:
    --------
    pd.DataFrame
        Le DataFrame enrichi avec de nouvelles colonnes descriptives.
    """
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df['hour'] = df[date_col].dt.hour
        df['dayofweek'] = df[date_col].dt.dayofweek
        print("💡 Nouvelles caractéristiques extraites : 'hour', 'dayofweek'")
    else:
        print(f"⚠️ La colonne '{date_col}' n'est pas au format Datetime. feature_engineering ignorée.")
    return df
