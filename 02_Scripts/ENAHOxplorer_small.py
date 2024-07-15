import pandas as pd
from openai import OpenAI
from scipy.spatial import distance
import tiktoken
import pickle
from pathlib import Path
from sklearn.mixture import GaussianMixture
import numpy as np

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class ENAHOxplorer():
    
    def __init__(_self, key_):
        _self.path_str = Path(__file__).parent.parent
        _self.dfvars = pd.read_pickle(_self.path_str / '01_Data/dfvars_notna_renames_fullEmb.pkl')
        _self.dfgrvars = pd.read_pickle(_self.path_str / '01_Data/dfgrvars.pkl')
        _self.client = OpenAI(api_key = key_)
        with open(_self.path_str / "01_Data/ENAHO_tree.pkl",'rb') as f:
            _self.tree = pickle.load(f)
    
    def get_embedding(_self, text, model="text-embedding-3-small"):
        if pd.isna(text):
            return [0]
        else:
            text = str(text).lower()
            return _self.client.embeddings.create(input = [text], model=model).data[0].embedding
    
    def _get_similitud(_self, x, emb):
        return (1-distance.cosine(x, emb))
    
    def cluster_near_one_gmm(_self, series: pd.Series, n_components: int = 2) -> pd.Index:
        # Asegurarse de que los datos están en el rango [0, 1]
        
        # Convertir la serie en un DataFrame necesario para GMM
        cos = series.values.reshape(-1, 1)
        data = np.sqrt(2-2*cos)
        # Realizar clustering usando Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data)
        labels = gmm.predict(data)
        
        # Identificar los clusters únicos
        unique_labels = set(labels)
        
        # Encontrar el cluster más cercano a 1
        closest_cluster = None
        min_distance = float('inf')
        for label in unique_labels:
            cluster_values = series[labels == label]
            distance_to_one = abs(cluster_values.mean() - 1)
            if distance_to_one < min_distance:
                min_distance = distance_to_one
                closest_cluster = label
        
        # Devolver los índices de los valores del cluster más cercano a 1
        # return series[labels == closest_cluster].index
        return (labels == closest_cluster)
