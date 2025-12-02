import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class DistanceApproach():
    
    def __init__(self, returns, sector_map=None, sector_neutral=False):
        """
        returns: DataFrame of log returns for all assets
        sector_map: dict like {"Tech": ["AAPL","MSFT"], ...}
        sector_neutral: whether to search pairs within each sector
        """
        self.returns = returns
        self.sector_map = sector_map
        self.sector_neutral = sector_neutral

    @staticmethod
    def angular_distance(returns):
        corr = returns.corr()
        return np.sqrt(0.5 * (1 - corr))

    @staticmethod
    def absolute_angular_distance(returns):
        corr = returns.corr()
        return np.sqrt(1 - np.abs(corr))

    @staticmethod
    def squared_angular_distance(returns):
        corr = returns.corr()
        return np.sqrt(1 - corr**2)

    def select(self, metric="angular", top_k=10):
        """
        Returns a list of (asset1, asset2) pairs OR a dict by sector if sector-neutral.
        """

        if metric == "angular":
            distances = self.angular_distance(self.returns)
        elif metric == "absolute_angular":
            distances = self.absolute_angular_distance(self.returns)
        elif metric == "squared_angular":
            distances = self.squared_angular_distance(self.returns)
        else:
            raise ValueError("Unknown metric")

        distances.values[np.triu_indices_from(distances)] = np.inf
        distances = distances.stack()  # (i, j) -> distance

        if self.sector_neutral:
            return self._sector_neutral(distances, top_k)
        else:
            return self._global(distances, top_k)

    def _global(self, distances, top_k):
        return distances.nsmallest(top_k).index.tolist()

    def _sector_neutral(self, distances, top_k):
        results = {}
        for sector, stocks in self.sector_map.items():
            sub = distances.loc[stocks, stocks]
            sub = sub[sub < np.inf]
            results[sector] = sub.nsmallest(top_k).index.tolist()
        return results



class MLApproach():
    
    def __init__(self, returns, n_components=3, n_clusters=5, random_state=0):
        self.returns = returns
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _compute_pca_scores(self):
        X = self.returns.values
        X = (X - X.mean(axis=0)) / X.std(axis=0)  
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        scores = pca.fit_transform(X.T)
        return pd.DataFrame(scores, index=self.returns.columns)

    def select(self, top_k=5):
        scores = self._compute_pca_scores()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(scores)

        scores["cluster"] = labels

        pairs = []

        for cluster_id in range(self.n_clusters):
            names = scores.index[scores["cluster"] == cluster_id]

            if len(names) < 2:
                continue

            sub = self.returns[names].corr()

            sub.values[np.triu_indices_from(sub)] = -np.inf

            best = sub.stack().sort_values(ascending=False).head(top_k)
            cluster_pairs = best.index.tolist()
            pairs.extend(cluster_pairs)

        return pairs
