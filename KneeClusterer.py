import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans, DBSCAN # Will deal with DBscan Variants later, need to read the papers
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from kneed.knee_locator import KneeLocator # python implementation of the kneedle, paper: https://raghavan.usc.edu//papers/kneedle-simplex11.pdf
from collections.abc import Iterable

# Will implement our own Kneelocator in the future, probably

SCORE_MAP = {"calinski_harabasz": calinski_harabasz_score,
             "davies_bouldin": davies_bouldin_score,
             "silhouette": silhouette_score}

CLUSTERER_MAP = {"kmeans": KMeans,
                 "bisecting_kmeans": BisectingKMeans,
                 "minibatch_kmeans": MiniBatchKMeans,
                 "DBscan": DBSCAN}

CLUSTERER_VAR_NAME = {"kmeans": "n_clusters",
                      "bisecting_kmeans": "n_clusters",
                      "minibatch_kmeans": "n_clusters",
                      "DBscan": "eps"}

score_to_knee_locator_kwargs = {"calinski_harabasz": {
                                        "curve": "concave",
                                        "direction": "increasing",},
                                "silhouette": {
                                        "curve": "concave",
                                        "direction": "increasing",},
                                "davies_bouldin": {
                                        "curve": "concave",
                                        "direction": "decreasing",}}

def print_v(x, verbose = 0):
    if verbose:
        print(x)

# Use this if we want a functional inferface instead

# Knees and elbows are interchangable and depends in the kwargs
# We are usually finding elbows here
def kneed_knee_finder(x, y, **kwargs):
    knee_locator = KneeLocator(x,y, **kwargs)
    return knee_locator.knee

class KneeClusterer():
    
    def __init__(
        self,
        range_param = 10, # Int, float or iterable, is k for Kmeans and epsilon for DBscan
        estimator = "kmeans",
        scorer = "calinski_harabasz",
        knee_locator = KneeLocator,
        estimator_kwargs = {},
        scorer_kwargs = {},
        knee_locator_kwargs = {}, # Arguements here overide the "automatically" decided kwargs, you have to provide all kwargs a custom knee finder is used
        float_range_num = 100, # Used when range_param is float (epsilon for DBscan), used to determine how many points are evaluated
        tuned_var = None, # For custom estimators, provide the variable name to be tested as a str
        verbose = 0,
    ):
        self.float_range_num = float_range_num
        self._set_range(range_param)
        self.estimator = CLUSTERER_MAP[estimator] if isinstance(estimator, str) else estimator
        self.scorer = SCORE_MAP[scorer] if isinstance(scorer, str) else scorer
        self.knee_locator = knee_locator
        self.estimator_kwargs = estimator_kwargs
        self.scorer_kwargs = scorer_kwargs # Used by silhouette score
        # Automatically determine some kwargs for the knee locator, if custom scorer is used, then user will have to determine the kwargs themselves
        self.knee_locator_kwargs = score_to_knee_locator_kwargs[estimator] | knee_locator_kwargs if estimator in CLUSTERER_MAP.keys else knee_locator_kwargs 
        self.verbose = verbose
        self.tuned_var = CLUSTERER_VAR_NAME[estimator] if None else tuned_var,
        self.knee = None
        self.best_estimator = None
        
        
    def fit(self, X, y, sample_weight): # y is ignored, following the sklearn interface
        
        # Fit all candidate clusterers once
        scores = []
        for x in self.range:
            estimator_args = self.estimator_kwargs | {self.tuned_var: x}
            labels = self.estimator(**estimator_args).fit_predict(X, y, sample_weight) # Use y for api consistance, it doesnt matter
            score = self.scorer(X, labels, **self.scorer_kwargs)
            scores.append(score)
            
        # Determine clusterer to be used by calculating elbow
        knee_locator = self.knee_locator(self.range, scores, **self.knee_locator_kwargs)
        knee = knee_locator.knee
        
        # Fit the best clusterer one last time
        # Yes we can store previous estimators in a list and return by indexing, that works as well
        estimator_args_best = self.estimator_kwargs | {self.tuned_var: knee}
        return self.estimator(**estimator_args_best).fit(X, y, sample_weight) # Returns a fitted estimator
        
    def fit_predict(self, X, y, sample_weight):
        estimator = self.fit(X, y, sample_weight=sample_weight)
        return estimator.predict(X)
        

    def _set_range(self, range_param):
        if isinstance(range_param, Iterable):
            self.range_param = range_param
        elif isinstance(range_param, int) and range_param > 2:
            self.range = range(2,range_param)
        elif isinstance(range_param, float):
            self.range = np.linspace(0, range_param, self.float_range_num)
        else:
            raise ValueError(f"Invalid range arguement provided:  {range_param}")
            

