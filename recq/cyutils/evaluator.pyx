import numpy as np
cimport numpy as np
from .evaluator cimport Recommend, CppEvaluator

def cy_recommend(np.ndarray[float, ndim=2] ratings, int k, int n_thread):
    ratings = np.ascontiguousarray(ratings)
    return Recommend(&ratings[0, 0], ratings.shape[0], ratings.shape[1], k, n_thread)

cdef class CyEvaluator:
    cdef CppEvaluator c_eval  # Hold a C++ instance which we're wrapping
    
    def __init__(self, vector[unordered_set[int]] eval_set,
                     vector[string] metrics,
                     vector[int] ks,
                     int n_thread,
                     int n_group,
                     np.ndarray[int, ndim=1] i_groups,
                     np.ndarray[int, ndim=1] i_degrees,
                     np.ndarray[float, ndim=4] metric_values):

        
        self.c_eval = CppEvaluator(eval_set, metrics, ks,
                        n_thread, n_group, &i_groups[0], &i_degrees[0], &metric_values[0, 0, 0, 0])

    def eval(self, np.ndarray[float, ndim=2] batch_ratings, np.ndarray[int, ndim=1] batch_users):
        batch_ratings = np.ascontiguousarray(batch_ratings)
        batch_users = np.ascontiguousarray(batch_users)
        self.c_eval.Eval(&batch_ratings[0, 0], batch_ratings.shape[0], batch_ratings.shape[1], &batch_users[0])
    
    def eval_top_k(self, vector[vector[int]] batch_top_k, np.ndarray[int, ndim=1] batch_users):
        # batch_top_k.shape[1] >= max(ks)
        batch_users = np.ascontiguousarray(batch_users)
        self.c_eval.EvalTopK(batch_top_k, &batch_users[0])

