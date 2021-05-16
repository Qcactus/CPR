from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string

cdef extern from "include/evaluator.h":
    cdef vector[vector[int]] Recommend(float *ratings,
                       int n_user,
                       int n_item,
                       int k,
                       int n_thread)

    cdef cppclass CppEvaluator:
        CppEvaluator() except +
        CppEvaluator(vector[unordered_set[int]] eval_set,
                     vector[string] metrics,
                     vector[int] ks,
                     int n_thread,
                     int n_group,
                     int *i_groups,
                     int *i_degrees,
                     float *metric_values) except +

        void Eval(float *ratings, int n_user, int n_item, int *users)

        void EvalTopK(vector[vector[int]] top_ks, int *users)
 