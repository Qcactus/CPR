from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

cdef extern from "include/labeler.h":
    cdef cppclass CppLabeler:
        CppLabeler() except +
        CppLabeler(vector[unordered_set[int]] dataset,
                    float *labels,
                    int n_step,
                    int batch_sample_size,
                    int n_thread) except +

        void Label(int *users, int *items)