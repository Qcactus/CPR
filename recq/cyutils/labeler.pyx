import numpy as np
cimport numpy as np
from .labeler cimport CppLabeler


cdef class CyLabeler:
    cdef CppLabeler c_labeler
    
    def __init__(self, vector[unordered_set[int]] dataset,
                    np.ndarray[float, ndim=1] labels,
                    int n_step,
                    int batch_sample_size,
                    int n_thread):
        labels = np.ascontiguousarray(labels)
        self.c_labeler = CppLabeler(dataset, &labels[0], 
                                    n_step, batch_sample_size, n_thread)
    
    def label(self, np.ndarray[int, ndim=1] users,
                np.ndarray[int, ndim=1] items):
        users = np.ascontiguousarray(users)
        items = np.ascontiguousarray(items)
        self.c_labeler.Label(&users[0], &items[0])
