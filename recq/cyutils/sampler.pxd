from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

cdef extern from "include/sampler.h":
    cdef cppclass CppCPRSampler:
        CppCPRSampler() except +
        CppCPRSampler(vector[unordered_set[int]] train,
                    int *u_interacts,
                    int *i_interacts,
                    int *users,
                    int *items,
                    int n_step,
                    int *batch_sample_sizes,
                    int sizes_len,
                    int n_thread) except +

        int Sample(int *interact_idx, int interact_idx_len, int *batch_choice_sizes)

    cdef cppclass CppPairNegSampler:
        CppPairNegSampler() except +
        CppPairNegSampler(vector[unordered_set[int]] train,
                    int *negs,
                    int n_step,
                    int batch_sample_size,
                    int n_thread) except +

        int Sample(int *users, int *items, int items_len)

    cdef cppclass CppPointNegSampler:
        CppPointNegSampler() except +
        CppPointNegSampler(vector[unordered_set[int]] train,
                    int *neg_users,
                    int *neg_items,
                    int n_step,
                    int batch_sample_size,
                    int n_thread) except +

        int Sample(int *rand_users, int *rand_items, int rand_len)

    cdef cppclass CppDICENegSampler:
        CppDICENegSampler() except +
        CppDICENegSampler(vector[unordered_set[int]] train,
                    int n_item,
                    float min_size,
                    int *i_degrees,
                    int *neg_items,
                    float *neg_mask,
                    int n_thread) except +

        void Sample(int *users, int *items, int len, float *rand, float margin)
 