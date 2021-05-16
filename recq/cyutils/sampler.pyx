import numpy as np
cimport numpy as np
from .sampler cimport CppCPRSampler, CppPairNegSampler, CppPointNegSampler, CppDICENegSampler


cdef class CyCPRSampler:
    cdef CppCPRSampler c_cross_sampler
    
    def __init__(self, vector[unordered_set[int]] train,
                    np.ndarray[int, ndim=1] u_interacts,
                    np.ndarray[int, ndim=1] i_interacts,
                    np.ndarray[int, ndim=1] users,
                    np.ndarray[int, ndim=1] items,
                    int n_step,
                    np.ndarray[int, ndim=1] batch_sample_sizes,
                    int n_thread):
        u_interacts = np.ascontiguousarray(u_interacts)
        i_interacts = np.ascontiguousarray(i_interacts)
        batch_sample_sizes = np.ascontiguousarray(batch_sample_sizes)
        self.c_cross_sampler = CppCPRSampler(train, 
                                            &u_interacts[0], 
                                            &i_interacts[0], 
                                            &users[0], 
                                            &items[0],
                                            n_step, 
                                            &batch_sample_sizes[0], 
                                            batch_sample_sizes.shape[0], 
                                            n_thread)
    
    def sample(self, np.ndarray[int, ndim=1] interact_idx, np.ndarray[int, ndim=1] batch_choice_sizes):
        interact_idx = np.ascontiguousarray(interact_idx)
        return self.c_cross_sampler.Sample(&interact_idx[0], interact_idx.shape[0], &batch_choice_sizes[0])

cdef class CyPairNegSampler:
    cdef CppPairNegSampler c_pair_neg_sampler
    
    def __init__(self, vector[unordered_set[int]] train,
                    np.ndarray[int, ndim=1] negs,
                    int n_step,
                    int batch_sample_size,
                    int n_thread):
        self.c_pair_neg_sampler = CppPairNegSampler(train, 
                                                    &negs[0], 
                                                    n_step, 
                                                    batch_sample_size, 
                                                    n_thread)
    
    def sample(self, np.ndarray[int, ndim=1] users,
                np.ndarray[int, ndim=1] items):
        users = np.ascontiguousarray(users)
        items = np.ascontiguousarray(items)
        return self.c_pair_neg_sampler.Sample(&users[0], &items[0], items.shape[0])

cdef class CyPointNegSampler:
    cdef CppPointNegSampler c_point_neg_sampler
    
    def __init__(self, vector[unordered_set[int]] train,
                    np.ndarray[int, ndim=1] neg_users,
                    np.ndarray[int, ndim=1] neg_items,
                    int n_step,
                    int batch_sample_size,
                    int n_thread):
        self.c_point_neg_sampler = CppPointNegSampler(train, 
                                                    &neg_users[0], 
                                                    &neg_items[0],
                                                    n_step, 
                                                    batch_sample_size, 
                                                    n_thread)
    
    def sample(self, np.ndarray[int, ndim=1] rand_users,
                np.ndarray[int, ndim=1] rand_items):
        rand_users = np.ascontiguousarray(rand_users)
        rand_items = np.ascontiguousarray(rand_items)
        return self.c_point_neg_sampler.Sample(&rand_users[0], &rand_items[0], rand_users.shape[0])

cdef class CyDICENegSampler:
    cdef CppDICENegSampler c_dice_sampler
    
    def __init__(self, vector[unordered_set[int]] train,
                    int n_item,
                    float min_size,
                    np.ndarray[int, ndim=1] i_degrees,
                    np.ndarray[int, ndim=1] neg_items,
                    np.ndarray[float, ndim=1] neg_mask,
                    int n_thread):
        self.c_dice_sampler = CppDICENegSampler(train,
                                            n_item,
                                            min_size,
                                            &i_degrees[0],
                                            &neg_items[0],
                                            &neg_mask[0],
                                            n_thread)
    
    def sample(self, np.ndarray[int, ndim=1] users,
                np.ndarray[int, ndim=1] items,
                np.ndarray[float, ndim=1] rand,
                float margin):
        users = np.ascontiguousarray(users)
        items = np.ascontiguousarray(items)
        self.c_dice_sampler.Sample(&users[0], &items[0], items.shape[0], &rand[0], margin)