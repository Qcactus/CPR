#ifndef LABELER_h
#define LABELER_h
#include <vector>
#include <unordered_set>
#include <iostream>
#include "thread_pool.h"

using namespace std;

void Label1Batch(int *users,
                 int *items,
                 const vector<unordered_set<int>> &dataset,
                 float *labels,
                 int batch_sample_size)
{
    for (int i = 0; i < batch_sample_size; i++)
    {
        if (dataset[users[i]].find(items[i]) == dataset[users[i]].end())
        {
            labels[i] = 0;
        }
        else
        {
            labels[i] = 1;
        }
    }
}

class CppLabeler
{
public:
    CppLabeler(){};
    CppLabeler(const vector<unordered_set<int>> &dataset,
               float *labels,
               int n_step,
               int batch_sample_size,
               int n_thread)
        : dataset(dataset),
          labels(labels),
          n_step(n_step),
          batch_sample_size(batch_sample_size),
          n_thread(n_thread) {}

    void Label(int *users, int *items)
    {
        ThreadPool pool(n_thread);
        vector<future<void>> results;
        for (int i = 0; i < n_step; i++)
        {
            results.emplace_back(
                pool.enqueue(Label1Batch,
                             users + i * batch_sample_size,
                             items + i * batch_sample_size,
                             cref(dataset),
                             labels + i * batch_sample_size,
                             batch_sample_size));
        }
        for (auto &&result : results)
        {
            result.get();
        }
    }

    vector<unordered_set<int>> dataset;
    float *labels;
    int n_step;
    int batch_sample_size;
    int n_thread;
};

#endif