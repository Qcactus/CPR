#ifndef SAMPLE_H
#define SAMPLE_H
#include <vector>
#include <unordered_set>
#include <iostream>
#include "thread_pool.h"

using namespace std;

int CPRSample1Batch(int *interact_idx,
                    const vector<unordered_set<int>> &train,
                    int *u_interacts,
                    int *i_interacts,
                    int *users,
                    int *items,
                    int k, // k-interaction samples
                    int batch_choice_size,
                    int batch_sample_size)
{
    int len = 0;
    int curr_u, next_i;
    for (int i = 0; i < batch_choice_size; i++)
    {
        bool flag = true;
        for (int j = 0; j < k; j++)
        {
            curr_u = u_interacts[interact_idx[i + batch_choice_size * j]];
            next_i = i_interacts[interact_idx[i + batch_choice_size * ((j + 1) % k)]];
            if (train[curr_u].find(next_i) != train[curr_u].end())
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            for (int j = 0; j < k; j++)
            {
                users[len + batch_sample_size * j] = u_interacts[interact_idx[i + batch_choice_size * j]];
                items[len + batch_sample_size * j] = i_interacts[interact_idx[i + batch_choice_size * j]];
            }
            len++;
            if (len == batch_sample_size)
                break;
        }
    }
    if (len < batch_sample_size)
        return -1;
    return 0;
}

class CppCPRSampler
{
public:
    CppCPRSampler(){};
    CppCPRSampler(const vector<unordered_set<int>> &train,
                  int *u_interacts,
                  int *i_interacts,
                  int *users,
                  int *items,
                  int n_step,
                  int *batch_sample_sizes,
                  int sizes_len,
                  int n_thread)
        : train(train),
          u_interacts(u_interacts),
          i_interacts(i_interacts),
          users(users),
          items(items),
          n_step(n_step),
          batch_sample_sizes(batch_sample_sizes),
          sizes_len(sizes_len),
          n_thread(n_thread) {}

    int Sample(int *interact_idx, int interact_idx_len, int *batch_choice_sizes)
    {
        ThreadPool pool(n_thread);
        vector<future<int>> results;
        int *interact_pt = interact_idx;
        int *u_pt = users;
        int *i_pt = items;

        for (int i = 0; i < n_step; i++)
        {
            for (int j = 0; j < sizes_len; j++)
            {
                results.emplace_back(
                    pool.enqueue(CPRSample1Batch,
                                 interact_pt,
                                 cref(train),
                                 u_interacts,
                                 i_interacts,
                                 u_pt,
                                 i_pt,
                                 j + 2,
                                 batch_choice_sizes[j],
                                 batch_sample_sizes[j]));
                interact_pt += (j + 2) * batch_choice_sizes[j];
                u_pt += (j + 2) * batch_sample_sizes[j];
                i_pt += (j + 2) * batch_sample_sizes[j];
            }
        }

        int status = 0;
        for (auto &&result : results)
        {
            if (result.get() == -1)
            {
                status = -1;
            }
        }
        return status;
    }

    vector<unordered_set<int>> train;
    int *u_interacts;
    int *i_interacts;
    int *users;
    int *items;
    int n_step;
    int *batch_sample_sizes;
    int sizes_len;
    int n_thread;
};

int PairNegSample1Batch(int *items,
                        int *users,
                        const vector<unordered_set<int>> &train,
                        int *negs,
                        int batch_choice_size,
                        int batch_sample_size)
{
    int len = 0;
    for (int i = 0; i < batch_choice_size; i++)
    {
        if (train[users[len]].find(items[i]) == train[users[len]].end())
        {
            negs[len++] = items[i];
            if (len == batch_sample_size)
                break;
        }
    }
    if (len < batch_sample_size)
        return -1;
    return 0;
}

class CppPairNegSampler
{
public:
    CppPairNegSampler(){};
    CppPairNegSampler(const vector<unordered_set<int>> &train,
                      int *negs,
                      int n_step,
                      int batch_sample_size,
                      int n_thread)
        : train(train),
          negs(negs),
          n_step(n_step),
          batch_sample_size(batch_sample_size),
          n_thread(n_thread) {}

    int Sample(int *users, int *items, int items_len)
    {
        int batch_choice_size = items_len / n_step;
        ThreadPool pool(n_thread);
        vector<future<int>> results;
        for (int i = 0; i < n_step; i++)
        {
            results.emplace_back(
                pool.enqueue(PairNegSample1Batch,
                             items + i * batch_choice_size,
                             users + i * batch_sample_size,
                             cref(train),
                             negs + i * batch_sample_size,
                             batch_choice_size,
                             batch_sample_size));
        }

        int status = 0;
        for (auto &&result : results)
        {
            if (result.get() == -1)
            {
                status = -1;
            }
        }
        return status;
    }

    vector<unordered_set<int>> train;
    int *negs;
    int n_step;
    int batch_sample_size;
    int n_thread;
};

int PointNegSample1Batch(int *rand_users,
                         int *rand_items,
                         const vector<unordered_set<int>> &train,
                         int *neg_users,
                         int *neg_items,
                         int batch_choice_size,
                         int batch_sample_size)
{
    int len = 0;
    for (int i = 0; i < batch_choice_size; i++)
    {
        if (train[rand_users[i]].find(rand_items[i]) == train[rand_users[i]].end())
        {
            neg_users[len] = rand_users[i];
            neg_items[len] = rand_items[i];
            len++;
            if (len == batch_sample_size)
                break;
        }
    }
    if (len < batch_sample_size)
        return -1;
    return 0;
}

class CppPointNegSampler
{
public:
    CppPointNegSampler(){};
    CppPointNegSampler(const vector<unordered_set<int>> &train,
                       int *neg_users,
                       int *neg_items,
                       int n_step,
                       int batch_sample_size,
                       int n_thread)
        : train(train),
          neg_users(neg_users),
          neg_items(neg_items),
          n_step(n_step),
          batch_sample_size(batch_sample_size),
          n_thread(n_thread) {}

    int Sample(int *rand_users, int *rand_items, int rand_len)
    {
        int batch_choice_size = rand_len / n_step;
        ThreadPool pool(n_thread);
        vector<future<int>> results;
        for (int i = 0; i < n_step; i++)
        {
            results.emplace_back(
                pool.enqueue(PointNegSample1Batch,
                             rand_users + i * batch_choice_size,
                             rand_items + i * batch_choice_size,
                             cref(train),
                             neg_users + i * batch_sample_size,
                             neg_items + i * batch_sample_size,
                             batch_choice_size,
                             batch_sample_size));
        }

        int status = 0;
        for (auto &&result : results)
        {
            if (result.get() == -1)
            {
                status = -1;
            }
        }
        return status;
    }

    vector<unordered_set<int>> train;
    int *neg_users;
    int *neg_items;
    int n_step;
    int batch_sample_size;
    int n_thread;
};

void DICESample1(float rand1,
                 float rand2,
                 int user,
                 int item,
                 int n_item,
                 float margin,
                 float min_size,
                 const vector<unordered_set<int>> &train,
                 int *i_degrees,
                 int *neg_item,
                 float *neg_mask)
{
    // vector<int> pop_items;
    // vector<int> unpop_items;
    int pop_items[n_item];
    int unpop_items[n_item];
    int n_pop = 0, n_unpop = 0;
    for (int i = 0; i < n_item; i++)
    {
        if (train[user].find(i) != train[user].end())
            continue;
        if (i_degrees[i] > i_degrees[item] + margin)
            // pop_items.emplace_back(i);
            pop_items[n_pop++] = i;
        if (i_degrees[i] < i_degrees[item] / 2.0)
            // unpop_items.emplace_back(i);
            unpop_items[n_unpop++] = i;
    }
    if (n_pop < min_size)
    {
        *(neg_item) = unpop_items[(int)floor(rand1 * n_unpop)];
        *(neg_mask) = 0;
    }
    else if (n_unpop < min_size)
    {
        *(neg_item) = pop_items[(int)floor(rand1 * n_pop)];
        *(neg_mask) = 1;
    }
    else
    {
        if (rand2 < 0.5)
        {
            *(neg_item) = unpop_items[(int)floor(rand1 * n_unpop)];
            *(neg_mask) = 0;
        }
        else
        {
            *(neg_item) = pop_items[(int)floor(rand1 * n_pop)];
            *(neg_mask) = 1;
        }
    }
}
class CppDICENegSampler
{
public:
    CppDICENegSampler(){};
    CppDICENegSampler(const vector<unordered_set<int>> &dataset,
                      int n_item,
                      float min_size,
                      int *i_degrees,
                      int *neg_items,
                      float *neg_mask,
                      int n_thread)
        : dataset(dataset),
          n_item(n_item),
          min_size(min_size),
          i_degrees(i_degrees),
          neg_items(neg_items),
          neg_mask(neg_mask),
          n_thread(n_thread) {}

    void Sample(int *users, int *items, int len, float *rand, float margin)
    {
        ThreadPool pool(n_thread);
        vector<future<void>> results;
        for (int i = 0; i < len; i++)
        {
            results.emplace_back(
                pool.enqueue(DICESample1,
                             rand[i],
                             rand[len + i],
                             users[i],
                             items[i],
                             n_item,
                             margin,
                             min_size,
                             cref(dataset),
                             i_degrees,
                             neg_items + i,
                             neg_mask + i));
        }

        for (auto &&result : results)
        {
            result.get();
        }
    }

    vector<unordered_set<int>> dataset;
    int n_item;
    float min_size;
    int *i_degrees;
    int *neg_items;
    float *neg_mask;
    int n_thread;
};

#endif