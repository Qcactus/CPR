#ifndef EVALUATOR_H
#define EVALUATOR_H
#include <vector>
#include <unordered_set>
#include <functional>
#include <map>
#include <iostream>
#include <numeric> // std::iota
#include <algorithm>
#include "thread_pool.h"
#include "metric.h"

using namespace std;

const map<string, function<float(const vector<int> &, const unordered_set<int> &, const int *)>>
    funcMap = {{"Recall", Recall}, {"Precision", Precision}, {"NDCG", NDCG}, {"Rec", Rec}, {"Hits", Hits}, {"ARP", ARP}};

vector<int> Recommend1D(float *ratings_1d,
                        int n_item,
                        int k)
{
    vector<int> rec_1d(k);
    // initialize original index locations
    vector<int> idx(n_item);
    iota(idx.begin(), idx.end(), 0);
    partial_sort_copy(idx.begin(), idx.end(), rec_1d.begin(), rec_1d.end(),
                      [&ratings_1d](int i1, int i2) { return ratings_1d[i1] > ratings_1d[i2]; });
    return rec_1d;
}

void Eval1D(float *ratings_1d,
            const vector<int> &top_k,
            int n_item,
            const unordered_set<int> &eval_set_1d,
            float *metric_values,
            const vector<string> &metrics,
            const vector<int> &ks,
            int n_col,
            int *i_groups,
            int *i_degrees)
{
    if (eval_set_1d.size() == 0)
    {
        for (size_t i = 0; i < metrics.size() * ks.size() * n_col; i++)
        {
            metric_values[i] = NAN;
        }
        return;
    }

    vector<int> rec_1d;
    if (ratings_1d)
    {
        int max_k = *max_element(ks.begin(), ks.end());
        rec_1d = Recommend1D(ratings_1d, n_item, max_k);
    }
    else
    {
        rec_1d = top_k;
    }
    for (size_t k = 0; k < ks.size(); k++)
    {
        vector<int> rec_k(rec_1d.begin(), rec_1d.begin() + ks[k]); // top k recommendations
        // calculate metrics without dividing groups
        for (size_t m = 0; m < metrics.size(); m++)
        {
            *(metric_values + (m * ks.size() + k) * n_col) = funcMap.at(metrics[m])(rec_k, eval_set_1d, i_degrees);
        }
        // calculate metrics in groups
        if (n_col > 1)
        {
            for (int i = 0; i < n_col - 1; i++)
            {
                vector<int> rec_k_i; // top k rec in group i
                for (auto &r : rec_k)
                {
                    if (i_groups[r] == i)
                    {
                        rec_k_i.emplace_back(r);
                    }
                }
                unordered_set<int> eval_set_i; // truth in group i
                for (auto &t : eval_set_1d)
                {
                    if (i_groups[t] == i)
                    {
                        eval_set_i.insert(t);
                    }
                }
                for (size_t m = 0; m < metrics.size(); m++)
                {
                    *(metric_values + (m * ks.size() + k) * n_col + i + 1) = funcMap.at(metrics[m])(rec_k_i, eval_set_i, i_degrees);
                }
            }
        }
    }
}

vector<vector<int>> Recommend(float *ratings,
                              int n_user,
                              int n_item,
                              int k,
                              int n_thread)
{
    vector<vector<int>> rec;
    ThreadPool pool(n_thread);
    vector<future<vector<int>>> results;

    for (int i = 0; i < n_user; i++)
    {
        results.emplace_back(
            pool.enqueue(Recommend1D, ratings + n_item * i, n_item, k));
    }

    for (auto &&result : results)
    {

        rec.emplace_back(result.get());
    }
    return rec;
}

class CppEvaluator
{
public:
    CppEvaluator(){};
    CppEvaluator(const vector<unordered_set<int>> &eval_set,
                 const vector<string> &metrics,
                 // List of k values. For each metric and each k, metric@k
                 // will be calculated when calling function update_metrics.
                 const vector<int> &ks,
                 const int n_thread,
                 const int n_group,
                 // Index list of the group to which each item belongs.
                 int *i_groups,
                 int *i_degrees,
                 float *metric_values)
        : eval_set(eval_set),
          metrics(metrics), ks(ks), n_thread(n_thread),
          n_group(n_group), i_groups(i_groups),
          i_degrees(i_degrees), metric_values(metric_values)
    {
        n_metric = metrics.size();
        n_k = ks.size();
        if (n_group == 1)
            n_col = 1;
        else
            n_col = n_group + 1;
    }

    void Eval(float *ratings, int n_user, int n_item, int *users)
    {
        ThreadPool pool(n_thread);
        vector<future<void>> results;

        for (int i = 0; i < n_user; i++)
        {
            results.emplace_back(
                pool.enqueue(Eval1D, ratings + n_item * i, vector<int>(), n_item, eval_set[users[i]], metric_values + users[i] * n_metric * n_k * n_col, metrics, ks, n_col, i_groups, i_degrees));
        }

        for (auto &&result : results)
        {
            result.get();
        }
    }
    void EvalTopK(const vector<vector<int>> &top_ks, int *users)
    {
        int n_user = top_ks.size();
        ThreadPool pool(n_thread);
        vector<future<void>> results;

        for (int i = 0; i < n_user; i++)
        {
            results.emplace_back(
                pool.enqueue(Eval1D, nullptr, top_ks[i], 0, eval_set[users[i]], metric_values + users[i] * n_metric * n_k * n_col, metrics, ks, n_col, i_groups, i_degrees));
        }

        for (auto &&result : results)
        {
            result.get();
        }
    }

    vector<unordered_set<int>> eval_set;
    vector<string> metrics;
    int n_metric;
    vector<int> ks;
    int n_k;
    int n_thread;
    int n_group;
    int n_col;
    int *i_groups;
    int *i_degrees;
    float *metric_values;
};

#endif