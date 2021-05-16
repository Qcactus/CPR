#ifndef METRIC_H
#define METRIC_H

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

using namespace std;

float Precision(const vector<int> &rec, const unordered_set<int> &truth, const int *i_degrees)
{
    if (rec.size() == 0)
    {
        return NAN;
    }
    int hits = 0;
    for (auto r : rec)
    {
        if (truth.find(r) != truth.end())
        {
            hits++;
        }
    }
    return 1.0 * hits / rec.size();
}

float Recall(const vector<int> &rec, const unordered_set<int> &truth, const int *i_degrees)
{
    if (truth.size() == 0)
    {
        return NAN;
    }
    int hits = 0;
    for (auto r : rec)
    {
        if (truth.find(r) != truth.end())
        {
            hits++;
        }
    }
    return 1.0 * hits / truth.size();
}

float NDCG(const vector<int> &rec, const unordered_set<int> &truth, const int *i_degrees)
{
    if (rec.size() == 0 || truth.size() == 0)
    {
        return NAN;
    }
    float iDCG = 0;
    float DCG = 0;
    for (unsigned int i = 0; i < rec.size(); i++)
    {
        if (truth.find(rec[i]) != truth.end())
        {
            DCG += 1.0 / log2(i + 2);
        }
        if (i < truth.size())
        {
            iDCG += 1.0 / log2(i + 2);
        }
    }
    return DCG / iDCG;
}

float Rec(const vector<int> &rec, const unordered_set<int> &truth, const int *i_degrees)
{
    return rec.size();
}

float Hits(const vector<int> &rec, const unordered_set<int> &truth, const int *i_degrees)
{
    int hits = 0;
    for (auto r : rec)
    {
        if (truth.find(r) != truth.end())
        {
            hits++;
        }
    }
    return 1.0 * hits;
}

float ARP(const vector<int> &rec, const unordered_set<int> &truth, const int *i_degrees)
{
    if (rec.size() == 0)
    {
        return NAN;
    }
    float pop = 0;
    for (auto r : rec)
    {
        pop += *(i_degrees + r);
    }
    return pop / rec.size();
}
#endif