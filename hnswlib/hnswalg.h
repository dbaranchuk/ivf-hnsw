#pragma once

#include "visited_list_pool.h"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)

#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

using namespace std;

inline bool exists_test(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

#define DEBUG_LIB 1
namespace hnswlib {
    typedef unsigned int idx_t;
    typedef unsigned char uint8_t;

    struct HierarchicalNSW
    {
        // Fields
        size_t maxelements_;
        size_t cur_element_count;

        size_t efConstruction_;
        int maxlevel_;

        VisitedListPool *visitedlistpool;

        mutex cur_element_count_guard_;
        mutex MaxLevelGuard_;

        idx_t enterpoint_node;

        size_t dist_calc;

        char *data_level0_memory_;

        vector<char> elementLevels;

        size_t d_;
        size_t data_size_;
        size_t offsetData;
        size_t size_data_per_element;
        size_t M_;
        size_t maxM_;
        size_t size_links_level0;


        mutex global;
        size_t ef_;
        float hops0 = 0.0;

    public:
        HierarchicalNSW(const string &infoLocation, const string &dataLocation, const string &edgeLocation);
        HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM, size_t efConstruction = 500);
        ~HierarchicalNSW();

        inline char *getDataByInternalId(idx_t internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element + offsetData);
        }

        inline uint8_t *get_linklist0(idx_t internal_id) const {
            return (uint8_t *) (data_level0_memory_ + internal_id * size_data_per_element);
        }

        //std::priority_queue<std::pair<dist_t, idx_t>, vector<pair<dist_t, idx_t>>, CompareByFirst>
        std::priority_queue<std::pair<float, idx_t>> searchBaseLayer(idx_t ep, void *datapoint, size_t ef);

        void getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, const int NN);

        void mutuallyConnectNewElement(void *datapoint, idx_t cur_c, std::priority_queue<std::pair<float, idx_t>> topResults, int level);

        void addPoint(void *datapoint, idx_t label);

        std::priority_queue<std::pair<float, idx_t >> searchKnn(void *query_data, int k);

        void SaveInfo(const string &location);
        void SaveEdges(const string &location);

        void LoadInfo(const string &location);
        void LoadData(const string &location);
        void LoadEdges(const string &location);
        
        float fstdistfunc(const void *x, const void *y);
    };
}
