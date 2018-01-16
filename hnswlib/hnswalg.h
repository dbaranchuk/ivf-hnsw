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

#include <faiss/Heap.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

using namespace std;

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

namespace hnswlib {
    typedef unsigned int idx_t;
    typedef unsigned char uint8_t;

    struct HierarchicalNSW
    {
        // Fields
        size_t maxelements_;
        size_t cur_element_count;

        size_t efConstruction_;
        //int maxlevel_;

        VisitedListPool *visitedlistpool;

        std::mutex cur_element_count_guard_;
        //std::mutex MaxLevelGuard_;

        idx_t enterpoint_node;

        size_t dist_calc;

        char *data_level0_memory_;

        size_t d_;
        size_t data_size_;
        size_t offset_data;
        size_t size_data_per_element;
        size_t M_;
        size_t maxM_;
        size_t size_links_level0;

        mutex global;
        size_t efSearch;

    public:
        HierarchicalNSW(const string &infoLocation, const string &dataLocation, const string &edgeLocation);
        HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM, size_t efConstruction = 500);
        ~HierarchicalNSW();

        inline float *getDataByInternalId(idx_t internal_id) const {
            return (float *) (data_level0_memory_ + internal_id * size_data_per_element + offset_data);
        }

        inline uint8_t *get_linklist0(idx_t internal_id) const {
            return (uint8_t *) (data_level0_memory_ + internal_id * size_data_per_element);
        }

        std::priority_queue<std::pair<float, idx_t>> searchBaseLayer(const float *x, size_t ef);

        void getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, const int NN);

        void mutuallyConnectNewElement(const float *x, idx_t id, std::priority_queue<std::pair<float, idx_t>> topResults);

        void addPoint(const float *point, int label);

        std::priority_queue<std::pair<float, idx_t >> searchKnn(const float *query_data, int k);

        void SaveInfo(const string &location);
        void SaveEdges(const string &location);

        void LoadInfo(const string &location);
        void LoadData(const string &location);
        void LoadEdges(const string &location);
        
        float fstdistfunc(const float *x, const float *y);
    };
}
