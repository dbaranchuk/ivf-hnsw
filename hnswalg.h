#pragma once

#include <sparsehash/dense_hash_map>

#include "hnswlib.h"
#include "visited_list_pool.h"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <map>
#include <cmath>
#include <queue>

using google::dense_hash_map;
using google::dense_hash_set;

enum ParameterIndex{
    i_threshold = 0,
    i_maxelements = 1,
    i_M = 2,
    i_maxM = 3,
    i_maxM0 = 4,
    i_size_links_level0 = 5,
    i_size_data_per_element = 6,
    i_offsetData = 7,
    i_size_links_per_element = 8,
    i_partOffset = 9
};


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
    typedef unsigned int tableint;
    typedef unsigned char linklistsizeint;

    template<typename dist_t, typename vtype>
    class HierarchicalNSW
    {
    public:
        HierarchicalNSW(SpaceInterface<dist_t> *s) {}

        HierarchicalNSW(SpaceInterface<dist_t> *s, const string &infoLocation, const string &dataLocation,
                        const string &edgeLocation, bool nmslib = false)
        {
            LoadInfo(infoLocation, s);
            data_level0_memory_ = (char *) malloc(total_size);
            LoadData(dataLocation);
            LoadEdges(edgeLocation);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::map<size_t, std::pair<size_t, size_t>> &M_map, size_t efConstruction = 200)
        {
            space = s;
            data_size_ = s->get_data_size();

            efConstruction_ = efConstruction;

            for (auto p : M_map){
                params[i_maxelements] = p.first;
                params[i_M] = p.second.first;
                params[i_maxM] = p.second.second;
                params[i_size_links_level0] = params[i_maxM]* sizeof(tableint) + sizeof(linklistsizeint);
                params[i_size_data_per_element] = params[i_size_links_level0] + data_size_;
                params[i_offsetData] = params[i_size_links_level0];
                params[i_partOffset] = total_size;

                total_size = params[i_maxelements] * params[i_size_data_per_element];
                maxelements_ = p.first;
            }
            elementLevels = vector<char>(maxelements_);
            for (int i = 0; i < maxelements_; i++)
                elementLevels[i] = 0;

            data_level0_memory_ = (char *) malloc(total_size);

            cout << "Size Mb: " << total_size / (1000 * 1000) << "\n";
            cur_element_count = 0;

            visitedlistpool = new VisitedListPool(1, maxelements_);
            visitedsetpool = new VisitedSetPool(1);
            //initializations for special treatment of the first node
            enterpoint_node = -1;
        }

        ~HierarchicalNSW()
        {
            free(data_level0_memory_);

            delete visitedsetpool;
            delete visitedlistpool;
        }
        // Fields
        SpaceInterface<dist_t> *space;

        size_t maxelements_;
        size_t cur_element_count;
        size_t efConstruction_;

        VisitedListPool *visitedlistpool;
        VisitedSetPool *visitedsetpool;

        mutex cur_element_count_guard_;
        mutex MaxLevelGuard_;

        tableint enterpoint_node;

        size_t dist_calc;

        char *data_level0_memory_;

        vector<char> elementLevels;

        size_t params[8];

        size_t data_size_;
        size_t total_size;

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + params[i_partOffset] +
                    internal_id * params[i_size_data_per_element] + params[i_offsetData]);
        }

        inline linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + params[i_partOffset] + internal_id * params[i_size_data_per_element]);
        };

        std::priority_queue<std::pair<dist_t, tableint  >> searchBaseLayer(tableint ep, void *datapoint, int ef)
        {
            VisitedSet *vs = visitedsetpool->getFreeVisitedSet();

            std::priority_queue<std::pair<dist_t, tableint  >> topResults;
            std::priority_queue<std::pair<dist_t, tableint >> candidateSet;
            dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(ep));

            topResults.emplace(dist, ep);
            candidateSet.emplace(-dist, ep);
            vs->insert(ep);

            dist_t lowerBound = dist;

            while (!candidateSet.empty()) {

                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                linklistsizeint *ll_cur = get_linklist0(curNodeNum);
                linklistsizeint size = *ll_cur;
                tableint *data = (tableint *) (ll_cur + 1);

                _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

                for (linklistsizeint j = 0; j < size; ++j) {
                    tableint tnum = *(data + j);
                    _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);
                    if (vs->count(tnum) == 0){
                        vs->insert(tnum);
                        dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(tnum));
                        if (topResults.top().first > dist || topResults.size() < ef) {
                            candidateSet.emplace(-dist, tnum);
                            _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
                            topResults.emplace(dist, tnum);
                            if (topResults.size() > ef) {
                                topResults.pop();
                            }
                            lowerBound = topResults.top().first;
                        }
                    }
                }
            }
            visitedsetpool->releaseVisitedSet(vs);
            return topResults;
        }

        struct CompareByFirst {
            constexpr bool operator()(pair<dist_t, tableint> const &a,
                                      pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep, void *datapoint, size_t ef)
        {
            VisitedList *vl = visitedlistpool->getFreeVisitedList();
            //VisitedSet *vs = visitedsetpool->getFreeVisitedSet();
            vl_type *massVisited = vl->mass;
            vl_type currentV = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> topResults;
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(ep));

            dist_calc++;
            topResults.emplace(dist, ep);
            candidateSet.emplace(-dist, ep);
            massVisited[ep] = currentV;
//          vs->insert(ep);
            dist_t lowerBound = dist;

            while (!candidateSet.empty()) {
                hops0 += 1.0 / 10000;
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if (-curr_el_pair.first > lowerBound)
                    break;

                candidateSet.pop();
                tableint curNodeNum = curr_el_pair.second;

                linklistsizeint *ll_cur = get_linklist0(curNodeNum);
                linklistsizeint size = *ll_cur;
                tableint *data = (tableint *)(ll_cur + 1);

                _mm_prefetch((char *) (massVisited + *data), _MM_HINT_T0);
                _mm_prefetch((char *) (massVisited + *data + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

                for (linklistsizeint j = 0; j < size; ++j) {
                    int tnum = *(data + j);

                    _mm_prefetch((char *) (massVisited + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);

//                    if (vs->count(tnum) == 0){
//                        vs->insert(tnum);
                    if (!(massVisited[tnum] == currentV)) {
                        massVisited[tnum] = currentV;

                        dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(tnum));
                        dist_calc++;

                        if (topResults.top().first > dist || topResults.size() < ef) {
                            candidateSet.emplace(-dist, tnum);

                            _mm_prefetch(get_linklist0(candidateSet.top().second), _MM_HINT_T0);
                            topResults.emplace(dist, tnum);

                            if (topResults.size() > ef)
                                topResults.pop();

                            lowerBound = topResults.top().first;
                        }
                    }
                }
            }
            //visitedsetpool->releaseVisitedSet(vs);
            visitedlistpool->releaseVisitedList(vl);
            return topResults;
        }

        void getNeighborsByHeuristic(std::priority_queue<std::pair<dist_t, tableint>> &topResults, const int NN) {
            if (topResults.size() < NN)
                return;

            std::priority_queue<std::pair<dist_t, tableint>> resultSet;
            std::priority_queue<std::pair<dist_t, tableint>> templist;
            vector<std::pair<dist_t, tableint>> returnlist;
            while (topResults.size() > 0) {
                resultSet.emplace(-topResults.top().first, topResults.top().second);
                topResults.pop();
            }

            while (resultSet.size()) {
                if (returnlist.size() >= NN)
                    break;
                std::pair<dist_t, tableint> curen = resultSet.top();
                dist_t dist_to_query = -curen.first;
                resultSet.pop();
                bool good = true;
                for (std::pair<dist_t, tableint> curen2 : returnlist) {
                    dist_t curdist = space->fstdistfunc(getDataByInternalId(curen2.second), getDataByInternalId(curen.second));
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) returnlist.push_back(curen);
            }
            for (std::pair<dist_t, tableint> curen2 : returnlist)
                topResults.emplace(-curen2.first, curen2.second);
        }

        void mutuallyConnectNewElement(void *datapoint, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>> topResults)
        {
            std::cout << "HUI\n";
            size_t Mmax = params[i_maxM0];
            size_t M = params[i_M];

            getNeighborsByHeuristic(topResults, M);

            while (topResults.size() > M) {
                throw exception();
                topResults.pop();
            }

            vector<tableint> rez(M);
            while (topResults.size() > 0) {
                rez.push_back(topResults.top().second);
                topResults.pop();
            }
            {
                linklistsizeint *ll_cur = get_linklist0(cur_c);

                if (*ll_cur) {
                    cout << *ll_cur << "\n";
                    cout << (int) elementLevels[cur_c] << "\n";
                    throw runtime_error("Should be blank");
                }
                *ll_cur = rez.size();

                tableint *data = (tableint *)(ll_cur + 1);
                for (int idx = 0; idx < rez.size(); idx++) {
                    if (data[idx])
                        throw runtime_error("Should be blank");
                    data[idx] = rez[idx];
                }
            }
            for (int idx = 0; idx < rez.size(); idx++) {
                if (rez[idx] == cur_c)
                    throw runtime_error("Connection to the same element");

                linklistsizeint *ll_other = get_linklist0(rez[idx]);
                linklistsizeint sz_link_list_other = *ll_other;

                if (sz_link_list_other > Mmax || sz_link_list_other < 0)
                    throw runtime_error("Bad sz_link_list_other");

                if (sz_link_list_other < Mmax) {
                    tableint *data = (tableint *) (ll_other + 1);
                    data[sz_link_list_other] = cur_c;
                    *ll_other = sz_link_list_other + 1;
                } else {
                    // finding the "weakest" element to replace it with the new one
                    tableint *data = (tableint *) (ll_other + 1);
                    dist_t d_max = space->fstdistfunc(getDataByInternalId(cur_c), getDataByInternalId(rez[idx]));
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (int j = 0; j < sz_link_list_other; j++)
                        candidates.emplace(space->fstdistfunc(getDataByInternalId(data[j]),
                                                              getDataByInternalId(rez[idx])), data[j]);

                    getNeighborsByHeuristic(candidates, Mmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    *ll_other = indx;
                }
            }
        }

        mutex global;
        size_t ef_;

        float nev9zka = 0.0;
        tableint enterpoint0;
        float hops0 = 0.0;

        void addPoint(void *datapoint, labeltype label)
        {
            tableint cur_c = 0;
            {
                unique_lock <mutex> lock(cur_element_count_guard_);
                if (cur_element_count >= maxelements_) {
                    cout << "The number of elements exceeds the specified limit\n";
                    throw runtime_error("The number of elements exceeds the specified limit");
                };
                cur_c = cur_element_count;
                cur_element_count++;
            }

            memset((char *) get_linklist0(cur_c), 0, params[i_size_data_per_element]);
            memcpy(getDataByInternalId(cur_c), datapoint, data_size_);

            tableint currObj = enterpoint_node;
            enterpoint_node = 0;
            std::priority_queue<std::pair<dist_t, tableint>> topResults = searchBaseLayer(currObj, datapoint,
                                                                                          efConstruction_);
            mutuallyConnectNewElement(datapoint, cur_c, topResults);
        };

        std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(void *query_data, int k)
        {
            dist_t curdist = space->fstdistfunc(query_data, getDataByInternalId(enterpoint_node));
            dist_calc++;

            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> tmpTopResults = searchBaseLayerST(
                    enterpoint_node, query_data, ef_);
            std::priority_queue<std::pair<dist_t, labeltype >> results;

            // Remove clusters as answers
            std::priority_queue<std::pair<dist_t, tableint >> topResults;
            while (tmpTopResults.size() > 0) {
                std::pair<dist_t, tableint> rez = tmpTopResults.top();
                topResults.push(rez);
                tmpTopResults.pop();
            }

            while (topResults.size() > k)
                topResults.pop();

            //while (topResults.size() > 0) {
            //    std::pair<dist_t, tableint> rez = topResults.top();
            //    results.push(std::pair<dist_t, labeltype>(rez.first, rez.second-maxclusters_));
            //    topResults.pop();
            //}
            return topResults;
        };


        void SaveInfo(const string &location) {
            cout << "Saving info to " << location << endl;
            std::ofstream output(location, std::ios::binary);
            streampos position;

            writeBinaryPOD(output, enterpoint_node);
            output.write((char *) params, 8 * sizeof(size_t));
            output.close();
        }


        void SaveEdges(const string &location)
        {
            cout << "Saving edges to " << location << endl;
            FILE *fout = fopen(location.c_str(), "wb");

            for (tableint i = 0; i < maxelements_; i++) {
                linklistsizeint *ll_cur = get_linklist0(i);
                int size = *ll_cur;

                fwrite((int *)&size, sizeof(int), 1, fout);
                tableint *data = (tableint *)(ll_cur + 1);
                fwrite(data, sizeof(tableint), *ll_cur, fout);
            }
        }

        void LoadInfo(const string &location, SpaceInterface<dist_t> *s)
        {
            cout << "Loading info from " << location << endl;
            std::ifstream input(location, std::ios::binary);
            streampos position;

            space = s;
            data_size_ = s->get_data_size();

            readBinaryPOD(input, enterpoint_node);

            input.read((char *) params, 8*sizeof(size_t));

            efConstruction_ = 0;
            maxelements_ = params[i_maxelements];
            total_size = params[i_maxelements] * params[i_size_data_per_element];

            cur_element_count = maxelements_;
            visitedlistpool = new VisitedListPool(1, maxelements_);
            visitedsetpool = new VisitedSetPool(1);

            elementLevels = vector<char>(maxelements_);
            for (int i = 0; i < maxelements_; i++)
                elementLevels[i] = 0;

            cout << "Predicted size=" << total_size / (1000 * 1000) << "\n";
            input.close();
        }

        void LoadData(const string &location)
        {
            cout << "Loading data from " << location << endl;
            FILE *fin = fopen(location.c_str(), "rb");
            int dim;
            const int D = space->get_data_dim();
            vtype mass[D];
            //unsigned char mass[D];
            for (tableint i = 0; i < maxelements_; i++) {
                fread((int *) &dim, sizeof(int), 1, fin);
                if (dim != D)
                    cerr << "Wront data dim" << endl;

                fread(mass, sizeof(vtype), dim, fin);
                //fread(mass, sizeof(unsigned char), dim, fin);
                memset((char *) get_linklist0(i), 0, params[i_size_data_per_element]);
                memcpy(getDataByInternalId(i), mass, data_size_);
            }


        }

        void LoadEdges(const string &location)
        {
            cout << "Loading edges from " << location << endl;
            FILE *fin = fopen(location.c_str(), "rb");
            int size;

            for (tableint i = 0; i < maxelements_; i++) {
                fread((int *)&size, sizeof(int), 1, fin);
                linklistsizeint *ll_cur = get_linklist0(i);
                *ll_cur = size;
                tableint *data = (tableint *)(ll_cur + 1);

                fread((tableint *)data, sizeof(tableint), size, fin);
            }
        }
    };
}
