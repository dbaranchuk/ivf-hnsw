#pragma once

#include <sparsehash/dense_hash_map>

#include "visited_list_pool.h"
#include "hnswlib.h"
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
            total_size = 0;
            maxelements_ = 0;
            parts_num = M_map.size();
            params_num = 10;
            params = new size_t[parts_num * params_num];
            int i = 0;
            for (auto p : M_map){
                params[i*params_num + i_threshold] = p.first;
                params[i*params_num + i_maxelements] = i ? p.first - params[(i-1)*params_num + i_threshold] : p.first;
                params[i*params_num + i_M] = p.second.first;
                params[i*params_num + i_maxM] = p.second.second;
                params[i*params_num + i_maxM0] = p.second.second;//(i == parts_num-1) ? p.second : 2 * p.second;
                params[i*params_num + i_size_links_level0] = params[i*params_num + i_maxM0]* sizeof(tableint) + sizeof(linklistsizeint);
                params[i*params_num + i_size_data_per_element] = params[i*params_num + i_size_links_level0] + data_size_;
                params[i*params_num + i_offsetData] = params[i*params_num + i_size_links_level0];
                params[i*params_num + i_size_links_per_element] = params[i*params_num + i_maxM] * sizeof(tableint) + sizeof(linklistsizeint);
                params[i*params_num + i_partOffset] = total_size;

                total_size += params[i*params_num + i_maxelements] * params[i*params_num + i_size_data_per_element];
                maxelements_ += params[i*params_num + i_maxelements];
                i++;
            }
            elementLevels = vector<char>(maxelements_);
            std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;
            data_level0_memory_ = (char *) malloc(total_size);
            std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;

            cout << "Size Mb: " << total_size / (1000 * 1000) << "\n";
            cur_element_count = 0;

            //visitedlistpool = new VisitedListPool(1, maxelements_);
            visitedsetpool = new VisitedSetPool(1);
            //initializations for special treatment of the first node
            enterpoint_node = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * params[0*params_num + i_maxelements]);
            mult_ = 1 / log(1.0 * params[0*params_num + i_M]);//M_);
        }

        ~HierarchicalNSW()
        {
            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (elementLevels[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visitedsetpool;
            //delete visitedlistpool;
            delete params;
        }
        // Fields
        SpaceInterface<dist_t> *space;

        size_t maxelements_;
        size_t cur_element_count;

        size_t efConstruction_;
        double mult_;
        int maxlevel_;

        VisitedListPool *visitedlistpool;
        VisitedSetPool *visitedsetpool;

        mutex cur_element_count_guard_;
        mutex MaxLevelGuard_;

        tableint enterpoint_node;

        size_t dist_calc;

        char *data_level0_memory_;
        char **linkLists_;

        vector<char> elementLevels;

        size_t *params;
        size_t parts_num;
        size_t params_num;

        size_t data_size_;
        size_t total_size;
        std::default_random_engine generator = std::default_random_engine(100);


//        inline labeltype *getExternalLabelPointer(tableint internal_id)
//        {
//            if (internal_id < maxclusters_)
//                return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_cluster_ + label_offset_cluster_);
//            else {
//                tableint internal_element_id = internal_id - maxclusters_;
//                return (labeltype *) (data_level0_memory_ + maxclusters_ * size_data_per_cluster_ +
//                                      internal_element_id * size_data_per_element_ + label_offset_);
//            }
//        }

        inline size_t *getParametersByInternalId(tableint internal_id)
        {
            size_t *param = params;
            while (internal_id >= param[i_threshold]) param += params_num;
            return param;
        };

        inline char *getDataByInternalId(tableint internal_id)
        {
            size_t *param = getParametersByInternalId(internal_id);
            tableint ref_id = (param == params) ? internal_id : internal_id - (param - params_num)[i_threshold];
            return (data_level0_memory_ + param[i_partOffset] + ref_id * param[i_size_data_per_element] + param[i_offsetData]);
        }

        inline linklistsizeint *get_linklist0(tableint internal_id)
        {
            size_t *param = getParametersByInternalId(internal_id);
            tableint ref_id = (param == params) ? internal_id : internal_id - (param - params_num)[i_threshold];

            return (linklistsizeint *) (data_level0_memory_ + param[i_partOffset] + ref_id * param[i_size_data_per_element]);
        };

        inline linklistsizeint* get_linklist(tableint cur_c, int level)
        {
            size_t *param = getParametersByInternalId(cur_c);
            //In Smart hnsw only clusters on the above levels
            return (linklistsizeint *)(linkLists_[cur_c] + (level - 1) * param[i_size_links_per_element]);
        };


        std::priority_queue<std::pair<dist_t, tableint  >> searchBaseLayer(tableint ep, void *datapoint, int level, int ef)
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

                linklistsizeint *ll_cur = level ? get_linklist(curNodeNum, level) : get_linklist0(curNodeNum);
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
        searchBaseLayerST(tableint ep, void *datapoint, size_t ef, int q_idx = -1)
        {
            //VisitedList *vl = visitedlistpool->getFreeVisitedList();
            VisitedSet *vs = visitedsetpool->getFreeVisitedSet();

            //vl_type *massVisited = vl->mass;
            //vl_type currentV = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> topResults;
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t dist;
            if (q_idx != -1)
                dist = space->fstdistfuncST(getDataByInternalId(ep));
            else
                dist = space->fstdistfunc(datapoint, getDataByInternalId(ep));

            dist_calc++;
            topResults.emplace(dist, ep);
            candidateSet.emplace(-dist, ep);
            //massVisited[ep] = currentV;
            vs->insert(ep);
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

                //_mm_prefetch((char *) (massVisited + *data), _MM_HINT_T0);
                //_mm_prefetch((char *) (massVisited + *data + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

                for (linklistsizeint j = 0; j < size; ++j) {
                    int tnum = *(data + j);

                    //_mm_prefetch((char *) (massVisited + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);

                    if (vs->count(tnum) == 0){
                        vs->insert(tnum);
                    //if (!(massVisited[tnum] == currentV)) {
                    //    massVisited[tnum] = currentV;

                        dist_t dist;
                        if (q_idx != -1)
                            dist = space->fstdistfuncST(getDataByInternalId(tnum));
                        else
                            dist = space->fstdistfunc(datapoint, getDataByInternalId(tnum));

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
            visitedsetpool->releaseVisitedSet(vs);
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
                                       std::priority_queue<std::pair<dist_t, tableint>> topResults, int level)
        {
            size_t *param = getParametersByInternalId(cur_c);
            size_t curMmax = level ? param[i_maxM] : param[i_maxM0];
            size_t curM = param[i_M];

            getNeighborsByHeuristic(topResults, curM);

            while (topResults.size() > curM) {
                throw exception();
                topResults.pop();
            }

            vector<tableint> rez;
            rez.reserve(curM);
            while (topResults.size() > 0) {
                rez.push_back(topResults.top().second);
                topResults.pop();
            }
            {
                linklistsizeint *ll_cur = level ? get_linklist(cur_c, level) : get_linklist0(cur_c);

                if (*ll_cur) {
                    cout << *ll_cur << "\n";
                    cout << (int) elementLevels[cur_c] << "\n";
                    cout << level << "\n";
                    throw runtime_error("Should be blank");
                }
                *ll_cur = rez.size();

                tableint *data = (tableint *)(ll_cur + 1);
                for (int idx = 0; idx < rez.size(); idx++) {
                    if (data[idx])
                        throw runtime_error("Should be blank");
                    if (level > elementLevels[rez[idx]])
                        throw runtime_error("Bad level");
                    data[idx] = rez[idx];
                }
            }
            for (int idx = 0; idx < rez.size(); idx++) {
                if (rez[idx] == cur_c)
                    throw runtime_error("Connection to the same element");

                auto rezParam = getParametersByInternalId(rez[idx]);
                size_t rezMmax = level ? rezParam[i_maxM] : rezParam[i_maxM0];
                linklistsizeint *ll_other = level ? get_linklist(rez[idx], level) : get_linklist0(rez[idx]);

                if (level > elementLevels[rez[idx]])
                    throw runtime_error("Bad level");

                linklistsizeint sz_link_list_other = *ll_other;

                if (sz_link_list_other > rezMmax || sz_link_list_other < 0)
                    throw runtime_error("Bad sz_link_list_other");

                if (sz_link_list_other < rezMmax) {
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
                        candidates.emplace(space->fstdistfunc(getDataByInternalId(data[j]), getDataByInternalId(rez[idx])), data[j]);

                    getNeighborsByHeuristic(candidates, rezMmax);

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
        // My
        float nev9zka = 0.0;
        tableint enterpoint0;
        float hops = 0.0;
        float hops0 = 0.0;

        void setElementLevels(const vector<size_t> &elements_per_level, bool one_layer)
        {
            if (elements_per_level.size() == 0) {
                std::uniform_real_distribution<double> distribution(0.0, 1.0);
                for (size_t i = 0; i < params[0*params_num + i_maxelements]; ++i) {
                    elementLevels[i] = 0; //(int) (-log(distribution(generator)) * mult_) + 1;
                }
                for (int i = params[0*params_num + i_maxelements]; i < maxelements_; i++)
                    elementLevels[i] = 0;
            } else{
                for (size_t i = 0; i < maxelements_; ++i) {
                    if (one_layer){
                        elementLevels[i] = 0;
                        continue;
                    }
                    if (i < elements_per_level[5])
                        elementLevels[i] = 5;
                    else if (i < elements_per_level[5] + elements_per_level[4])
                        elementLevels[i] = 4;
                    else if (i < elements_per_level[5] + elements_per_level[4] +
                                 elements_per_level[3])
                        elementLevels[i] = 3;
                    else if (i < elements_per_level[5] + elements_per_level[4] +
                                 elements_per_level[3] + elements_per_level[2])
                        elementLevels[i] = 2;
                    else if (i < elements_per_level[5] + elements_per_level[4] +
                                 elements_per_level[3] + elements_per_level[2] + elements_per_level[1])
                        elementLevels[i] = 1;
                    else
                        elementLevels[i] = 0;
                }
            }
        }

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
            int curlevel = elementLevels[cur_c];

            unique_lock <mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();


            auto curParam = getParametersByInternalId(cur_c);
            memset((char *) get_linklist0(cur_c), 0, curParam[i_size_data_per_element]);

//            // Initialisation of the data and label
//            //memcpy(getExternalLabelPointer(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), datapoint, data_size_);

            if (curlevel) {
                // Above levels contain only clusters
                linkLists_[cur_c] = (char *) malloc(curParam[i_size_links_per_element] * curlevel);
                memset(linkLists_[cur_c], 0, curParam[i_size_links_per_element] * curlevel);
            }

            tableint currObj = enterpoint_node;
            if (currObj != -1) {
                if (curlevel < maxlevelcopy) {
                    dist_t curdist = space->fstdistfunc(datapoint, getDataByInternalId(currObj));
                    for (int level = maxlevelcopy; level > curlevel; level--) {

                        bool changed = true;
                        while (changed) {
                            changed = false;
                            linklistsizeint *data = get_linklist(currObj, level);
                            linklistsizeint size = *data;
                            tableint *datal = (tableint *) (data + 1);
                            for (linklistsizeint i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > maxelements_)
                                    throw runtime_error("cand error");
                                dist_t d = space->fstdistfunc(datapoint, getDataByInternalId(cand));
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                for (int level = 0; level <= min(curlevel, maxlevelcopy); level++) {
                    if (level > maxlevelcopy || level < 0)
                        throw runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>> topResults = searchBaseLayer(currObj, datapoint,
                                                                                                    level, efConstruction_);
                    mutuallyConnectNewElement(datapoint, cur_c, topResults, level);
                }

            } else {
                // Do nothing for the first element
                enterpoint_node = 0;
                maxlevel_ = curlevel;

            }
            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node = cur_c;
                maxlevel_ = curlevel;
            }
        };

        std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(void *query_data, int k, int q_idx = -1)
        {
            tableint currObj = enterpoint_node;
            dist_t curdist;

            if (q_idx != -1)
                curdist = space->fstdistfuncST(getDataByInternalId(enterpoint_node));
            else
                curdist = space->fstdistfunc(query_data, getDataByInternalId(enterpoint_node));

            dist_calc++;
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    linklistsizeint *data = get_linklist(currObj, level);
                    linklistsizeint size = *data;
                    tableint *datal = (tableint *) (data + 1);
                    for (linklistsizeint i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > maxelements_)
                            throw runtime_error("cand error");

                        dist_t d;
                        if (q_idx != -1)
                            d = space->fstdistfuncST(getDataByInternalId(cand));
                        else
                            d = space->fstdistfunc(query_data, getDataByInternalId(cand));

                        dist_calc++;
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                            hops += 1.f / 10000;
                        }
                    }
                }
            }
            enterpoint0 = currObj;

            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> tmpTopResults = searchBaseLayerST(
                    currObj, query_data, ef_, q_idx);
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


        void printListsize()
        {
            float av_M = 0;
            int numLinks[32];
            for (int i = 0; i < 32; i++)
                numLinks[i] = 0;

            for (int i = 0; i < maxelements_; i++){
                linklistsizeint *ll_cur = get_linklist0(i);
                numLinks[*ll_cur - 1]++;
                av_M += (1.0 * *ll_cur) / maxelements_;
                //if (i % 10000 != 0)
                //    continue;
                //cout << "Element #" << i << " M:" << (int) *ll_cur << endl;
            }

            cout << "Average number of links: " << endl;
            cout << "Links distribution" << endl;
            for (int i = 0; i < 32; i++){
                cout << " Number of elements with " << i+1 << " links: " << numLinks[i] << endl;
            }
            int part_1 = 0;
            for (int i = 24; i < 32; i++)
                part_1 += numLinks[i];
            cout << "Part Mmax = 32: " << part_1 << endl;

            int part_2 = 0;
            for (int i = 16; i < 24; i++)
                part_2 += numLinks[i];
            cout << "Part Mmax = 24: " << part_2 << endl;

            int part_3 = 0;
            for (int i = 10; i < 16; i++)
                part_3 += numLinks[i];
            cout << "Part Mmax = 16: " << part_3 << endl;

            int part_4 = 0;
            for (int i = 7; i < 10; i++)
                part_4 += numLinks[i];
            cout << "Part Mmax = 10: " << part_4 << endl;

            int part_5 = 0;
            for (int i = 5; i < 7; i++)
                part_5 += numLinks[i];
            cout << "Part Mmax = 7: " << part_5 << endl;

            int part_6 = 0;
            for (int i = 0; i < 5; i++)
                part_6 += numLinks[i];
            cout << "Part Mmax = 5: " << part_6 << endl;
        }


        void SaveInfo(const string &location) {
            cout << "Saving info to " << location << endl;
            std::ofstream output(location, std::ios::binary);
            streampos position;

            writeBinaryPOD(output, enterpoint_node);
            writeBinaryPOD(output, parts_num);
            writeBinaryPOD(output, params_num);
            output.write((char *) params, parts_num * params_num * sizeof(size_t));

            for (size_t i = 0; i < params[0*params_num + i_maxelements]; ++i) {
                unsigned int linkListSize = elementLevels[i] > 0 ? params[0*params_num + i_size_links_per_element] * elementLevels[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write((char *)linkLists_[i], linkListSize);
            }
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
            readBinaryPOD(input, parts_num);
            readBinaryPOD(input, params_num);
            cout << enterpoint_node << " " << parts_num << " " << params_num << endl;
            //enterpoint_node  = 0;
            params = new size_t[params_num*parts_num];
            input.read((char *) params, parts_num*params_num*sizeof(size_t));

            efConstruction_ = 0;
            total_size = 0;
            maxelements_ = 0;
            for (size_t i = 0; i < parts_num; i++) {
                maxelements_ += params[i*params_num + i_maxelements];
                total_size += params[i*params_num + i_maxelements] * params[i*params_num + i_size_data_per_element];
            }
            cur_element_count = maxelements_;
            //visitedlistpool = new VisitedListPool(1, maxelements_);
            visitedsetpool = new VisitedSetPool(1);

            /** Hierarcy **/
            linkLists_ = (char **) malloc(sizeof(void *) * (params[0*params_num + i_maxelements]));

            elementLevels = vector<char>(maxelements_);
            maxlevel_ = 0;

            for (size_t i = 0; i < params[0*params_num + i_maxelements]; i++) {
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    elementLevels[i] = 0;
                    linkLists_[i] = nullptr;
                } else {
                    elementLevels[i] = linkListSize / params[0*params_num + i_size_links_per_element];
                    linkLists_[i] = (char *) malloc(linkListSize);
                    input.read(linkLists_[i], linkListSize);
                }
                if (elementLevels[i] > maxlevel_){
                    maxlevel_ = elementLevels[i];
                }
            }
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
            float massf[D];
            unsigned char mass_code[data_size_];

            for (tableint i = 0; i < maxelements_; i++) {
                fread((int *) &dim, sizeof(int), 1, fin);
                if (dim != D)
                    cerr << "Wront data dim" << endl;

                fread((vtype *)mass, sizeof(vtype), dim, fin);
                for (int j = 0; j < D; j++)
                    massf[j] = (1.0)*mass[j];

                //dynamic_cast<NewL2SpacePQ *>(space)->pq->compute_code(massf, mass_code);

                // Initialisation of the data and label
                //memcpy(getExternalLabelPointer(cur_c), &label, sizeof(labeltype));
                memset((char *) get_linklist0(i), 0, getParametersByInternalId(i)[i_size_data_per_element]);
                memcpy(getDataByInternalId(i), mass_code, data_size_);
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


        void check(bool *map, tableint ep)
        {
            queue<tableint> q;
            q.push(ep);

            while(!q.empty()) {
                linklistsizeint *ll_cur = get_linklist0(q.front());
                linklistsizeint size = *ll_cur;
                tableint *data = (tableint *) (ll_cur + 1);
                q.pop();

                for (tableint l = 0; l < size; l++) {
                    if (map[*(data + l)]) continue;
                    map[*(data + l)] = true;
                    q.push(*(data + l));
                }
            }
        }

        int check_connectivity(unsigned int *massQA, int qsize)
        {
            bool *map = new bool[maxelements_];
            memset((char *)map, 0, sizeof(bool)*maxelements_);

            //#pragma parallel for num_thereds(32)
            map[0] = true;
            check(map, 0);

            int num_unreachable = 0;
            int num_unreachable_gt = 0;

            #pragma omp parallel for num_threads(32)
            for (tableint i = 0; i < maxelements_; i++){
                if (!map[i]) {
                    for (int q = 0; q < qsize; q++){
                        if (i == massQA[1000*q]) {
                            num_unreachable_gt++;
                            break;
                        }
                    }
                    num_unreachable++;
                }
            }
            cout << "Number of unreachable gt nodes: " << num_unreachable_gt << endl;
            cout << "Total number of unreachable nodes: " << num_unreachable << endl;
            delete map;
        }

        void printNumElements()
        {
            vector<int> counters(maxlevel_+1);
            for (int i = 0; i < maxlevel_+1; i++)
                counters[i] = 0;

            for(int i = 0; i < maxelements_; i++)
                counters[elementLevels[i]]++;

            for (int i = 0; i < maxlevel_+1; i++)
                cout << "Number of elements on the " << i << " level: " << counters[i] << endl;
        }


        double computeError(dense_hash_set<labeltype> &v1, dense_hash_set<labeltype>&v2)
        {
            double error = 0.0;
            size_t n1 = v1.size(), n2 = v2.size();

            for (int i = 0; i < maxelements_; i++) {
                linklistsizeint *ll_cur = get_linklist0(i);
                size_t size = *ll_cur;
                tableint *data = (tableint *) (ll_cur + 1);

                int deg1 = 0, deg2 = 0;
                for (int j = 0; j < size; j++) {
                    deg1 += v1.count(*(data + j));
                    deg2 += v2.count(*(data + j));
                }
                int term1 = deg1 * ((long) (log2(n1 / (deg1 + 1))) + 1);
                int term2 = deg2 * ((long) (log2(n2 / (deg2 + 1))) + 1);
                error += (double) (term1 + term2) / maxelements_;
            }
            return error;
        }

        double computeMoveGain(tableint id, double error, dense_hash_set<labeltype> &v1, dense_hash_set<labeltype>&v2, bool isInFirst)
        {
            double gain = 0.0;
            size_t n1 = v1.size(), n2 = v2.size();

            for (int i = 0; i < maxelements_; i++) {
                linklistsizeint *ll_cur = get_linklist0(i);
                size_t size = *ll_cur;
                tableint *data = (tableint *) (ll_cur + 1);

                int deg1 = 0, deg2 = 0;
                for (int j = 0; j < size; j++) {
                    if (*(data + j) == id){
                        deg1 += !isInFirst;
                        deg2 += isInFirst;
                    }
                    deg1 += v1.count(*(data + j));
                    deg2 += v2.count(*(data + j));
                }
                int term1 = deg1 * ((long)(log2((n1 + !isInFirst - isInFirst) / (deg1+1))) + 1);
                int term2 = deg2 * ((long)(log2((n2 - !isInFirst + isInFirst)/ (deg2+1))) + 1);
                gain += (double) (term1 + term2) / maxelements_;
            }
            gain = error - gain;
            //cout << "Gain #" << id << ": " << gain << endl;
            return gain;
        }

        void recursive_reorder(labeltype *start, size_t n)
        {
            size_t n1 = n/2, n2 = n - n1;
            dense_hash_set<labeltype> v1(n1), v2(n2);
            v1.set_empty_key(NULL);
            v2.set_empty_key(NULL);

            //Init
            for (int i = 0; i < n1; i++)
                v1.insert(*(start + i));
            for (int i = n1; i < n; i++)
                v2.insert(*(start + i));

            double error = computeError(v1, v2);
            cout << "Current Error: " << error << endl;

            priority_queue<std::pair<double, labeltype>> s1, s2;

            cout << "Compute Move Gains S1" << endl;
            //#pragma omp parallel for num_threads(2)
            for (int i = 0; i < 20000; i++) {
                tableint id = *(start + i);
                double gain = computeMoveGain(id, error, v1, v2, true);
                auto element = std::pair<double, labeltype>(gain, id);
                s1.push(element) ;
            }
            cout << "Compute Move Gains S2" << endl;
            //#pragma omp parallel for num_threads(4)
            for (int i = n1; i < n1+20000; i++)
                s2.push(std::pair<double, labeltype>(computeMoveGain(*(start + i), error, v1, v2, false), *(start + i)));

            cout << "Swap good candidates" << endl;
            int num_swaps = 0;
            while (!s1.empty() && !s2.empty()){
                if ((s1.top().first + s2.top().first) > 0){
                    v1.set_deleted_key(NULL);
                    v1.erase(s1.top().second);
                    v1.clear_deleted_key();
                    v1.insert(s2.top().second);

                    v2.set_deleted_key(NULL);
                    v2.erase(s2.top().second);
                    v2.clear_deleted_key();
                    v2.insert(s1.top().second);

                    num_swaps++;
                }
                s1.pop();
                s2.pop();
            }
            cout << "Number of swaps: " << num_swaps << endl;
            cout << "Final Error: " << computeError(v1, v2) << endl;
        }

        void reorder_graph()
        {
            labeltype *labels = new labeltype[maxelements_];
            for (int i = 0; i < maxelements_; i++){
                labels[i] = i;
            }
            recursive_reorder(labels, maxelements_);
            delete labels;
        }
    };
}
