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
void writeBinaryPOD(std::ostream &out, const T &podRef) {
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

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t>
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

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t maxElements, const std::map<size_t, size_t> &M_map, size_t efConstruction = 200,
                        size_t maxClusters = 0, size_t M_cluster = 0): elementLevels(maxElements)
        {
            maxelements_ = maxElements;

            space = s;
            data_size_ = s->get_data_size();

            efConstruction_ = efConstruction;

            total_size = 0;
            parts_num = M_map.size();
            params_num = 10;
            params = new size_t[parts_num * params_num];
            int i = 0;
            for (auto p : M_map){
                params[i*params_num + i_threshold] = p.first;
                params[i*params_num + i_maxelements] = i ? p.first - params[(i-1)*params_num + i_threshold] : p.first;
                params[i*params_num + i_M] = p.second;
                params[i*params_num + i_maxM] = p.second;
                params[i*params_num + i_maxM0] = 1 * p.second;
                params[i*params_num + i_size_links_level0] = params[i*params_num + i_maxM0]* sizeof(tableint) + sizeof(linklistsizeint);
                params[i*params_num + i_size_data_per_element] = params[i*params_num + i_size_links_level0] + data_size_;
                params[i*params_num + i_offsetData] = params[i*params_num + i_size_links_level0];
                params[i*params_num + i_size_links_per_element] = params[i*params_num + i_maxM] * sizeof(tableint) + sizeof(linklistsizeint);
                params[i*params_num + i_partOffset] = total_size;

                total_size += params[i*params_num + i_maxelements] * params[i*params_num + i_size_data_per_element];
                i++;
            }
            offsetLevel0_ = 0;

            std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;
            data_level0_memory_ = (char *) malloc(total_size);
            std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;

            cout << "Size Mb: " << total_size / (1000 * 1000) << "\n";
            cur_element_count = 0;

            visitedlistpool = new VisitedListPool(1, maxelements_);
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
            delete visitedlistpool;
            delete space;
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
        mutex cur_element_count_guard_;
        mutex MaxLevelGuard_;

        tableint enterpoint_node;

        size_t dist_calc;
        size_t offsetLevel0_;

        char *data_level0_memory_;
        char **linkLists_;

        vector<char> elementLevels;

        size_t *params;
        size_t parts_num;
        size_t params_num;

        size_t data_size_;
        size_t total_size;
        //size_t label_offset_;
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
            int i = 0;
            while (internal_id >= params[i*params_num + i_threshold]) ++i;
            return (params + i*params_num);
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

            return (linklistsizeint *) (data_level0_memory_ + param[i_partOffset] +
                                        ref_id * param[i_size_data_per_element] + offsetLevel0_);
        };

        inline linklistsizeint* get_linklist(tableint cur_c, int level)
        {
            size_t *param = getParametersByInternalId(cur_c);
            //In Smart hnsw only clusters on the above levels
            return (linklistsizeint *)(linkLists_[cur_c] + (level - 1) * param[i_size_links_per_element]);
        };


        std::priority_queue<std::pair<dist_t, tableint  >> searchBaseLayer(tableint ep, void *datapoint, int level)
        {
            VisitedList *vl = visitedlistpool->getFreeVisitedList();
            dense_hash_set<tableint> *setVisited = vl->getVisitedSet();

            std::priority_queue<std::pair<dist_t, tableint  >> topResults;
            std::priority_queue<std::pair<dist_t, tableint >> candidateSet;
            dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(ep));

            topResults.emplace(dist, ep);
            candidateSet.emplace(-dist, ep);
            setVisited->insert(ep);

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
                    if (setVisited->count(tnum) == 0){
                        setVisited->insert(tnum);
                        dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(tnum));
                        if (topResults.top().first > dist || topResults.size() < efConstruction_) {
                            candidateSet.emplace(-dist, tnum);
                            _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
                            topResults.emplace(dist, tnum);
                            if (topResults.size() > efConstruction_) {
                                topResults.pop();
                            }
                            lowerBound = topResults.top().first;
                        }
                    }
                }
            }
            visitedlistpool->releaseVisitedList(vl);

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
            VisitedList *vl = visitedlistpool->getFreeVisitedList();
            dense_hash_set<tableint> *setVisited = vl->getVisitedSet();
            //setVisited->set_empty_key(NULL);
            //vl_type *massVisited = vl->mass;
            //vl_type currentV = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> topResults;
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t dist;
            if (q_idx != -1)
                dist = space->fstdistfuncST(q_idx, getDataByInternalId(ep));
            else
                dist = space->fstdistfunc(datapoint, getDataByInternalId(ep));

            dist_calc++;
            topResults.emplace(dist, ep);
            candidateSet.emplace(-dist, ep);
            //massVisited[ep] = currentV;
            setVisited->insert(ep);
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

                    if (setVisited->count(tnum) == 0){
                        setVisited->insert(tnum);
                    //if (!(massVisited[tnum] == currentV)) {
                    //    massVisited[tnum] = currentV;

                        dist_t dist;
                        if (q_idx != -1)
                            dist = space->fstdistfuncST(q_idx, getDataByInternalId(tnum));
                        else
                            dist = space->fstdistfunc(datapoint, getDataByInternalId(tnum));

                        dist_calc++;
                        if (topResults.top().first > dist || topResults.size() < ef) {
                            candidateSet.emplace(-dist, tnum);

                            _mm_prefetch(get_linklist0(candidateSet.top().second), _MM_HINT_T0);\
                            topResults.emplace(dist, tnum);

                            if (topResults.size() > ef)
                                topResults.pop();

                            lowerBound = topResults.top().first;
                        }
                    }
                }
            }
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
                    if (curdist < dist_to_query) { // 75 gradus
                        good = false;
                        break;
                    }
                }
                if (good) returnlist.push_back(curen);
            }
            //if (returnlist.size() != NN)
            //    std::cout << returnlist.size() << std::endl;
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

        void setElementLevels(const vector<size_t> &elements_per_level)
        {
            if (elements_per_level.size() == 0) {
                std::uniform_real_distribution<double> distribution(0.0, 1.0);
                for (size_t i = 0; i < maxelements_; ++i)
                    elementLevels[i] = 0; // (int) (-log(distribution(generator)) * mult_);
            } else{
                for (size_t i = 0; i < maxelements_; ++i){
                    if (i < elements_per_level[6])
                        elementLevels[i] = 0;
                    else if (i < elements_per_level[6] + elements_per_level[5])
                        elementLevels[i] = 0;
                    else if (i < elements_per_level[6] + elements_per_level[5] + elements_per_level[4])
                        elementLevels[i] = 0;
                    else if (i < elements_per_level[6] + elements_per_level[5] + elements_per_level[4] +
                                 elements_per_level[3])
                        elementLevels[i] = 0;
                    else if (i < elements_per_level[6] + elements_per_level[5] + elements_per_level[4] +
                                 elements_per_level[3] + elements_per_level[2])
                        elementLevels[i] = 0;
                    else if (i < elements_per_level[6] + elements_per_level[5] + elements_per_level[4] +
                                 elements_per_level[3] + elements_per_level[2] + elements_per_level[1])
                        elementLevels[i] = 0;
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
            tableint currObj = enterpoint_node;

            auto curParam = getParametersByInternalId(cur_c);
            memset((char *) get_linklist0(cur_c), 0, curParam[i_size_data_per_element]);

            // Initialisation of the data and label
            //memcpy(getExternalLabelPointer(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), datapoint, data_size_);

            if (curlevel) {
                // Above levels contain only clusters
                auto param = getParametersByInternalId(cur_c);

                linkLists_[cur_c] = (char *) malloc(param[i_size_links_per_element] * curlevel);
                memset(linkLists_[cur_c], 0, param[i_size_links_per_element] * curlevel);
            }

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
                                                                                                    level);
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

        std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(void *query_data, int k, std::unordered_set<int> &cluster_idx_set, int q_idx = -1)
        {
            tableint currObj = enterpoint_node;
            dist_t curdist;

            if (q_idx != -1)
                curdist = space->fstdistfuncST(q_idx, getDataByInternalId(enterpoint_node));
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
                            d = space->fstdistfuncST(q_idx, getDataByInternalId(cand));
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
                //if (getExternalLabel(rez.second) >= maxclusters_)
                if (cluster_idx_set.count(rez.second) == 0)
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
            for (int i = 0; i < maxelements_; i++){
                if (i % 1000 != 0)
                    continue;

                linklistsizeint *ll_cur = get_linklist0(i);
                if (*ll_cur != 32)
                cout << "Element #" << i << " M:" << (int) *ll_cur << endl;
            }
        }


        void SaveInfo(const string &location)
        {
            cout << "Saving info to " << location << endl;
            std::ofstream output(location, std::ios::binary);
            streampos position;

            cout << enterpoint_node << endl;
            cout << maxlevel_ << endl;
            cout << total_size << endl;
            cout << cur_element_count << endl;
            cout << maxelements_ << endl;

            writeBinaryPOD(output, parts_num);
            writeBinaryPOD(output, params_num);
            output.write((char *)params, parts_num * params_num *sizeof(size_t));
            output.close();
        }


        void SaveEdges(const string &location)
        {
            cout << "Saving index to " << location << endl;
            FILE *fout = fopen(location.c_str(), "wb");

            for (tableint i = 0; i < maxelements_; i++) {
                linklistsizeint *ll_cur = get_linklist0(i);
                size_t size = *ll_cur;

                fwrite(&size, sizeof(size_t), 1, fout);
                tableint *data = (tableint *)(ll_cur + 1);
                fwrite(data, sizeof(tableint), *ll_cur, fout);
            }
        }

        void LoadInfo(const string &location, SpaceInterface<dist_t> *s, const int ep = 0)
        {
            cout << "Loading info from " << location << endl;
            std::ifstream input(location, std::ios::binary);
            streampos position;

            space = s;
            data_size_ = s->get_data_size();

            readBinaryPOD(input, parts_num);
            readBinaryPOD(input, params_num);
            params = new size_t[params_num*parts_num];
            input.read((char *) params, parts_num*params_num*sizeof(size_t));
            input.close();

            efConstruction_ = 240;
            enterpoint_node = ep;
            total_size = 0;
            maxelements_ = 0;
            for (size_t i = 0; i < parts_num; i++) {
                maxelements_ += params[i*params_num + i_maxelements];
                total_size += params[i*params_num + i_maxelements] * params[i*params_num + i_size_data_per_element];
            }
            cur_element_count = maxelements_;

            visitedlistpool = new VisitedListPool(1, maxelements_);

            //linkLists_ = (char **) malloc(sizeof(void *) * (params[0*params_num + i_threshold]));

            elementLevels = vector<char>(maxelements_);
            maxlevel_ = 0;
            for (int i = 0; i < maxelements_; i++) {
                elementLevels[i] = 0;
                if (elementLevels[i] > maxlevel_){
                    maxlevel_ = elementLevels[i];
                }
            }

            cout << "Predicted size=" << total_size / (1000 * 1000) << "\n";
        }

        void LoadData(const string &location)
        {
            cout << "Loading data from " << location << endl;
            FILE *fin = fopen(location.c_str(), "rb");
            int dim;
            const int D = space->get_data_dim();
            char *massb[D];

            for (tableint i = 0; i < maxelements_; i++) {
                fread((int *) &dim, sizeof(int), 1, fin);
                if (dim != D)
                    cerr << "Wront data dim" << endl;

                fread((char *)massb, sizeof(char), dim, fin);
                memcpy(getDataByInternalId(i), massb, data_size_);
            }
        }

        void LoadEdges(const string &location)
        {
            cout << "Loading edges from " << location << endl;
            FILE *fin = fopen(location.c_str(), "rb");
            size_t size;

            for (tableint i = 0; i < maxelements_; i++) {
                fread((size_t *)&size, sizeof(size_t), 1, fin);
                linklistsizeint *ll_cur = get_linklist0(i);
                *ll_cur = size;
                tableint *data = (tableint *)(ll_cur + 1);

                fread((tableint *)data, sizeof(tableint), size, fin);
            }
        }


//        void PrevLoadIndex(const string &location, SpaceInterface<dist_t> *s)
//        {
//            cout << "Loading index from " << location << endl;
//            std::ifstream input(location, std::ios::binary);
//            streampos position;
//
//            readBinaryPOD(input, offsetLevel0_);
//            readBinaryPOD(input, maxelements_);
//            readBinaryPOD(input, cur_element_count);
//            readBinaryPOD(input, size_data_per_element_);
//            readBinaryPOD(input, label_offset_);
//            readBinaryPOD(input, offsetData_);
//            readBinaryPOD(input, maxlevel_);
//            readBinaryPOD(input, enterpoint_node);
//
//            readBinaryPOD(input, maxM_);
//            readBinaryPOD(input, maxM0_);
//            readBinaryPOD(input, M_);
//            readBinaryPOD(input, mult_);
//            readBinaryPOD(input, efConstruction_);
//
//            // Clusters
//            readBinaryPOD(input, maxclusters_);
//            readBinaryPOD(input, size_data_per_cluster_);
//            readBinaryPOD(input, label_offset_cluster_);
//            readBinaryPOD(input, offsetData_cluster_);
//            readBinaryPOD(input, maxM_cluster_);
//            readBinaryPOD(input, maxM0_cluster_);
//            readBinaryPOD(input, M_cluster_);
//
//
//            space = s;
//            data_size_ = s->get_data_size();
//
//            data_level0_memory_ = (char *) malloc(maxclusters_ * size_data_per_cluster_ +
//                                                  maxelements_ * size_data_per_element_);
//            input.read(data_level0_memory_, maxclusters_ * size_data_per_cluster_  +
//                                            maxelements_ * size_data_per_element_);
//
//
//            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
//            size_links_per_cluster_ = maxM_cluster_ * sizeof(tableint) + sizeof(linklistsizeint);
//
//            visitedlistpool = new VisitedListPool(1, maxclusters_ + maxelements_);
//            linkLists_ = (char **) malloc(sizeof(void *) * (maxclusters_ /*+ maxelements*/));
//
//            elementLevels = vector<char>(maxclusters_ + maxelements_);
//            ef_ = 10;
//
//            for (size_t i = 0; i < maxclusters_; i++) {
//                unsigned int linkListSize;
//                readBinaryPOD(input, linkListSize);
//                if (linkListSize == 0) {
//                    elementLevels[i] = 0;
//                    //linkLists_[i] = nullptr;
//                } else {
//                    elementLevels[i] = linkListSize / size_links_per_cluster_;
//                    linkLists_[i] = (char *) malloc(linkListSize);
//                    input.read((char *)linkLists_[i], linkListSize);
//                }
//            }
//            for (size_t i = maxclusters_; i < maxelements_; i++) {
//                unsigned int linkListSize;
//                readBinaryPOD(input, linkListSize);
//                if (linkListSize == 0) {
//                    elementLevels[i] = 0;
//                    //linkLists_[i] = nullptr;
//                } else {
//                    elementLevels[i] = linkListSize / size_links_per_element_;
//                    linkLists_[i] = (char *) malloc(linkListSize);
//                    input.read(linkLists_[i], linkListSize);
//                }
//            }
//
//            input.close();
//            size_t total_size = maxclusters_ * size_data_per_cluster + maxelements_ * size_data_per_element;
//            cout << "Loaded index, predicted size=" << total_size / (1000 * 1000) << "\n";
//            return;
//        }
//        void SaveIndex(const string &location)
//        {
//            cout << "Saving index to " << location.c_str() << "\n";
//            std::ofstream output(location, std::ios::binary);
//            streampos position;
//
//            writeBinaryPOD(output, offsetLevel0_);
//            writeBinaryPOD(output, maxelements_);
//            writeBinaryPOD(output, cur_element_count);
//            writeBinaryPOD(output, maxlevel_);
//            writeBinaryPOD(output, enterpoint_node);
//
//            writeBinaryPOD(output, mult_);
//            writeBinaryPOD(output, efConstruction_);
//
//            writeBinaryPOD(output, total_size);
//            writeBinaryPOD(output, parts_num);
//            writeBinaryPOD(output, params_num);
//            output.write((char *)params, parts_num * params_num *sizeof(size_t));
//
////            for (size_t i = 0; i < params_size; i++) {
////                writeBinaryPOD(output, params[i] + i_threshold);
////                writeBinaryPOD(output, params[i] + i_M);
////                writeBinaryPOD(output, params[i] + i_maxM);
////                writeBinaryPOD(output, params[i] + i_maxM0);
////            }
//
//            output.write(data_level0_memory_, total_size);
//
////            for (size_t i = 0; i < params[0*params_num + i_threshold]; ++i) {
////                unsigned int linkListSize = elementLevels[i] > 0 ? params[0*params_num + i_size_links_per_element] * elementLevels[i] : 0;
////                writeBinaryPOD(output, linkListSize);
////                if (linkListSize)
////                    output.write((char *)linkLists_[i], linkListSize);
////            }
//            output.close();
//        }
//
//        void LoadIndex(const string &location, SpaceInterface<dist_t> *s)
//        {
//            cout << "Loading index from " << location << endl;
//            std::ifstream input(location, std::ios::binary);
//            streampos position;
//
//            readBinaryPOD(input, offsetLevel0_);
//            readBinaryPOD(input, maxelements_);
//            readBinaryPOD(input, cur_element_count);
//            readBinaryPOD(input, maxlevel_);
//            readBinaryPOD(input, enterpoint_node);
//
//            readBinaryPOD(input, mult_);
//            readBinaryPOD(input, efConstruction_);
//
//            space = s;
//            data_size_ = s->get_data_size();
//
//            // Params
//            readBinaryPOD(input, total_size);
//            readBinaryPOD(input, parts_num);
//            readBinaryPOD(input, params_num);
//            params = new size_t[params_num*parts_num];
//            input.read((char *) params, parts_num*params_num*sizeof(size_t));
//
//            size_t threshold, M, maxM, maxM0;
//            total_size = 0;
//
////            for (size_t i = 0; i < params_size; i++) {
////                readBinaryPOD(input, threshold);
////                readBinaryPOD(input, M);
////                readBinaryPOD(input, maxM);
////                readBinaryPOD(input, maxM0);
////
////                params[i][i_M] = M;
////                params[i][i_threshold] = threshold;
////                params[i][i_maxelements] = i ? params[i-1][i_threshold] - threshold : threshold;
////                params[i][i_maxM] = M;
////                params[i][i_maxM0] = 2 * M;
////                params[i][i_size_links_level0] = params[i][i_maxM0]* sizeof(tableint) + sizeof(linklistsizeint);
////                params[i][i_size_data_per_element] = params[i][i_size_links_level0] + data_size_;
////                params[i][i_offsetData] = params[i][i_size_links_level0];
////                params[i][i_size_links_per_element] = params[i][i_maxM] * sizeof(tableint) + sizeof(linklistsizeint);
////
////                total_size += params[i][i_maxelements] * params[i][i_size_data_per_element];
////            }
//
//            data_level0_memory_ = (char *) malloc(total_size);
//            input.read(data_level0_memory_, total_size);
//
//            visitedlistpool = new VisitedListPool(1, maxelements_);
////            linkLists_ = (char **) malloc(sizeof(void *) * (params[0*params_num + i_threshold]));
//
//            elementLevels = vector<char>(maxelements_);
//            for (int i = 0; i < maxelements_; i++)
//                elementLevels[i] = 0;
//
////            for (size_t i = 0; i < params[0*params_num + i_threshold]; i++) {
////                unsigned int linkListSize;
////                readBinaryPOD(input, linkListSize);
////                if (linkListSize == 0) {
////                    elementLevels[i] = 0;
////                    linkLists_[i] = nullptr;
////                } else {
////                    elementLevels[i] = linkListSize / params[0*params_num + i_size_links_per_element];
////                    linkLists_[i] = (char *) malloc(linkListSize);
////                    input.read((char *)linkLists_[i], linkListSize);
////                }
////            }
//
//            input.close();
//            cout << "Loaded index, predicted size=" << total_size / (1000 * 1000) << "\n";
//            return;
//        }
    };
}
