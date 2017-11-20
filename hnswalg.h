#pragma once

#include "hnswlib.h"
#include "visited_list_pool.h"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <array>
#include <map>
#include <cmath>
#include <queue>


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
            LoadData(dataLocation);
            LoadEdges(edgeLocation);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t maxelements, size_t M, size_t maxM, size_t efConstruction = 200)
        {
            space = s;
            data_size_ = s->get_data_size();

            efConstruction_ = efConstruction;
            maxelements_ = maxelements;
            M_ = M;
            maxM_ = maxM;
            size_links_level0 = maxM * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element = size_links_level0 + data_size_;
            offsetData = size_links_level0;

            std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;
            data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);
            std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;

            cout << "Size Mb: " << (maxelements_ * size_data_per_element) / (1000 * 1000) << "\n";
            cur_element_count = 0;

            visitedlistpool = new VisitedListPool(1, maxelements_);
            visitedsetpool = new VisitedSetPool(1);
            //initializations for special treatment of the first node
            enterpoint_node = -1;
            maxlevel_ = -1;

            elementLevels = vector<char>(maxelements_);
            for (size_t i = 0; i < maxelements_; ++i)
                elementLevels[i] = 0;
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
        int maxlevel_;

        VisitedListPool *visitedlistpool;
        VisitedSetPool *visitedsetpool;

        mutex cur_element_count_guard_;
        mutex MaxLevelGuard_;

        tableint enterpoint_node;

        size_t dist_calc;

        char *data_level0_memory_;

        vector<char> elementLevels;

        size_t data_size_;
        size_t offsetData;
        size_t size_data_per_element;
        size_t M_;
        size_t maxM_;
        size_t size_links_level0;

        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_level0_memory_ + internal_id * size_data_per_element + offsetData);
        }

        inline linklistsizeint *get_linklist0(tableint internal_id) const
        {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element);
        };

        std::priority_queue<std::pair<dist_t, tableint  >> searchBaseLayer(tableint ep, void *datapoint, int level, int ef)
        {
            VisitedList *vl = visitedlistpool->getFreeVisitedList();
            vl_type *massVisited = vl->mass;
            vl_type currentV = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint  >> topResults;
            std::priority_queue<std::pair<dist_t, tableint >> candidateSet;
            dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(ep));

            topResults.emplace(dist, ep);
            candidateSet.emplace(-dist, ep);
            massVisited[ep] = currentV;

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
                    if (!(massVisited[tnum] == currentV)) {
                        massVisited[tnum] = currentV;
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
        searchBaseLayerST(tableint ep, void *datapoint, size_t ef)
        {
            VisitedList *vl = visitedlistpool->getFreeVisitedList();
            vl_type *massVisited = vl->mass;
            vl_type currentV = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> topResults;
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t dist = space->fstdistfunc(datapoint, getDataByInternalId(ep));
            dist_calc++;

            topResults.emplace(dist, ep);
            candidateSet.emplace(-dist, ep);
            massVisited[ep] = currentV;
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
                    dist_t curdist = space->fstdistfunc(getDataByInternalId(curen2.second),
                                                        getDataByInternalId(curen.second));
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
            size_t curMmax = maxM_;
            size_t curM = M_;

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
                linklistsizeint *ll_cur = get_linklist0(cur_c);

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

                size_t rezMmax = maxM_;
                linklistsizeint *ll_other = get_linklist0(rez[idx]);

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

                    getNeighborsByHeuristicMerge(candidates, rezMmax);

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
            int curlevel = elementLevels[cur_c];

            unique_lock <mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();

            memset((char *) get_linklist0(cur_c), 0, size_data_per_element);
            memcpy(getDataByInternalId(cur_c), datapoint, data_size_);

            tableint currObj = enterpoint_node;
            if (currObj != -1) {
                if (curlevel < maxlevelcopy) {
                    dist_t curdist = space->fstdistfunc(datapoint, getDataByInternalId(currObj));
                    for (int level = maxlevelcopy; level > curlevel; level--) {

                        bool changed = true;
                        while (changed) {
                            changed = false;
                            linklistsizeint *data = get_linklist0(currObj);
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
            dist_t curdist = space->fstdistfunc(query_data, getDataByInternalId(enterpoint_node));

            dist_calc++;
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    linklistsizeint *data = get_linklist0(currObj);
                    linklistsizeint size = *data;
                    tableint *datal = (tableint *) (data + 1);
                    for (linklistsizeint i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > maxelements_)
                            throw runtime_error("cand error");

                        dist_t d = space->fstdistfunc(query_data, getDataByInternalId(cand));
                        dist_calc++;

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
            enterpoint0 = currObj;

            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> tmpTopResults = searchBaseLayerST(
                    currObj, query_data, ef_);
            //std::priority_queue<std::pair<dist_t, labeltype >> results;

            // Remove clusters as answers
            std::priority_queue<std::pair<dist_t, tableint >> topResults;
            while (tmpTopResults.size() > 0) {
                std::pair<dist_t, tableint> rez = tmpTopResults.top();
                topResults.push(rez);
                tmpTopResults.pop();
            }

            while (topResults.size() > k)
                topResults.pop();

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
            }

            std::cout << "Links distribution" << std::endl;
            for (int i = 0; i < 32; i++){
                cout << " Number of elements with " << i+1 << " links: " << numLinks[i] << endl;
            }
        }


        void SaveInfo(const string &location) {
            cout << "Saving info to " << location << endl;
            std::ofstream output(location, std::ios::binary);
            streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, enterpoint_node);
            writeBinaryPOD(output, data_size_);
            writeBinaryPOD(output, offsetData);
            writeBinaryPOD(output, size_data_per_element);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, maxM_);
            writeBinaryPOD(output, size_links_level0);

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

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, enterpoint_node);
            readBinaryPOD(input, data_size_);
            readBinaryPOD(input, offsetData);
            readBinaryPOD(input, size_data_per_element);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, size_links_level0);

            data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);

            efConstruction_ = 0;
            cur_element_count = maxelements_;

            visitedlistpool = new VisitedListPool(1, maxelements_);
            visitedsetpool = new VisitedSetPool(1);

            elementLevels = vector<char>(maxelements_);
            for (size_t i = 0; i < maxelements_; ++i)
                elementLevels[i] = 0;
            maxlevel_ = 0;

            cout << "Predicted size=" << maxelements_ * size_data_per_element / (1000 * 1000) << "\n";
            input.close();
        }

        void LoadData(const string &location)
        {
            cout << "Loading data from " << location << endl;
            FILE *fin = fopen(location.c_str(), "rb");
            int dim;
            const int D = space->get_data_dim();
            vtype mass[D];
            for (tableint i = 0; i < maxelements_; i++) {
                fread((int *) &dim, sizeof(int), 1, fin);
                if (dim != D)
                    cerr << "Wront data dim" << endl;

                fread(mass, sizeof(vtype), dim, fin);
                memset((char *) get_linklist0(i), 0, size_data_per_element);
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

        void getNeighborsByHeuristicMerge(std::priority_queue<std::pair<dist_t, tableint>> &topResults, const int NN) {
            if (topResults.size() < NN)
                return;

            std::priority_queue<std::pair<dist_t, tableint>> resultSet;
            std::priority_queue<std::pair<dist_t, tableint>> templist;
            std::vector<std::pair<dist_t, tableint>> returnlist;
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
                if (good)
                    returnlist.push_back(curen);
                else
                    templist.emplace(curen);
            }

            while (returnlist.size() < NN && templist.size() > 0) {
                returnlist.push_back(templist.top());
                templist.pop();
            }
            for (std::pair<dist_t, tableint> curen2 : returnlist)
                topResults.emplace(-curen2.first, curen2.second);
        }

        void merge(const HierarchicalNSW<dist_t, vtype> *hnsw)
        {
            int counter = 0;
//#pragma omp parallel for num_threads(32)

            for (int i = 0; i < maxelements_; i++){
                float *data = (float *) getDataByInternalId(i);

                linklistsizeint *ll1 = get_linklist0(i);
                linklistsizeint *ll2 = hnsw->get_linklist0(maxelements_- 1 - i);

                float identity = space->fstdistfunc((void *)data, (void *)hnsw->getDataByInternalId(maxelements_- 1 - i));
                if (identity > 0.0000001){
                    std::cout << "Merging different points\n";
                    exit(1);
                }

                size_t size1 = *ll1;
                size_t size2 = *ll2;
                labeltype *links1 = (labeltype *)(ll1 + 1);
                labeltype *links2 = (labeltype *)(ll2 + 1);
                std::unordered_set<labeltype> links;
                for (labeltype link = 0; link < size1; link++)
                    links.insert(links1[link]);
                for (labeltype link = 0; link < size2; link++)
                    links.insert(maxelements_- 1 - links2[link]);

                if (links.size() <= maxM_){
                    int indx = 0;
                    for (labeltype link : links)
                        links1[indx++] = link;
                    *ll1 = indx;
                } else {
                    std::priority_queue<std::pair<dist_t, tableint>> topResults;

                    for (labeltype link : links){
                        float *point = (float *) getDataByInternalId(link);
                        dist_t dist = space->fstdistfunc((void *)data, (void *)point);
                        topResults.emplace(std::make_pair(dist, link));
                    }

                    getNeighborsByHeuristicMerge(topResults, maxM_);

                    int indx = 0;
                    while (topResults.size() > 0) {
                        links1[indx++] = topResults.top().second;
                        topResults.pop();
                    }
                    *ll1 = indx;
                }

                if (*ll1 < maxM_)
                    counter++;
            }
            std::cout << counter << std::endl;
        }
    };
}
