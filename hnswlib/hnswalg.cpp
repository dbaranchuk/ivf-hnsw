#include "hnswalg.h"

namespace hnswlib {

    HierarchicalNSW::HierarchicalNSW(const string &infoLocation, const string &dataLocation, const string &edgeLocation)
    {
        LoadInfo(infoLocation);
        LoadData(dataLocation);
        LoadEdges(edgeLocation);
    }

    HierarchicalNSW::HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM, size_t efConstruction)
{
    d_ = d;
    data_size_ = d * sizeof(float);

    efConstruction_ = efConstruction;
    ef_ = efConstruction;

    maxelements_ = maxelements;
    M_ = M;
    maxM_ = maxM;
    size_links_level0 = maxM * sizeof(idx_t) + sizeof(uint8_t);
    size_data_per_element = size_links_level0 + data_size_;
    offsetData = size_links_level0;

    std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;
    data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);
    std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;

    std::cout << "Size Mb: " << (maxelements_ * size_data_per_element) / (1000 * 1000) << std::endl;
    cur_element_count = 0;

    visitedlistpool = new VisitedListPool(1, maxelements_);
    //initializations for special treatment of the first node
    enterpoint_node = -1;
    maxlevel_ = -1;
}

HierarchicalNSW::~HierarchicalNSW()
{
    free(data_level0_memory_);
    delete visitedlistpool;
}


std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::searchBaseLayer(const float *point, size_t ef)
{
    VisitedList *vl = visitedlistpool->getFreeVisitedList();
    vl_type *massVisited = vl->mass;
    vl_type currentV = vl->curV;
    std::priority_queue<std::pair<float, idx_t >> topResults;
    std::priority_queue<std::pair<float, idx_t >> candidateSet;

    float dist = fstdistfunc(point, getDataByInternalId(enterpoint_node));
    dist_calc++;

    topResults.emplace(dist, enterpoint_node);
    candidateSet.emplace(-dist, enterpoint_node);
    massVisited[enterpoint_node] = currentV;
    float lowerBound = dist;

    while (!candidateSet.empty())
    {
        std::pair<float, idx_t> curr_el_pair = candidateSet.top();
        if (-curr_el_pair.first > lowerBound)
            break;

        candidateSet.pop();
        idx_t curNodeNum = curr_el_pair.second;

        uint8_t *ll_cur = get_linklist0(curNodeNum);
        uint8_t size = *ll_cur;
        idx_t *data = (idx_t *)(ll_cur + 1);

        _mm_prefetch((char *) (massVisited + *data), _MM_HINT_T0);
        _mm_prefetch((char *) (massVisited + *data + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

        for (uint8_t j = 0; j < size; ++j) {
            int tnum = *(data + j);

            _mm_prefetch((char *) (massVisited + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);

            if (!(massVisited[tnum] == currentV)) {
                massVisited[tnum] = currentV;

                float dist = fstdistfunc(point, getDataByInternalId(tnum));
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

    std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::search(const float *point, size_t k, size_t ef)
    {
        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;
        std::priority_queue<std::pair<float, idx_t >> topResults;
        std::priority_queue<std::less, std::pair<float, idx_t >> candidateSet;

        float dist = fstdistfunc(point, getDataByInternalId(enterpoint_node));
        dist_calc++;

        topResults.emplace(dist, enterpoint_node);
        candidateSet.emplace(dist, enterpoint_node);
        massVisited[enterpoint_node] = currentV;
        float lowerBound = dist;

        while (!candidateSet.empty())
        {
            std::pair<float, idx_t> curr_el_pair = candidateSet.top();
            if (curr_el_pair.first > lowerBound)
                break;

            candidateSet.pop();
            idx_t curNodeNum = curr_el_pair.second;

            uint8_t *ll_cur = get_linklist0(curNodeNum);
            uint8_t size = *ll_cur;
            idx_t *data = (idx_t *)(ll_cur + 1);

            _mm_prefetch((char *) (massVisited + *data), _MM_HINT_T0);
            _mm_prefetch((char *) (massVisited + *data + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

            for (uint8_t j = 0; j < size; ++j) {
                int tnum = *(data + j);

                _mm_prefetch((char *) (massVisited + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);

                if (!(massVisited[tnum] == currentV)) {
                    massVisited[tnum] = currentV;

                    float dist = fstdistfunc(point, getDataByInternalId(tnum));
                    dist_calc++;

                    if (topResults.top().first > dist || topResults.size() < ef) {
                        candidateSet.emplace(dist, tnum);

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

void HierarchicalNSW::getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, const int NN)
{
    if (topResults.size() < NN)
        return;

    std::priority_queue<std::pair<float, idx_t>> resultSet;
    std::priority_queue<std::pair<float, idx_t>> templist;
    std::vector<std::pair<float, idx_t>> returnlist;
    while (topResults.size() > 0) {
        resultSet.emplace(-topResults.top().first, topResults.top().second);
        topResults.pop();
    }

    while (resultSet.size()) {
        if (returnlist.size() >= NN)
            break;
        std::pair<float, idx_t> curen = resultSet.top();
        float dist_to_query = -curen.first;
        resultSet.pop();
        bool good = true;
        for (std::pair<float, idx_t> curen2 : returnlist) {
            float curdist = fstdistfunc(getDataByInternalId(curen2.second),
                                         getDataByInternalId(curen.second));
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) returnlist.push_back(curen);
    }
    for (std::pair<float, idx_t> curen2 : returnlist)
        topResults.emplace(-curen2.first, curen2.second);
}

void HierarchicalNSW::mutuallyConnectNewElement(const float *point, idx_t cur_c,
                               std::priority_queue<std::pair<float, idx_t>> topResults)
{
    getNeighborsByHeuristic(topResults, M_);

    while (topResults.size() > M_) {
        throw exception();
        topResults.pop();
    }

    std::vector<idx_t> rez;
    rez.reserve(M_);
    while (topResults.size() > 0) {
        rez.push_back(topResults.top().second);
        topResults.pop();
    }
    {
        uint8_t *ll_cur = get_linklist0(cur_c);
        if (*ll_cur) {
            std::cout << *ll_cur << std::endl;
            throw runtime_error("Should be blank");
        }
        *ll_cur = rez.size();

        idx_t *data = (idx_t *)(ll_cur + 1);
        for (int idx = 0; idx < rez.size(); idx++) {
            if (data[idx])
                throw runtime_error("Should be blank");
            data[idx] = rez[idx];
        }
    }
    for (int idx = 0; idx < rez.size(); idx++) {
        if (rez[idx] == cur_c)
            throw runtime_error("Connection to the same element");

        size_t rezMmax = maxM_;
        uint8_t *ll_other = get_linklist0(rez[idx]);
        uint8_t sz_link_list_other = *ll_other;

        if (sz_link_list_other > rezMmax || sz_link_list_other < 0)
            throw runtime_error("Bad sz_link_list_other");

        if (sz_link_list_other < rezMmax) {
            idx_t *data = (idx_t *) (ll_other + 1);
            data[sz_link_list_other] = cur_c;
            *ll_other = sz_link_list_other + 1;
        } else {
            // finding the "weakest" element to replace it with the new one
            idx_t *data = (idx_t *) (ll_other + 1);
            float d_max = fstdistfunc(getDataByInternalId(cur_c), getDataByInternalId(rez[idx]));
            // Heuristic:
            std::priority_queue<std::pair<float, idx_t>> candidates;
            candidates.emplace(d_max, cur_c);

            for (int j = 0; j < sz_link_list_other; j++)
                candidates.emplace(fstdistfunc(getDataByInternalId(data[j]), getDataByInternalId(rez[idx])), data[j]);

            getNeighborsByHeuristic(candidates, rezMmax); // Merge

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

void HierarchicalNSW::addPoint(const float *point)
{
    idx_t cur_c = 0;
    {
        unique_lock <mutex> lock(cur_element_count_guard_);
        if (cur_element_count >= maxelements_) {
            cout << "The number of elements exceeds the specified limit\n";
            throw runtime_error("The number of elements exceeds the specified limit");
        };
        cur_c = cur_element_count;
        cur_element_count++;
    }
    int curlevel = 0;

    unique_lock <mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy)
        templock.unlock();

    memset((char *) get_linklist0(cur_c), 0, size_data_per_element);
    memcpy(getDataByInternalId(cur_c), point, data_size_);

    if (enterpoint_node != -1) {
        std::priority_queue<std::pair<float, idx_t>> topResults = searchBaseLayer(point, efConstruction_);
        mutuallyConnectNewElement(point, cur_c, topResults);
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

std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::searchKnn(const float *query, int k)
{
    auto topResults = search(query, k, ef_);
    while (topResults.size() > k)
        topResults.pop();

    return topResults;
};

void HierarchicalNSW::SaveInfo(const string &location)
{
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


void HierarchicalNSW::SaveEdges(const string &location)
{
    cout << "Saving edges to " << location << endl;
    FILE *fout = fopen(location.c_str(), "wb");

    for (idx_t i = 0; i < maxelements_; i++) {
        uint8_t *ll_cur = get_linklist0(i);
        int size = *ll_cur;

        fwrite((int *)&size, sizeof(int), 1, fout);
        idx_t *data = (idx_t *)(ll_cur + 1);
        fwrite(data, sizeof(idx_t), *ll_cur, fout);
    }
}

void HierarchicalNSW::LoadInfo(const string &location)
{
    cout << "Loading info from " << location << endl;
    std::ifstream input(location, std::ios::binary);
    streampos position;

    readBinaryPOD(input, maxelements_);
    readBinaryPOD(input, enterpoint_node);
    readBinaryPOD(input, data_size_);
    readBinaryPOD(input, offsetData);
    readBinaryPOD(input, size_data_per_element);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, size_links_level0);

    d_ = data_size_ / sizeof(float);
    data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);

    efConstruction_ = 0;
    cur_element_count = maxelements_;

    visitedlistpool = new VisitedListPool(1, maxelements_);
    maxlevel_ = 0;

    cout << "Predicted size=" << maxelements_ * size_data_per_element / (1000 * 1000) << "\n";
    input.close();
}
    
    /** TODO float -> vtype **/ 
void HierarchicalNSW::LoadData(const string &location)
{
    cout << "Loading data from " << location << endl;
    FILE *fin = fopen(location.c_str(), "rb");
    int dim;
    float mass[d_];
    for (idx_t i = 0; i < maxelements_; i++) {
        fread((int *) &dim, sizeof(int), 1, fin);
        if (dim != d_)
            cerr << "Wront data dim" << endl;

        fread(mass, sizeof(float), dim, fin);
        memset((char *) get_linklist0(i), 0, size_data_per_element);
        memcpy(getDataByInternalId(i), mass, data_size_);
    }
}

void HierarchicalNSW::LoadEdges(const string &location)
{
    cout << "Loading edges from " << location << endl;
    FILE *fin = fopen(location.c_str(), "rb");
    int size;

    for (idx_t i = 0; i < maxelements_; i++) {
        fread((int *)&size, sizeof(int), 1, fin);
        uint8_t *ll_cur = get_linklist0(i);
        *ll_cur = size;
        idx_t *data = (idx_t *)(ll_cur + 1);

        fread((idx_t *)data, sizeof(idx_t), size, fin);
    }
}

float HierarchicalNSW::fstdistfunc(const float *x, const float *y)
{
    float PORTABLE_ALIGN32 TmpRes[8];
#ifdef USE_AVX
    size_t qty16 = d_ >> 4;

            const float *pEnd1 = x + (qty16 << 4);

            __m256 diff, v1, v2;
            __m256 sum = _mm256_set1_ps(0);

            while (x < pEnd1) {
                v1 = _mm256_loadu_ps(x);
                x += 8;
                v2 = _mm256_loadu_ps(y);
                y += 8;
                diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

                v1 = _mm256_loadu_ps(x);
                x += 8;
                v2 = _mm256_loadu_ps(y);
                y += 8;
                diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
            }

            _mm256_store_ps(TmpRes, sum);
            float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

            return (res);
#else
    size_t qty16 = d_ >> 4;

    const float *pEnd1 = x + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (x < pEnd1) {
        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return (res);
#endif
}
}