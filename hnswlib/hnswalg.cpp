#include "hnswalg.h"

namespace hnswlib {

    HierarchicalNSW::HierarchicalNSW(const std::string &infoLocation,
                                     const std::string &dataLocation,
                                     const std::string &edgeLocation)
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
    efSearch = efConstruction;

    maxelements_ = maxelements;
    M_ = M;
    maxM_ = maxM;
    size_links_level0 = maxM * sizeof(idx_t) + sizeof(uint8_t);
    size_data_per_element = size_links_level0 + data_size_;
    offset_data = size_links_level0;

    std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;
    data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);
    std::cout << (data_level0_memory_ ? 1 : 0) << std::endl;

    std::cout << "Size Mb: " << (maxelements_ * size_data_per_element) / (1000 * 1000) << std::endl;

    visitedlistpool = new VisitedListPool(1, maxelements_);

    enterpoint_node = 0;
    cur_element_count = 0;
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
        size_t size = *ll_cur;
        idx_t *data = (idx_t *)(ll_cur + 1);

        _mm_prefetch((char *) (massVisited + *data), _MM_HINT_T0);
        _mm_prefetch((char *) (massVisited + *data + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

        for (size_t j = 0; j < size; ++j) {
            size_t tnum = *(data + j);

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


void HierarchicalNSW::getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN)
{
    if (topResults.size() < NN)
        return;

    std::priority_queue<std::pair<float, idx_t>> resultSet;
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
    for (std::pair<float, idx_t> elem : returnlist)
        topResults.emplace(-elem.first, elem.second);
}

void HierarchicalNSW::mutuallyConnectNewElement(const float *point, idx_t cur_c,
                               std::priority_queue<std::pair<float, idx_t>> topResults)
{
    getNeighborsByHeuristic(topResults, M_);

    std::vector<idx_t> res;
    res.reserve(M_);
    while (topResults.size() > 0) {
        res.push_back(topResults.top().second);
        topResults.pop();
    }
    {
        uint8_t *ll_cur = get_linklist0(cur_c);
        if (*ll_cur)
            throw std::runtime_error("Should be blank");

        *ll_cur = res.size();

        idx_t *data = (idx_t *)(ll_cur + 1);
        for (size_t idx = 0; idx < res.size(); idx++) {
            if (data[idx])
                throw std::runtime_error("Should be blank");
            data[idx] = res[idx];
        }
    }
    for (size_t idx = 0; idx < res.size(); idx++) {
        if (res[idx] == cur_c)
            throw std::runtime_error("Connection to the same element");

        size_t resMmax = maxM_;
        uint8_t *ll_other = get_linklist0(res[idx]);
        uint8_t sz_link_list_other = *ll_other;

        if (sz_link_list_other > resMmax || sz_link_list_other < 0)
            throw std::runtime_error("Bad sz_link_list_other");

        if (sz_link_list_other < resMmax) {
            idx_t *data = (idx_t *) (ll_other + 1);
            data[sz_link_list_other] = cur_c;
            *ll_other = sz_link_list_other + 1;
        } else {
            // finding the "weakest" element to replace it with the new one
            idx_t *data = (idx_t *) (ll_other + 1);
            float d_max = fstdistfunc(getDataByInternalId(cur_c), getDataByInternalId(res[idx]));
            // Heuristic:
            std::priority_queue<std::pair<float, idx_t>> candidates;
            candidates.emplace(d_max, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++)
                candidates.emplace(fstdistfunc(getDataByInternalId(data[j]), getDataByInternalId(res[idx])), data[j]);

            getNeighborsByHeuristic(candidates, resMmax);

            size_t indx = 0;
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
    if (cur_element_count >= maxelements_) {
        std::cout << "The number of elements exceeds the specified limit\n";
        throw std::runtime_error("The number of elements exceeds the specified limit");
    }
    idx_t cur_c = cur_element_count++;
    memset((char *) get_linklist0(cur_c), 0, size_data_per_element);
    memcpy(getDataByInternalId(cur_c), point, data_size_);

    // Do nothing for the first element
    if (cur_c != 0) {
        std::priority_queue <std::pair<float, idx_t>> topResults = searchBaseLayer(point, efConstruction_);
        mutuallyConnectNewElement(point, cur_c, topResults);
    }
};

std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::searchKnn(const float *query, size_t k)
{
    auto topResults = searchBaseLayer(query, efSearch);
    while (topResults.size() > k)
        topResults.pop();

    return topResults;
};

void HierarchicalNSW::SaveInfo(const std::string &location)
{
    std::cout << "Saving info to " << location << std::endl;
    std::ofstream output(location, std::ios::binary);

    writeBinaryPOD(output, maxelements_);
    writeBinaryPOD(output, enterpoint_node);
    writeBinaryPOD(output, data_size_);
    writeBinaryPOD(output, offset_data);
    writeBinaryPOD(output, size_data_per_element);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, maxM_);
    writeBinaryPOD(output, size_links_level0);
}


void HierarchicalNSW::SaveEdges(const std::string &location)
{
    std::cout << "Saving edges to " << location << std::endl;
    std::ofstream output(location, std::ios::binary);

    for (size_t i = 0; i < maxelements_; i++) {
        uint8_t *ll_cur = get_linklist0(i);
        uint32_t size = *ll_cur;

        output.write((char *) &size, sizeof(uint32_t));
        idx_t *data = (idx_t *)(ll_cur + 1);
        output.write((char *) data, sizeof(idx_t) * size);
    }
}

void HierarchicalNSW::LoadInfo(const std::string &location)
{
    std::cout << "Loading info from " << location << std::endl;
    std::ifstream input(location, std::ios::binary);

    readBinaryPOD(input, maxelements_);
    readBinaryPOD(input, enterpoint_node);
    readBinaryPOD(input, data_size_);
    readBinaryPOD(input, offset_data);
    readBinaryPOD(input, size_data_per_element);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, size_links_level0);

    d_ = data_size_ / sizeof(float);
    data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);

    efConstruction_ = 0;
    cur_element_count = maxelements_;

    visitedlistpool = new VisitedListPool(1, maxelements_);
}

void HierarchicalNSW::LoadData(const std::string &location)
{
    std::cout << "Loading data from " << location << std::endl;
    std::ifstream input(location, std::ios::binary);

    uint32_t dim;
    float mass[d_];
    for (size_t i = 0; i < maxelements_; i++) {
        input.read((char *) &dim, sizeof(uint32_t));
        if (dim != d_) {
            std::cout << "Wront data dim" << std::endl;
            exit(1);
        }
        input.read((char *) mass, dim * sizeof(float));
        memcpy(getDataByInternalId(i), mass, data_size_);
    }
}

void HierarchicalNSW::LoadEdges(const std::string &location)
{
    std::cout << "Loading edges from " << location << std::endl;
    std::ifstream input(location, std::ios::binary);

    uint32_t size;

    for (size_t i = 0; i < maxelements_; i++) {
        input.read((char *) &size, sizeof(uint32_t));

        uint8_t *ll_cur = get_linklist0(i);
        *ll_cur = size;
        idx_t *data = (idx_t *)(ll_cur + 1);

        input.read((char *) data, size * sizeof(idx_t));
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