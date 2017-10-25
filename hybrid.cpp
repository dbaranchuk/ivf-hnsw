#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <faiss/ProductQuantizer.h>
#include <faiss/index_io.h>
#include "hnswIndexPQ.h"
#include "hnswlib.h"

#include <cmath>
#include <map>
#include <set>
#include <unordered_set>
using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }
    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};


/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/


#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif



/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
size_t getPeakRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
	PROCESS_MEMORY_COUNTERS info;
	GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
	return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
	struct psinfo psinfo;
	int fd = -1;
	if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
		return (size_t)0L;      /* Can't open? */
	if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
	{
		close(fd);
		return (size_t)0L;      /* Can't read? */
	}
	close(fd);
	return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
	return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
size_t getCurrentRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
	PROCESS_MEMORY_COUNTERS info;
	GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
	return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
	struct mach_task_basic_info info;
	mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
		(task_info_t)&info, &infoCount) != KERN_SUCCESS)
		return (size_t)0L;      /* Can't access? */
	return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t)0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1)
    {
        fclose(fp);
        return (size_t)0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
	return (size_t)0L;          /* Unsupported. */
#endif
}


template <typename dist_t>
static void get_gt(unsigned int *massQA, size_t qsize, vector<std::priority_queue< std::pair<dist_t, labeltype >>> &answers,
                   size_t gt_dim, size_t k = 1)
{
    (vector<std::priority_queue< std::pair<dist_t, labeltype >>>(qsize)).swap(answers);
    std::cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[gt_dim*i + j]);
        }
    }
}

template <typename format>
static void loadXvecs(const char *path, format *mass, const int n, const int d)
{
    ifstream input(path, ios::binary);
    for (int i = 0; i < n; i++) {
        int in = 0;
        input.read((char *)&in, sizeof(int));
        if (in != d) {
            cout << "file error\n";
            exit(1);
        }
        input.read((char *)(mass + i*d), in*sizeof(format));
    }
    input.close();
}

static void check_precomputing(Index *index, const char *path_data, const char *path_precomputed_idxs,
                               size_t vecdim, size_t ncentroids, size_t vecsize,
                               std::set<idx_t> gt_mistakes, std::set<idx_t> gt_correct)
{
    size_t batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * vecdim);
    std::vector<idx_t> idx_batch(batch_size);

//    int counter = 0;
    std::vector<float> mistake_dst;
    std::vector<float> correct_dst;
    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        readXvec<float>(base_input, batch.data(), vecdim, batch_size);

        printf("%.1f %c \n", (100.*b)/(vecsize/batch_size), '%');

        for (int i = 0; i < batch_size; i++) {
            int elem = batch_size*b + i;
            //float min_dist = 1000000;
            //int min_centroid = 100000000;

            if (gt_mistakes.count(elem) == 0 &&
                gt_correct.count(elem) == 0)
                continue;

            float *data = batch.data() + i*vecdim;
            for (int j = 0; j < ncentroids; j++) {
                float *centroid = (float *) index->quantizer->getDataByInternalId(j);
                float dist = faiss::fvec_L2sqr(data, centroid, vecdim);
                //if (dist < min_dist){
                //    min_dist = dist;
                //    min_centroid = j;
                //}
                if (gt_mistakes.count(elem) != 0)
                    mistake_dst.push_back(dist);
                if (gt_correct.count(elem) != 0)
                    correct_dst.push_back(dist);
            }
//            if (min_centroid != idx_batch[i]){
//                std::cout << "Element: " << elem << " True centroid: " << min_centroid << " Precomputed centroid:" << idx_batch[i] << std::endl;
//                counter++;
//            }
        }
    }

    std::cout << "Correct distance distribution\n";
    for (int i = 0; i < correct_dst.size(); i++)
        std::cout << correct_dst[i] << std::endl;

    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Mistake distance distribution\n";
    for (int i = 0; i < mistake_dst.size(); i++)
        std::cout << mistake_dst[i] << std::endl;

    idx_input.close();
    base_input.close();
}

void compute_vector(float *vector, const float *p1, const float *p2, const int d)
{
    for (int i = 0; i < d; i++)
        vector[i] = p1[i] - p2[i];
}

void normalize_vector(float *vector, const int d)
{
    float norm = sqrt(faiss::fvec_norm_L2sqr(vector, d));
    for (int i = 0; i < d; i++)
        vector[i] /= norm;
}

double compute_quantization_error(const float *reconstructed_x, const float *x,
                                 const int vecdim, const int n)
{
    double error = 0.0;
    for (int i = 0; i < n; i++)
        error += faiss::fvec_L2sqr(reconstructed_x + i*vecdim, x + i * vecdim, vecdim);
    return error / n;
}


void collect_groups(const char *path_groups, const char *path_data, const char *path_precomputed_idxs,
                 std::unordered_set<idx_t> centroid_nums, const int vecdim, const int vecsize)
{
    std::unordered_map<idx_t, std::vector<float>> data;
    const int ncentroids = centroid_nums.size();

    const int batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * vecdim);
    std::vector<idx_t> idx_batch(batch_size);

    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        readXvec<float>(base_input, batch.data(), vecdim, batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            if (centroid_nums.count(idx_batch[i]) == 0)
                continue;

            for (int d = 0; d < vecdim; d++)
                data[idx_batch[i]].push_back(batch[i * vecdim + d]);
        }
        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();


    FILE *fout = fopen(path_groups, "wb");
    for (idx_t centroid_num : centroid_nums) {
        fwrite(&centroid_num, sizeof(int), 1, fout);
        auto group_data = data[centroid_num];
        int groupsize = group_data.size() / vecdim;
        fwrite(&groupsize, sizeof(int), 1, fout);
        for (int i = 0; i < groupsize; i++) {
            fwrite(&vecdim, sizeof(int), 1, fout);
            fwrite(group_data.data() + i * vecdim, sizeof(float), vecdim, fout);
        }
    }
    fclose(fout);
}

void save_groups(Index *index, const char *path_groups, const char *path_data,
                 const char *path_precomputed_idxs, const int vecdim, const int vecsize)
{
    const int ncentroids = 999973;
    std::vector<std::vector<float>> data(ncentroids);

    const int batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * vecdim);
    std::vector<idx_t> idx_batch(batch_size);

    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        readXvec<float>(base_input, batch.data(), vecdim, batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            if (idx_batch[i] < 900000)
                continue;

            idx_t cur_idx = idx_batch[i];
            for (int d = 0; d < vecdim; d++)
                data[cur_idx].push_back(batch[i * vecdim + d]);
        }

        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();

    FILE *fout = fopen(path_groups, "wb");
    for (int i = 900000; i < ncentroids; i++) {
        int groupsize = data[i].size() / vecdim;

        if (groupsize != index->ids[i].size()){
            std::cout << "Wrong groupsize: " << groupsize << " vs "
                      << index->ids[i].size() <<std::endl;
            exit(1);
        }

        fwrite(&groupsize, sizeof(int), 1, fout);
        fwrite(data[i].data(), sizeof(float), data[i].size(), fout);
    }
}

void compute_subcentroids(float *subcentroids, const float *centroid,
                          const float *centroid_vectors,
                          const float alpha, const int vecdim,
                          const int ncentroids, const int groupsize)
{
    for (int c = 0; c < ncentroids; c++) {
        const float *centroid_vector = centroid_vectors + c * vecdim;
        float *subcentroid = subcentroids + c * vecdim;

        float check_norm = faiss::fvec_norm_L2sqr(centroid_vector, vecdim);
        if (c != 0 && !(0.99999 <  check_norm < 1.00001)){
            std::cout << "Centroid " << c << " has wrong norm: " << check_norm << std::endl;
            exit(1);
        }
        for (int i = 0; i < vecdim; i++)
            subcentroid[i] = centroid_vector[i] * alpha + centroid[i];
    }
}

float compute_alpha(const float *centroid_vectors, const float *point_vectors,
                    const float *centroid,
                    const int vecdim, const int ncentroids, const int groupsize)
{
    int counter_positive = 0;
    int counter_negative = 0;
    float positive_alpha = 0.0;
    float negative_alpha = 0.0;

    for (int i = 0; i < groupsize; i++) {
        const float *point_vector = point_vectors + i*vecdim;
        std::priority_queue<std::pair<float, float>> max_heap;

        for (int c = 0; c < ncentroids; c++){
            const float *centroid_vector = centroid_vectors + c*vecdim;
            float alpha = faiss::fvec_inner_product (centroid_vector, point_vector, vecdim);

            std::vector<float> subcentroid(vecdim);
            for (int d = 0; d < vecdim; d++)
                subcentroid[d] = centroid_vector[d] * alpha + centroid[d];

            float dist = faiss::fvec_L2sqr(point_vector, subcentroid.data(), vecdim);
            max_heap.emplace(std::make_pair(-dist, alpha));
        }
        float optim_alpha = max_heap.top().second;
        if (optim_alpha < 0) {
            counter_negative++;
            negative_alpha += optim_alpha;
        } else {
            counter_positive++;
            positive_alpha += optim_alpha;
        }
    }
    positive_alpha /= counter_positive;
    negative_alpha /= counter_negative;

//    if (counter_positive == 0)
//        std::cout << "Positive Alpha: every alpha is negative" << std::endl;
//    else
//        std::cout << "Positive Alpha: " << positive_alpha << std::endl;
//
//    if (counter_negative == 0)
//        std::cout << "Negative Alpha: every alphas is positive" << std::endl;
//    else
//        std::cout << "Negative Alpha: " << negative_alpha << std::endl;

    return (counter_positive > counter_negative) ? positive_alpha : negative_alpha;
}

float compute_idxs(std::vector<std::vector<idx_t>> &idxs,
                  const float *points, const float *subcentroids,
                  const int vecdim, const int ncentroids, const int groupsize)
{
    float av_dist = 0.0;
    for (int i = 0; i < groupsize; i++) {
        const float *point = points + i * vecdim;
        std::priority_queue<std::pair<float, idx_t>> max_heap;

        for (int c = 0; c < ncentroids; c++) {
            const float *subcentroid = subcentroids + c * vecdim;
            float dist = faiss::fvec_L2sqr(subcentroid, point, vecdim);
            max_heap.emplace(std::make_pair(-dist, c));
        }
        idx_t idx = max_heap.top().second;
        idxs[idx].push_back(i);
        av_dist += -max_heap.top().first;
    }
    return av_dist / groupsize;
    //std::cout << "[Modified] Average Distance: " << av_dist / groupsize << std::endl;
}


void check_idea(Index *index, const char *path_centroids,
                const char *path_precomputed_idxs, const char *path_data,
                const int vecsize, const int vecdim)
{
    StopW stopw = StopW();
    const bool include_zero_centroid = false;
    const int nc = 128;
    const int maxM = 128;
    const char *path_groups = "/home/dbaranchuk/data/groups/groups999973.dat";

    if (!exists_test(path_groups)) {
        std::cout << "Precompute Group file first\n";

        std::unordered_set<idx_t> centroid_nums;
        for (idx_t i = 50000; i < 150000; i += 10)
            centroid_nums.insert(i);

        collect_groups(path_groups, path_data, path_precomputed_idxs, centroid_nums, vecdim, vecsize);
    }

    std::ifstream input(path_groups, ios::binary);

    double baseline_average = 0.0;
    double modified_average = 0.0;

    double baseline_error = 0.0;
    double modified_error = 0.0;

    const int ngroups = 999973;

    int j1 = 0;
//#pragma omp parallel for num_threads(24)
    for (int g = 0; g < ngroups; g++) {
        /** Read Original vectors from Group file**/
        idx_t centroid_num;
        int groupsize;
        std::vector<float> data;
//#pragma omp critical
        {
            //input.read((char *) &centroid_num, sizeof(idx_t));
            input.read((char *) &groupsize, sizeof(int));
            //std::cout << centroid_num << " " << groupsize << std::endl;
            data.reserve(groupsize * vecdim);
            input.read((char *) data.data(), groupsize * vecdim * sizeof(float));
            //readXvecs<float>(input, data.data(), vecdim, groupsize);
            centroid_num = j1++;

            if (groupsize != index->ids[centroid_num].size()){
                std::cout << "Wrong groupsize\n";
                exit(1);
            }
        }

        if (groupsize == 0)
            continue;

        /** Find NN centroids to source centroid **/
        const float *centroid = (float *) index->quantizer->getDataByInternalId(centroid_num);
        int nc = groupsize;
//        auto nn_centroids_raw = index->quantizer->searchKnn((void *) centroid, nc + 1);
//
//        /** Remove source centroid from consideration **/
//        std::priority_queue<std::pair<float, idx_t>> nn_centroids_before_heuristic;
//        while (nn_centroids_raw.size() > 1) {
//            nn_centroids_before_heuristic.emplace(nn_centroids_raw.top());
//            nn_centroids_raw.pop();
//        }

        /** Pruning **/
        //index->quantizer->getNeighborsByHeuristicMerge(nn_centroids_before_heuristic, maxM);
        //size_t ncentroids = nn_centroids_before_heuristic.size() + include_zero_centroid;
        //std::cout << "Number of centroids after pruning: " << ncentroids << std::endl;

//        if (ncentroids > nc + include_zero_centroid) {
//            std::cout << "Wrong number of nn centroids\n";
//            exit(1);
//        }

//        std::vector<idx_t> nn_centroids(ncentroids);
        //std::vector<std::pair<float, idx_t>> nn_centroids(ncentroids);

//        if (include_zero_centroid)
//            nn_centroids[0] = centroid_num;
//
//        while (nn_centroids_before_heuristic.size() > 0) {
//            nn_centroids[nn_centroids_before_heuristic.size() -
//                         !include_zero_centroid] = nn_centroids_before_heuristic.top().second;
//            nn_centroids_before_heuristic.pop();
//        }

        /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
        //std::vector<float> normalized_centroid_vectors(ncentroids * vecdim);
        std::vector<float> point_vectors(groupsize * vecdim);

//        for (int i = 0; i < ncentroids; i++) {
//            float *neighbor_centroid = (float *) index->quantizer->getDataByInternalId(nn_centroids[i]);
//            compute_vector(normalized_centroid_vectors.data() + i * vecdim, neighbor_centroid, centroid, vecdim);
//
//            /** Normalize them **/
//            if (include_zero_centroid && (i == 0)) continue;
//            normalize_vector(normalized_centroid_vectors.data() + i * vecdim, vecdim);
//        }

        double av_dist = 0.0;
        for (int i = 0; i < groupsize; i++) {
            compute_vector(point_vectors.data() + i * vecdim, data.data() + i * vecdim, centroid, vecdim);
            av_dist += faiss::fvec_norm_L2sqr(point_vectors.data() + i * vecdim, vecdim);
        }
        //std::cout << "[Baseline] Average Distance: " << av_dist / groupsize << std::endl;
        baseline_average += av_dist / groupsize;

//        /** Find alphas for vectors **/
//        float alpha = compute_alpha(normalized_centroid_vectors.data(), point_vectors.data(),
//                                    centroid, vecdim, ncentroids, groupsize);
//
//        /** Compute final subcentroids **/
//        std::vector<float> subcentroids(ncentroids * vecdim);
//        compute_subcentroids(subcentroids.data(), centroid, normalized_centroid_vectors.data(),
//                             alpha, vecdim, ncentroids, groupsize);
//
//        /** Compute sub idxs for group points **/
//        std::vector<std::vector<idx_t>> idxs(ncentroids);
//        modified_average += compute_idxs(idxs, data.data(), subcentroids.data(),
//                                         vecdim, ncentroids, groupsize);

//        /** Baseline Quantization Error **/
//        {
//            std::vector<idx_t> keys(groupsize);
//            for (int i = 0; i < groupsize; i++)
//                keys[i] = centroid_num;
//
//            std::vector<uint8_t> codes = index->codes[centroid_num];
//
//            std::vector<float> decoded_residuals(groupsize * vecdim);
//            index->pq->decode(codes.data(), decoded_residuals.data(), groupsize);
//
//            std::vector<float> reconstructed_x(groupsize * vecdim);
//            index->reconstruct(groupsize, reconstructed_x.data(), decoded_residuals.data(), keys.data());
//
//            double error = compute_quantization_error(reconstructed_x.data(), data.data(), vecdim, groupsize);
//            //std::cout << "[Baseline] Quantization Error: " << error << std::endl;
//            baseline_error += error;
//        }
//        /** Modified Quantization Error **/
//        {
//            std::vector<float> reconstructed_x(groupsize * vecdim);
//
//            for (int c = 0; c < ncentroids; c++){
//                float *subcentroid = subcentroids.data() + c * vecdim;
//                std::vector<idx_t> idx = idxs[c];
//
//                for (idx_t id : idx) {
//                    float *point = data.data() + id * vecdim;
//
//                    float residual[vecdim];
//                    for (int j = 0; j < vecdim; j++)
//                        residual[j] = point[j] - subcentroid[j];
//
//                    uint8_t code[index->pq->code_size];
//                    index->pq->compute_code(residual, code);
//
//                    float decoded_residual[vecdim];
//                    index->pq->decode(code, decoded_residual);
//
//                    float *rx = reconstructed_x.data() + id * vecdim;
//                    for (int j = 0; j < vecdim; j++)
//                        rx[j] = subcentroid[j] + decoded_residual[j];
//                }
//            }
//            double error = compute_quantization_error(reconstructed_x.data(), data.data(), vecdim, groupsize);
//            //std::cout << "[Modified] Quantization Error: " << error << std::endl;
//            modified_error += error;
//        }
    }
//    std::cout << "Average ncentroids: " << average_nc / ngroups << std::endl;
    std::cout << "[Global Baseline] Average Distance: " << baseline_average / ngroups << std::endl;
    std::cout << "[Global Modified] Average Distance: " << modified_average / ngroups << std::endl;
//    std::cout << "[Global Baseline] Average Error: " << baseline_error / ngroups << std::endl;
//    std::cout << "[Global Modified] Average Error: " << modified_error / ngroups << std::endl;

    std::cout << "Time(s): " << stopw.getElapsedTimeMicro() / 1000000 << std::endl;
    input.close();
}

void compute_average_distance(const char *path_data, const char *path_centroids, const char *path_precomputed_idxs,
                              const int ncentroids, const int vecdim, const int vecsize)
{
    std::ifstream centroids_input(path_centroids, ios::binary);
    std::vector<float> centroids(ncentroids*vecdim);
    readXvec<float>(centroids_input, centroids.data(), vecdim, ncentroids);
    centroids_input.close();

    const int batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * vecdim);
    std::vector<idx_t> idx_batch(batch_size);

    double average_dist = 0.0;
    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        readXvec<float>(base_input, batch.data(), vecdim, batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            const float *centroid = centroids + idx_batch[i] * vecdim;
            average_dist += faiss::fvec_L2sqr(batch.data() + i*vecdim, centroid, vecdim);
        }

        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();

    std::cout << "Average: " << average_dist / 1000000000 << std::endl;
}

void hybrid_test(const char *path_centroids,
                 const char *path_index, const char *path_precomputed_idxs,
                 const char *path_pq, const char *path_norm_pq,
                 const char *path_learn, const char *path_data, const char *path_q,
                 const char *path_gt, const char *path_info, const char *path_edges,
                 const int k,
                 const int vecsize,
                 const int ncentroids,
                 const int qsize,
                 const int vecdim,
                 const int efConstruction,
                 const int efSearch,
                 const int M,
                 const int M_PQ,
                 const int nprobes,
                 const int max_codes)
{
    cout << "Loading GT:\n";
    const int gt_dim = 1;
    idx_t *massQA = new idx_t[qsize * gt_dim];
    loadXvecs<idx_t>(path_gt, massQA, qsize, gt_dim);

    cout << "Loading queries:\n";
    float massQ[qsize * vecdim];
    loadXvecs<float>(path_q, massQ, qsize, vecdim);

    SpaceInterface<float> *l2space = new L2Space(vecdim);

    /** Create Index **/
    Index *index = new Index(vecdim, ncentroids, M_PQ, 8);
    index->buildQuantizer(l2space, path_centroids, path_info, path_edges, 500);
    index->precompute_idx(vecsize, path_data, path_precomputed_idxs);

    /** Train PQ **/
    std::ifstream learn_input(path_learn, ios::binary);
    int nt = 65536;
    std::vector<float> trainvecs(nt * vecdim);

    readXvec<float>(learn_input, trainvecs.data(), vecdim, nt);
    learn_input.close();

    /** Train residual PQ **/
    if (exists_test(path_pq)) {
        std::cout << "Loading PQ codebook from " << path_pq << std::endl;
        read_pq(path_pq, index->pq);
    }
    else {
        index->train_residual_pq(nt, trainvecs.data());
        std::cout << "Saving PQ codebook to " << path_pq << std::endl;
        write_pq(path_pq, index->pq);
    }

    /** Train norm PQ **/
    if (exists_test(path_norm_pq)) {
        std::cout << "Loading norm PQ codebook from " << path_norm_pq << std::endl;
        read_pq(path_norm_pq, index->norm_pq);
    }
    else {
        index->train_norm_pq(nt, trainvecs.data());
        std::cout << "Saving norm PQ codebook to " << path_norm_pq << std::endl;
        write_pq(path_norm_pq, index->norm_pq);
    }


    if (exists_test(path_index)){
        /** Load Index **/
        std::cout << "Loading index from " << path_index << std::endl;
        //index->read(path_index);
        compute_average_distance(path_data, path_centroids, path_precomputed_idxs, ncentroids, vecdim, vecsize);

        //save_groups(index, "/home/dbaranchuk/data/groups/groups999973.dat", path_data,
        //            path_precomputed_idxs, vecdim, vecsize);
        //check_idea(index, path_centroids, path_precomputed_idxs, path_data, vecsize, vecdim);
    } else {
        /** Add elements **/
        size_t batch_size = 1000000;
        std::ifstream base_input(path_data, ios::binary);
        std::ifstream idx_input(path_precomputed_idxs, ios::binary);
        std::vector<float> batch(batch_size * vecdim);
        std::vector<idx_t> idx_batch(batch_size);
        std::vector<idx_t> ids(vecsize);

        for (int b = 0; b < (vecsize / batch_size); b++) {
            readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
            readXvec<float>(base_input, batch.data(), vecdim, batch_size);
            for (size_t i = 0; i < batch_size; i++)
                ids[batch_size*b + i] = batch_size*b + i;

            printf("%.1f %c \n", (100.*b)/(vecsize/batch_size), '%');
            index->add(batch_size, batch.data(), ids.data() + batch_size*b, idx_batch.data());
        }
        idx_input.close();
        base_input.close();

        /** Save index, pq and norm_pq **/
        std::cout << "Saving index to " << path_index << std::endl;
        std::cout << "       pq to " << path_pq << std::endl;
        std::cout << "       norm pq to " << path_norm_pq << std::endl;
        index->write(path_index);
    }
//    /** Compute centroid norms **/
//    std::cout << "Computing centroid norms"<< std::endl;
//    index->compute_centroid_norm_table();

    /** Compute centroid sizes **/
    //std::cout << "Computing centroid sizes"<< std::endl;
    //index->compute_centroid_size_table(path_data, path_precomputed_idxs);

    /** Compute centroid vars **/
    //std::cout << "Computing centroid vars"<< std::endl;
    //index->compute_centroid_var_table(path_data, path_precomputed_idxs);

    //const char *path_index_new = "/home/dbaranchuk/hybrid8M_PQ16_new.index";
    //index->write(path_index_new);

    /** Update centroids **/
    //std::cout << "Update centroids" << std::endl;
    //index->update_centroids(path_data, path_precomputed_idxs,
    //                        "/home/dbaranchuk/data/updated_centroids4M.fvecs");

    /** Parse groundtruth **/
//    vector<std::priority_queue< std::pair<float, labeltype >>> answers;
//    std::cout << "Parsing gt:\n";
//    get_gt<float>(massQA, qsize, answers, gt_dim);

    /** Set search parameters **/
//    int correct = 0;
//    idx_t results[k];
//
//    index->max_codes = max_codes;
//    index->nprobe = nprobes;
//    index->quantizer->ef_ = efSearch;

    /** Search **/
//    StopW stopw = StopW();
//    for (int i = 0; i < qsize; i++) {
//        index->search(massQ+i*vecdim, k, results);
//
//        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
//        unordered_set<labeltype> g;
//
//        while (gt.size()) {
//            g.insert(gt.top().second);
//            gt.pop();
//        }
//
//        for (int j = 0; j < k; j++)
//            if (g.count(results[j]) != 0){
//                correct++;
//                break;
//            }
//    }
    /**Represent results**/
//    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
//    std::cout << "Recall@" << k << ": " << 1.0f*correct / qsize << std::endl;
//    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;


    //std::cout << "Check precomputed idxs"<< std::endl;
    //check_precomputing(index, path_data, path_precomputed_idxs, vecdim, ncentroids, vecsize, gt_mistakes, gt_correct);

    delete index;
    delete massQA;
    delete l2space;
}