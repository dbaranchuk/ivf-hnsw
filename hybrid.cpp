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

#include <map>
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

//template <typename format>
//static void readXvec(std::ifstream &input, format *mass, const int d, const int n = 1)
//{
//    int in = 0;
//    for (int i = 0; i < n; i++) {
//        input.read((char *) &in, sizeof(int));
//        if (in != d) {
//            std::cout << "file error\n";
//            exit(1);
//        }
//        input.read((char *)(mass+i*d), in * sizeof(format));
//    }
//}

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
                               size_t vecdim, size_t ncentroids, size_t vecsize)
{
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

        int counter = 0;
        for (int i = 0; i < batch_size; i++) {
            float min_dist = 10000;
            float min_centroid = 1000000;

            float *data = batch.data() + i*vecdim;
#pragma omp parallel for num_threads(20)
            for (int j = 0; j < ncentroids; j++) {
                float *centroid = (float *) index->quantizer->getDataByInternalId(j);
                float dist = faiss::fvec_L2sqr(data, centroid, vecdim);
                if (dist < min_dist){
                    min_dist = dist;
                    min_centroid = j;
                }
            }
            if (min_centroid != idx_batch[i]){
                std::cout << batch_size*b + i << " " << min_centroid << " " << idx_batch[i] << std::endl;
                counter++;
            }
        }
        double error = counter * (100.0) / batch_size;
        std::cout << "Percentage of incorrect centroids: " << error << "%\n";
    }
    idx_input.close();
    base_input.close();
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
        index->read(path_index);
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

    std::cout << "Check precomputed idxs"<< std::endl;
    check_precomputing(index, path_data, path_precomputed_idxs, vecdim, ncentroids, vecsize);

    /** Compute centroid norms **/
    std::cout << "Computing centroid norms"<< std::endl;
    index->compute_centroid_norm_table();

    /** Parse groundtruth **/
    vector<std::priority_queue< std::pair<float, labeltype >>> answers;
    std::cout << "Parsing gt:\n";
    get_gt<float>(massQA, qsize, answers, gt_dim);

    /** Set search parameters **/
    int correct = 0;
    idx_t results[k];

    index->max_codes = max_codes;
    index->nprobe = nprobes;
    index->quantizer->ef_ = efSearch;

    /** Search **/
    StopW stopw = StopW();
    for (int i = 0; i < qsize; i++) {
        index->search(massQ+i*vecdim, k, results);

        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        int prev_correct = correct;
        for (int j = 0; j < k; j++){
            if (g.count(results[j]) != 0){
                correct++;
                break;
            }
        }
        if (prev_correct == correct){
            std::cout << i << " " << answers[i].top().second << std::endl;
        }
    }

    /**Represent results**/
    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
    std::cout << "Recall@" << k << ": " << 1.0f*correct / qsize << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;

    delete index;
    delete massQA;
    delete l2space;
}