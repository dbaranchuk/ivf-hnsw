#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <faiss/ProductQuantizer.h>
#include <faiss/index_io.h>
#include "hnswIndexPQ.h"
#include "hnswIndexPQ_new.h"
#include "hnswlib.h"

#include <cmath>
#include <map>
#include <set>
#include <unordered_set>
using namespace std;
using namespace hnswlib;

//class StopW {
//    std::chrono::steady_clock::time_point time_begin;
//public:
//    StopW() {
//        time_begin = std::chrono::steady_clock::now();
//    }
//    float getElapsedTimeMicro() {
//        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
//        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
//    }
//    void reset() {
//        time_begin = std::chrono::steady_clock::now();
//    }
//
//};


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

void save_groups(Index *index, const char *path_groups, const char *path_data,
                 const char *path_precomputed_idxs, const int vecdim, const int vecsize)
{
    const int ncentroids = 999973;
    std::vector<std::vector<float>> data(ncentroids);
    std::vector<std::vector<idx_t>> idxs(ncentroids);

    const int batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * vecdim);
    std::vector<idx_t> idx_batch(batch_size);

    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        readXvec<float>(base_input, batch.data(), vecdim, batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            //if (idx_batch[i] < 900000)
            //    continue;

            idx_t cur_idx = idx_batch[i];
            //for (int d = 0; d < vecdim; d++)
            //    data[cur_idx].push_back(batch[i * vecdim + d]);
            idxs[cur_idx].push_back(b*batch_size + i);
        }

        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();

    //FILE *fout = fopen(path_groups, "wb");
    const char *path_idxs = "/home/dbaranchuk/data/groups/sift1B_idxs9993127.ivecs";
    FILE *fout = fopen(path_idxs, "wb");

//    size_t counter = 0;
    for (int i = 0; i < ncentroids; i++) {
        int groupsize = data[i].size() / vecdim;
//        counter += idxs[i].size();

        if (groupsize != index->ids[i].size()){
            std::cout << "Wrong groupsize: " << groupsize << " vs "
                      << index->ids[i].size() <<std::endl;
            exit(1);
        }

        fwrite(&groupsize, sizeof(int), 1, fout);
        fwrite(idxs[i].data(), sizeof(idx_t), idxs[i].size(), fout);
        //fwrite(data[i].data(), sizeof(float), data[i].size(), fout);
    }
//    if (counter != 9993127){
//        std::cout << "Wrong poitns num\n";
//        exit(1);
//    }
}


void check_groups(const char *path_data, const char *path_precomputed_idxs,
                  const char *path_groups, const char *path_groups_idxs)
{

    const int vecsize = 1000000000;
    const int d = 128;
    /** Read Group **/
    std::ifstream input_groups(path_groups, ios::binary);
    std::ifstream input_groups_idxs(path_groups_idxs, ios::binary);

    int groupsize, check_groupsize;
    input_groups.read((char *) &groupsize, sizeof(int));
    input_groups_idxs.read((char *) &check_groupsize, sizeof(int));
    if (groupsize != check_groupsize){
        std::cout << "Wrong groupsizes: " << groupsize << " " << check_groupsize << std::endl;
        exit(1);
    }

    std::vector<uint8_t> group_b(groupsize*d);
    std::vector<float> group(groupsize*d);
    std::vector<idx_t> group_idxs(groupsize);

    //input_groups.read((char *) group.data(), groupsize * d * sizeof(float));
    input_groups.read((char *) group_b.data(), groupsize * d * sizeof(uint8_t));
    for (int i = 0; i < groupsize*d; i++)
        group[i] = (1.0)*group_b[i];

    input_groups_idxs.read((char *) group_idxs.data(), groupsize * sizeof(idx_t));

    input_groups.close();
    input_groups_idxs.close();

    /** Make set of idxs **/
    std::unordered_set<idx_t > idx_set;
    for (int i = 0; i < groupsize; i++)
        idx_set.insert(group_idxs[i]);

    /** Loop **/
    const int batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * d);
    std::vector<idx_t> idx_batch(batch_size);

    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        //readXvec<float>(base_input, batch.data(), d, batch_size);
        readXvecFvec<uint8_t>(base_input, batch.data(), d, batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            if (idx_set.count(b*batch_size + i) == 0)
                continue;

            const float *x = batch.data() + i*d;
            for (int j = 0; j < groupsize; j++){
                if (group_idxs[j] != b*batch_size + i)
                    continue;

                const float *y = group.data() + j * d;

                std::cout << faiss::fvec_L2sqr(x, y, d) << std::endl;
                break;
            }
        }

        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();
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
            const float *centroid = centroids.data() + idx_batch[i] * vecdim;
            average_dist += faiss::fvec_L2sqr(batch.data() + i*vecdim, centroid, vecdim);
        }

        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();

    std::cout << "Average: " << average_dist / 1000000000 << std::endl;
}


void compute_average_distance_sift(const char *path_data, const char *path_centroids, const char *path_precomputed_idxs,
                              const int ncentroids, const int vecdim, const int vecsize)
{
    std::ifstream centroids_input(path_centroids, ios::binary);
    std::vector<float> centroids(ncentroids*vecdim);
    readXvec<float>(centroids_input, centroids.data(), vecdim, ncentroids);
    centroids_input.close();

    const int batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float > batch(batch_size * vecdim);
    std::vector<idx_t> idx_batch(batch_size);

    double average_dist = 0.0;
    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        readXvecFvec<uint8_t>(base_input, batch.data(), vecdim, batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            const float *centroid = centroids.data() + idx_batch[i] * vecdim;
            average_dist += faiss::fvec_L2sqr(batch.data() + i*vecdim, centroid, vecdim);
        }

        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();

    std::cout << "Average: " << average_dist / vecsize << std::endl;
}

void save_groups_sift(const char *path_groups, const char *path_data, const char *path_precomputed_idxs,
                      const int ncentroids, const int vecdim, const int vecsize)
{
    //std::vector<std::vector<uint8_t >> data(ncentroids);
    std::vector<std::vector<idx_t>> idxs(ncentroids);

    const int batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<uint8_t > batch(batch_size * vecdim);
    std::vector<idx_t> idx_batch(batch_size);

    for (int b = 0; b < (vecsize / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
    //    readXvec<uint8_t >(base_input, batch.data(), vecdim, batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            idx_t cur_idx = idx_batch[i];
      //      for (int d = 0; d < vecdim; d++)
      //          data[cur_idx].push_back(batch[i * vecdim + d]);
            idxs[cur_idx].push_back(b*batch_size + i);
        }

        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
    }
    idx_input.close();
    base_input.close();

    //FILE *fout = fopen(path_groups, "wb");
    const char *path_idxs = "/home/dbaranchuk/data/groups/sift1B_idxs.ivecs";
    FILE *fout = fopen(path_idxs, "wb");

    size_t counter = 0;
    for (int i = 0; i < ncentroids; i++) {
        int groupsize = idxs[i].size();//data[i].size() / vecdim;
        counter += groupsize;

        fwrite(&groupsize, sizeof(int), 1, fout);
        fwrite(idxs[i].data(), sizeof(idx_t), idxs[i].size(), fout);
        //fwrite(data[i].data(), sizeof(uint8_t), data[i].size(), fout);
    }
    if (counter != vecsize){
        std::cout << "Wrong poitns num\n";
        exit(1);
    }
}

void random_subset(const float *x, float *x_out, int d, int nx, int sub_nx)
{
    int seed = 1234;
    std::vector<int> perm (nx);
    faiss::rand_perm (perm.data (), nx, seed);

    for (idx_t i = 0; i < sub_nx; i++)
        memcpy (x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
}


//template <typename format>
//void readBvec(std::ifstream &input, float *data, const int d, const int n = 1)
//{
//    int in = 0;
//    format mass[d];
//
//    for (int i = 0; i < n; i++) {
//        input.read((char *) &in, sizeof(int));
//        if (in != d) {
//            std::cout << "file error\n";
//            exit(1);
//        }
//        input.read((char *)(mass), in * sizeof(format));
//        for (size_t j = 0; j < d; j++) {
//            data[i * d + j] = (1.0) * mass[j];
//        }
//    }
//}

void check_groupsizes(Index *index, int ncentroids)
{
    std::vector < size_t > groupsizes(ncentroids);

    int sparse_counter = 0;
    int big_counter = 0;
    int small_counter = 0;
    int other_counter = 0;
    int giant_counter = 0;
    for (int i = 0; i < ncentroids; i++){
        int groupsize = index->norm_codes[i].size();
        if (groupsize < 100)
            sparse_counter++;
        else if (groupsize > 100 && groupsize < 500)
            small_counter++;
        else if (groupsize > 1500 && groupsize < 3000)
            big_counter++;
        else if (groupsize > 3000)
            giant_counter++;
        else
            other_counter++;
    }

    std::cout << "Number of clusters with size < 100: " << sparse_counter << std::endl;
    std::cout << "Number of clusters with size > 100 && < 500 : " << small_counter << std::endl;

    std::cout << "Number of clusters with size > 1500 && < 3000: " << big_counter << std::endl;
    std::cout << "Number of clusters with size > 3000: " << giant_counter << std::endl;

    std::cout << "Number of clusters with size > 500 && < 1500: " << other_counter << std::endl;
}


enum class Dataset
{
    DEEP1B,
    SIFT1B
};

void hybrid_test(const char *path_centroids,
                 const char *path_index, const char *path_precomputed_idxs,
                 const char *path_pq, const char *path_norm_pq,
                 const char *path_learn, const char *path_data, const char *path_q,
                 const char *path_gt, const char *path_info, const char *path_edges,
                 const char *path_groups, const char *path_idxs,
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
                 const int max_codes,
                 const int nsubcentroids)
{
//    compute_average_distance_sift("/home/dbaranchuk/data/bigann/bigann_base.bvecs",
//                                  "/home/dbaranchuk/data/sift1B_centroids1M.fvecs",
//                                  "/home/dbaranchuk/sift1B_precomputed_idxs_993127.ivecs",
//                                  993127, 128, vecsize);
//    save_groups_sift("/home/dbaranchuk/data/groups/sift1B_groups.bvecs",
//                     "/home/dbaranchuk/data/bigann/bigann_base.bvecs",
//                     "/home/dbaranchuk/sift1B_precomputed_idxs_993127.ivecs",
//                     993127, 128, vecsize);
//    exit(0);
//    check_groups(path_data, path_precomputed_idxs, path_groups, path_idxs);
//    exit(0);
    Dataset dataset = Dataset::DEEP1B;

    cout << "Loading GT:\n";
    int gt_dim;
    switch(dataset){
        case Dataset::SIFT1B:
            gt_dim = 1000;
            break;
        case Dataset::DEEP1B:
            gt_dim = 1;
    }
    idx_t *massQA = new idx_t[qsize * gt_dim];
    loadXvecs<idx_t>(path_gt, massQA, qsize, gt_dim);

    cout << "Loading queries:\n";
    float massQ[qsize * vecdim];
    std::ifstream query_input(path_q, ios::binary);
    switch(dataset){
        case Dataset::SIFT1B:
            readXvecFvec<uint8_t >(query_input, massQ, vecdim, qsize);
            break;
        case Dataset::DEEP1B:
            readXvec<float >(query_input, massQ, vecdim, qsize);
            break;
    }
    query_input.close();

    SpaceInterface<float> *l2space = new L2Space(vecdim);

    /** Create Index **/
    ModifiedIndex *index = new ModifiedIndex(vecdim, ncentroids, M_PQ, 8, nsubcentroids);
    //Index *index = new Index(vecdim, ncentroids, M_PQ, 8);
    index->buildQuantizer(l2space, path_centroids, path_info, path_edges, 500);
    //index->precompute_idx(vecsize, path_data, path_precomputed_idxs);

    /** Train PQ **/
    std::ifstream learn_input(path_learn, ios::binary);
    int nt = 10000000;
    int sub_nt = 262144;//65536;
    std::vector<float> trainvecs(nt * vecdim);
    switch(dataset){
        case Dataset::SIFT1B:
            readXvecFvec<uint8_t >(learn_input, trainvecs.data(), vecdim, nt);
            break;
        case Dataset::DEEP1B:
            readXvec<float>(learn_input, trainvecs.data(), vecdim, nt);
            break;
    }
    learn_input.close();

    /** Set Random Subset of 65536 trainvecs **/
    std::vector<float> trainvecs_rnd_subset(sub_nt * vecdim);
    random_subset(trainvecs.data(), trainvecs_rnd_subset.data(), vecdim, nt, sub_nt);

    /** Train residual PQ **/
    if (exists_test(path_pq)) {
        std::cout << "Loading PQ codebook from " << path_pq << std::endl;
        read_pq(path_pq, index->pq);
    }
    else {
        std::cout << "Training PQ codebook " << std::endl;
        index->train_residual_pq(sub_nt, trainvecs_rnd_subset.data());
        std::cout << "Saving PQ codebook to " << path_pq << std::endl;
        write_pq(path_pq, index->pq);
    }

    /** Train norm PQ **/
    if (exists_test(path_norm_pq)) {
        std::cout << "Loading norm PQ codebook from " << path_norm_pq << std::endl;
        read_pq(path_norm_pq, index->norm_pq);
    }
    else {
        index->train_norm_pq(sub_nt, trainvecs_rnd_subset.data());
        std::cout << "Saving norm PQ codebook to " << path_norm_pq << std::endl;
        write_pq(path_norm_pq, index->norm_pq);
    }


    if (exists_test(path_index)){
        /** Load Index **/
        std::cout << "Loading index from " << path_index << std::endl;
        index->read(path_index);
    } else {
        /** Add elements **/
        index->add<uint8_t>(path_groups, path_idxs);

//        size_t batch_size = 1000000;
//        std::ifstream base_input(path_data, ios::binary);
//        std::ifstream idx_input(path_precomputed_idxs, ios::binary);
//        std::vector<float> batch(batch_size * vecdim);
//        std::vector<idx_t> idx_batch(batch_size);
//        std::vector<idx_t> ids(vecsize);
//
//        for (int b = 0; b < (vecsize / batch_size); b++) {
//            readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
//            //readXvec<float>(base_input, batch.data(), vecdim, batch_size);
//            readXvecFvec<uint8_t>(base_input, batch.data(), vecdim, batch_size);
//            for (size_t i = 0; i < batch_size; i++)
//                ids[batch_size*b + i] = batch_size*b + i;
//
//            printf("%.1f %c \n", (100.*b)/(vecsize/batch_size), '%');
//            index->add(batch_size, batch.data(), ids.data() + batch_size*b, idx_batch.data());
//        }
//        idx_input.close();
//        base_input.close();

        /** Save index, pq and norm_pq **/
        std::cout << "Saving index to " << path_index << std::endl;
        std::cout << "       pq to " << path_pq << std::endl;
        std::cout << "       norm pq to " << path_norm_pq << std::endl;
        index->write(path_index);
    }
    /** Computing Centroid Norms **/
    std::cout << "Computing centroid norms"<< std::endl;
    index->compute_centroid_norms();
    index->compute_s_c();

    /** Parse groundtruth **/
    vector<std::priority_queue< std::pair<float, labeltype >>> answers;
    std::cout << "Parsing gt:\n";
    get_gt<float>(massQA, qsize, answers, gt_dim);

    /** Compute Graphic **/
    //index->compute_graphic(massQ, massQA, gt_dim, qsize);

    /** Set search parameters **/
    int correct = 0;
    idx_t results[k];

    index->max_codes = max_codes;
    index->nprobe = nprobes;
    index->quantizer->ef_ = efSearch;

    /** Search **/
    StopW stopw = StopW();
    for (int i = 0; i < 1/*qsize*/; i++) {
        index->search(massQ+i*vecdim, k, results);

        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        for (int j = 0; j < k; j++)
            if (g.count(results[j]) != 0){
                correct++;
                break;
            }
    }

    /**Represent results**/
    float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
    std::cout << "Recall@" << k << ": " << 1.0f*correct / qsize << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;
    std::cout << "Average max_codes: " << index->average_max_codes / 10000 << std::endl;

    //check_groupsizes(index, ncentroids);
    //std::cout << "Check precomputed idxs"<< std::endl;
    //check_precomputing(index, path_data, path_precomputed_idxs, vecdim, ncentroids, vecsize, gt_mistakes, gt_correct);

    delete index;
    delete massQA;
    delete l2space;
}