#pragma once

#include <queue>
#include <limits>
#include <cmath>
#include <chrono>
#include <sys/time.h>

//#include <faiss/utils.h>

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

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

template <typename format>
void readXvec(std::ifstream &input, format *data, const int d, const int n = 1)
{
    int in = 0;
    for (int i = 0; i < n; i++) {
        input.read((char *) &in, sizeof(int));
        if (in != d) {
            std::cout << "file error\n";
            exit(1);
        }
        input.read((char *)(data + i*d), in * sizeof(format));
    }
}

template <typename format>
void readXvecFvec(std::ifstream &input, float *data, const int d, const int n = 1)
{
    int in = 0;
    format mass[d];

    for (int i = 0; i < n; i++) {
        input.read((char *) &in, sizeof(int));
        if (in != d) {
            std::cout << "file error\n";
            exit(1);
        }
        input.read((char *)mass, in * sizeof(format));
        for (int j = 0; j < d; j++)
            data[i*d + j] = (1.0)*mass[j];
    }
}

void random_subset(const float *x, float *x_out, int d, int nx, int sub_nx)
{
    int seed = 1234;
    std::vector<int> perm (nx);
    //faiss::rand_perm (perm.data (), nx, seed);

    for (int i = 0; i < sub_nx; i++)
        memcpy (x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
}


enum class Dataset
{
    DEEP1B,
    SIFT1B
};















//void compute_average_distance(const char *path_data, const char *path_centroids, const char *path_precomputed_idxs,
//                              const int ncentroids, const int vecdim, const int vecsize)
//{
//    std::ifstream centroids_input(path_centroids, ios::binary);
//    std::vector<float> centroids(ncentroids*vecdim);
//    readXvec<float>(centroids_input, centroids.data(), vecdim, ncentroids);
//    centroids_input.close();
//
//    const int batch_size = 1000000;
//    std::ifstream base_input(path_data, ios::binary);
//    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
//    std::vector<float> batch(batch_size * vecdim);
//    std::vector<idx_t> idx_batch(batch_size);
//
//    double average_dist = 0.0;
//    for (int b = 0; b < (vecsize / batch_size); b++) {
//        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
//        readXvec<float>(base_input, batch.data(), vecdim, batch_size);
//
//        for (size_t i = 0; i < batch_size; i++) {
//            const float *centroid = centroids.data() + idx_batch[i] * vecdim;
//            average_dist += faiss::fvec_L2sqr(batch.data() + i*vecdim, centroid, vecdim);
//        }
//
//        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
//    }
//    idx_input.close();
//    base_input.close();
//
//    std::cout << "Average: " << average_dist / 1000000000 << std::endl;
//}
//
//
//void compute_average_distance_sift(const char *path_data, const char *path_centroids, const char *path_precomputed_idxs,
//                              const int ncentroids, const int vecdim, const int vecsize)
//{
//    std::ifstream centroids_input(path_centroids, ios::binary);
//    std::vector<float> centroids(ncentroids*vecdim);
//    readXvec<float>(centroids_input, centroids.data(), vecdim, ncentroids);
//    centroids_input.close();
//
//    const int batch_size = 1000000;
//    std::ifstream base_input(path_data, ios::binary);
//    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
//    std::vector<float > batch(batch_size * vecdim);
//    std::vector<idx_t> idx_batch(batch_size);
//
//    double average_dist = 0.0;
//    for (int b = 0; b < (vecsize / batch_size); b++) {
//        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
//        readXvecFvec<uint8_t>(base_input, batch.data(), vecdim, batch_size);
//
//        for (size_t i = 0; i < batch_size; i++) {
//            const float *centroid = centroids.data() + idx_batch[i] * vecdim;
//            average_dist += faiss::fvec_L2sqr(batch.data() + i*vecdim, centroid, vecdim);
//        }
//
//        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
//    }
//    idx_input.close();
//    base_input.close();
//
//    std::cout << "Average: " << average_dist / vecsize << std::endl;
//}
//static void check_precomputing(IndexIVF_HNSW *index, const char *path_data, const char *path_precomputed_idxs,
//                               size_t vecdim, size_t ncentroids, size_t vecsize,
//                               std::set<idx_t> gt_mistakes, std::set<idx_t> gt_correct)
//{
//    size_t batch_size = 1000000;
//    std::ifstream base_input(path_data, ios::binary);
//    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
//    std::vector<float> batch(batch_size * vecdim);
//    std::vector<idx_t> idx_batch(batch_size);
//
////    int counter = 0;
//    std::vector<float> mistake_dst;
//    std::vector<float> correct_dst;
//    for (int b = 0; b < (vecsize / batch_size); b++) {
//        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
//        readXvec<float>(base_input, batch.data(), vecdim, batch_size);
//
//        printf("%.1f %c \n", (100.*b)/(vecsize/batch_size), '%');
//
//        for (int i = 0; i < batch_size; i++) {
//            int elem = batch_size*b + i;
//            //float min_dist = 1000000;
//            //int min_centroid = 100000000;
//
//            if (gt_mistakes.count(elem) == 0 &&
//                gt_correct.count(elem) == 0)
//                continue;
//
//            float *data = batch.data() + i*vecdim;
//            for (int j = 0; j < ncentroids; j++) {
//                float *centroid = (float *) index->quantizer->getDataByInternalId(j);
//                float dist = faiss::fvec_L2sqr(data, centroid, vecdim);
//                //if (dist < min_dist){
//                //    min_dist = dist;
//                //    min_centroid = j;
//                //}
//                if (gt_mistakes.count(elem) != 0)
//                    mistake_dst.push_back(dist);
//                if (gt_correct.count(elem) != 0)
//                    correct_dst.push_back(dist);
//            }
////            if (min_centroid != idx_batch[i]){
////                std::cout << "Element: " << elem << " True centroid: " << min_centroid << " Precomputed centroid:" << idx_batch[i] << std::endl;
////                counter++;
////            }
//        }
//    }
//
//    std::cout << "Correct distance distribution\n";
//    for (int i = 0; i < correct_dst.size(); i++)
//        std::cout << correct_dst[i] << std::endl;
//
//    std::cout << std::endl << std::endl << std::endl;
//    std::cout << "Mistake distance distribution\n";
//    for (int i = 0; i < mistake_dst.size(); i++)
//        std::cout << mistake_dst[i] << std::endl;
//
//    idx_input.close();
//    base_input.close();
//}
//









//void save_groups(IndexIVF_HNSW *index, const char *path_groups, const char *path_data,
//                 const char *path_precomputed_idxs, const int vecdim, const int vecsize)
//{
//    const int ncentroids = 999973;
//    std::vector<std::vector<float>> data(ncentroids);
//    std::vector<std::vector<idx_t>> idxs(ncentroids);
//
//    const int batch_size = 1000000;
//    std::ifstream base_input(path_data, ios::binary);
//    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
//    std::vector<float> batch(batch_size * vecdim);
//    std::vector<idx_t> idx_batch(batch_size);
//
//    for (int b = 0; b < (vecsize / batch_size); b++) {
//        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
//        readXvec<float>(base_input, batch.data(), vecdim, batch_size);
//
//        for (size_t i = 0; i < batch_size; i++) {
//            //if (idx_batch[i] < 900000)
//            //    continue;
//
//            idx_t cur_idx = idx_batch[i];
//            //for (int d = 0; d < vecdim; d++)
//            //    data[cur_idx].push_back(batch[i * vecdim + d]);
//            idxs[cur_idx].push_back(b*batch_size + i);
//        }
//
//        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
//    }
//    idx_input.close();
//    base_input.close();
//
//    //FILE *fout = fopen(path_groups, "wb");
//    const char *path_idxs = "/home/dbaranchuk/data/groups/sift1B_idxs9993127.ivecs";
//    FILE *fout = fopen(path_idxs, "wb");
//
////    size_t counter = 0;
//    for (int i = 0; i < ncentroids; i++) {
//        int groupsize = data[i].size() / vecdim;
////        counter += idxs[i].size();
//
//        if (groupsize != index->ids[i].size()){
//            std::cout << "Wrong groupsize: " << groupsize << " vs "
//                      << index->ids[i].size() <<std::endl;
//            exit(1);
//        }
//
//        fwrite(&groupsize, sizeof(int), 1, fout);
//        fwrite(idxs[i].data(), sizeof(idx_t), idxs[i].size(), fout);
//        //fwrite(data[i].data(), sizeof(float), data[i].size(), fout);
//    }
////    if (counter != 9993127){
////        std::cout << "Wrong poitns num\n";
////        exit(1);
////    }
//}




//void save_groups_sift(const char *path_groups, const char *path_data, const char *path_precomputed_idxs,
//                      const int ncentroids, const int vecdim, const int vecsize)
//{
//    //std::vector<std::vector<uint8_t >> data(ncentroids);
//    std::vector<std::vector<idx_t>> idxs(ncentroids);
//
//    const int batch_size = 1000000;
//    std::ifstream base_input(path_data, ios::binary);
//    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
//    std::vector<uint8_t > batch(batch_size * vecdim);
//    std::vector<idx_t> idx_batch(batch_size);
//
//    for (int b = 0; b < (vecsize / batch_size); b++) {
//        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
//    //    readXvec<uint8_t >(base_input, batch.data(), vecdim, batch_size);
//
//        for (size_t i = 0; i < batch_size; i++) {
//            idx_t cur_idx = idx_batch[i];
//      //      for (int d = 0; d < vecdim; d++)
//      //          data[cur_idx].push_back(batch[i * vecdim + d]);
//            idxs[cur_idx].push_back(b*batch_size + i);
//        }
//
//        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
//    }
//    idx_input.close();
//    base_input.close();
//
//    //FILE *fout = fopen(path_groups, "wb");
//    const char *path_idxs = "/home/dbaranchuk/data/groups/sift1B_idxs.ivecs";
//    FILE *fout = fopen(path_idxs, "wb");
//
//    size_t counter = 0;
//    for (int i = 0; i < ncentroids; i++) {
//        int groupsize = idxs[i].size();//data[i].size() / vecdim;
//        counter += groupsize;
//
//        fwrite(&groupsize, sizeof(int), 1, fout);
//        fwrite(idxs[i].data(), sizeof(idx_t), idxs[i].size(), fout);
//        //fwrite(data[i].data(), sizeof(uint8_t), data[i].size(), fout);
//    }
//    if (counter != vecsize){
//        std::cout << "Wrong poitns num\n";
//        exit(1);
//    }
//}

























//void check_groups(const char *path_data, const char *path_precomputed_idxs,
//                  const char *path_groups, const char *path_groups_idxs)
//{
//
//    const int vecsize = 1000000000;
//    const int d = 128;
//    /** Read Group **/
//    std::ifstream input_groups(path_groups, ios::binary);
//    std::ifstream input_groups_idxs(path_groups_idxs, ios::binary);
//
//    int groupsize, check_groupsize;
//    input_groups.read((char *) &groupsize, sizeof(int));
//    input_groups_idxs.read((char *) &check_groupsize, sizeof(int));
//    if (groupsize != check_groupsize){
//        std::cout << "Wrong groupsizes: " << groupsize << " " << check_groupsize << std::endl;
//        exit(1);
//    }
//
//    std::vector<uint8_t> group_b(groupsize*d);
//    std::vector<float> group(groupsize*d);
//    std::vector<idx_t> group_idxs(groupsize);
//
//    //input_groups.read((char *) group.data(), groupsize * d * sizeof(float));
//    input_groups.read((char *) group_b.data(), groupsize * d * sizeof(uint8_t));
//    for (int i = 0; i < groupsize*d; i++)
//        group[i] = (1.0)*group_b[i];
//
//    input_groups_idxs.read((char *) group_idxs.data(), groupsize * sizeof(idx_t));
//
//    input_groups.close();
//    input_groups_idxs.close();
//
//    /** Make set of idxs **/
//    std::unordered_set<idx_t > idx_set;
//    for (int i = 0; i < groupsize; i++)
//        idx_set.insert(group_idxs[i]);
//
//    /** Loop **/
//    const int batch_size = 1000000;
//    std::ifstream base_input(path_data, ios::binary);
//    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
//    std::vector<float> batch(batch_size * d);
//    std::vector<idx_t> idx_batch(batch_size);
//
//    for (int b = 0; b < (vecsize / batch_size); b++) {
//        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
//        //readXvec<float>(base_input, batch.data(), d, batch_size);
//        readXvecFvec<uint8_t>(base_input, batch.data(), d, batch_size);
//
//        for (size_t i = 0; i < batch_size; i++) {
//            if (idx_set.count(b*batch_size + i) == 0)
//                continue;
//
//            const float *x = batch.data() + i*d;
//            for (int j = 0; j < groupsize; j++){
//                if (group_idxs[j] != b*batch_size + i)
//                    continue;
//
//                const float *y = group.data() + j * d;
//
//                std::cout << faiss::fvec_L2sqr(x, y, d) << std::endl;
//                break;
//            }
//        }
//
//        if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / (vecsize / batch_size), '%');
//    }
//    idx_input.close();
//    base_input.close();
//}












//void check_groupsizes(IndexIVF_HNSW *index, int ncentroids)
//{
//    std::vector < size_t > groupsizes(ncentroids);
//
//    int sparse_counter = 0;
//    int big_counter = 0;
//    int small_counter = 0;
//    int other_counter = 0;
//    int giant_counter = 0;
//    for (int i = 0; i < ncentroids; i++){
//        int groupsize = index->norm_codes[i].size();
//        if (groupsize < 100)
//            sparse_counter++;
//        else if (groupsize > 100 && groupsize < 500)
//            small_counter++;
//        else if (groupsize > 1500 && groupsize < 3000)
//            big_counter++;
//        else if (groupsize > 3000)
//            giant_counter++;
//        else
//            other_counter++;
//    }
//
//    std::cout << "Number of clusters with size < 100: " << sparse_counter << std::endl;
//    std::cout << "Number of clusters with size > 100 && < 500 : " << small_counter << std::endl;
//
//    std::cout << "Number of clusters with size > 1500 && < 3000: " << big_counter << std::endl;
//    std::cout << "Number of clusters with size > 3000: " << giant_counter << std::endl;
//
//    std::cout << "Number of clusters with size > 500 && < 1500: " << other_counter << std::endl;
//}