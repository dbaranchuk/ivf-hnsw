#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include "hnswlib.h"
#include <faiss/ProductQuantizer.h>
#include "hnswIndexPQ.h"

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

template <typename dist_t, typename vtype>
static float test_approx(vtype *massQ, size_t qsize, HierarchicalNSW<dist_t, vtype> &appr_alg,
                         size_t vecdim, vector<std::priority_queue< std::pair<dist_t, labeltype >>> &answers,
                         size_t k, bool pq = false)
{
	size_t correct = 0;

	//uncomment to test in parallel mode:
	//#pragma omp parallel for
	for (int i = 0; i < qsize; i++) {
		std::priority_queue< std::pair<dist_t, labeltype >> result;
        if (pq)
            result = appr_alg.searchKnn(massQ + vecdim*i, k, i);
        else
            result = appr_alg.searchKnn(massQ + vecdim*i, k);

		std::priority_queue< std::pair<dist_t, labeltype >> gt(answers[i]);
		unordered_set <labeltype> g;

        float dist2gt = appr_alg.space->fstdistfunc((void*)(massQ + vecdim*i),//appr_alg.getDataByInternalId(gt.top().second),
                                                     appr_alg.getDataByInternalId(appr_alg.enterpoint0));
        appr_alg.nev9zka += dist2gt / qsize;

		while (gt.size()) {
			g.insert(gt.top().second);
			gt.pop();
		}

		while (result.size()) {
			if (g.find(result.top().second) != g.end())
				correct++;
			result.pop();
		}
	}
	return 1.0f*correct / qsize;
}

template <typename dist_t, typename vtype>
static void test_vs_recall(vtype *massQ, size_t qsize, HierarchicalNSW<dist_t, vtype> &appr_alg,
                           size_t vecdim, vector<std::priority_queue< std::pair<dist_t, labeltype >>> &answers,
                           size_t k, bool pq = false)
{
	vector<size_t> efs; //= {30, 100, 460};
    if (k < 30) {
        for (int i = k; i < 30; i++) efs.push_back(i);
        for (int i = 30; i < 100; i += 10) efs.push_back(i);
        for (int i = 100; i <= 780; i += 40) efs.push_back(i);
    }
	else if (k < 100) {
        for (int i = k; i < 100; i += 10) efs.push_back(i);
        for (int i = 100; i <= 780; i += 40) efs.push_back(i);
    } else
        for (int i = k; i <= 780; i += 40) efs.push_back(i);

	for (size_t ef : efs) {
		appr_alg.ef_ = ef;
        appr_alg.dist_calc = 0;
        appr_alg.nev9zka = 0.0;
        appr_alg.hops = 0.0;
        appr_alg.hops0 = 0.0;
		StopW stopw = StopW();
		float recall = test_approx<dist_t, vtype>(massQ, qsize, appr_alg, vecdim, answers, k, pq);
		float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
		float avr_dist_count = appr_alg.dist_calc*1.f / qsize;
		cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\t" << avr_dist_count << " dcs\t" << appr_alg.hops0 + appr_alg.hops << " hps\n";
		if (recall > 1.0) {
			cout << recall << "\t" << time_us_per_query << " us\t" << avr_dist_count << " dcs\n";
			break;
		}
    }
    cout << "Average hops on levels 1+: " << appr_alg.hops << endl;
    cout << "Average distance from 0 level entry point to query: " << appr_alg.nev9zka << endl;
}


/**
 * Main SIFT Test Function
*/

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

//template <typename format>
//static void readXvec(ifstream &input, format *mass, const int d)
//{
//    int in = 0;
//    input.read((char *) &in, sizeof(int));
//    if (in != d) {
//        cout << "file error\n";
//        exit(1);
//    }
//    input.read((char *) mass, in * sizeof(format));
//}

template<typename dist_t, typename vtype>
static void _hnsw_test(const char *path_codebooks, const char *path_tables, const char *path_data, const char *path_q,
                       const char *path_gt, const char *path_info, const char *path_edges,
                       L2SpaceType l2SpaceType,
                       const int k, const int vecsize, const int qsize,
                       const int vecdim, const int efConstruction, const int M, bool one_layer)
{
    const int M_PQ = 16;
    const bool PQ = (path_codebooks && path_tables);

    const int specsize = 5000000;//101917929;
    //const map<size_t, pair<size_t, size_t>> M_map = {{vecsize, {M, 2*M}}};
    //const map<size_t, pair<size_t, size_t>> M_map = {{specsize, {16, 32}}, {vecsize, {M, 2*M}}};
    const map<size_t, pair<size_t, size_t>> M_map = {{100000000, {16, 32}}, {200000000, {12, 24}}, {400000000, {8, 16}},
                                                     {500000000, {6, 10}}, {900000000, {5, 7}}, {vecsize, {5, 5}}};
    //const map<size_t, pair<size_t, size_t>> M_map = {{50000000, {18, 32}}, {100000000, {16, 28}}, {150000000, {14, 24}},
    //                                                 {200000000, {10, 16}}, {vecsize, {6, 8}}};
    //                                                 {800000000, {4, 8}}, {900000000, {4, 6}}, {1000000000, {4, 4}}};
    //const map<size_t, pair<size_t, size_t>> M_map = {{100000000, {16, 32}},{200000000, {12, 24}},{400000000, {8, 16}},{vecsize, {6, 10}}};
                                                     //{600000000, {5, 10}},{800000000, {5, 10}},{900000000, {5, 7}},{vecsize, {5, 6}}};
    //
    const vector<size_t> elements_per_level;// = {100000000, 5000000, 250000, 12500, 625, 32};
    //const map<size_t, pair<size_t, size_t>> M_map = {{5263157, {16, 32}}, {vecsize, {M, 2*M}}};
    cout << "Loading GT:\n";
    const int gt_dim = 1000;
    unsigned int *massQA = new unsigned int[qsize * gt_dim];
    loadXvecs<unsigned int>(path_gt, massQA, qsize, gt_dim);

    cout << "Loading queries:\n";
    vtype massQ[qsize * vecdim];
    loadXvecs<vtype>(path_q, massQ, qsize, vecdim);

    SpaceInterface<dist_t> *l2space;

    switch(l2SpaceType) {
        case L2SpaceType::PQ:
            l2space = dynamic_cast<SpaceInterface<dist_t> *>(new L2SpacePQ(vecdim, M_PQ, 256));
            dynamic_cast<L2SpacePQ *>(l2space)->set_codebooks(path_codebooks);
            dynamic_cast<L2SpacePQ *>(l2space)->set_construction_tables(path_tables);
            dynamic_cast<L2SpacePQ *>(l2space)->compute_query_tables((unsigned char *) massQ, qsize);
            break;
        case L2SpaceType::Float:
            l2space = dynamic_cast<SpaceInterface<dist_t> *>(new L2Space(vecdim));
            break;
        case L2SpaceType::Int:
            l2space = dynamic_cast<SpaceInterface<dist_t> *>(new L2SpaceI(vecdim));
            break;
    }

    HierarchicalNSW<dist_t, vtype> *appr_alg;
    if (exists_test(path_info) && exists_test(path_edges)) {
        appr_alg = new HierarchicalNSW<dist_t, vtype>(l2space, path_info, path_data, path_edges);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building index:\n";
        size_t j1 = 0;
        appr_alg = new HierarchicalNSW<dist_t, vtype>(l2space, M_map, efConstruction);
        appr_alg->setElementLevels(elements_per_level, one_layer);

        StopW stopw = StopW();
        StopW stopw_full = StopW();

        cout << "Adding elements\n";
        ifstream input(path_data, ios::binary);

        vtype mass[PQ ? M_PQ : vecdim];
        readXvec<vtype>(input, mass, (PQ ? M_PQ : vecdim));
        appr_alg->addPoint((void *) (mass), j1);

        size_t report_every = 1000000;
#pragma omp parallel for num_threads(30)
        for (int i = 1; i < vecsize; i++) {
            vtype mass[PQ ? M_PQ : vecdim];
#pragma omp critical
            {
                readXvec<vtype>(input, mass, (PQ ? M_PQ : vecdim));
                if (++j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) (mass), (size_t) j1);
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->SaveInfo(path_info);
        appr_alg->SaveEdges(path_edges);
    }
    appr_alg->printListsize();
    //appr_alg->reorder_graph();
    //appr_alg->check_connectivity(massQA, qsize);
    appr_alg->printNumElements();

    vector<std::priority_queue< std::pair<dist_t, labeltype >>> answers;

    cout << "Parsing gt:\n";
    get_gt<dist_t>(massQA, qsize, answers, gt_dim);

    cout << "Loaded gt\n";
    test_vs_recall<dist_t, vtype>(massQ, qsize, *appr_alg, vecdim, answers, k, PQ);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    delete massQA;
    delete l2space;
}

void hnsw_test(const char *l2space_type,
               const char *path_codebooks, const char *path_tables, const char *path_data, const char *path_q,
               const char *path_gt, const char *path_info, const char *path_edges,
               const int k, const int vecsize, const int qsize,
               const int vecdim, const int efConstruction, const int M, bool one_layer)
{
    if (!strcmp (l2space_type, "int")) {
        _hnsw_test<int, unsigned char>(path_codebooks, path_tables, path_data, path_q,
                        path_gt, path_info, path_edges,
                        (path_codebooks && path_tables) ? L2SpaceType::PQ : L2SpaceType::Int,
                        k, vecsize, qsize, vecdim, efConstruction, M, one_layer);
    } else if (!strcmp (l2space_type, "float"))
        _hnsw_test<float, float>(path_codebooks, path_tables, path_data, path_q,
                          path_gt, path_info, path_edges,
                          L2SpaceType::Float,
                          k, vecsize, qsize, vecdim, efConstruction, M, one_layer);
}














static void ____hnsw_test(const char *path_data, const char *path_q,
                          const char *path_gt, const char *path_info, const char *path_edges,
                          const int k, const int vecsize, const int qsize,
                          const int vecdim, const int efConstruction, const int M)
{
    const int M_PQ = 16;

    cout << "Loading GT:\n";
    const int gt_dim = 1000;
    unsigned int *massQA = new unsigned int[qsize * gt_dim];
    loadXvecs<unsigned int>(path_gt, massQA, qsize, gt_dim);

    cout << "Loading queries:\n";
    float massQ[qsize * vecdim];
    loadXvecs<float>(path_q, massQ, qsize, vecdim);

    SpaceInterface<float> *l2space = new L2Space(vecdim);


    Index *index = new Index(vecdim, 1000000, M_PQ, 8);
    index->buildQuantizer(l2space, "/sata2/dbaranchuk/deep/deep_base_1m_clusters.fvecs", path_info, path_edges);


    size_t batch_size = 1000000;
    FILE *fout = fopen("/sata2/dbaranchuk/precomputed_idxs.ivecs", "wb");

    std::ifstream input(path_data, ios::binary);

    float *batch = new float[batch_size * vecdim];
    idx_t *precomputed_idx = new idx_t[batch_size];
    for (int i = 0; i < vecsize / batch_size; i++) {
        std::cout << "Batch number: " << i+1 << " of " << vecsize / batch_size << std::endl;

        readXvec(input, batch, vecdim, batch_size);
        index->assign(batch_size, batch, precomputed_idx);

        fwrite((idx_t *) &batch_size, sizeof(idx_t), 1, fout);
        fwrite(precomputed_idx, sizeof(idx_t), batch_size, fout);
    }
    delete precomputed_idx;
    delete batch;

    input.close();
    fclose(fout);

//    std::ifstream idx_input("/sata2/dbaranchuk/precomputed_idxs.ivecs", ios::binary);
//    //idx_t *precomputed_idx = new idx_t[vecsize];
//    readXvec(idx_input, precomputed_idx, batch_size, vecsize/batch_size);
//    //index->assign(path_data, precomputed_idx, vecsize);
//    idx_input.close();
//
//
//    std::ifstream learn_input("/sata2/dbaranchuk/bigann/bigann_learn.bvecs", ios::binary);
//    int nt = 1000000;
//    std::vector<float> trainvecs(nt * vecdim);
//
//    readXvec<vtype>(learn_input, trainvecs.data(), vecdim, nt);
//    index->pq = faiss::ProductQuantizer(vecdim, M_PQ, 8);
//    index->code_size = index->pq.code_size;
//    index->verbose = true;
//    index->train_residual(nt, trainvecs.data());


    //appr_alg->printListsize();
    //appr_alg->reorder_graph();
    //appr_alg->check_connectivity(massQA, qsize);
    //appr_alg->printNumElements();

    vector<std::priority_queue< std::pair<float, labeltype >>> answers;

    cout << "Parsing gt:\n";
    get_gt<float>(massQA, qsize, answers, gt_dim);

    //cout << "Loaded gt\n";
    //test_vs_recall<dist_t, vtype>(massQ, qsize, *appr_alg, vecdim, answers, k, PQ);
    //cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    delete index;
    delete massQA;
    delete l2space;
}


void ___hnsw_test(const char *l2space_type,
                  const char *path_codebooks, const char *path_tables, const char *path_data, const char *path_q,
                  const char *path_gt, const char *path_info, const char *path_edges,
                  const int k, const int vecsize, const int qsize,
                  const int vecdim, const int efConstruction, const int M, bool one_layer)
{

    ____hnsw_test(path_data, path_q, path_gt, path_info, path_edges, k, vecsize, qsize, vecdim, efConstruction, M);
}