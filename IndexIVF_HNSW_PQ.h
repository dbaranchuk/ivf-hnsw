#pragma once

#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>

#include "L2space.h"
#include "hnswalg.h"
#include <faiss/ProductQuantizer.h>
#include <faiss/utils.h>
#include <faiss/index_io.h>
#include <faiss/Heap.h>

using namespace std;

typedef unsigned int idx_t;
typedef unsigned char uint8_t;

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

namespace hnswlib {
    void read_pq(const char *path, faiss::ProductQuantizer *_pq);
    void write_pq(const char *path, faiss::ProductQuantizer *_pq);

    struct IndexIVF_HNSW_PQ
	{
		size_t d;
		size_t nc;
        size_t code_size;

        /** Query members **/
        size_t nprobe = 16;
        size_t max_codes = 10000;

		faiss::ProductQuantizer *norm_pq;
        faiss::ProductQuantizer *pq;

        std::vector < std::vector<idx_t> > ids;
        std::vector < std::vector<uint8_t> > codes;
        std::vector < std::vector<uint8_t> > norm_codes;

        std::vector < float > centroid_norms;
		HierarchicalNSW<float, float> *quantizer;

    public:
		IndexIVF_HNSW_PQ(size_t dim, size_t ncentroids,
			  size_t bytes_per_code, size_t nbits_per_idx);


		~IndexIVF_HNSW_PQ();

		void buildQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                            const char *path_info, const char *path_edges, int efSearch);


		void assign(size_t n, const float *data, idx_t *idxs);


		void add(idx_t n, float * x, const idx_t *xids, const idx_t *idx);

        struct CompareByFirst {
            constexpr bool operator()(std::pair<float, idx_t> const &a,
                                      std::pair<float, idx_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        double average_max_codes = 0;

		void search (float *x, idx_t k, float *distances, long *labels);

        void train_norm_pq(idx_t n, const float *x);

        void train_residual_pq(idx_t n, const float *x);


        void precompute_idx(size_t n, const char *path_data, const char *fo_name);


        void write(const char *path_index);

        void read(const char *path_index);

        void compute_centroid_norms();

        void compute_graphic(float *x, const idx_t *groundtruth, size_t gt_dim, size_t qsize);

        void compute_s_c();

    private:
        std::vector < float > dis_table;
        std::vector < float > norms;

        float fstdistfunc(uint8_t *code);
    public:
        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys);

		void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys);
	};



    struct ModifiedIndex
    {
        size_t d;             /** Vector Dimension **/
        size_t nc;            /** Number of Centroids **/
        size_t nsubc;         /** Number of Subcentroids **/
        size_t code_size;     /** PQ Code Size **/

        /** Query members **/
        size_t nprobe = 16;
        size_t max_codes = 10000;

        /** NEW **/
        std::vector < std::vector < idx_t > > ids;
        std::vector < std::vector < uint8_t > > codes;
        std::vector < std::vector < uint8_t > > norm_codes;
        std::vector < std::vector < idx_t > > nn_centroid_idxs;
        std::vector < std::vector < idx_t > > group_sizes;
        std::vector < float > alphas;

        faiss::ProductQuantizer *norm_pq;
        faiss::ProductQuantizer *pq;

        HierarchicalNSW<float, float> *quantizer;

        std::vector< std::vector<float> > s_c;
    public:

        ModifiedIndex(size_t dim, size_t ncentroids, size_t bytes_per_code,
                      size_t nbits_per_idx, size_t nsubcentroids);

        ~ModifiedIndex();

        void buildQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                            const char *path_info, const char *path_edges, int efSearch);


        void assign(size_t n, const float *data, idx_t *idxs);
        template<typename ptype>
        void add(const char *path_groups, const char *path_idxs);
        double average_max_codes = 0;

        void search(float *x, idx_t k, float *distances, long *labels);

        int counter_reused = 0;
        int counter_computed = 0;
        int filter_points = 0;

        void searchGF(float *x, idx_t k, float *distances, long *labels);

        void searchG(float *x, idx_t k, float *distances, long *labels);
        void write(const char *path_index);
        void read(const char *path_index);
        void train_pq(const size_t n, const float *x);

        void compute_centroid_norms();

        void compute_s_c();
    private:
        std::vector<float> q_s;

        std::vector<float> dis_table;
        std::vector<float> norms;
        std::vector<float> centroid_norms;

        float fstdistfunc(uint8_t *code);
    public:
        void compute_residuals(size_t n, float *residuals, const float *points, const float *subcentroids, const idx_t *keys);

        void reconstruct(size_t n, float *x, const float *decoded_residuals, const float *subcentroids, const idx_t *keys);

        /** NEW **/
        void sub_vectors(float *target, const float *x, const float *y);
        void compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                      const float *points, const int groupsize);

        void compute_vectors(float *target, const float *x, const float *centroid, const int n);

        float compute_alpha(const float *centroid_vectors, const float *points,
                            const float *centroid, const float *centroid_vector_norms_L2sqr,
                            const int groupsize);

        struct CompareByFirst {
            constexpr bool operator()(std::pair<float, idx_t> const &a,
                                      std::pair<float, idx_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };
    };
}
