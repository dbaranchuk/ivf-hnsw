
#ifndef IVF_HNSW_LIB_IVF_HNSW_H
#define IVF_HNSW_LIB_IVF_HNSW_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>

//#include "hnswlib/hnswlib.h"
#include <hnswlib/hnswlib.h>

#include <faiss/ProductQuantizer.h>
#include <faiss/index_io.h>
#include <faiss/utils.h>
#include <faiss/Heap.h>

#include "utils.h"

typedef unsigned int idx_t;
typedef unsigned char uint8_t;

namespace ivfhnsw {

    /** Structures for an IVF-HNSW index and IVF-HNSW + Grouping (+ Pruning) index
      *
      * Supports adding vertices and searching them.
      *
      * Currently only asymmetric queries are supported:
      * database-to-database queries are not implemented.
    */
    struct IndexIVF_HNSW 
    {
        size_t d;             /** Vector Dimension **/
        size_t nc;            /** Number of Centroids **/
        size_t code_size;     /** PQ Code Size **/

        /** Search parameters **/
        size_t nprobe;
        size_t max_codes;

        /** Fine Product Quantizers **/
        faiss::ProductQuantizer *norm_pq;
        faiss::ProductQuantizer *pq;

        std::vector<std::vector<idx_t> > ids;
        std::vector<std::vector<uint8_t> > codes;
        std::vector<std::vector<uint8_t> > norm_codes;

        std::vector<float> centroid_norms;
        hnswlib::HierarchicalNSW *quantizer;

    public:
        IndexIVF_HNSW(size_t dim, size_t ncentroids,
                      size_t bytes_per_code, size_t nbits_per_idx);
        
        ~IndexIVF_HNSW();

        
        /** Construct HNSW Coarse Quantizer **/
        void buildCoarseQuantizer(const char *path_clusters,
                                  const char *path_info, const char *path_edges,
                                  int M, int efConstruction);

        void assign(size_t n, const float *data, idx_t *idxs);

        void add_batch(size_t n, const float *x, const idx_t *xids, const idx_t *idx);

        double average_max_codes = 0;

        void search(float *x, idx_t k, float *distances, long *labels);

        void train_pq(idx_t n, const float *x);

        void write(const char *path_index);

        void read(const char *path_index);

        void compute_centroid_norms();

    private:
        std::vector<float> query_table;
        std::vector<float> norms;

        float fstdistfunc(uint8_t *code);

        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys);

        void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys);
    };




    struct IndexIVF_HNSW_Grouping
    {
        size_t d;             /** Vector Dimension **/
        size_t nc;            /** Number of Centroids **/
        size_t nsubc;         /** Number of Subcentroids **/
        size_t code_size;     /** PQ Code Size **/

        /** Search Parameters **/
        size_t nprobe = 16;
        size_t max_codes = 10000;
        bool isPruning = true;

        /** NEW **/
        std::vector<std::vector<idx_t> > ids;
        std::vector<std::vector<uint8_t> > codes;
        std::vector<std::vector<uint8_t> > norm_codes;

        std::vector<std::vector<idx_t> > nn_centroid_idxs;
        std::vector<std::vector<idx_t> > group_sizes;
        std::vector<float> alphas;

        /** Product Quantizers for data compression **/
        faiss::ProductQuantizer *norm_pq;
        faiss::ProductQuantizer *pq;

        /** Coarse Quantizer based on HNSW [Y.Malkov]**/
        hnswlib::HierarchicalNSW *quantizer;

        /** Distances from region centroids to their subcentroids **/
        std::vector<std::vector<float> > centroid_subcentroid_distances;
        std::vector<std::vector<float> > s_c;
    public:

        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids);
        ~IndexIVF_HNSW_Grouping();

        void buildCoarseQuantizer(const char *path_clusters, const char *path_info,
                                  const char *path_edges, int M, int efConstruction);

        void assign(size_t n, const float *data, idx_t *idxs);

        void add_group(int centroid_num, int groupsize,
                       const float *data, const idx_t *idxs,
                       double &baseline_average, double &modified_average);

        double average_max_codes = 0;
        int counter_reused = 0;
        int counter_computed = 0;
        int filter_points = 0;

        void search(float *x, idx_t k, float *distances, long *labels);

        void write(const char *path_index);

        void read(const char *path_index);

        void train_pq(const size_t n, const float *x);

        void compute_centroid_norms();

        void compute_s_c();

    private:
        std::vector<float> q_s;

        std::vector<float> query_table;
        std::vector<float> norms;
        std::vector<float> centroid_norms;

        float fstdistfunc(uint8_t *code);

    public:
        void compute_residuals(size_t n, float *residuals, const float *points,
                               const float *subcentroids, const idx_t *keys);

        void reconstruct(size_t n, float *x, const float *decoded_residuals,
                         const float *subcentroids, const idx_t *keys);

        void sub_vectors(float *target, const float *x, const float *y);

        void compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                      const float *points, const int groupsize);

        void compute_vectors(float *target, const float *x, const float *centroid, const int n);

        float compute_alpha(const float *centroid_vectors, const float *points,
                            const float *centroid, const float *centroid_vector_norms_L2sqr,
                            const int groupsize);
    };
}

#endif // IVF_HNSW_LIB_IVF_HNSW_H