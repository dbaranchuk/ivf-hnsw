
#ifndef IVF_HNSW_LIB_IVF_HNSW_H
#define IVF_HNSW_LIB_IVF_HNSW_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>

#include <faiss/ProductQuantizer.h>
#include <faiss/index_io.h>
#include <faiss/utils.h>
#include <faiss/Heap.h>

#include "Index.h"

namespace ivfhnsw {
    /*************************************/
    /** Structure for an IVF-HNSW index **/
    /*************************************/
    struct IndexIVF_HNSW: Index
    {
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

    public:
        IndexIVF_HNSW(size_t dim, size_t ncentroids,
                      size_t bytes_per_code, size_t nbits_per_idx);
        
        ~IndexIVF_HNSW();

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

        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys);

        void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys);
    };
}

#endif // IVF_HNSW_LIB_IVF_HNSW_H