//
// Created by dbaranchuk on 23.12.17.
//

#ifndef IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
#define IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>

#include <faiss/index_io.h>
#include <faiss/Heap.h>

#include "Index.h"

namespace ivfhnsw{
    /*********************************************************/
    /** Structure for IVF_HNSW + Grouping( + Pruning) index **/
    /*********************************************************/
    struct IndexIVF_HNSW_Grouping: Index
    {
        size_t nsubc;         /** Number of Subcentroids **/
        bool isPruning = true;

        /** NEW **/
        std::vector<std::vector<idx_t> > ids;
        std::vector<std::vector<uint8_t> > codes;
        std::vector<std::vector<uint8_t> > norm_codes;

        std::vector<std::vector<idx_t> > nn_centroid_idxs;
        std::vector<std::vector<idx_t> > group_sizes;
        std::vector<float> alphas;

    public:

        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids);

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

        void compute_centroid_dists();

    private:
        std::vector<float> q_s;
        std::vector<float> norms;
        std::vector<float> centroid_norms;               /** Region centroids L2 square norms **/
        std::vector<std::vector<float> > centroid_dists; /** Distances from region centroids to their subcentroids **/

    public:
        void compute_residuals(size_t n, float *residuals, const float *points,
                               const float *subcentroids, const idx_t *keys);

        void reconstruct(size_t n, float *x, const float *decoded_residuals,
                         const float *subcentroids, const idx_t *keys);

        void compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                      const float *points, const int groupsize);

        float compute_alpha(const float *centroid_vectors, const float *points,
                            const float *centroid, const float *centroid_vector_norms_L2sqr,
                            const int groupsize);
    };
}
#endif //IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
