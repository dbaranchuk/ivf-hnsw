#ifndef IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
#define IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H

#include "IndexIVF_HNSW.h"

namespace ivfhnsw{
    //=======================================
    // IVF_HNSW + Grouping( + Pruning) index
    //=======================================
    struct IndexIVF_HNSW_Grouping: IndexIVF_HNSW
    {
        size_t nsubc;         ///< Number of subcentroids per group
        bool do_pruning;      ///< Number of subcentroids per group

        std::vector<std::vector<idx_t> > nn_centroid_idxs; ///< indices of the nsubc nearest centroids for each centroid
        std::vector<std::vector<idx_t> > group_sizes;      ///< sizes of subgroups for each group
        std::vector<float> alphas;    ///< Coefficients that determine the location of subcentroids

    public:
        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids);

        void add_group(int centroid_num, int groupsize,
                       const float *data, const idx_t *idxs,
                       double &baseline_average, double &modified_average);


        void search(size_t k, const float *x, float *distances, long *labels);

        void write(const char *path_index);
        void read(const char *path_index);

        void train_pq(size_t n, const float *x);

        void compute_centroid_dists();

    protected:
        std::vector<float> q_s;
        std::vector<std::vector<float> > centroid_dists; /** Distances from region centroids to their subcentroids **/

    public:
        void compute_residuals(size_t n, const float *x, float *residuals,
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
