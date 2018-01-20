#ifndef IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
#define IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H

#include "IndexIVF_HNSW.h"

namespace ivfhnsw{
    //=======================================
    // IVF_HNSW + Grouping( + Pruning) index
    //=======================================
    struct IndexIVF_HNSW_Grouping: IndexIVF_HNSW
    {
        size_t nsubc;         ///< Number of sub-centroids per group
        bool do_pruning;      ///< Turn on/off pruning

        std::vector<std::vector<idx_t> > nn_centroid_idxs;    ///< Indices of the <nsubc> nearest centroids for each centroid
        std::vector<std::vector<int> > subgroup_sizes;        ///< Sizes of sub-groups for each group
        std::vector<float> alphas;    ///< Coefficients that determine the location of sub-centroids

        float global_numerator = 0.0;
        float global_denominator = 0.0;

    public:
        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                               size_t nbits_per_idx, size_t nsubcentroids);

        /** Add <group_size> vectors of dimension <d> from the <group_idx>-th group to the index.
          *
          * @param group_idx         index of the group
          * @param group_size        number of base vectors in the group
          * @param x                 base vectors to add (size: group_size * d)
          * @param ids               ids to store for the vectors (size: groups_size)
          * TODO: remove last two parameters
        */
        void add_group(int group_idx, int group_size,
                       const float *x, const idx_t *ids,
                       double &baseline_average, double &modified_average);

        void search(size_t k, const float *x, float *distances, long *labels);

        void write(const char *path_index);
        void read(const char *path_index);

        void train_pq(size_t n, const float *x);

        /// Compute distances between the group centroid and its <subc> nearest neighbors in the HNSW graph
        void compute_inter_centroid_dists();

    protected:
        /// Distances to the coarse centroids. Used for distance computation between a query and base points
        std::vector<float> query_centroid_dists;

        /// Distances between coarse centroids and their sub-centroids
        std::vector<std::vector<float>> inter_centroid_dists;

    private:
        void compute_residuals(size_t n, const float *x, float *residuals,
                               const float *subcentroids, const idx_t *keys);

        void reconstruct(size_t n, float *x, const float *decoded_residuals,
                         const float *subcentroids, const idx_t *keys);

        void compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                      const float *points, int group_size);

        float compute_alpha(const float *centroid_vectors, const float *points,
                            const float *centroid, const float *centroid_vector_norms_L2sqr, int group_size);
    };
}
#endif //IVF_HNSW_LIB_INDEXIVF_HNSW_GROUPING_H
