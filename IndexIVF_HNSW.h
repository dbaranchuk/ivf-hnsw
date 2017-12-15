#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>

#include <map>
#include <set>

#include "hnswlib/hnswlib.h"
#include "utils.h"

#include <faiss/ProductQuantizer.h>
//#include <faiss/utils.h>
#include <faiss/index_io.h>
#include <faiss/Heap.h>

using namespace hnswlib;

typedef unsigned int idx_t;
typedef unsigned char uint8_t;

namespace ivfhnsw {

/** Abstract structure for an index
 *
 * Supports adding vertices and searching them.
 *
 * Currently only asymmetric queries are supported:
 * database-to-database queries are not implemented.
 */

//    struct Index {
//
//        typedef long idx_t;    ///< all indices are this type
//
//        int d;                 ///< vector dimension
//        idx_t ntotal;          ///< total nb of indexed vectors
//        bool verbose;          ///< verbosity level
//
//        /// set if the Index does not require training, or if training is done already
//        bool is_trained;
//
//        explicit Index (idx_t d = 0):
//                d(d),
//                ntotal(0),
//                verbose(false),
//                is_trained(true),
//
//        virtual ~Index () {  }
//
//
//        /** Perform training on a representative set of vectors
//         *
//         * @param n      nb of training vectors
//         * @param x      training vecors, size n * d
//         */
//        virtual void train(idx_t /*n*/, const float* /*x*/) {
//            // does nothing by default
//        }
//
//        /** Add n vectors of dimension d to the index.
//         *
//         * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
//         * This function slices the input vectors in chuncks smaller than
//         * blocksize_add and calls add_core.
//         * @param x      input matrix, size n * d
//         */
//        virtual void add (idx_t n, const float *x) = 0;
//
//        /** Same as add, but stores xids instead of sequential ids.
//         *
//         * The default implementation fails with an assertion, as it is
//         * not supported by all indexes.
//         *
//         * @param xids if non-null, ids to store for the vectors (size n)
//         */
//        virtual void add_with_ids (idx_t n, const float * x, const long *xids);
//
//        /** query n vectors of dimension d to the index.
//         *
//         * return at most k vectors. If there are not enough results for a
//         * query, the result array is padded with -1s.
//         *
//         * @param x           input vectors to search, size n * d
//         * @param labels      output labels of the NNs, size n*k
//         * @param distances   output pairwise distances, size n*k
//         */
//        virtual void search (idx_t n, const float *x, idx_t k,
//                             float *distances, idx_t *labels) const = 0;
//
//        /** query n vectors of dimension d to the index.
//         *
//         * return all vectors with distance < radius. Note that many
//         * indexes do not implement the range_search (only the k-NN search
//         * is mandatory).
//         *
//         * @param x           input vectors to search, size n * d
//         * @param radius      search radius
//         * @param result      result table
//         */
//        virtual void range_search (idx_t n, const float *x, float radius,
//                                   RangeSearchResult *result) const;
//
//        /** return the indexes of the k vectors closest to the query x.
//         *
//         * This function is identical as search but only return labels of neighbors.
//         * @param x           input vectors to search, size n * d
//         * @param labels      output labels of the NNs, size n*k
//         */
//        void assign (idx_t n, const float * x, idx_t * labels, idx_t k = 1);
//
//        /** Reconstruct a stored vector (or an approximation if lossy coding)
//         *
//         * this function may not be defined for some indexes
//         * @param key         id of the vector to reconstruct
//         * @param recons      reconstucted vector (size d)
//         */
//        virtual void reconstruct (idx_t key, float * recons) const;
//
//
//        /** Reconstruct vectors i0 to i0 + ni - 1
//         *
//         * this function may not be defined for some indexes
//         * @param recons      reconstucted vector (size ni * d)
//         */
//        virtual void reconstruct_n (idx_t i0, idx_t ni, float *recons) const;
//
//
//        /** Computes a residual vector after indexing encoding.
//         *
//         * The residual vector is the difference between a vector and the
//         * reconstruction that can be decoded from its representation in
//         * the index. The residual can be used for multiple-stage indexing
//         * methods, like IndexIVF's methods.
//         *
//         * @param x           input vector, size d
//         * @param residual    output residual vector, size d
//         * @param key         encoded index, as returned by search and assign
//         */
//        void compute_residual (const float * x, float * residual, idx_t key) const;
//
//        /** Display the actual class name and some more info */
//        void display () const;
//    };

    struct IndexIVF_HNSW 
    {
        size_t d;             /** Vector Dimension **/
        size_t nc;            /** Number of Centroids **/
        size_t code_size;     /** PQ Code Size **/

        /** Search parameters **/
        size_t nprobe = 16;
        size_t max_codes = 10000;

        /** Fine Product Quantizers **/
        faiss::ProductQuantizer *norm_pq;
        faiss::ProductQuantizer *pq;

        std::vector<std::vector<idx_t> > ids;
        std::vector<std::vector<uint8_t> > codes;
        std::vector<std::vector<uint8_t> > norm_codes;

        std::vector<float> centroid_norms;
        HierarchicalNSW<float, float> *quantizer;

    public:
        IndexIVF_HNSW(size_t dim, size_t ncentroids,
                      size_t bytes_per_code, size_t nbits_per_idx);
        
        ~IndexIVF_HNSW();

        
        /** Construct HNSW Coarse Quantizer **/
        void buildCoarseQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                                  const char *path_info, const char *path_edges,
                                  int M, int efConstruction);

        void assign(size_t n, const float *data, idx_t *idxs);

        template<typename ptype>
        void add(size_t n, const char *path_data, const char *path_precomputed_idxs);

        void add_batch(size_t n, const float *x, const idx_t *xids, const idx_t *idx);

        double average_max_codes = 0;

        void search(float *x, idx_t k, float *distances, long *labels);

        void train_pq(idx_t n, const float *x);

        template<typename ptype>
        void precompute_idx(size_t n, const char *path_data, const char *path_precomputed_idxs);

        void write(const char *path_index);

        void read(const char *path_index);

        void compute_centroid_norms();

        void compute_s_c();

    private:
        std::vector<float> query_table;
        std::vector<float> norms;

        float fstdistfunc(uint8_t *code);

        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys);

        void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys);
    };

/** TODO CPP **/











//    struct IndexIVF_HNSW_Grouping
//    {
//        size_t d;             /** Vector Dimension **/
//        size_t nc;            /** Number of Centroids **/
//        size_t nsubc;         /** Number of Subcentroids **/
//        size_t code_size;     /** PQ Code Size **/
//
//        /** Search Parameters **/
//        size_t nprobe = 16;
//        size_t max_codes = 10000;
//        bool isPruning = true;
//
//        /** NEW **/
//        std::vector<std::vector<idx_t> > ids;
//        std::vector<std::vector<uint8_t> > codes;
//        std::vector<std::vector<uint8_t> > norm_codes;
//
//        std::vector<std::vector<idx_t> > nn_centroid_idxs;
//        std::vector<std::vector<idx_t> > group_sizes;
//        std::vector<float> alphas;
//
//        /** Product Quantizers for data compression **/
//        faiss::ProductQuantizer *norm_pq;
//        faiss::ProductQuantizer *pq;
//
//        /** Coarse Quantizer based on HNSW [Y.Malkov]**/
//        HierarchicalNSW<float, float> *quantizer;
//
//        /** Distances from region centroids to their subcentroids **/
//        std::vector<std::vector<float> > centroid_subcentroid_distances;
//        std::vector<std::vector<float> > s_c;
//    public:
//
//        IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
//                               size_t nbits_per_idx, size_t nsubcentroids);
//
//        ~IndexIVF_HNSW_Grouping();
//
//        void buildCoarseQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
//                                  const char *path_info, const char *path_edges,
//                                  int M, int efConstruction);
//
//        void assign(size_t n, const float *data, idx_t *idxs);
//
//        template<typename ptype>
//        void add(const char *path_groups, const char *path_idxs);
//
//        double average_max_codes = 0;
//        int counter_reused = 0;
//        int counter_computed = 0;
//        int filter_points = 0;
//
//        void search(float *x, idx_t k, float *distances, long *labels);
//
//        void write(const char *path_index);
//
//        void read(const char *path_index);
//
//        void train_pq(const size_t n, const float *x);
//
//        void compute_centroid_norms();
//
//        void compute_s_c();
//
//    private:
//        std::vector<float> q_s;
//
//        std::vector<float> query_table;
//        std::vector<float> norms;
//        std::vector<float> centroid_norms;
//
//        float fstdistfunc(uint8_t *code);
//
//    public:
//        void compute_residuals(size_t n, float *residuals, const float *points, const float *subcentroids,
//                               const idx_t *keys);
//
//        void
//        reconstruct(size_t n, float *x, const float *decoded_residuals, const float *subcentroids, const idx_t *keys);
//
//        void sub_vectors(float *target, const float *x, const float *y);
//
//        void compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
//                                      const float *points, const int groupsize);
//
//        void compute_vectors(float *target, const float *x, const float *centroid, const int n);
//
//        float compute_alpha(const float *centroid_vectors, const float *points,
//                            const float *centroid, const float *centroid_vector_norms_L2sqr,
//                            const int groupsize);
//    };
//
///** IVF + HNSW + Grouping + Pruning **/
//    IndexIVF_HNSW_Grouping::IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
//                                 size_t nbits_per_idx, size_t nsubcentroids = 64) :
//            d(dim), nc(ncentroids), nsubc(nsubcentroids) {
//        codes.resize(nc);
//        norm_codes.resize(nc);
//        ids.resize(nc);
//        alphas.resize(nc);
//        nn_centroid_idxs.resize(nc);
//        group_sizes.resize(nc);
//
//        pq = new faiss::ProductQuantizer(d, bytes_per_code, nbits_per_idx);
//        norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);
//
//        query_table.resize(pq->ksub * pq->M);
//
//        norms.resize(65536);
//        code_size = pq->code_size;
//
//        /** Compute centroid norms **/
//        s_c.resize(nc);
//        q_s.resize(nc);
//        std::fill(q_s.begin(), q_s.end(), 0);
//    }
//
//
//    IndexIVF_HNSW_Grouping::~IndexIVF_HNSW_Grouping() {
//        delete pq;
//        delete norm_pq;
//        delete quantizer;
//    }
//
//    void IndexIVF_HNSW_Grouping::buildCoarseQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
//                                                      const char *path_info, const char *path_edges,
//                                                      int M, int efConstruction=500)
//    {
//        if (exists_test(path_info) && exists_test(path_edges)) {
//            quantizer = new HierarchicalNSW<float, float>(l2space, path_info, path_clusters, path_edges);
//            quantizer->ef_ = efConstruction;
//            return;
//        }
//        quantizer = new HierarchicalNSW<float, float>(l2space, nc, M, 2*M, efConstruction);
//        quantizer->ef_ = efConstruction;
//
//        std::cout << "Constructing quantizer\n";
//        int j1 = 0;
//        std::ifstream input(path_clusters, ios::binary);
//
//        float mass[d];
//        readXvec<float>(input, mass, d);
//        quantizer->addPoint((void *) (mass), j1);
//
//        size_t report_every = 100000;
//#pragma omp parallel for
//        for (int i = 1; i < nc; i++) {
//            float mass[d];
//#pragma omp critical
//            {
//                readXvec<float>(input, mass, d);
//                if (++j1 % report_every == 0)
//                    std::cout << j1 / (0.01 * nc) << " %\n";
//            }
//            quantizer->addPoint((void *) (mass), (size_t) j1);
//        }
//        input.close();
//        quantizer->SaveInfo(path_info);
//        quantizer->SaveEdges(path_edges);
//    }
//
//
//    void IndexIVF_HNSW_Grouping::assign(size_t n, const float *data, idx_t *idxs) {
//#pragma omp parallel for
//        for (int i = 0; i < n; i++)
//            idxs[i] = quantizer->searchKnn(const_cast<float *>(data + i * d), 1).top().second;
//    }
//
//    template<typename ptype>
//    void IndexIVF_HNSW_Grouping::add(const char *path_groups, const char *path_idxs) {
//        size_t maxM = 32;
//        StopW stopw = StopW();
//
//        double baseline_average = 0.0;
//        double modified_average = 0.0;
//
//        std::ifstream input_groups(path_groups, ios::binary);
//        std::ifstream input_idxs(path_idxs, ios::binary);
//
//        /** Vectors for construction **/
//        std::vector<std::vector<float> > centroid_vector_norms_L2sqr(nc);
//
//        /** Find NN centroids to source centroid **/
//        std::cout << "Find NN centroids to source centroids\n";
//
//#pragma omp parallel for
//        for (int i = 0; i < nc; i++) {
//            const float *centroid = (float *) quantizer->getDataByInternalId(i);
//            std::priority_queue<std::pair<float, idx_t>> nn_centroids_raw = quantizer->searchKnn((void *) centroid,
//                                                                                                 nsubc + 1);
//            /** Without heuristic **/
//            centroid_vector_norms_L2sqr[i].resize(nsubc);
//            nn_centroid_idxs[i].resize(nsubc);
//            while (nn_centroids_raw.size() > 1) {
//                centroid_vector_norms_L2sqr[i][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
//                nn_centroid_idxs[i][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
//                nn_centroids_raw.pop();
//            }
//        }
//
//        /** Adding groups to index **/
//        std::cout << "Adding groups to index\n";
//        int j1 = 0;
//#pragma omp parallel for reduction(+:baseline_average, modified_average)
//        for (int c = 0; c < nc; c++) {
//            /** Read Original vectors from Group file**/
//            idx_t centroid_num;
//            int groupsize;
//            std::vector<float> data;
//            std::vector<ptype> pdata;
//            std::vector<idx_t> idxs;
//
//#pragma omp critical
//            {
//                int check_groupsize;
//                input_groups.read((char *) &groupsize, sizeof(int));
//                input_idxs.read((char *) &check_groupsize, sizeof(int));
//                if (check_groupsize != groupsize) {
//                    std::cout << "Wrong groupsizes: " << groupsize << " vs "
//                              << check_groupsize << std::endl;
//                    exit(1);
//                }
//
//                data.resize(groupsize * d);
//                pdata.resize(groupsize * d);
//                idxs.resize(groupsize);
//
//                input_groups.read((char *) pdata.data(), groupsize * d * sizeof(ptype));
//                for (int i = 0; i < groupsize * d; i++)
//                    data[i] = (1.0) * pdata[i];
//
//                input_idxs.read((char *) idxs.data(), groupsize * sizeof(idx_t));
//
//                centroid_num = j1++;
//                if (j1 % 10000 == 0) {
//                    std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
//                              << (100. * j1) / 1000000 << "%" << std::endl;
//                }
//            }
//
//            if (groupsize == 0)
//                continue;
//
//            const float *centroid = (float *) quantizer->getDataByInternalId(centroid_num);
//            const float *centroid_vector_norms = centroid_vector_norms_L2sqr[centroid_num].data();
//            const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
//
//            /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
//            std::vector<float> centroid_vectors(nsubc * d);
//            for (int subc = 0; subc < nsubc; subc++) {
//                float *neighbor_centroid = (float *) quantizer->getDataByInternalId(nn_centroids[subc]);
//                sub_vectors(centroid_vectors.data() + subc * d, neighbor_centroid, centroid);
//            }
//
//            /** Find alphas for vectors **/
//            alphas[centroid_num] = compute_alpha(centroid_vectors.data(), data.data(), centroid,
//                                                 centroid_vector_norms, groupsize);
//
//            /** Compute final subcentroids **/
//            std::vector<float> subcentroids(nsubc * d);
//            for (int subc = 0; subc < nsubc; subc++) {
//                const float *centroid_vector = centroid_vectors.data() + subc * d;
//                float *subcentroid = subcentroids.data() + subc * d;
//                faiss::fvec_madd(d, centroid, alphas[centroid_num], centroid_vector, subcentroid);
//            }
//
//            /** Find subcentroid idx **/
//            std::vector<idx_t> subcentroid_idxs(groupsize);
//            compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), groupsize);
//
//            /** Compute Residuals **/
//            std::vector<float> residuals(groupsize * d);
//            compute_residuals(groupsize, residuals.data(), data.data(), subcentroids.data(), subcentroid_idxs.data());
//
//            /** Compute Codes **/
//            std::vector<uint8_t> xcodes(groupsize * code_size);
//            pq->compute_codes(residuals.data(), xcodes.data(), groupsize);
//
//            /** Decode Codes **/
//            std::vector<float> decoded_residuals(groupsize * d);
//            pq->decode(xcodes.data(), decoded_residuals.data(), groupsize);
//
//            /** Reconstruct Data **/
//            std::vector<float> reconstructed_x(groupsize * d);
//            reconstruct(groupsize, reconstructed_x.data(), decoded_residuals.data(),
//                        subcentroids.data(), subcentroid_idxs.data());
//
//            /** Compute norms **/
//            std::vector<float> norms(groupsize);
//            faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, groupsize);
//
//            /** Compute norm codes **/
//            std::vector<uint8_t> xnorm_codes(groupsize);
//            norm_pq->compute_codes(norms.data(), xnorm_codes.data(), groupsize);
//
//            /** Distribute codes **/
//            std::vector<std::vector<idx_t> > construction_ids(nsubc);
//            std::vector<std::vector<uint8_t> > construction_codes(nsubc);
//            std::vector<std::vector<uint8_t> > construction_norm_codes(nsubc);
//            for (int i = 0; i < groupsize; i++) {
//                const idx_t idx = idxs[i];
//                const idx_t subcentroid_idx = subcentroid_idxs[i];
//
//                construction_ids[subcentroid_idx].push_back(idx);
//                construction_norm_codes[subcentroid_idx].push_back(xnorm_codes[i]);
//                for (int j = 0; j < code_size; j++)
//                    construction_codes[subcentroid_idx].push_back(xcodes[i * code_size + j]);
//
//                const float *subcentroid = subcentroids.data() + subcentroid_idx * d;
//                const float *point = data.data() + i * d;
//                baseline_average += faiss::fvec_L2sqr(centroid, point, d);
//                modified_average += faiss::fvec_L2sqr(subcentroid, point, d);
//            }
//            /** Add codes **/
//            for (int subc = 0; subc < nsubc; subc++) {
//                idx_t subcsize = construction_norm_codes[subc].size();
//                group_sizes[centroid_num].push_back(subcsize);
//
//                for (int i = 0; i < subcsize; i++) {
//                    ids[centroid_num].push_back(construction_ids[subc][i]);
//                    for (int j = 0; j < code_size; j++)
//                        codes[centroid_num].push_back(construction_codes[subc][i * code_size + j]);
//                    norm_codes[centroid_num].push_back(construction_norm_codes[subc][i]);
//                }
//            }
//        }
//        std::cout << "[Baseline] Average Distance: " << baseline_average / 1000000000 << std::endl;
//        std::cout << "[Modified] Average Distance: " << modified_average / 1000000000 << std::endl;
//
//        input_groups.close();
//        input_idxs.close();
//
//        std::cout << "Computing centroid norms"<< std::endl;
//        compute_centroid_norms();
//    }
//
//    void IndexIVF_HNSW_Grouping::search(float *x, idx_t k, float *distances, long *labels)
//    {
//        std::vector<float> r;
//        std::vector<idx_t> subcentroid_nums;
//        subcentroid_nums.reserve(nsubc * nprobe);
//        idx_t keys[nprobe];
//        float q_c[nprobe];
//
//        /** Find NN Centroids **/
//        auto coarse = quantizer->searchKnn(x, nprobe);
//        for (int i = nprobe - 1; i >= 0; i--) {
//            auto elem = coarse.top();
//            q_c[i] = elem.first;
//            keys[i] = elem.second;
//
//            /** Add q_c to precomputed q_s **/
//            idx_t key = keys[i];
//            q_s[key] = q_c[i];
//            subcentroid_nums.push_back(key);
//
//            coarse.pop();
//        }
//
//        /** Compute Query Table **/
//        pq->compute_inner_prod_table(x, query_table.data());
//
//        /** Prepare max heap with \k answers **/
//        faiss::maxheap_heapify(k, distances, labels);
//
//        /** Pruning **/
//        double threshold = 0.0;
//        if (isPruning) {
//            int ncode = 0;
//            int normalize = 0;
//
//            r.resize(nsubc * nprobe);
//            for (int i = 0; i < nprobe; i++) {
//                idx_t centroid_num = keys[i];
//
//                if (norm_codes[centroid_num].size() == 0)
//                    continue;
//                ncode += norm_codes[centroid_num].size();
//
//                float *subr = r.data() + i * nsubc;
//                const idx_t *groupsizes = group_sizes[centroid_num].data();
//                const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
//                float alpha = alphas[centroid_num];
//
//                for (int subc = 0; subc < nsubc; subc++) {
//                    if (groupsizes[subc] == 0)
//                        continue;
//
//                    idx_t subcentroid_num = nn_centroids[subc];
//
//                    if (q_s[subcentroid_num] < 0.00001) {
//                        const float *nn_centroid = (float *) quantizer->getDataByInternalId(subcentroid_num);
//                        q_s[subcentroid_num] = faiss::fvec_L2sqr(x, nn_centroid, d);
//                        subcentroid_nums.push_back(subcentroid_num);
//                        counter_computed++;
//                    } else counter_reused++;
//
//                    subr[subc] = (1 - alpha)*(q_c[i] - alpha * s_c[centroid_num][subc]) + alpha*q_s[subcentroid_num];
//                    threshold += subr[subc];
//                    normalize++;
//                }
//                if (ncode >= 2 * max_codes)
//                    break;
//            }
//            threshold /= normalize;
//        }
//
//        int ncode = 0;
//        for (int i = 0; i < nprobe; i++) {
//            idx_t centroid_num = keys[i];
//            if (norm_codes[centroid_num].size() == 0)
//                continue;
//
//            const idx_t *groupsizes = group_sizes[centroid_num].data();
//            const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
//            float alpha = alphas[centroid_num];
//            float fst_term = (1 - alpha) * (q_c[i] - centroid_norms[centroid_num]);
//
//            const uint8_t *norm_code = norm_codes[centroid_num].data();
//            uint8_t *code = codes[centroid_num].data();
//            const idx_t *id = ids[centroid_num].data();
//
//            for (int subc = 0; subc < nsubc; subc++) {
//                idx_t groupsize = groupsizes[subc];
//                if (groupsize == 0)
//                    continue;
//
//                if (isPruning && r[i * nsubc + subc] > threshold) {
//                    code += groupsize * code_size;
//                    norm_code += groupsize;
//                    id += groupsize;
//                    filter_points += groupsize;
//                    continue;
//                }
//
//                idx_t subcentroid_num = nn_centroids[subc];
//                if (q_s[subcentroid_num] < 0.00001) {
//                    const float *nn_centroid = (float *) quantizer->getDataByInternalId(subcentroid_num);
//                    q_s[subcentroid_num] = faiss::fvec_L2sqr(x, nn_centroid, d);
//                    subcentroid_nums.push_back(subcentroid_num);
//                    counter_computed++;
//                } else counter_reused += !isPruning;
//
//                float snd_term = alpha * (q_s[subcentroid_num] - centroid_norms[subcentroid_num]);
//                norm_pq->decode(norm_code, norms.data(), groupsize);
//
//                for (int j = 0; j < groupsize; j++) {
//                    float q_r = fstdistfunc(code + j * code_size);
//                    float dist = fst_term + snd_term - 2 * q_r + norms[j];
//                    if (dist < distances[0]) {
//                        faiss::maxheap_pop(k, distances, labels);
//                        faiss::maxheap_push(k, distances, labels, dist, id[j]);
//                    }
//                }
//                /** Shift to the next group **/
//                code += groupsize * code_size;
//                norm_code += groupsize;
//                id += groupsize;
//                ncode += groupsize;
//            }
//            if (ncode >= max_codes)
//                break;
//        }
//        average_max_codes += ncode;
//
//        /** Zero subcentroids **/
//        for (idx_t subcentroid_num : subcentroid_nums)
//            q_s[subcentroid_num] = 0;
//    }
//
//    void IndexIVF_HNSW_Grouping::write(const char *path_index) {
//        FILE *fout = fopen(path_index, "wb");
//
//        fwrite(&d, sizeof(size_t), 1, fout);
//        fwrite(&nc, sizeof(size_t), 1, fout);
//        fwrite(&nsubc, sizeof(size_t), 1, fout);
//
//        idx_t size;
//        /** Save Vector Indexes per  **/
//        for (size_t i = 0; i < nc; i++) {
//            size = ids[i].size();
//            fwrite(&size, sizeof(idx_t), 1, fout);
//            fwrite(ids[i].data(), sizeof(idx_t), size, fout);
//        }
//        /** Save PQ Codes **/
//        for (int i = 0; i < nc; i++) {
//            size = codes[i].size();
//            fwrite(&size, sizeof(idx_t), 1, fout);
//            fwrite(codes[i].data(), sizeof(uint8_t), size, fout);
//        }
//        /** Save Norm Codes **/
//        for (int i = 0; i < nc; i++) {
//            size = norm_codes[i].size();
//            fwrite(&size, sizeof(idx_t), 1, fout);
//            fwrite(norm_codes[i].data(), sizeof(uint8_t), size, fout);
//        }
//        /** Save NN Centroid Indexes **/
//        for (int i = 0; i < nc; i++) {
//            size = nn_centroid_idxs[i].size();
//            fwrite(&size, sizeof(idx_t), 1, fout);
//            fwrite(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fout);
//        }
//        /** Write Group Sizes **/
//        for (int i = 0; i < nc; i++) {
//            size = group_sizes[i].size();
//            fwrite(&size, sizeof(idx_t), 1, fout);
//            fwrite(group_sizes[i].data(), sizeof(idx_t), size, fout);
//        }
//        /** Save Alphas **/
//        fwrite(alphas.data(), sizeof(float), nc, fout);
//
//        /** Save Centroid Norms **/
//        fwrite(centroid_norms.data(), sizeof(float), nc, fout);
//
//        fclose(fout);
//    }
//
//    void IndexIVF_HNSW_Grouping::read(const char *path_index) {
//        FILE *fin = fopen(path_index, "rb");
//
//        fread(&d, sizeof(size_t), 1, fin);
//        fread(&nc, sizeof(size_t), 1, fin);
//        fread(&nsubc, sizeof(size_t), 1, fin);
//
//        ids.resize(nc);
//        codes.resize(nc);
//        norm_codes.resize(nc);
//        nn_centroid_idxs.resize(nc);
//        group_sizes.resize(nc);
//
//        idx_t size;
//        /** Read Indexes **/
//        for (size_t i = 0; i < nc; i++) {
//            fread(&size, sizeof(idx_t), 1, fin);
//            ids[i].resize(size);
//            fread(ids[i].data(), sizeof(idx_t), size, fin);
//        }
//        /** Read Codes **/
//        for (size_t i = 0; i < nc; i++) {
//            fread(&size, sizeof(idx_t), 1, fin);
//            codes[i].resize(size);
//            fread(codes[i].data(), sizeof(uint8_t), size, fin);
//        }
//        /** Read Norm Codes **/
//        for (size_t i = 0; i < nc; i++) {
//            fread(&size, sizeof(idx_t), 1, fin);
//            norm_codes[i].resize(size);
//            fread(norm_codes[i].data(), sizeof(uint8_t), size, fin);
//        }
//        /** Read NN Centroid Indexes **/
//        for (size_t i = 0; i < nc; i++) {
//            fread(&size, sizeof(idx_t), 1, fin);
//            nn_centroid_idxs[i].resize(size);
//            fread(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fin);
//        }
//        /** Read Group Sizes **/
//        for (size_t i = 0; i < nc; i++) {
//            fread(&size, sizeof(idx_t), 1, fin);
//            group_sizes[i].resize(size);
//            fread(group_sizes[i].data(), sizeof(idx_t), size, fin);
//        }
//
//        /** Read Alphas **/
//        alphas.resize(nc);
//        fread(alphas.data(), sizeof(float), nc, fin);
//
//        /** Read Centroid Norms **/
//        centroid_norms.resize(nc);
//        fread(centroid_norms.data(), sizeof(float), nc, fin);
//
//        fclose(fin);
//    }
//
//    void IndexIVF_HNSW_Grouping::train_pq(const size_t n, const float *x) {
//        std::vector<float> train_subcentroids;
//        std::vector<idx_t> train_subcentroid_idxs;
//
//        std::vector<float> train_residuals;
//        std::vector<idx_t> assigned(n);
//        assign(n, x, assigned.data());
//
//        std::unordered_map<idx_t, std::vector<float>> group_map;
//
//        for (int i = 0; i < n; i++) {
//            idx_t key = assigned[i];
//            for (int j = 0; j < d; j++)
//                group_map[key].push_back(x[i * d + j]);
//        }
//
//        /** Train Residual PQ **/
//        std::cout << "Training Residual PQ codebook " << std::endl;
//        for (auto p : group_map) {
//            const idx_t centroid_num = p.first;
//            const float *centroid = (float *) quantizer->getDataByInternalId(centroid_num);
//            const vector<float> data = p.second;
//            const int groupsize = data.size() / d;
//
//            std::vector<idx_t> nn_centroids(nsubc);
//            std::vector<float> centroid_vector_norms(nsubc);
//            auto nn_centroids_raw = quantizer->searchKnn((void *) centroid, nsubc + 1);
//
//            while (nn_centroids_raw.size() > 1) {
//                centroid_vector_norms[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
//                nn_centroids[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
//                nn_centroids_raw.pop();
//            }
//
//            /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
//            std::vector<float> centroid_vectors(nsubc * d);
//            for (int i = 0; i < nsubc; i++) {
//                const float *neighbor_centroid = (float *) quantizer->getDataByInternalId(nn_centroids[i]);
//                sub_vectors(centroid_vectors.data() + i * d, neighbor_centroid, centroid);
//            }
//
//            /** Find alphas for vectors **/
//            float alpha = compute_alpha(centroid_vectors.data(), data.data(), centroid,
//                                        centroid_vector_norms.data(), groupsize);
//
//            /** Compute final subcentroids **/
//            std::vector<float> subcentroids(nsubc * d);
//            for (int subc = 0; subc < nsubc; subc++) {
//                const float *centroid_vector = centroid_vectors.data() + subc * d;
//                float *subcentroid = subcentroids.data() + subc * d;
//
//                faiss::fvec_madd(d, centroid, alpha, centroid_vector, subcentroid);
//            }
//
//            /** Find subcentroid idx **/
//            std::vector<idx_t> subcentroid_idxs(groupsize);
//            compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), groupsize);
//
//            /** Compute Residuals **/
//            std::vector<float> residuals(groupsize * d);
//            compute_residuals(groupsize, residuals.data(), data.data(), subcentroids.data(),
//                              subcentroid_idxs.data());
//
//            for (int i = 0; i < groupsize; i++) {
//                train_subcentroid_idxs.push_back(subcentroid_idxs[i]);
//                for (int j = 0; j < d; j++) {
//                    train_subcentroids.push_back(subcentroids[i * d + j]);
//                    train_residuals.push_back(residuals[i * d + j]);
//                }
//            }
//        }
//
//        printf("Training %zdx%zd product quantizer on %ld vectors in %dD\n",
//               pq->M, pq->ksub, train_residuals.size() / d, d);
//        pq->verbose = true;
//        pq->train(n, train_residuals.data());
//
//        /** Norm PQ **/
//        std::cout << "Training Norm PQ codebook " << std::endl;
//        std::vector<float> train_norms;
//        const float *residuals = train_residuals.data();
//        const float *subcentroids = train_subcentroids.data();
//        const idx_t *subcentroid_idxs = train_subcentroid_idxs.data();
//
//        for (auto p : group_map) {
//            const vector<float> data = p.second;
//            const int groupsize = data.size() / d;
//
//            /** Compute Codes **/
//            std::vector<uint8_t> xcodes(groupsize * code_size);
//            pq->compute_codes(residuals, xcodes.data(), groupsize);
//
//            /** Decode Codes **/
//            std::vector<float> decoded_residuals(groupsize * d);
//            pq->decode(xcodes.data(), decoded_residuals.data(), groupsize);
//
//            /** Reconstruct Data **/
//            std::vector<float> reconstructed_x(groupsize * d);
//            reconstruct(groupsize, reconstructed_x.data(), decoded_residuals.data(),
//                        subcentroids, subcentroid_idxs);
//
//            /** Compute norms **/
//            std::vector<float> group_norms(groupsize);
//            faiss::fvec_norms_L2sqr(group_norms.data(), reconstructed_x.data(), d, groupsize);
//
//            for (int i = 0; i < groupsize; i++)
//                train_norms.push_back(group_norms[i]);
//
//            residuals += groupsize * d;
//            subcentroids += groupsize * d;
//            subcentroid_idxs += groupsize;
//        }
//        printf("Training %zdx%zd product quantizer on %ld vectors in 1D\n", norm_pq->M, norm_pq->ksub,
//               train_norms.size());
//        norm_pq->verbose = true;
//        norm_pq->train(n, train_norms.data());
//    }
//
//    void IndexIVF_HNSW_Grouping::compute_centroid_norms()
//    {
//        centroid_norms.resize(nc);
//        #pragma omp parallel for
//        for (int i = 0; i < nc; i++) {
//            const float *centroid = (float *) quantizer->getDataByInternalId(i);
//            centroid_norms[i] = faiss::fvec_norm_L2sqr(centroid, d);
//        }
//    }
//
//    void IndexIVF_HNSW_Grouping::compute_s_c() {
//        for (int i = 0; i < nc; i++) {
//            const float *centroid = (float *) quantizer->getDataByInternalId(i);
//            s_c[i].resize(nsubc);
//            for (int subc = 0; subc < nsubc; subc++) {
//                idx_t subc_idx = nn_centroid_idxs[i][subc];
//                const float *subcentroid = (float *) quantizer->getDataByInternalId(subc_idx);
//                s_c[i][subc] = faiss::fvec_L2sqr(subcentroid, centroid, d);
//            }
//        }
//    }
//
//    float IndexIVF_HNSW_Grouping::fstdistfunc(uint8_t *code) {
//        float result = 0.;
//        int dim = code_size >> 2;
//        int m = 0;
//        for (int i = 0; i < dim; i++) {
//            result += query_table[pq->ksub * m + code[m]]; m++;
//            result += query_table[pq->ksub * m + code[m]]; m++;
//            result += query_table[pq->ksub * m + code[m]]; m++;
//            result += query_table[pq->ksub * m + code[m]]; m++;
//        }
//        return result;
//    }
//
//    void IndexIVF_HNSW_Grouping::compute_residuals(size_t n, float *residuals, const float *points, const float *subcentroids,
//                                          const idx_t *keys) {
//        //#pragma omp parallel for num_threads(16)
//        for (idx_t i = 0; i < n; i++) {
//            const float *subcentroid = subcentroids + keys[i] * d;
//            const float *point = points + i * d;
//            for (int j = 0; j < d; j++) {
//                residuals[i * d + j] = point[j] - subcentroid[j];
//            }
//        }
//    }
//
//    void IndexIVF_HNSW_Grouping::reconstruct(size_t n, float *x, const float *decoded_residuals, const float *subcentroids,
//                                    const idx_t *keys) {
////            #pragma omp parallel for num_threads(16)
//        for (idx_t i = 0; i < n; i++) {
//            const float *subcentroid = subcentroids + keys[i] * d;
//            const float *decoded_residual = decoded_residuals + i * d;
//            for (int j = 0; j < d; j++)
//                x[i * d + j] = subcentroid[j] + decoded_residual[j];
//        }
//    }
//
//    void IndexIVF_HNSW_Grouping::sub_vectors(float *target, const float *x, const float *y) {
//        for (int i = 0; i < d; i++)
//            target[i] = x[i] - y[i];
//    }
//
//    void IndexIVF_HNSW_Grouping::compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
//                                                 const float *points, const int groupsize) {
////            #pragma omp parallel for num_threads(16)
//        for (int i = 0; i < groupsize; i++) {
//            std::priority_queue<std::pair<float, idx_t>> max_heap;
//            for (int subc = 0; subc < nsubc; subc++) {
//                const float *subcentroid = subcentroids + subc * d;
//                const float *point = points + i * d;
//                float dist = faiss::fvec_L2sqr(subcentroid, point, d);
//                max_heap.emplace(std::make_pair(-dist, subc));
//            }
//            subcentroid_idxs[i] = max_heap.top().second;
//        }
//
//    }
//
//    void IndexIVF_HNSW_Grouping::compute_vectors(float *target, const float *x, const float *centroid, const int n) {
////            #pragma omp parallel for num_threads(16)
//        for (int i = 0; i < n; i++)
//            for (int j = 0; j < d; j++)
//                target[i * d + j] = x[i * d + j] - centroid[j];
//    }
//
//    float IndexIVF_HNSW_Grouping::compute_alpha(const float *centroid_vectors, const float *points,
//                                       const float *centroid, const float *centroid_vector_norms_L2sqr,
//                                       const int groupsize) {
//        int counter_positive = 0;
//        int counter_negative = 0;
//
//        float positive_numerator = 0.;
//        float positive_denominator = 0.;
//
//        float negative_numerator = 0.;
//        float negative_denominator = 0.;
//
//        float positive_alpha = 0.0;
//        float negative_alpha = 0.0;
//
//        std::vector<float> point_vectors(groupsize * d);
//        compute_vectors(point_vectors.data(), points, centroid, groupsize);
//
//        for (int i = 0; i < groupsize; i++) {
//            const float *point_vector = point_vectors.data() + i * d;
//            const float *point = points + i * d;
//
//            std::priority_queue<std::pair<float, std::pair<float, float>>> max_heap;
//
//            for (int subc = 0; subc < nsubc; subc++) {
//                const float *centroid_vector = centroid_vectors + subc * d;
//                const float centroid_vector_norm_L2sqr = centroid_vector_norms_L2sqr[subc];
//
//                float numerator = faiss::fvec_inner_product(centroid_vector, point_vector, d);
//                float denominator = centroid_vector_norm_L2sqr;
//                float alpha = numerator / denominator;
//
//                std::vector<float> subcentroid(d);
//                faiss::fvec_madd(d, centroid, alpha, centroid_vector, subcentroid.data());
//
//                float dist = faiss::fvec_L2sqr(point, subcentroid.data(), d);
//                max_heap.emplace(std::make_pair(-dist, std::make_pair(numerator, denominator)));
//            }
//            float optim_numerator = max_heap.top().second.first;
//            float optim_denominator = max_heap.top().second.second;
//            if (optim_numerator < 0) {
//                counter_negative++;
//                negative_numerator += optim_numerator;
//                negative_denominator += optim_denominator;
//                //negative_alpha += optim_numerator / optim_denominator;
//            } else {
//                counter_positive++;
//                positive_numerator += optim_numerator;
//                positive_denominator += optim_denominator;
//                //positive_alpha += optim_numerator / optim_denominator;
//            }
//        }
//        //positive_alpha /= groupsize;
//        //negative_alpha /= groupsize;
//        positive_alpha = positive_numerator / positive_denominator;
//        negative_alpha = negative_numerator / negative_denominator;
//        return (counter_positive > counter_negative) ? positive_alpha : negative_alpha;
//    }

}