#ifndef IVF_HNSW_LIB_IVF_HNSW_H
#define IVF_HNSW_LIB_IVF_HNSW_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_map>

#include <faiss/index_io.h>
#include <faiss/Heap.h>
#include <faiss/ProductQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/FaissAssert.h>
#include <faiss/utils.h>

#include <hnswlib/hnswalg.h>
#include "utils.h"

namespace ivfhnsw {
    /** Index based on a inverted file (IVF) with Product Quantizer encoding.
      *
      * In the inverted file, the quantizer (an HNSW instance) provides a
      * quantization index for each vector to be added. The quantization
      * index maps to a list (aka inverted list or posting list), where the
      * id of the vector is then stored.
      *
      * At search time, the vector to be searched is also quantized, and
      * only the list corresponding to the quantization index is
      * searched. This speeds up the search by making it
      * non-exhaustive. This can be relaxed using multi-probe search: a few
      * (nprobe) quantization indices are selected and several inverted
      * lists are visited.
      *
      * Supports HNSW quantizer construction, PQ training, adding vertices,
      * serialization and searching.
      *
      * Each residual vector is encoded as a product quantizer code.
      *
      * Currently only asymmetric queries are supported:
      * database-to-database queries are not implemented.
    */
    struct IndexIVF_HNSW
    {
        typedef unsigned char uint8_t;  ///< all codes are this type
        typedef unsigned int idx_t;     ///< all indices are this type

        size_t d;             ///< Vector dimension
        size_t nc;            ///< Number of centroids
        size_t code_size;     ///< Code size per vector in bytes

        hnswlib::HierarchicalNSW *quantizer; ///< Quantizer that maps vectors to inverted lists (HNSW [Y.Malkov])

        //
        std::vector<idx_t> pq_idxs;
        std::vector<faiss::ProductQuantizer *> pqs;
        size_t npq;

        faiss::ProductQuantizer *norm_pq;    ///< Produces the norm codes of reconstructed base vectors

        size_t nprobe;        ///< Number of probes at search time
        size_t max_codes;     ///< Max number of codes to visit to do a query

        std::vector<std::vector<idx_t> > ids;           ///< Inverted lists for indexes
        std::vector<std::vector<uint8_t> > codes;       ///< PQ codes of residuals
        std::vector<std::vector<uint8_t> > norm_codes;  ///< PQ codes of norms of reconstructed base vectors

    protected:
        std::vector<float> norms;           ///< L2 square norms of reconstructed base vectors
        std::vector<float> centroid_norms;  ///< L2 square norms of coarse centroids

    public:
        explicit IndexIVF_HNSW(size_t dim, size_t ncentroids, size_t bytes_per_code, size_t nbits_per_idx);
        virtual ~IndexIVF_HNSW();

        /** Construct from stretch or load the existing quantizer (HNSW) instance
          *
          * if all files exist, quantizer will be loaded, else HNSW will be constructed
          * @param path_data           path to input vectors
          * @param path_info           path to parameters for HNSW
          * @param path_edges          path to edges for HNSW
          * @param M                   min number of edges per point, default: 16
          * @param efConstruction      max number of candidate vertices in queue to observe, default: 500
        */
        void build_quantizer(const char *path_data, const char *path_info, const char *path_edges,
                            int M=16, int efConstruction = 500);

        /** Return the indices of the k HNSW vertices closest to the query x.
          *
          * @param n           number of input vectors
          * @param x           query vectors, size n * d
          * @param labels      output labels of the nearest neighbours, size n * k
          * @param k           number of the closest HNSW vertices to the query x
        */
        void assign (size_t n, const float *x, idx_t *labels, size_t k = 1);

        /** Query n vectors of dimension d to the index.
         *
         * Return at most k vectors. If there are not enough results for a
         * query, the result array is padded with -1s.
         *
         * @param k           number of the closest vertices to search
         * @param x           query vectors, size n * d
         * @param distances   output pairwise distances, size n * k
         * @param labels      output labels of the nearest neighbours, size n * k
         */
        virtual void search(size_t k, const float *x, float *distances, long *labels);

        /** Add n vectors of dimension d to the index.
          *
          * @param n                 number of base vectors in a batch
          * @param x                 base vectors to add, size n * d
          * @param xids              ids to store for the vectors (size n)
          * @param precomputed_idx   if non-null, assigned idxs to store for the vectors (size n)
        */
        virtual void add_batch(size_t n, const float *x, const idx_t *xids, const idx_t *precomputed_idx = nullptr);

        /** Train product quantizers
          *
          * @param n     number of training vectors of dimension d
          * @param x     learn vectors, size n * d
        */
        virtual void train_pq(size_t n, const float *x);

        /// Write index to the path
        virtual void write(const char *path);

        /// Read index from the path
        virtual void read(const char *path);

        /// Compute norms of the HNSW vertices
        void compute_centroid_norms();

    protected:
        /// Size pq.M * pq.ksub
        std::vector<float> precomputed_table;

        /// L2 sqr distance function for PQ codes
        float pq_L2sqr(const uint8_t *code);

    private:
        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys);
        void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys);
    };
}
#endif //IVF_HNSW_LIB_INDEX_HNSW_H
