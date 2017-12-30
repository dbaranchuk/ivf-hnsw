
#ifndef IVF_HNSW_LIB_IVF_HNSW_H
#define IVF_HNSW_LIB_IVF_HNSW_H

#include <faiss/index_io.h>
#include <faiss/Heap.h>

#include "Index.h"


/** Index based on a inverted file (IVF)
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
 */
namespace ivfhnsw {
    /*************************************/
    /** Structure for an IVF-HNSW index **/
    /*************************************/
    struct IndexIVF_HNSW: Index
    {

    public:
        IndexIVF_HNSW(size_t dim, size_t ncentroids,
                      size_t bytes_per_code, size_t nbits_per_idx);

        void add_batch(size_t n, const float *x, const idx_t *xids, const idx_t *idx);

        void search(float *x, idx_t k, float *distances, long *labels);

        void train_pq(const size_t n, const float *x);

        void write(const char *path_index);
        void read(const char *path_index);

    private:
        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys);
        void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys);
    };
}

#endif // IVF_HNSW_LIB_IVF_HNSW_H