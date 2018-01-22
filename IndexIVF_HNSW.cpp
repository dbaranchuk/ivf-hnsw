#include "IndexIVF_HNSW.h"

namespace ivfhnsw {
    //=========================
    // IVF_HNSW implementation 
    //=========================
    IndexIVF_HNSW::IndexIVF_HNSW(size_t dim, size_t ncentroids, size_t bytes_per_code, size_t nbits_per_idx):
            d(dim), nc(ncentroids)
    {
        pq = new faiss::ProductQuantizer(d, bytes_per_code, nbits_per_idx);
        norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);

        code_size = pq->code_size;
        norms.resize(65536); // buffer for reconstructed base points at search time supposing that
                             // the max size of the list is less than 65536.
        precomputed_table.resize(pq->ksub * pq->M);

        codes.resize(nc);
        norm_codes.resize(nc);
        ids.resize(nc);
    }

    IndexIVF_HNSW::~IndexIVF_HNSW()
    {
        if (quantizer) delete quantizer;
        if (pq) delete pq;
        if (norm_pq) delete norm_pq;
        if (opq_matrix) delete opq_matrix;
    }

    /**
     * There has been removed parallel HNSW construction in order to make internal centroid ids equal to external ones.
     * Construction time is still acceptable: ~5 minutes for 1 million 96-d vectors on Intel Xeon E5-2650 V2 2.60GHz.
     */
    // TODO: paralyze in the right way
    void IndexIVF_HNSW::build_quantizer(const char *path_data, const char *path_info,
                                        const char *path_edges, int M, int efConstruction)
    {
        if (exists(path_info) && exists(path_edges)) {
            quantizer = new hnswlib::HierarchicalNSW(path_info, path_data, path_edges);
            quantizer->efSearch = efConstruction;
            return;
        }
        quantizer = new hnswlib::HierarchicalNSW(d, nc, M, 2 * M, efConstruction);

        std::cout << "Constructing quantizer\n";
        std::ifstream input(path_data, ios::binary);

        size_t report_every = 100000;
        for (int i = 0; i < nc; i++) {
            float mass[d];
            readXvec<float>(input, mass, d);
            if (i % report_every == 0)
                std::cout << i / (0.01 * nc) << " %\n";
            quantizer->addPoint(mass);
        }
        input.close();
        quantizer->SaveInfo(path_info);
        quantizer->SaveEdges(path_edges);
    }


    void IndexIVF_HNSW::assign(size_t n, const float *x, idx_t *labels, size_t k) {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
            labels[i] = quantizer->searchKnn(const_cast<float *>(x + i * d), k).top().second;
    }

    void IndexIVF_HNSW::add_batch(size_t n, const float *x, const idx_t *xids, const idx_t *precomputed_idx)
    {
        const idx_t *idx;
        // Check whether idxs are precomputed. If not, assign x
        if (precomputed_idx)
            idx = precomputed_idx;
        else {
            idx = new idx_t[n];
            assign(n, x, const_cast<idx_t *>(idx));
        }
        // Compute residuals for original vectors
        std::vector<float> residuals(n * d);
        compute_residuals(n, x, residuals.data(), idx);

        // If do_opq, rotate residuals
        if (do_opq){
            //std::vector<float> copy_residuals(n * d);
            //memcpy(copy_residuals.data(), residuals.data(), n * d * sizeof(float));
            opq_matrix->apply_noalloc(n, residuals.data(), residuals.data());
        }

        // Encode residuals
        std::vector <uint8_t> xcodes(n * code_size);
        pq->compute_codes(residuals.data(), xcodes.data(), n);

        // Decode residuals
        std::vector<float> decoded_residuals(n * d);
        pq->decode(xcodes.data(), decoded_residuals.data(), n);

        // Reverse rotation
        if (do_opq){
            //std::vector<float> copy_decoded_residuals(n * d);
            //memcpy(copy_decoded_residuals.data(), decoded_residuals.data(), n * d * sizeof(float));
            opq_matrix->transform_transpose(n, decoded_residuals.data(), decoded_residuals.data());
        }

        // Reconstruct original vectors 
        std::vector<float> reconstructed_x(n * d);
        reconstruct(n, reconstructed_x.data(), decoded_residuals.data(), idx);

        // Compute l2 square norms of reconstructed vectors
        std::vector<float> norms(n);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, n);

        // Encode norms
        std::vector <uint8_t> xnorm_codes(n);
        norm_pq->compute_codes(norms.data(), xnorm_codes.data(), n);

        // Add vector indices and PQ codes for residuals and norms to Index
        for (size_t i = 0; i < n; i++) {
            idx_t key = idx[i];
            idx_t id = xids[i];
            ids[key].push_back(id);
            uint8_t *code = xcodes.data() + i * code_size;
            for (size_t j = 0; j < code_size; j++)
                codes[key].push_back(code[j]);

            norm_codes[key].push_back(xnorm_codes[i]);
        }
        
        // Free memory, if it is allocated 
        if (idx != precomputed_idx)
            delete idx;
    }

    /** Search procedure
      *
      * During IVF-HNSW-PQ search we compute
      *
      *     d = || x - y_C - y_R ||^2
      *
      * where x is the query vector, y_C the coarse centroid, y_R the
      * refined PQ centroid. The expression can be decomposed as:
      *
      *    d = || x - y_C ||^2 - || y_C ||^2 + || y_C + y_R ||^2 - 2 * (x|y_R)
      *        -----------------------------   -----------------   -----------
      *                     term 1                   term 2           term 3
      *
      * We use the following decomposition:
      * - term 1 is the distance to the coarse centroid, that is computed
      *   during the 1st stage search in the HNSW graph, minus the norm of the coarse centroid
      * - term 2 is the L2 norm of the reconstructed base point, that is computed at construction time, quantized
      *   using separately trained product quantizer for such norms and stored along with the residual PQ codes.
      * - term 3 is the classical non-residual distance table.
      *
      * Norms of centroids are precomputed and saved without compression, as their memory consumption is negligible.
      * If it is necessary, the norms can be added to the term 3 and compressed to byte together. We do not think that
      * it will lead to considerable decrease in accuracy.
      *
      * Since y_R defined by a product quantizer, it is split across
      * sub-vectors and stored separately for each subvector.
      *
    */
    void IndexIVF_HNSW::search(size_t k, const float *x, float *distances, long *labels)
    {
        float query_centroid_dists[nprobe]; // Distances to the coarse centroids.
        idx_t centroid_idxs[nprobe];        // Indices of the nearest coarse centroids

        // For correct search using OPQ rotate a query
        const float *query = (do_opq) ? opq_matrix->apply(1, x) : x;

        // Find the nearest coarse centroids to the query
        auto coarse = quantizer->searchKnn(query, nprobe);
        for (int i = nprobe - 1; i >= 0; i--) {
            query_centroid_dists[i] = coarse.top().first;
            centroid_idxs[i] = coarse.top().second;
            coarse.pop();
        }

        // Precompute table
        pq->compute_inner_prod_table(query, precomputed_table.data());

        // Prepare max heap with k answers
        faiss::maxheap_heapify(k, distances, labels);

        int ncode = 0;
        for (int i = 0; i < nprobe; i++) {
            idx_t centroid_idx = centroid_idxs[i];
            int group_size = norm_codes[centroid_idx].size();
            if (group_size == 0)
                continue;

            const uint8_t *code = codes[centroid_idx].data();
            const uint8_t *norm_code = norm_codes[centroid_idx].data();
            const idx_t *id = ids[centroid_idx].data();
            float term1 = query_centroid_dists[i] - centroid_norms[centroid_idx];

            // Decode the norms of each vector in the list
            norm_pq->decode(norm_code, norms.data(), group_size);

            for (int j = 0; j < group_size; j++) {
                float term3 = 2 * pq_L2sqr(code + j * code_size);
                float dist = term1 + norms[j] - term3; //term2 = norms[j]
                if (dist < distances[0]) {
                    faiss::maxheap_pop(k, distances, labels);
                    faiss::maxheap_push(k, distances, labels, dist, id[j]);
                }
            }
            ncode += group_size;
            if (ncode >= max_codes)
                break;
        }
        if (do_opq)
            delete const_cast<float *>(query);
    }


    void IndexIVF_HNSW::train_pq(size_t n, const float *x)
    {
        // Assign train vectors 
        std::vector <idx_t> assigned(n);
        assign(n, x, assigned.data());

        // Compute residuals for original vectors
        std::vector<float> residuals(n * d);
        compute_residuals(n, x, residuals.data(), assigned.data());

        // Train OPQ rotation matrix and rotate residuals
        if (do_opq){
            faiss::OPQMatrix *matrix = new faiss::OPQMatrix(d, pq->M);

            std::cout << "Training OPQ Matrix" << std::endl;
            matrix->verbose = true;
            matrix->max_train_points = n;
            matrix->niter = 70;
            matrix->train(n, residuals.data());
            opq_matrix = matrix;
            opq_matrix->apply_noalloc(n, residuals.data(), residuals.data());
        }

        // Train residual PQ
        printf("Training %zdx%zd product quantizer on %ld vectors in %dD\n", pq->M, pq->ksub, n, d);
        pq->verbose = true;
        pq->train(n, residuals.data());

        // Encode residuals
        std::vector <uint8_t> xcodes(n * code_size);
        pq->compute_codes(residuals.data(), xcodes.data(), n);

        // Decode residuals
        std::vector<float> decoded_residuals(n * d);
        pq->decode(xcodes.data(), decoded_residuals.data(), n);

        // Reverse rotation
        if (do_opq)
            opq_matrix->transform_transpose(n, decoded_residuals.data(), decoded_residuals.data());

        // Reconstruct original vectors 
        std::vector<float> reconstructed_x(n * d);
        reconstruct(n, reconstructed_x.data(), decoded_residuals.data(), assigned.data());

        // Compute l2 square norms of reconstructed vectors
        std::vector<float> norms(n);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, n);

        // Train norm PQ
        printf("Training %zdx%zd product quantizer on %ld vectors in %dD\n", norm_pq->M, norm_pq->ksub, n, d);
        norm_pq->verbose = true;
        norm_pq->train(n, norms.data());
    }

    // Write index 
    void IndexIVF_HNSW::write(const char *path_index) {
        FILE *fout = fopen(path_index, "wb");

        fwrite(&d, sizeof(size_t), 1, fout);
        fwrite(&nc, sizeof(size_t), 1, fout);
        fwrite(&nprobe, sizeof(size_t), 1, fout);
        fwrite(&max_codes, sizeof(size_t), 1, fout);

        size_t size;
        // Save vector indices
        for (size_t i = 0; i < nc; i++) {
            size = ids[i].size();
            fwrite(&size, sizeof(size_t), 1, fout);
            fwrite(ids[i].data(), sizeof(idx_t), size, fout);
        }

        // Save PQ codes
        for (int i = 0; i < nc; i++) {
            size = codes[i].size();
            fwrite(&size, sizeof(size_t), 1, fout);
            fwrite(codes[i].data(), sizeof(uint8_t), size, fout);
        }

        // Save norm PQ codes
        for (int i = 0; i < nc; i++) {
            size = norm_codes[i].size();
            fwrite(&size, sizeof(size_t), 1, fout);
            fwrite(norm_codes[i].data(), sizeof(uint8_t), size, fout);
        }

        // Save centroid norms 
        fwrite(centroid_norms.data(), sizeof(float), nc, fout);
        fclose(fout);
    }

    // Read index 
    void IndexIVF_HNSW::read(const char *path_index)
    {
        FILE *fin = fopen(path_index, "rb");

        fread(&d, sizeof(size_t), 1, fin);
        fread(&nc, sizeof(size_t), 1, fin);
        fread(&nprobe, sizeof(size_t), 1, fin);
        fread(&max_codes, sizeof(size_t), 1, fin);

        size_t size;
        // Read vector indices
        ids = std::vector < std::vector < idx_t >> (nc);
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(size_t), 1, fin);
            ids[i].resize(size);
            fread(ids[i].data(), sizeof(idx_t), size, fin);
        }

        // Read PQ codes
        codes = std::vector<std::vector<uint8_t>> (nc);
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(size_t), 1, fin);
            codes[i].resize(size);
            fread(codes[i].data(), sizeof(uint8_t), size, fin);
        }

        // Read norm PQ codes
        norm_codes = std::vector<std::vector<uint8_t>> (nc);
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(size_t), 1, fin);
            norm_codes[i].resize(size);
            fread(norm_codes[i].data(), sizeof(uint8_t), size, fin);
        }

        // Read centroid norms 
        centroid_norms.resize(nc);
        fread(centroid_norms.data(), sizeof(float), nc, fin);
        fclose(fin);
    }

    void IndexIVF_HNSW::compute_centroid_norms()
    {
        centroid_norms.resize(nc);
        for (int i = 0; i < nc; i++) {
            const float *centroid = quantizer->getDataByInternalId(i);
            centroid_norms[i] = faiss::fvec_norm_L2sqr(centroid, d);
        }
    }

    void IndexIVF_HNSW::rotate_quantizer() {
        if (!do_opq){
            printf("OPQ encoding is turned off");
            abort();
        }
        std::vector<float> copy_centroid(d);
        for (int i = 0; i < nc; i++){
            float *centroid = quantizer->getDataByInternalId(i);
            memcpy(copy_centroid.data(), centroid, d * sizeof(float));
            opq_matrix->apply_noalloc(1, copy_centroid.data(), centroid);
        }
    }

    float IndexIVF_HNSW::pq_L2sqr(const uint8_t *code)
    {
        float result = 0.;
        int dim = code_size >> 2;
        int m = 0;
        for (int i = 0; i < dim; i++) {
            result += precomputed_table[pq->ksub * m + code[m]]; m++;
            result += precomputed_table[pq->ksub * m + code[m]]; m++;
            result += precomputed_table[pq->ksub * m + code[m]]; m++;
            result += precomputed_table[pq->ksub * m + code[m]]; m++;
        }
        return result;
    }

    // Private 
    void IndexIVF_HNSW::reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys)
    {
        for (idx_t i = 0; i < n; i++) {
            float *centroid = quantizer->getDataByInternalId(keys[i]);
            faiss::fvec_madd(d, decoded_residuals + i*d, 1., centroid, x + i*d);
        }
    }

    void IndexIVF_HNSW::compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys)
    {
        for (idx_t i = 0; i < n; i++) {
            float *centroid = quantizer->getDataByInternalId(keys[i]);
            faiss::fvec_madd(d, x + i*d, -1., centroid, residuals + i*d);
        }
    }
}