#include "IndexIVF_HNSW_Grouping.h"

namespace ivfhnsw
{
    //================================================
    // IVF_HNSW + grouping( + pruning) implementation
    //================================================
    IndexIVF_HNSW_Grouping::IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                                                   size_t nbits_per_idx, size_t nsubcentroids = 64):
           IndexIVF_HNSW(dim, ncentroids, bytes_per_code, nbits_per_idx), nsubc(nsubcentroids)
    {
        alphas.resize(nc);
        nn_centroid_idxs.resize(nc);
        subgroup_sizes.resize(nc);

        query_centroid_dists.resize(nc);
        std::fill(query_centroid_dists.begin(), query_centroid_dists.end(), 0);
    }

    void IndexIVF_HNSW_Grouping::add_group(int centroid_idx, int group_size,
                                           const float *data, const idx_t *idxs,
                                           double &baseline_average, double &modified_average)
    {
        // Find NN centroids to source centroid 
        const float *centroid = quantizer->getDataByInternalId(centroid_idx);
        std::priority_queue<std::pair<float, idx_t>> nn_centroids_raw = quantizer->searchKnn(centroid, nsubc + 1);

        std::vector<float> centroid_vector_norms_L2sqr(nsubc);
        nn_centroid_idxs[centroid_idx].resize(nsubc);
        while (nn_centroids_raw.size() > 1) {
            centroid_vector_norms_L2sqr[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
            nn_centroid_idxs[centroid_idx][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
            nn_centroids_raw.pop();
        }
        if (group_size == 0)
            return;

        const float *centroid_vector_norms = centroid_vector_norms_L2sqr.data();
        const idx_t *nn_centroids = nn_centroid_idxs[centroid_idx].data();

        // Compute centroid-neighbor_centroid and centroid-group_point vectors
        std::vector<float> centroid_vectors(nsubc * d);
        for (int subc = 0; subc < nsubc; subc++) {
            float *neighbor_centroid = quantizer->getDataByInternalId(nn_centroids[subc]);
            faiss::fvec_madd(d, neighbor_centroid, -1., centroid, centroid_vectors.data() + subc * d);
        }

        // Compute alpha for group vectors
        //alphas[centroid_idx] = compute_alpha(centroid_vectors.data(), data, centroid,
        //                                     centroid_vector_norms, group_size);

        alphas[centroid_idx] = 0.39817;// DEEP: 0.383882;

        // Compute final subcentroids
        std::vector<float> subcentroids(nsubc * d);
        for (int subc = 0; subc < nsubc; subc++) {
            const float *centroid_vector = centroid_vectors.data() + subc * d;
            float *subcentroid = subcentroids.data() + subc * d;
            faiss::fvec_madd(d, centroid, alphas[centroid_idx], centroid_vector, subcentroid);
        }

        // Find subcentroid idx
        std::vector<idx_t> subcentroid_idxs(group_size);
        compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data, group_size);

        // Compute residuals
        std::vector<float> residuals(group_size * d);
        compute_residuals(group_size, data, residuals.data(), subcentroids.data(), subcentroid_idxs.data());

        // Compute codes
        std::vector<uint8_t> xcodes(group_size * code_size);
        pq->compute_codes(residuals.data(), xcodes.data(), group_size);

        // Decode codes
        std::vector<float> decoded_residuals(group_size * d);
        pq->decode(xcodes.data(), decoded_residuals.data(), group_size);

        // Reconstruct data
        std::vector<float> reconstructed_x(group_size * d);
        reconstruct(group_size, reconstructed_x.data(), decoded_residuals.data(),
                    subcentroids.data(), subcentroid_idxs.data());

        // Compute norms 
        std::vector<float> norms(group_size);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, group_size);

        // Compute norm codes
        std::vector<uint8_t> xnorm_codes(group_size);
        norm_pq->compute_codes(norms.data(), xnorm_codes.data(), group_size);

        // Distribute codes
        std::vector<std::vector<idx_t> > construction_ids(nsubc);
        std::vector<std::vector<uint8_t> > construction_codes(nsubc);
        std::vector<std::vector<uint8_t> > construction_norm_codes(nsubc);
        for (int i = 0; i < group_size; i++) {
            idx_t idx = idxs[i];
            idx_t subcentroid_idx = subcentroid_idxs[i];

            construction_ids[subcentroid_idx].push_back(idx);
            construction_norm_codes[subcentroid_idx].push_back(xnorm_codes[i]);
            for (int j = 0; j < code_size; j++)
                construction_codes[subcentroid_idx].push_back(xcodes[i * code_size + j]);

            const float *subcentroid = subcentroids.data() + subcentroid_idx * d;
            const float *point = data + i * d;
            baseline_average += fvec_L2sqr(centroid, point, d);
            modified_average += fvec_L2sqr(subcentroid, point, d);
        }
        // Add codes to the index
        for (int subc = 0; subc < nsubc; subc++) {
            idx_t subgroup_size = construction_norm_codes[subc].size();
            subgroup_sizes[centroid_idx].push_back(subgroup_size);

            for (int i = 0; i < subgroup_size; i++) {
                ids[centroid_idx].push_back(construction_ids[subc][i]);
                for (int j = 0; j < code_size; j++)
                    codes[centroid_idx].push_back(construction_codes[subc][i * code_size + j]);
                norm_codes[centroid_idx].push_back(construction_norm_codes[subc][i]);
            }
        }
    }

    /** Search procedure
      *
      * During the IVF-HNSW-PQ + Grouping search we compute
      *
      *  d = || x - y_S - y_R ||^2
      *
      * where x is the query vector, y_S the coarse sub-centroid, y_R the
      * refined PQ centroid. The expression can be decomposed as:
      *
      *  d = (1 - α) * (|| x - y_C ||^2 - || y_C ||^2) + α * (|| x - y_N ||^2 - || y_N ||^2) + || y_S + y_R ||^2 - 2 * (x|y_R)
      *      -----------------------------------------   -----------------------------------   -----------------   -----------
      *                         term 1                                 term 2                        term 3          term 4
      *
      * We use the following decomposition:
      * - term 1 is the distance to the coarse centroid, that is computed
      *   during the 1st stage search in the HNSW graph, minus the norm of the coarse centroid.
      * - term 2 is the distance to y_N one of the <subc> nearest centroids,
      *   that is used for the sub-centroid computation, minus the norm of this centroid.
      * - term 3 is the L2 norm of the reconstructed base point, that is computed at construction time, quantized
      *   using separately trained product quantizer for such norms and stored along with the residual PQ codes.
      * - term 4 is the classical non-residual distance table.
      *
      * Norms of centroids are precomputed and saved without compression, as their memory consumption is negligible.
      * If it is necessary, the norms can be added to the term 3 and compressed to byte together. We do not think that
      * it will lead to considerable decrease in accuracy.
      *
      * Since y_R defined by a product quantizer, it is split across
      * sub-vectors and stored separately for each sub-vector.
      *
    */
    void IndexIVF_HNSW_Grouping::search(size_t k, const float *x, float *distances, long *labels)
    {
        std::vector<float> r;
        // Indices of coarse centroids, which distances to the query are computed during the search time
        std::vector<idx_t> used_centroid_idxs;
        used_centroid_idxs.reserve(nsubc * nprobe);
        idx_t centroid_idxs[nprobe]; // Indices of the nearest coarse centroids

        // Find the nearest coarse centroids to the query
        auto coarse = quantizer->searchKnn(x, nprobe);
        for (int i = nprobe - 1; i >= 0; i--) {
            idx_t centroid_idx = coarse.top().second;
            centroid_idxs[i] = centroid_idx;
            query_centroid_dists[centroid_idx] = coarse.top().first;
            used_centroid_idxs.push_back(centroid_idx);
            coarse.pop();
        }

        // Computing threshold for pruning
        double threshold = 0.0;
        if (do_pruning) {
            int ncode = 0;
            int nsubgroups = 0;

            r.resize(nsubc * nprobe);
            for (int i = 0; i < nprobe; i++) {
                idx_t centroid_idx = centroid_idxs[i];
                int group_size = norm_codes[centroid_idx].size();
                if (group_size == 0)
                    continue;

                float *subr = r.data() + i*nsubc;
                float alpha = alphas[centroid_idx];
                float q_c = query_centroid_dists[centroid_idx];

                for (int subc = 0; subc < nsubc; subc++) {
                    if (subgroup_sizes[centroid_idx][subc] == 0)
                        continue;

                    idx_t nn_centroid_idx = nn_centroid_idxs[centroid_idx][subc];
                    // Compute the distance to the coarse centroid if it is not computed
                    if (query_centroid_dists[nn_centroid_idx] < EPS) {
                        const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idx);
                        query_centroid_dists[nn_centroid_idx] = fvec_L2sqr(x, nn_centroid, d);
                        used_centroid_idxs.push_back(nn_centroid_idx);
                    }
                    // TODO: сделать красиво
                    subr[subc] = (1 - alpha) * (q_c - alpha * inter_centroid_dists[centroid_idx][subc])
                                 + alpha * query_centroid_dists[nn_centroid_idx];
                    threshold += subr[subc];
                    nsubgroups++;
                }
                ncode += group_size;
                if (ncode >= 2 * max_codes)
                    break;
            }
            threshold /= nsubgroups;
        }

        // Precompute table
        pq->compute_inner_prod_table(x, precomputed_table.data());

        // Prepare max heap with k answers
        faiss::maxheap_heapify(k, distances, labels);

        int ncode = 0;
        for (int i = 0; i < nprobe; i++) {
            idx_t centroid_idx = centroid_idxs[i];
            size_t group_size = norm_codes[centroid_idx].size();
            if (group_size == 0)
                continue;

            float alpha = alphas[centroid_idx];
            float term1 = (1 - alpha) * (query_centroid_dists[centroid_idx] - centroid_norms[centroid_idx]);

            const uint8_t *code = codes[centroid_idx].data();
            const uint8_t *norm_code = norm_codes[centroid_idx].data();
            const idx_t *id = ids[centroid_idx].data();

            for (int subc = 0; subc < nsubc; subc++) {
                int subgroup_size = subgroup_sizes[centroid_idx][subc];
                if (subgroup_size == 0)
                    continue;

                // Check pruning condition
                if (!do_pruning || r[i * nsubc + subc] < threshold) {
                    idx_t nn_centroid_idx = nn_centroid_idxs[centroid_idx][subc];

                    // Compute the distance to the coarse centroid if it is not compute
                    if (query_centroid_dists[nn_centroid_idx] < EPS) {
                        const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idx);
                        query_centroid_dists[nn_centroid_idx] = fvec_L2sqr(x, nn_centroid, d);
                        used_centroid_idxs.push_back(nn_centroid_idx);
                    }

                    float term2 = alpha * (query_centroid_dists[nn_centroid_idx] - centroid_norms[nn_centroid_idx]);
                    norm_pq->decode(norm_code, norms.data(), subgroup_size);

                    for (int j = 0; j < subgroup_size; j++) {
                        float term4 = 2 * pq_L2sqr(code + j * code_size);
                        float dist = term1 + term2 + norms[j] - term4; //term3 = norms[j]
                        if (dist < distances[0]) {
                            faiss::maxheap_pop(k, distances, labels);
                            faiss::maxheap_push(k, distances, labels, dist, id[j]);
                        }
                    }
                    ncode += subgroup_size;
                }
                // Shift to the next group
                code += subgroup_size * code_size;
                norm_code += subgroup_size;
                id += subgroup_size;
            }
            if (ncode >= max_codes)
                break;
        }

        // Zero computed dists for later queries
        for (idx_t used_centroid_idx : used_centroid_idxs)
            query_centroid_dists[used_centroid_idx] = 0;
    }

    // TODO: rewrite with writeXvec
    void IndexIVF_HNSW_Grouping::write(const char *path_index)
    {
        FILE *fout = fopen(path_index, "wb");

        fwrite(&d, sizeof(size_t), 1, fout);
        fwrite(&nc, sizeof(size_t), 1, fout);
        fwrite(&nsubc, sizeof(size_t), 1, fout);

        int size;
        // Save vector indices
        for (size_t i = 0; i < nc; i++) {
            size = ids[i].size();
            fwrite(&size, sizeof(int), 1, fout);
            fwrite(ids[i].data(), sizeof(idx_t), size, fout);
        }
        // Save PQ codes
        for (int i = 0; i < nc; i++) {
            size = codes[i].size();
            fwrite(&size, sizeof(int), 1, fout);
            fwrite(codes[i].data(), sizeof(uint8_t), size, fout);
        }
        // Save norm PQ codes
        for (int i = 0; i < nc; i++) {
            size = norm_codes[i].size();
            fwrite(&size, sizeof(int), 1, fout);
            fwrite(norm_codes[i].data(), sizeof(uint8_t), size, fout);
        }
        // Save NN centroid indices
        for (int i = 0; i < nc; i++) {
            size = nn_centroid_idxs[i].size();
            fwrite(&size, sizeof(int), 1, fout);
            fwrite(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fout);
        }
        // Write group sizes
        for (int i = 0; i < nc; i++) {
            size = subgroup_sizes[i].size();
            fwrite(&size, sizeof(int), 1, fout);
            fwrite(subgroup_sizes[i].data(), sizeof(int), size, fout);
        }
        // Save alphas
        fwrite(alphas.data(), sizeof(float), nc, fout);

        // Save centroid norms
        fwrite(centroid_norms.data(), sizeof(float), nc, fout);

        // Save inter centroid distances
        for (int i = 0; i < nc; i++) {
            size = inter_centroid_dists[i].size();
            fwrite(&size, sizeof(idx_t), 1, fout);
            fwrite(inter_centroid_dists[i].data(), sizeof(float), size, fout);
        }
        fclose(fout);
    }

    // TODO: rewrite with readXvec
    void IndexIVF_HNSW_Grouping::read(const char *path_index)
    {
        FILE *fin = fopen(path_index, "rb");

        fread(&d, sizeof(size_t), 1, fin);
        fread(&nc, sizeof(size_t), 1, fin);
        fread(&nsubc, sizeof(size_t), 1, fin);

        ids.resize(nc);
        codes.resize(nc);
        norm_codes.resize(nc);
        nn_centroid_idxs.resize(nc);
        subgroup_sizes.resize(nc);

        int size;
        // Read indices
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(int), 1, fin);
            ids[i].resize(size);
            fread(ids[i].data(), sizeof(idx_t), size, fin);
        }
        // Read PQ codes
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(int), 1, fin);
            codes[i].resize(size);
            fread(codes[i].data(), sizeof(uint8_t), size, fin);
        }
        // Read norm PQ codes
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(int), 1, fin);
            norm_codes[i].resize(size);
            fread(norm_codes[i].data(), sizeof(uint8_t), size, fin);
        }
        // Read NN centroid indices
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(int), 1, fin);
            nn_centroid_idxs[i].resize(size);
            fread(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fin);
        }
        // Read group sizes
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(int), 1, fin);
            subgroup_sizes[i].resize(size);
            fread(subgroup_sizes[i].data(), sizeof(int), size, fin);
        }

        // Read alphas
        alphas.resize(nc);
        fread(alphas.data(), sizeof(float), nc, fin);

        //REBUTTLE
        FILE *fout = fopen("deep1b_positive_alphas.dat", "wb");
        fwrite(alphas.data(), sizeof(float), nc, fout);
        fclose(fout);

        // Read centroid norms
        centroid_norms.resize(nc);
        fread(centroid_norms.data(), sizeof(float), nc, fin);

        // Read inter centroid distances
        inter_centroid_dists.resize(nc);
        for (int i = 0; i < nc; i++) {
            fread(&size, sizeof(idx_t), 1, fin);
            inter_centroid_dists[i].resize(size);
            fread(inter_centroid_dists[i].data(), sizeof(float), size, fin);
        }
        fclose(fin);
    }


    void IndexIVF_HNSW_Grouping::train_pq(size_t n, const float *x)
    {
        std::vector<float> train_subcentroids;
        std::vector<float> train_residuals;

        train_subcentroids.reserve(n*d);
        train_residuals.reserve(n*d);

        std::vector<idx_t> assigned(n);
        assign(n, x, assigned.data());

        std::unordered_map<idx_t, std::vector<float>> group_map;

        for (int i = 0; i < n; i++) {
            idx_t key = assigned[i];
            for (int j = 0; j < d; j++)
                group_map[key].push_back(x[i*d + j]);
        }

        double av_dist = 0.0;

        // Train Residual PQ
        std::cout << "Training Residual PQ codebook " << std::endl;
        for (auto p : group_map) {
            idx_t centroid_idx = p.first;
            const float *centroid = quantizer->getDataByInternalId(centroid_idx);
            const vector<float> data = p.second;
            int group_size = data.size() / d;

            std::vector<idx_t> nn_centroid_idxs(nsubc);
            std::vector<float> centroid_vector_norms(nsubc);
            auto nn_centroids_raw = quantizer->searchKnn(centroid, nsubc + 1);

            while (nn_centroids_raw.size() > 1) {
                centroid_vector_norms[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
                nn_centroid_idxs[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
                nn_centroids_raw.pop();
            }

            // Compute centroid-neighbor_centroid and centroid-group_point vectors
            std::vector<float> centroid_vectors(nsubc * d);
            for (int subc = 0; subc < nsubc; subc++) {
                const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idxs[subc]);
                faiss::fvec_madd(d, nn_centroid, -1., centroid, centroid_vectors.data() + subc * d);
            }

            // Find alphas for vectors
            float alpha = compute_alpha(centroid_vectors.data(), data.data(), centroid, centroid_vector_norms.data(), group_size);

            // Compute final subcentroids 
            std::vector<float> subcentroids(nsubc * d);
            for (int subc = 0; subc < nsubc; subc++)
                faiss::fvec_madd(d, centroid, alpha, centroid_vectors.data() + subc*d, subcentroids.data() + subc*d);

            // Find subcentroid idx
            std::vector<idx_t> subcentroid_idxs(group_size);
            compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), group_size);

            // Compute Residuals
            std::vector<float> residuals(group_size * d);
            compute_residuals(group_size, data.data(), residuals.data(), subcentroids.data(), subcentroid_idxs.data());

            //////////////////////
            float dists[group_size];
            faiss::fvec_norms_L2sqr(dists, residuals.data(), d, group_size);
            for (int i = 0; i < group_size; i++) {
                av_dist += dists[i];
            }
            //std::cout << group_size <<  " " << av_dist << std::endl;

            for (int i = 0; i < group_size; i++) {
                idx_t subcentroid_idx = subcentroid_idxs[i];
                for (int j = 0; j < d; j++) {
                    train_subcentroids.push_back(subcentroids[subcentroid_idx*d + j]);
                    train_residuals.push_back(residuals[i*d + j]);
                }
            }
        }
        std::cout << n << " " << av_dist/n << std::endl;

        printf("Training %zdx%zd PQ on %ld vectors in %dD\n", pq->M, pq->ksub, train_residuals.size() / d, d);
        pq->verbose = true;
        pq->train(n, train_residuals.data());

        // Norm PQ
        std::cout << "Training Norm PQ codebook " << std::endl;
        std::vector<float> train_norms;
        const float *residuals = train_residuals.data();
        const float *subcentroids = train_subcentroids.data();

        for (auto p : group_map) {
            const vector<float> data = p.second;
            int group_size = data.size() / d;

            // Compute Codes 
            std::vector<uint8_t> xcodes(group_size * code_size);
            pq->compute_codes(residuals, xcodes.data(), group_size);

            // Decode Codes 
            std::vector<float> decoded_residuals(group_size * d);
            pq->decode(xcodes.data(), decoded_residuals.data(), group_size);

            // Reconstruct Data 
            std::vector<float> reconstructed_x(group_size * d);
            for (idx_t i = 0; i < group_size; i++)
                faiss::fvec_madd(d, decoded_residuals.data() + i*d, 1., subcentroids+i*d, reconstructed_x.data() + i*d);

            // Compute norms 
            std::vector<float> group_norms(group_size);
            faiss::fvec_norms_L2sqr(group_norms.data(), reconstructed_x.data(), d, group_size);

            for (int i = 0; i < group_size; i++)
                train_norms.push_back(group_norms[i]);

            residuals += group_size * d;
            subcentroids += group_size * d;
        }
        printf("Training %zdx%zd PQ on %ld vectors in 1D\n", norm_pq->M, norm_pq->ksub, train_norms.size());
        norm_pq->verbose = true;
        norm_pq->train(n, train_norms.data());
    }

    void IndexIVF_HNSW_Grouping::compute_inter_centroid_dists()
    {
        inter_centroid_dists.resize(nc);
        for (int i = 0; i < nc; i++) {
            const float *centroid = quantizer->getDataByInternalId(i);
            inter_centroid_dists[i].resize(nsubc);
            for (int subc = 0; subc < nsubc; subc++) {
                idx_t nn_centroid_idx = nn_centroid_idxs[i][subc];
                const float *nn_centroid = quantizer->getDataByInternalId(nn_centroid_idx);
                inter_centroid_dists[i][subc] = fvec_L2sqr(nn_centroid, centroid, d);
            }
        }
    }

    void IndexIVF_HNSW_Grouping::compute_residuals(size_t n, const float *x, float *residuals,
                                                   const float *subcentroids, const idx_t *keys)
    {
        for (idx_t i = 0; i < n; i++) {
            const float *subcentroid = subcentroids + keys[i]*d;
            faiss::fvec_madd(d, x + i*d, -1., subcentroid, residuals + i*d);
        }
    }

    void IndexIVF_HNSW_Grouping::reconstruct(size_t n, float *x, const float *decoded_residuals,
                                             const float *subcentroids, const idx_t *keys)
    {
        for (idx_t i = 0; i < n; i++) {
            const float *subcentroid = subcentroids + keys[i] * d;
            faiss::fvec_madd(d, decoded_residuals + i*d, 1., subcentroid, x + i*d);
        }
    }

    void IndexIVF_HNSW_Grouping::compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                                          const float *x, int group_size)
    {
        for (int i = 0; i < group_size; i++) {
            float min_dist = 0.0;
            idx_t min_idx = -1;
            for (int subc = 0; subc < nsubc; subc++) {
                const float *subcentroid = subcentroids + subc * d;
                float dist = fvec_L2sqr(subcentroid, x + i*d, d);
                if (min_idx == -1 || dist < min_dist){
                    min_dist = dist;
                    min_idx = subc;
                }
            }
            subcentroid_idxs[i] = min_idx;
        }
    }

    float IndexIVF_HNSW_Grouping::compute_alpha(const float *centroid_vectors, const float *points,
                                                const float *centroid, const float *centroid_vector_norms_L2sqr,
                                                int group_size)
    {
        int counter_positive = 0;
        int counter_negative = 0;

        float positive_numerator = 0.;
        float positive_denominator = 0.;

        float negative_numerator = 0.;
        float negative_denominator = 0.;

        float positive_alpha = 0.0;
        float negative_alpha = 0.0;

        float group_numerator = 0.0;
        float group_denominator = 0.0;

        std::vector<float> point_vectors(group_size * d);
        for (int i = 0; i < group_size; i++)
            faiss::fvec_madd(d, points + i*d , -1., centroid, point_vectors.data() + i*d);

        for (int i = 0; i < group_size; i++) {
            const float *point_vector = point_vectors.data() + i * d;
            const float *point = points + i * d;

            std::priority_queue<std::pair<float, std::pair<float, float>>> maxheap;

            for (int subc = 0; subc < nsubc; subc++) {
                const float *centroid_vector = centroid_vectors + subc * d;
                const float centroid_vector_norm_L2sqr = centroid_vector_norms_L2sqr[subc];

                float numerator = faiss::fvec_inner_product(centroid_vector, point_vector, d);
                float denominator = centroid_vector_norm_L2sqr;
                float alpha = numerator / denominator;

                std::vector<float> subcentroid(d);
                faiss::fvec_madd(d, centroid, alpha, centroid_vector, subcentroid.data());

                float dist = fvec_L2sqr(point, subcentroid.data(), d);
                maxheap.emplace(-dist, std::make_pair(numerator, denominator));
            }
            float optim_numerator = 0.0;
            float optim_denominator = 0.0;
            //std::tie(optim_numerator, optim_denominator) = maxheap.top().second;

            //REBUTTAL
            while (maxheap.size() > 0){
                float numerator, denominator;
                std::tie(numerator, denominator) = maxheap.top().second;

                if (numerator > 0) {
                    optim_numerator = numerator;
                    optim_denominator = denominator;
                    break;
                }
                maxheap.pop();
            }
            group_numerator += optim_numerator;
            group_denominator += optim_denominator;

//            if (optim_numerator < 0) {
//                counter_negative++;
//                negative_numerator += optim_numerator;
//                negative_denominator += optim_denominator;
//            } else {
//                counter_positive++;
//                positive_numerator += optim_numerator;
//                positive_denominator += optim_denominator;
//            }
        }
        global_numerator += group_numerator;
        global_denominator += group_denominator;
        return (group_denominator > 0) ? group_numerator / group_denominator : 0.0;
//        positive_alpha = positive_numerator / positive_denominator;
//        negative_alpha = negative_numerator / negative_denominator;
//        return (counter_positive > counter_negative) ? positive_alpha : negative_alpha;
    }
}