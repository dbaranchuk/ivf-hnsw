//
// Created by dbaranchuk on 23.12.17.
//

#include "IndexIVF_HNSW_Grouping.h"


namespace ivfhnsw{
    /****************************************************/
    /** IVF_HNSW + grouping( + pruning) implementation **/
    /****************************************************/
    IndexIVF_HNSW_Grouping::IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                                                   size_t nbits_per_idx, size_t nsubcentroids = 64):
           IndexIVF_HNSW(dim, ncentroids, bytes_per_code, nbits_per_idx), nsubc(nsubcentroids)
    {
        alphas.resize(nc);
        nn_centroid_idxs.resize(nc);
        group_sizes.resize(nc);

        q_s.resize(nc);
        std::fill(q_s.begin(), q_s.end(), 0);

        do_pruning = false;
    }

    void IndexIVF_HNSW_Grouping::add_group(int centroid_num, int groupsize,
                                           const float *data, const idx_t *idxs,
                                           double &baseline_average, double &modified_average)
    {
        /** Find NN centroids to source centroid **/
        const float *centroid = quantizer->getDataByInternalId(centroid_num);
        std::priority_queue<std::pair<float, idx_t>> nn_centroids_raw = quantizer->searchKnn(centroid, nsubc + 1);
        /** Vectors for construction **/
        std::vector<float> centroid_vector_norms_L2sqr(nsubc);
        nn_centroid_idxs[centroid_num].resize(nsubc);
        while (nn_centroids_raw.size() > 1) {
            centroid_vector_norms_L2sqr[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
            nn_centroid_idxs[centroid_num][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
            nn_centroids_raw.pop();
        }
        if (groupsize == 0)
            return;

        const float *centroid_vector_norms = centroid_vector_norms_L2sqr.data();
        const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();

        /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
        std::vector<float> centroid_vectors(nsubc * d);
        for (int subc = 0; subc < nsubc; subc++) {
            float *neighbor_centroid = quantizer->getDataByInternalId(nn_centroids[subc]);
            faiss::fvec_madd(d, neighbor_centroid, -1., centroid, centroid_vectors.data() + subc * d);
        }

        /** Find alphas for vectors **/
        alphas[centroid_num] = compute_alpha(centroid_vectors.data(), data, centroid,
                                             centroid_vector_norms, groupsize);

        /** Compute final subcentroids **/
        std::vector<float> subcentroids(nsubc * d);
        for (int subc = 0; subc < nsubc; subc++) {
            const float *centroid_vector = centroid_vectors.data() + subc * d;
            float *subcentroid = subcentroids.data() + subc * d;
            faiss::fvec_madd(d, centroid, alphas[centroid_num], centroid_vector, subcentroid);
        }

        /** Find subcentroid idx **/
        std::vector<idx_t> subcentroid_idxs(groupsize);
        compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data, groupsize);

        /** Compute Residuals **/
        std::vector<float> residuals(groupsize * d);
        compute_residuals(groupsize, data, residuals.data(), subcentroids.data(), subcentroid_idxs.data());

        /** Compute Codes **/
        std::vector<uint8_t> xcodes(groupsize * code_size);
        pq->compute_codes(residuals.data(), xcodes.data(), groupsize);

        /** Decode Codes **/
        std::vector<float> decoded_residuals(groupsize * d);
        pq->decode(xcodes.data(), decoded_residuals.data(), groupsize);

        /** Reconstruct Data **/
        std::vector<float> reconstructed_x(groupsize * d);
        reconstruct(groupsize, reconstructed_x.data(), decoded_residuals.data(),
                    subcentroids.data(), subcentroid_idxs.data());

        /** Compute norms **/
        std::vector<float> norms(groupsize);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, groupsize);

        /** Compute norm codes **/
        std::vector<uint8_t> xnorm_codes(groupsize);
        norm_pq->compute_codes(norms.data(), xnorm_codes.data(), groupsize);

        /** Distribute codes **/
        std::vector<std::vector<idx_t> > construction_ids(nsubc);
        std::vector<std::vector<uint8_t> > construction_codes(nsubc);
        std::vector<std::vector<uint8_t> > construction_norm_codes(nsubc);
        for (int i = 0; i < groupsize; i++) {
            const idx_t idx = idxs[i];
            const idx_t subcentroid_idx = subcentroid_idxs[i];

            construction_ids[subcentroid_idx].push_back(idx);
            construction_norm_codes[subcentroid_idx].push_back(xnorm_codes[i]);
            for (int j = 0; j < code_size; j++)
                construction_codes[subcentroid_idx].push_back(xcodes[i * code_size + j]);

            const float *subcentroid = subcentroids.data() + subcentroid_idx * d;
            const float *point = data + i * d;
            baseline_average += fvec_L2sqr(centroid, point, d);
            modified_average += fvec_L2sqr(subcentroid, point, d);
        }
        /** Add codes **/
        for (int subc = 0; subc < nsubc; subc++) {
            idx_t subcsize = construction_norm_codes[subc].size();
            group_sizes[centroid_num].push_back(subcsize);

            for (int i = 0; i < subcsize; i++) {
                ids[centroid_num].push_back(construction_ids[subc][i]);
                for (int j = 0; j < code_size; j++)
                    codes[centroid_num].push_back(construction_codes[subc][i * code_size + j]);
                norm_codes[centroid_num].push_back(construction_norm_codes[subc][i]);
            }
        }
    }

    void IndexIVF_HNSW_Grouping::search(size_t k, const float *x, float *distances, long *labels)
    {
        std::vector<float> r;
        std::vector<idx_t> subcentroid_nums;
        subcentroid_nums.reserve(nsubc * nprobe);
        idx_t keys[nprobe];
        float q_c[nprobe];

        /** Find NN Centroids **/
        auto coarse = quantizer->searchKnn(x, nprobe);
        for (int i = nprobe - 1; i >= 0; i--) {
            auto elem = coarse.top();
            q_c[i] = elem.first;
            keys[i] = elem.second;

            /** Add q_c to precomputed q_s **/
            idx_t key = keys[i];
            q_s[key] = q_c[i];
            subcentroid_nums.push_back(key);

            coarse.pop();
        }

        /** Computing threshold for pruning **/
        double threshold = 0.0;
        if (do_pruning) {
            int ncode = 0;
            int normalize = 0;

            r.resize(nsubc * nprobe);
            for (int i = 0; i < nprobe; i++) {
                idx_t centroid_num = keys[i];
                size_t regionsize = norm_codes[centroid_num].size();
                if (regionsize == 0)
                    continue;
                ncode += regionsize;

                float *subr = r.data() + i*nsubc;
                const idx_t *groupsizes = group_sizes[centroid_num].data();
                const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
                float alpha = alphas[centroid_num];

                for (int subc = 0; subc < nsubc; subc++) {
                    if (groupsizes[subc] == 0)
                        continue;

                    idx_t subcentroid_num = nn_centroids[subc];

                    if (q_s[subcentroid_num] < EPS) {
                        const float *nn_centroid = quantizer->getDataByInternalId(subcentroid_num);
                        q_s[subcentroid_num] = fvec_L2sqr(x, nn_centroid, d);
                        subcentroid_nums.push_back(subcentroid_num);
                    }

                    subr[subc] = (1 - alpha)*(q_c[i] - alpha * centroid_dists[centroid_num][subc]) + alpha*q_s[subcentroid_num];
                    threshold += subr[subc];
                    normalize++;
                }
                if (ncode >= 2 * max_codes)
                    break;
            }
            threshold /= normalize;
        }

        /** Compute Query Table **/
        pq->compute_inner_prod_table(x, precomputed_table.data());

        /** Prepare max heap with \k answers **/
        faiss::maxheap_heapify(k, distances, labels);

        int ncode = 0;
        for (int i = 0; i < nprobe; i++) {
            idx_t centroid_num = keys[i];
            size_t regionsize = norm_codes[centroid_num].size();
            if (regionsize == 0)
                continue;

            const idx_t *groupsizes = group_sizes[centroid_num].data();
            const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
            float alpha = alphas[centroid_num];
            float fst_term = (1 - alpha) * (q_c[i] - centroid_norms[centroid_num]);

            const uint8_t *norm_code = norm_codes[centroid_num].data();
            uint8_t *code = codes[centroid_num].data();
            const idx_t *id = ids[centroid_num].data();

            for (int subc = 0; subc < nsubc; subc++) {
                idx_t groupsize = groupsizes[subc];
                if (groupsize == 0)
                    continue;

                if (do_pruning && r[i * nsubc + subc] > threshold) {
                    code += groupsize * code_size;
                    norm_code += groupsize;
                    id += groupsize;
                    continue;
                }

                idx_t subcentroid_num = nn_centroids[subc];
                if (q_s[subcentroid_num] < EPS) {
                    const float *nn_centroid = quantizer->getDataByInternalId(subcentroid_num);
                    q_s[subcentroid_num] = fvec_L2sqr(x, nn_centroid, d);
                    subcentroid_nums.push_back(subcentroid_num);
                }

                float snd_term = alpha * (q_s[subcentroid_num] - centroid_norms[subcentroid_num]);
                norm_pq->decode(norm_code, norms.data(), groupsize);

                for (int j = 0; j < groupsize; j++) {
                    float q_r = pq_L2sqr(code + j * code_size);
                    float dist = fst_term + snd_term - 2 * q_r + norms[j];
                    if (dist < distances[0]) {
                        faiss::maxheap_pop(k, distances, labels);
                        faiss::maxheap_push(k, distances, labels, dist, id[j]);
                    }
                }
                /** Shift to the next group **/
                code += groupsize * code_size;
                norm_code += groupsize;
                id += groupsize;
                ncode += groupsize;
            }
            if (ncode >= max_codes)
                break;
        }

        /** Zero subcentroids **/
        for (idx_t subcentroid_num : subcentroid_nums)
            q_s[subcentroid_num] = 0;
    }

    void IndexIVF_HNSW_Grouping::write(const char *path_index)
    {
        FILE *fout = fopen(path_index, "wb");

        fwrite(&d, sizeof(size_t), 1, fout);
        fwrite(&nc, sizeof(size_t), 1, fout);
        fwrite(&nsubc, sizeof(size_t), 1, fout);

        idx_t size;
        /** Save Vector Indexes per  **/
        for (size_t i = 0; i < nc; i++) {
            size = ids[i].size();
            fwrite(&size, sizeof(idx_t), 1, fout);
            fwrite(ids[i].data(), sizeof(idx_t), size, fout);
        }
        /** Save PQ Codes **/
        for (int i = 0; i < nc; i++) {
            size = codes[i].size();
            fwrite(&size, sizeof(idx_t), 1, fout);
            fwrite(codes[i].data(), sizeof(uint8_t), size, fout);
        }
        /** Save Norm Codes **/
        for (int i = 0; i < nc; i++) {
            size = norm_codes[i].size();
            fwrite(&size, sizeof(idx_t), 1, fout);
            fwrite(norm_codes[i].data(), sizeof(uint8_t), size, fout);
        }
        /** Save NN Centroid Indexes **/
        for (int i = 0; i < nc; i++) {
            size = nn_centroid_idxs[i].size();
            fwrite(&size, sizeof(idx_t), 1, fout);
            fwrite(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fout);
        }
        /** Write Group Sizes **/
        for (int i = 0; i < nc; i++) {
            size = group_sizes[i].size();
            fwrite(&size, sizeof(idx_t), 1, fout);
            fwrite(group_sizes[i].data(), sizeof(idx_t), size, fout);
        }
        /** Save Alphas **/
        fwrite(alphas.data(), sizeof(float), nc, fout);

        /** Save Centroid Norms **/
        fwrite(centroid_norms.data(), sizeof(float), nc, fout);

        /** Save Centroid Dists **/
        for (int i = 0; i < nc; i++) {
            size = centroid_dists[i].size();
            fwrite(&size, sizeof(idx_t), 1, fout);
            fwrite(centroid_dists[i].data(), sizeof(float), size, fout);
        }
        fclose(fout);
    }

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
        group_sizes.resize(nc);

        idx_t size;
        /** Read Ids **/
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(idx_t), 1, fin);
            ids[i].resize(size);
            fread(ids[i].data(), sizeof(idx_t), size, fin);
        }
        /** Read Codes **/
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(idx_t), 1, fin);
            codes[i].resize(size);
            fread(codes[i].data(), sizeof(uint8_t), size, fin);
        }
        /** Read Norm Codes **/
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(idx_t), 1, fin);
            norm_codes[i].resize(size);
            fread(norm_codes[i].data(), sizeof(uint8_t), size, fin);
        }
        /** Read NN Centroid Indexes **/
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(idx_t), 1, fin);
            nn_centroid_idxs[i].resize(size);
            fread(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fin);
        }
        /** Read Group Sizes **/
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(idx_t), 1, fin);
            group_sizes[i].resize(size);
            fread(group_sizes[i].data(), sizeof(idx_t), size, fin);
        }

        /** Read Alphas **/
        alphas.resize(nc);
        fread(alphas.data(), sizeof(float), nc, fin);

        /** Read Centroid Norms **/
        centroid_norms.resize(nc);
        fread(centroid_norms.data(), sizeof(float), nc, fin);

        /** Read Centroid Dists **/
        centroid_dists.resize(nc);
        for (int i = 0; i < nc; i++) {
            fread(&size, sizeof(idx_t), 1, fin);
            centroid_dists[i].resize(size);
            fread(centroid_dists[i].data(), sizeof(float), size, fin);
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

        /** Train Residual PQ **/
        std::cout << "Training Residual PQ codebook " << std::endl;
        for (auto p : group_map) {
            const idx_t centroid_num = p.first;
            const float *centroid = quantizer->getDataByInternalId(centroid_num);
            const vector<float> data = p.second;
            const int groupsize = data.size() / d;

            std::vector<idx_t> nn_centroids(nsubc);
            std::vector<float> centroid_vector_norms(nsubc);
            auto nn_centroids_raw = quantizer->searchKnn(centroid, nsubc + 1);

            while (nn_centroids_raw.size() > 1) {
                centroid_vector_norms[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
                nn_centroids[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
                nn_centroids_raw.pop();
            }

            /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
            std::vector<float> centroid_vectors(nsubc * d);
            for (int subc = 0; subc < nsubc; subc++) {
                const float *neighbor_centroid = quantizer->getDataByInternalId(nn_centroids[subc]);
                faiss::fvec_madd(d, neighbor_centroid, -1., centroid, centroid_vectors.data() + subc * d);
            }

            /** Find alphas for vectors **/
            float alpha = compute_alpha(centroid_vectors.data(), data.data(), centroid, centroid_vector_norms.data(), groupsize);

            /** Compute final subcentroids **/
            std::vector<float> subcentroids(nsubc * d);
            for (int subc = 0; subc < nsubc; subc++)
                faiss::fvec_madd(d, centroid, alpha, centroid_vectors.data() + subc*d, subcentroids.data() + subc*d);

            /** Find subcentroid idx **/
            std::vector<idx_t> subcentroid_idxs(groupsize);
            compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), groupsize);

            /** Compute Residuals **/
            std::vector<float> residuals(groupsize * d);
            compute_residuals(groupsize, data.data(), residuals.data(), subcentroids.data(), subcentroid_idxs.data());

            for (int i = 0; i < groupsize; i++) {
                int subcentroid_idx = subcentroid_idxs[i];
                for (int j = 0; j < d; j++) {
                    train_subcentroids.push_back(subcentroids[subcentroid_idx*d + j]);
                    train_residuals.push_back(residuals[i*d + j]);
                }
            }
        }
        printf("Training %zdx%zd PQ on %ld vectors in %dD\n", pq->M, pq->ksub, train_residuals.size() / d, d);
        pq->verbose = true;
        pq->train(n, train_residuals.data());

        /** Norm PQ **/
        std::cout << "Training Norm PQ codebook " << std::endl;
        std::vector<float> train_norms;
        const float *residuals = train_residuals.data();
        const float *subcentroids = train_subcentroids.data();

        for (auto p : group_map) {
            const vector<float> data = p.second;
            const int groupsize = data.size() / d;

            /** Compute Codes **/
            std::vector<uint8_t> xcodes(groupsize * code_size);
            pq->compute_codes(residuals, xcodes.data(), groupsize);

            /** Decode Codes **/
            std::vector<float> decoded_residuals(groupsize * d);
            pq->decode(xcodes.data(), decoded_residuals.data(), groupsize);

            /** Reconstruct Data **/
            std::vector<float> reconstructed_x(groupsize * d);
            for (idx_t i = 0; i < groupsize; i++)
                faiss::fvec_madd(d, decoded_residuals.data() + i*d, 1., subcentroids+i*d, reconstructed_x.data() + i*d);

            /** Compute norms **/
            std::vector<float> group_norms(groupsize);
            faiss::fvec_norms_L2sqr(group_norms.data(), reconstructed_x.data(), d, groupsize);

            for (int i = 0; i < groupsize; i++)
                train_norms.push_back(group_norms[i]);

            residuals += groupsize * d;
            subcentroids += groupsize * d;
        }
        printf("Training %zdx%zd PQ on %ld vectors in 1D\n", norm_pq->M, norm_pq->ksub, train_norms.size());
        norm_pq->verbose = true;
        norm_pq->train(n, train_norms.data());
    }

    void IndexIVF_HNSW_Grouping::compute_centroid_dists()
    {
        centroid_dists.resize(nc);
        for (int i = 0; i < nc; i++) {
            const float *centroid = quantizer->getDataByInternalId(i);
            centroid_dists[i].resize(nsubc);
            for (int subc = 0; subc < nsubc; subc++) {
                idx_t subc_idx = nn_centroid_idxs[i][subc];
                const float *subcentroid = quantizer->getDataByInternalId(subc_idx);
                centroid_dists[i][subc] = fvec_L2sqr(subcentroid, centroid, d);
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
                                                          const float *x, const int groupsize)
    {
        for (int i = 0; i < groupsize; i++) {
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
                                                const int groupsize)
    {
        int counter_positive = 0;
        int counter_negative = 0;

        float positive_numerator = 0.;
        float positive_denominator = 0.;

        float negative_numerator = 0.;
        float negative_denominator = 0.;

        float positive_alpha = 0.0;
        float negative_alpha = 0.0;

        std::vector<float> point_vectors(groupsize * d);
        for (int i = 0; i < groupsize; i++)
            faiss::fvec_madd(d, points + i*d , -1., centroid, point_vectors.data() + i*d);

        for (int i = 0; i < groupsize; i++) {
            const float *point_vector = point_vectors.data() + i * d;
            const float *point = points + i * d;

            std::priority_queue<std::pair<float, std::pair<float, float>>> max_heap;

            for (int subc = 0; subc < nsubc; subc++) {
                const float *centroid_vector = centroid_vectors + subc * d;
                const float centroid_vector_norm_L2sqr = centroid_vector_norms_L2sqr[subc];

                float numerator = faiss::fvec_inner_product(centroid_vector, point_vector, d);
                float denominator = centroid_vector_norm_L2sqr;
                float alpha = numerator / denominator;

                std::vector<float> subcentroid(d);
                faiss::fvec_madd(d, centroid, alpha, centroid_vector, subcentroid.data());

                float dist = fvec_L2sqr(point, subcentroid.data(), d);
                max_heap.emplace(-dist, std::make_pair(numerator, denominator));
            }
            float optim_numerator = max_heap.top().second.first;
            float optim_denominator = max_heap.top().second.second;
            if (optim_numerator < 0) {
                counter_negative++;
                negative_numerator += optim_numerator;
                negative_denominator += optim_denominator;
                //negative_alpha += optim_numerator / optim_denominator;
            } else {
                counter_positive++;
                positive_numerator += optim_numerator;
                positive_denominator += optim_denominator;
                //positive_alpha += optim_numerator / optim_denominator;
            }
        }
        //positive_alpha /= groupsize;
        //negative_alpha /= groupsize;
        positive_alpha = positive_numerator / positive_denominator;
        negative_alpha = negative_numerator / negative_denominator;
        return (counter_positive > counter_negative) ? positive_alpha : negative_alpha;
    }
}