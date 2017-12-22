#include "IndexIVF_HNSW.h"

namespace ivfhnsw {

/** Common IndexIVF + HNSW **/

/** Public **/
    IndexIVF_HNSW::IndexIVF_HNSW(size_t dim, size_t ncentroids, size_t bytes_per_code, size_t nbits_per_idx) :
            d(dim), nc(ncentroids)
    {
        codes.resize(ncentroids);
        norm_codes.resize(ncentroids);
        ids.resize(ncentroids);

        pq = new faiss::ProductQuantizer(dim, bytes_per_code, nbits_per_idx);
        norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);

        query_table.resize(pq->ksub * pq->M);

        norms.resize(65536);
        code_size = pq->code_size;
    }


    IndexIVF_HNSW::~IndexIVF_HNSW() {
        delete pq;
        delete norm_pq;
        delete quantizer;
    }


    void IndexIVF_HNSW::buildCoarseQuantizer(const char *path_clusters, const char *path_info,
                                             const char *path_edges, int M, int efConstruction = 500)
    {
        if (exists_test(path_info) && exists_test(path_edges)) {
            quantizer = new hnswlib::HierarchicalNSW(path_info, path_clusters, path_edges);
            quantizer->ef_ = efConstruction;
            return;
        }
        quantizer = new hnswlib::HierarchicalNSW(d, nc, M, 2 * M, efConstruction);
        quantizer->ef_ = efConstruction;

        std::cout << "Constructing quantizer\n";
        int j1 = 0;
        std::ifstream input(path_clusters, ios::binary);

        float mass[d];
        readXvec<float>(input, mass, d);
        quantizer->addPoint((void *) (mass), j1);

        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < nc; i++) {
            float mass[d];
#pragma omp critical
            {
                readXvec<float>(input, mass, d);
                if (++j1 % report_every == 0)
                    std::cout << j1 / (0.01 * nc) << " %\n";
            }
            quantizer->addPoint((void *) (mass), (size_t) j1);
        }
        input.close();
        quantizer->SaveInfo(path_info);
        quantizer->SaveEdges(path_edges);
    }


    void IndexIVF_HNSW::assign(size_t n, const float *data, idx_t *idxs)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            idxs[i] = quantizer->searchKnn(const_cast<float *>(data + i * d), 1).top().second;

        }
    }


    void IndexIVF_HNSW::add_batch(size_t n, const float *x, const idx_t *xids, const idx_t *idx) {
        /** Compute residuals for original vectors **/
        std::vector<float> residuals(n * d);
        compute_residuals(n, x, residuals.data(), idx);

        /** Encode Residuals **/
        std::vector <uint8_t> xcodes(n * code_size);
        pq->compute_codes(residuals.data(), xcodes.data(), n);

        /** Decode Residuals **/
        std::vector<float> decoded_residuals(n * d);
        pq->decode(xcodes.data(), decoded_residuals.data(), n);

        /** Reconstruct original vectors **/
        std::vector<float> reconstructed_x(n * d);
        reconstruct(n, reconstructed_x.data(), decoded_residuals.data(), idx);

        /** Compute l2 square norms for reconstructed vectors **/
        std::vector<float> norms(n);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, n);

        /** Encode these norms**/
        std::vector <uint8_t> xnorm_codes(n);
        norm_pq->compute_codes(norms.data(), xnorm_codes.data(), n);

        /** Add vector indecies and PQ codes for residuals and norms to Index **/
        for (size_t i = 0; i < n; i++) {
            idx_t key = idx[i];
            idx_t id = xids[i];
            ids[key].push_back(id);
            uint8_t *code = xcodes.data() + i * code_size;
            for (size_t j = 0; j < code_size; j++)
                codes[key].push_back(code[j]);

            norm_codes[key].push_back(xnorm_codes[i]);
        }
    }


    void IndexIVF_HNSW::search(float *x, idx_t k, float *distances, long *labels)
    {
        long keys[nprobe];
        float q_c[nprobe];

        /** Find NN Centroids **/
//        auto coarse = quantizer->searchKnn(x, nprobe);
//        for (int i = nprobe - 1; i >= 0; i--) {
//            auto elem = coarse.top();
//            q_c[i] = elem.first;
//            keys[i] = elem.second;
//            coarse.pop();
//        }

        quantizer->search(x, q_c, keys, nprobe, quantizer->ef_);

        //for (int i = 0; i < nprobe; i++)
        //         std:: cout << q_c[i] << ' ' << keys[i] << std::endl;

        /** Compute Query Table **/
        pq->compute_inner_prod_table(x, query_table.data());

        /** Prepare max heap with \k answers **/
        faiss::maxheap_heapify(k, distances, labels);

        int ncode = 0;
        for (int i = 0; i < nprobe; i++) {
            idx_t key = keys[0];//keys[i];

            std::vector <uint8_t> code = codes[key];
            std::vector <uint8_t> norm_code = norm_codes[key];
            float term1 = q_c[0] - centroid_norms[key];
            int ncodes = norm_code.size();

            std:: cout << q_c[0] << ' ' << keys[0] << std::endl;
            faiss::minheap_pop(nprobe-i, q_c, keys);
            norm_pq->decode(norm_code.data(), norms.data(), ncodes);

            for (int j = 0; j < ncodes; j++) {
                float q_r = fstdistfunc(code.data() + j * code_size);
                float dist = term1 - 2 * q_r + norms[j];
                idx_t label = ids[key][j];
                if (dist < distances[0]) {
                    faiss::maxheap_pop(k, distances, labels);
                    faiss::maxheap_push(k, distances, labels, dist, label);
                }
            }
            ncode += ncodes;
            if (ncode >= max_codes)
                break;
        }
        std:: cout << "HUI" << std::endl;
        average_max_codes += ncode;
    }


    void IndexIVF_HNSW::train_pq(idx_t n, const float *x)
    {
        /** Assign train vectors **/
        std::vector <idx_t> assigned(n);
        assign(n, x, assigned.data());

        /** Compute residuals for original vectors **/
        std:
        vector<float> residuals(n * d);
        compute_residuals(n, x, residuals.data(), assigned.data());

        /** Train Residual PQ **/
        printf("Training %zdx%zd product quantizer on %ld vectors in %dD\n", pq->M, pq->ksub, n, d);
        pq->verbose = true;
        pq->train(n, residuals.data());

        /** Encode Residuals **/
        std::vector <uint8_t> xcodes(n * code_size);
        pq->compute_codes(residuals.data(), xcodes.data(), n);

        /** Decode Residuals **/
        std::vector<float> decoded_residuals(n * d);
        pq->decode(xcodes.data(), decoded_residuals.data(), n);

        /** Reconstruct original vectors **/
        std::vector<float> reconstructed_x(n * d);
        reconstruct(n, reconstructed_x.data(), decoded_residuals.data(), assigned.data());

        /** Compute l2 square norms for reconstructed vectors **/
        std::vector<float> norms(n);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), d, n);

        /** Train Norm PQ **/
        printf("Training %zdx%zd product quantizer on %ld vectors in %dD\n", norm_pq->M, norm_pq->ksub, n, d);
        norm_pq->verbose = true;
        norm_pq->train(n, norms.data());
    }

    /** Write index **/
    void IndexIVF_HNSW::write(const char *path_index) {
        FILE *fout = fopen(path_index, "wb");

        fwrite(&d, sizeof(size_t), 1, fout);
        fwrite(&nc, sizeof(size_t), 1, fout);
        fwrite(&nprobe, sizeof(size_t), 1, fout);
        fwrite(&max_codes, sizeof(size_t), 1, fout);

        size_t size;
        for (size_t i = 0; i < nc; i++) {
            size = ids[i].size();
            fwrite(&size, sizeof(size_t), 1, fout);
            fwrite(ids[i].data(), sizeof(idx_t), size, fout);
        }

        for (int i = 0; i < nc; i++) {
            size = codes[i].size();
            fwrite(&size, sizeof(size_t), 1, fout);
            fwrite(codes[i].data(), sizeof(uint8_t), size, fout);
        }

        for (int i = 0; i < nc; i++) {
            size = norm_codes[i].size();
            fwrite(&size, sizeof(size_t), 1, fout);
            fwrite(norm_codes[i].data(), sizeof(uint8_t), size, fout);
        }

        /** Save Centroid Norms **/
        fwrite(centroid_norms.data(), sizeof(float), nc, fout);
        fclose(fout);
    }

    /** Read index **/
    void IndexIVF_HNSW::read(const char *path_index)
    {
        FILE *fin = fopen(path_index, "rb");

        fread(&d, sizeof(size_t), 1, fin);
        fread(&nc, sizeof(size_t), 1, fin);
        fread(&nprobe, sizeof(size_t), 1, fin);
        fread(&max_codes, sizeof(size_t), 1, fin);

        ids = std::vector < std::vector < idx_t >> (nc);
        codes = std::vector < std::vector < uint8_t >> (nc);
        norm_codes = std::vector < std::vector < uint8_t >> (nc);

        size_t size;
        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(size_t), 1, fin);
            ids[i].resize(size);
            fread(ids[i].data(), sizeof(idx_t), size, fin);
        }

        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(size_t), 1, fin);
            codes[i].resize(size);
            fread(codes[i].data(), sizeof(uint8_t), size, fin);
        }

        for (size_t i = 0; i < nc; i++) {
            fread(&size, sizeof(size_t), 1, fin);
            norm_codes[i].resize(size);
            fread(norm_codes[i].data(), sizeof(uint8_t), size, fin);
        }

        /** Read Centroid Norms **/
        centroid_norms.resize(nc);
        fread(centroid_norms.data(), sizeof(float), nc, fin);
        fclose(fin);
    }

/** Private **/
    void IndexIVF_HNSW::compute_centroid_norms()
    {
        centroid_norms.resize(nc);
#pragma omp parallel for
        for (int i = 0; i < nc; i++) {
            float *c = (float *) quantizer->getDataByInternalId(i);
            centroid_norms[i] = faiss::fvec_norm_L2sqr(c, d);
        }
    }


    float IndexIVF_HNSW::fstdistfunc(uint8_t *code) {
        float result = 0.;
        int dim = code_size >> 2;
        int m = 0;
        for (int i = 0; i < dim; i++) {
            result += query_table[pq->ksub * m + code[m]];
            m++;
            result += query_table[pq->ksub * m + code[m]];
            m++;
            result += query_table[pq->ksub * m + code[m]];
            m++;
            result += query_table[pq->ksub * m + code[m]];
            m++;
        }
        return result;
    }

    void IndexIVF_HNSW::reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys) {
        for (idx_t i = 0; i < n; i++) {
            float *centroid = (float *) quantizer->getDataByInternalId(keys[i]);
            for (int j = 0; j < d; j++)
                x[i * d + j] = centroid[j] + decoded_residuals[i * d + j];
        }
    }


    void IndexIVF_HNSW::compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys) {
        for (idx_t i = 0; i < n; i++) {
            float *centroid = (float *) quantizer->getDataByInternalId(keys[i]);
            for (int j = 0; j < d; j++) {
                residuals[i * d + j] = x[i * d + j] - centroid[j];
            }
        }
    }


    /***************************************/
    /** IVF + HNSW + Grouping( + Pruning) **/
    /***************************************/
    IndexIVF_HNSW_Grouping::IndexIVF_HNSW_Grouping(size_t dim, size_t ncentroids, size_t bytes_per_code,
                                                   size_t nbits_per_idx, size_t nsubcentroids = 64):
            d(dim), nc(ncentroids), nsubc(nsubcentroids)
    {
        codes.resize(nc);
        norm_codes.resize(nc);
        ids.resize(nc);
        alphas.resize(nc);
        nn_centroid_idxs.resize(nc);
        group_sizes.resize(nc);

        pq = new faiss::ProductQuantizer(d, bytes_per_code, nbits_per_idx);
        norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);

        query_table.resize(pq->ksub * pq->M);

        norms.resize(65536);
        code_size = pq->code_size;

        /** Compute centroid norms **/
        s_c.resize(nc);
        q_s.resize(nc);
        std::fill(q_s.begin(), q_s.end(), 0);
    }


    IndexIVF_HNSW_Grouping::~IndexIVF_HNSW_Grouping()
    {
        delete pq;
        delete norm_pq;
        delete quantizer;
    }

    void IndexIVF_HNSW_Grouping::buildCoarseQuantizer(const char *path_clusters,
                                                      const char *path_info, const char *path_edges,
                                                      int M, int efConstruction=500)
    {
        if (exists_test(path_info) && exists_test(path_edges)) {
            quantizer = new hnswlib::HierarchicalNSW(path_info, path_clusters, path_edges);
            quantizer->ef_ = efConstruction;
            return;
        }
        quantizer = new hnswlib::HierarchicalNSW(d, nc, M, 2*M, efConstruction);
        quantizer->ef_ = efConstruction;

        std::cout << "Constructing quantizer\n";
        int j1 = 0;
        std::ifstream input(path_clusters, ios::binary);

        float mass[d];
        readXvec<float>(input, mass, d);
        quantizer->addPoint((void *) (mass), j1);

        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < nc; i++) {
            float mass[d];
#pragma omp critical
            {
                readXvec<float>(input, mass, d);
                if (++j1 % report_every == 0)
                    std::cout << j1 / (0.01 * nc) << " %\n";
            }
            quantizer->addPoint((void *) (mass), (size_t) j1);
        }
        input.close();
        quantizer->SaveInfo(path_info);
        quantizer->SaveEdges(path_edges);
    }


    void IndexIVF_HNSW_Grouping::assign(size_t n, const float *data, idx_t *idxs)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            idxs[i] = quantizer->searchKnn(const_cast<float *>(data + i * d), 1).top().second;
    }

    void IndexIVF_HNSW_Grouping::add_group(int centroid_num, int groupsize,
                                           const float *data, const idx_t *idxs,
                                           double &baseline_average, double &modified_average)
    {
        if (groupsize == 0)
            return;

        /** Find NN centroids to source centroid **/
        const float *centroid = (float *) quantizer->getDataByInternalId(centroid_num);
        std::priority_queue<std::pair<float, idx_t>> nn_centroids_raw = quantizer->searchKnn((void *) centroid,
                                                                                                 nsubc + 1);
        /** Vectors for construction **/
        std::vector<float> centroid_vector_norms_L2sqr(nsubc);
        nn_centroid_idxs[centroid_num].resize(nsubc);
        while (nn_centroids_raw.size() > 1) {
            centroid_vector_norms_L2sqr[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
            nn_centroid_idxs[centroid_num][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
            nn_centroids_raw.pop();
        }

        const float *centroid_vector_norms = centroid_vector_norms_L2sqr.data();
        const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();

        /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
        std::vector<float> centroid_vectors(nsubc * d);
        for (int subc = 0; subc < nsubc; subc++) {
            float *neighbor_centroid = (float *) quantizer->getDataByInternalId(nn_centroids[subc]);
            sub_vectors(centroid_vectors.data() + subc * d, neighbor_centroid, centroid);
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
        compute_residuals(groupsize, residuals.data(), data, subcentroids.data(), subcentroid_idxs.data());

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
            baseline_average += faiss::fvec_L2sqr(centroid, point, d);
            modified_average += faiss::fvec_L2sqr(subcentroid, point, d);
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

    void IndexIVF_HNSW_Grouping::search(float *x, idx_t k, float *distances, long *labels)
    {
        std::vector<float> r;
        std::vector<idx_t> subcentroid_nums;
        subcentroid_nums.reserve(nsubc * nprobe);
        idx_t keys[nprobe];
        float q_c[nprobe];
        const float eps = 0.00001;

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

        /** Compute Query Table **/
        pq->compute_inner_prod_table(x, query_table.data());

        /** Prepare max heap with \k answers **/
        faiss::maxheap_heapify(k, distances, labels);

        /** Pruning **/
        double threshold = 0.0;
        if (isPruning) {
            int ncode = 0;
            int normalize = 0;

            r.resize(nsubc * nprobe);
            for (int i = 0; i < nprobe; i++) {
                idx_t centroid_num = keys[i];

                if (norm_codes[centroid_num].size() == 0)
                    continue;
                ncode += norm_codes[centroid_num].size();

                float *subr = r.data() + i * nsubc;
                const idx_t *groupsizes = group_sizes[centroid_num].data();
                const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
                float alpha = alphas[centroid_num];

                for (int subc = 0; subc < nsubc; subc++) {
                    if (groupsizes[subc] == 0)
                        continue;

                    idx_t subcentroid_num = nn_centroids[subc];

                    if (q_s[subcentroid_num] < eps) {
                        const float *nn_centroid = (float *) quantizer->getDataByInternalId(subcentroid_num);
                        q_s[subcentroid_num] = faiss::fvec_L2sqr(x, nn_centroid, d);
                        subcentroid_nums.push_back(subcentroid_num);
                        counter_computed++;
                    } else counter_reused++;

                    subr[subc] = (1 - alpha)*(q_c[i] - alpha * s_c[centroid_num][subc]) + alpha*q_s[subcentroid_num];
                    threshold += subr[subc];
                    normalize++;
                }
                if (ncode >= 2 * max_codes)
                    break;
            }
            threshold /= normalize;
        }

        int ncode = 0;
        for (int i = 0; i < nprobe; i++) {
            idx_t centroid_num = keys[i];
            if (norm_codes[centroid_num].size() == 0)
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

                if (isPruning && r[i * nsubc + subc] > threshold) {
                    code += groupsize * code_size;
                    norm_code += groupsize;
                    id += groupsize;
                    filter_points += groupsize;
                    continue;
                }

                idx_t subcentroid_num = nn_centroids[subc];
                if (q_s[subcentroid_num] < eps) {
                    const float *nn_centroid = (float *) quantizer->getDataByInternalId(subcentroid_num);
                    q_s[subcentroid_num] = faiss::fvec_L2sqr(x, nn_centroid, d);
                    subcentroid_nums.push_back(subcentroid_num);
                    counter_computed++;
                } else counter_reused += !isPruning;

                float snd_term = alpha * (q_s[subcentroid_num] - centroid_norms[subcentroid_num]);
                norm_pq->decode(norm_code, norms.data(), groupsize);

                for (int j = 0; j < groupsize; j++) {
                    float q_r = fstdistfunc(code + j * code_size);
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
        average_max_codes += ncode;

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
        /** Read Indexes **/
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

        fclose(fin);
    }


    void IndexIVF_HNSW_Grouping::train_pq(const size_t n, const float *x)
    {
        std::vector<float> train_subcentroids;
        std::vector<idx_t> train_subcentroid_idxs;

        std::vector<float> train_residuals;
        std::vector<idx_t> assigned(n);
        assign(n, x, assigned.data());

        std::unordered_map<idx_t, std::vector<float>> group_map;

        for (int i = 0; i < n; i++) {
            idx_t key = assigned[i];
            for (int j = 0; j < d; j++)
                group_map[key].push_back(x[i * d + j]);
        }

        /** Train Residual PQ **/
        std::cout << "Training Residual PQ codebook " << std::endl;
        for (auto p : group_map) {
            const idx_t centroid_num = p.first;
            const float *centroid = (float *) quantizer->getDataByInternalId(centroid_num);
            const vector<float> data = p.second;
            const int groupsize = data.size() / d;

            std::vector<idx_t> nn_centroids(nsubc);
            std::vector<float> centroid_vector_norms(nsubc);
            auto nn_centroids_raw = quantizer->searchKnn((void *) centroid, nsubc + 1);

            while (nn_centroids_raw.size() > 1) {
                centroid_vector_norms[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
                nn_centroids[nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
                nn_centroids_raw.pop();
            }

            /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
            std::vector<float> centroid_vectors(nsubc * d);
            for (int i = 0; i < nsubc; i++) {
                const float *neighbor_centroid = (float *) quantizer->getDataByInternalId(nn_centroids[i]);
                sub_vectors(centroid_vectors.data() + i * d, neighbor_centroid, centroid);
            }

            /** Find alphas for vectors **/
            float alpha = compute_alpha(centroid_vectors.data(), data.data(), centroid,
                                        centroid_vector_norms.data(), groupsize);

            /** Compute final subcentroids **/
            std::vector<float> subcentroids(nsubc * d);
            for (int subc = 0; subc < nsubc; subc++) {
                const float *centroid_vector = centroid_vectors.data() + subc * d;
                float *subcentroid = subcentroids.data() + subc * d;

                faiss::fvec_madd(d, centroid, alpha, centroid_vector, subcentroid);
            }

            /** Find subcentroid idx **/
            std::vector<idx_t> subcentroid_idxs(groupsize);
            compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), groupsize);

            /** Compute Residuals **/
            std::vector<float> residuals(groupsize * d);
            compute_residuals(groupsize, residuals.data(), data.data(), subcentroids.data(),
                              subcentroid_idxs.data());

            for (int i = 0; i < groupsize; i++) {
                train_subcentroid_idxs.push_back(subcentroid_idxs[i]);
                for (int j = 0; j < d; j++) {
                    train_subcentroids.push_back(subcentroids[i * d + j]);
                    train_residuals.push_back(residuals[i * d + j]);
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
        const idx_t *subcentroid_idxs = train_subcentroid_idxs.data();

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
            reconstruct(groupsize, reconstructed_x.data(), decoded_residuals.data(),
                        subcentroids, subcentroid_idxs);

            /** Compute norms **/
            std::vector<float> group_norms(groupsize);
            faiss::fvec_norms_L2sqr(group_norms.data(), reconstructed_x.data(), d, groupsize);

            for (int i = 0; i < groupsize; i++)
                train_norms.push_back(group_norms[i]);

            residuals += groupsize * d;
            subcentroids += groupsize * d;
            subcentroid_idxs += groupsize;
        }
        printf("Training %zdx%zd PQ on %ld vectors in 1D\n", norm_pq->M, norm_pq->ksub, train_norms.size());
        norm_pq->verbose = true;
        norm_pq->train(n, train_norms.data());
    }


    void IndexIVF_HNSW_Grouping::compute_centroid_norms()
    {
        centroid_norms.resize(nc);
#pragma omp parallel for
        for (int i = 0; i < nc; i++) {
            const float *centroid = (float *) quantizer->getDataByInternalId(i);
            centroid_norms[i] = faiss::fvec_norm_L2sqr(centroid, d);
        }
    }


    void IndexIVF_HNSW_Grouping::compute_s_c()
    {
        for (int i = 0; i < nc; i++) {
            const float *centroid = (float *) quantizer->getDataByInternalId(i);
            s_c[i].resize(nsubc);
            for (int subc = 0; subc < nsubc; subc++) {
                idx_t subc_idx = nn_centroid_idxs[i][subc];
                const float *subcentroid = (float *) quantizer->getDataByInternalId(subc_idx);
                s_c[i][subc] = faiss::fvec_L2sqr(subcentroid, centroid, d);
            }
        }
    }


    float IndexIVF_HNSW_Grouping::fstdistfunc(uint8_t *code)
    {
        float result = 0.;
        int dim = code_size >> 2;
        int m = 0;
        for (int i = 0; i < dim; i++) {
            result += query_table[pq->ksub * m + code[m]]; m++;
            result += query_table[pq->ksub * m + code[m]]; m++;
            result += query_table[pq->ksub * m + code[m]]; m++;
            result += query_table[pq->ksub * m + code[m]]; m++;
        }
        return result;
    }


    void IndexIVF_HNSW_Grouping::compute_residuals(size_t n, float *residuals, const float *points,
                                                   const float *subcentroids, const idx_t *keys)
    {
        //#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const float *subcentroid = subcentroids + keys[i] * d;
            const float *point = points + i * d;
            for (int j = 0; j < d; j++) {
                residuals[i * d + j] = point[j] - subcentroid[j];
            }
        }
    }

    void IndexIVF_HNSW_Grouping::reconstruct(size_t n, float *x, const float *decoded_residuals,
                                             const float *subcentroids, const idx_t *keys)
    {
//            #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const float *subcentroid = subcentroids + keys[i] * d;
            const float *decoded_residual = decoded_residuals + i * d;
            for (int j = 0; j < d; j++)
                x[i * d + j] = subcentroid[j] + decoded_residual[j];
        }
    }

    void IndexIVF_HNSW_Grouping::sub_vectors(float *target, const float *x, const float *y) {
        for (int i = 0; i < d; i++)
            target[i] = x[i] - y[i];
    }


    void IndexIVF_HNSW_Grouping::compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                                          const float *points, const int groupsize)
    {
//            #pragma omp parallel for num_threads(16)
        for (int i = 0; i < groupsize; i++) {
            std::priority_queue<std::pair<float, idx_t>> max_heap;
            for (int subc = 0; subc < nsubc; subc++) {
                const float *subcentroid = subcentroids + subc * d;
                const float *point = points + i * d;
                float dist = faiss::fvec_L2sqr(subcentroid, point, d);
                max_heap.emplace(std::make_pair(-dist, subc));
            }
            subcentroid_idxs[i] = max_heap.top().second;
        }
    }


    void IndexIVF_HNSW_Grouping::compute_vectors(float *target, const float *x, const float *centroid, const int n)
    {
//            #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                target[i * d + j] = x[i * d + j] - centroid[j];
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
        compute_vectors(point_vectors.data(), points, centroid, groupsize);

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

                float dist = faiss::fvec_L2sqr(point, subcentroid.data(), d);
                max_heap.emplace(std::make_pair(-dist, std::make_pair(numerator, denominator)));
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