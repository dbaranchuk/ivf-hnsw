//
// Created by dbaranchuk on 15.12.17.
//

#include <IndexIVF_HNSW.h>

/** Common IndexIVF + HNSW **/

/** Public **/
IndexIVF_HNSW::IndexIVF_HNSW(size_t dim, size_t ncentroids, size_t bytes_per_code, size_t nbits_per_idx):
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

void IndexIVF_HNSW::buildCoarseQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                                         const char *path_info, const char *path_edges,
                                         int M, int efConstruction=500)
{
    if (exists_test(path_info) && exists_test(path_edges)) {
        quantizer = new HierarchicalNSW<float, float>(l2space, path_info, path_clusters, path_edges);
        quantizer->ef_ = efConstruction;
        return;
    }
    quantizer = new HierarchicalNSW<float, float>(l2space, nc, M, 2*M, efConstruction);
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

void IndexIVF_HNSW::assign(size_t n, const float *data, idx_t *idxs) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        idxs[i] = quantizer->searchKnn(const_cast<float *>(data + i * d), 1).top().second;
}

template<typename ptype>
void IndexIVF_HNSW::add(size_t n, const char *path_data, const char *path_precomputed_idxs)
{
    if (!exists_test(path_precomputed_idxs))
        precompute_idx<ptype>(n, path_data, path_precomputed_idxs);

    StopW stopw = StopW();

    const size_t batch_size = 1000000;
    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * d);
    std::vector<idx_t> idx_batch(batch_size);
    std::vector<idx_t> ids_batch(batch_size);

    for (int b = 0; b < (n / batch_size); b++) {
        readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
        readXvecFvec<ptype>(base_input, batch.data(), d, batch_size);

        for (size_t i = 0; i < batch_size; i++)
            ids_batch[i] = batch_size*b + i;

        if (b % 10 == 0)
            std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                      << (100.*b)/(n / batch_size) << "%" << std::endl;

        add_batch(batch_size, batch.data(), ids_batch.data(), idx_batch.data());
    }
    idx_input.close();
    base_input.close();

    std::cout << "Computing centroid norms"<< std::endl;
    compute_centroid_norms();
}

void IndexIVF_HNSW::add_batch(size_t n, const float *x, const idx_t *xids, const idx_t *idx)
{
    /** Compute residuals for original vectors **/
    std::vector<float> residuals(n * d);
    compute_residuals(n, x, residuals.data(), idx);

    /** Encode Residuals **/
    std::vector<uint8_t> xcodes(n * code_size);
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
    std::vector<uint8_t> xnorm_codes(n);
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

void IndexIVF_HNSW::search(float *x, idx_t k, float *distances, long *labels) {
    idx_t keys[nprobe];
    float q_c[nprobe];

    /** Find NN Centroids **/
    auto coarse = quantizer->searchKnn(x, nprobe);
    for (int i = nprobe - 1; i >= 0; i--) {
        auto elem = coarse.top();
        q_c[i] = elem.first;
        keys[i] = elem.second;
        coarse.pop();
    }

    /** Compute Query Table **/
    pq->compute_inner_prod_table(x, query_table.data());

    /** Prepare max heap with \k answers **/
    faiss::maxheap_heapify(k, distances, labels);

    int ncode = 0;
    for (int i = 0; i < nprobe; i++) {
        idx_t key = keys[i];
        std::vector<uint8_t> code = codes[key];
        std::vector<uint8_t> norm_code = norm_codes[key];
        float term1 = q_c[i] - centroid_norms[key];
        int ncodes = norm_code.size();

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
    average_max_codes += ncode;
}

void IndexIVF_HNSW::train_pq(idx_t n, const float *x)
{
    /** Assign train vectors **/
    std::vector<idx_t> assigned(n);
    assign(n, x, assigned.data());

    /** Compute residuals for original vectors **/
    std:vector<float> residuals(n * d);
    compute_residuals(n, x, residuals.data(), assigned.data());

    /** Train Residual PQ **/
    printf("Training %zdx%zd product quantizer on %ld vectors in %dD\n", pq->M, pq->ksub, n, d);
    pq->verbose = true;
    pq->train(n, residuals.data());

    /** Encode Residuals **/
    std::vector<uint8_t> xcodes(n * code_size);
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


template<typename ptype>
void IndexIVF_HNSW::precompute_idx(size_t n, const char *path_data, const char *path_precomputed_idxs)
{
    std::cout << "Precomputing indexes" << std::endl;
    size_t batch_size = 1000000;
    FILE *fout = fopen(path_precomputed_idxs, "wb");

    std::ifstream input(path_data, ios::binary);

    /** TODO **/
    //std::ofstream output(path_precomputed_idxs, ios::binary);

    std::vector<float> batch(batch_size * d);
    std::vector<idx_t> precomputed_idx(batch_size);

    for (int i = 0; i < n / batch_size; i++) {
        std::cout << "Batch number: " << i + 1 << " of " << n / batch_size << std::endl;
        readXvecFvec<ptype>(input, batch.data(), d, batch_size);
        assign(batch_size, batch.data(), precomputed_idx.data());

        fwrite((idx_t *) &batch_size, sizeof(idx_t), 1, fout);
        fwrite(precomputed_idx.data(), sizeof(idx_t), batch_size, fout);
    }
    input.close();
    fclose(fout);
}


void IndexIVF_HNSW::write(const char *path_index)
{
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

void IndexIVF_HNSW::read(const char *path_index)
{
    FILE *fin = fopen(path_index, "rb");

    fread(&d, sizeof(size_t), 1, fin);
    fread(&nc, sizeof(size_t), 1, fin);
    fread(&nprobe, sizeof(size_t), 1, fin);
    fread(&max_codes, sizeof(size_t), 1, fin);

    ids = std::vector<std::vector<idx_t>>(nc);
    codes = std::vector<std::vector<uint8_t>>(nc);
    norm_codes = std::vector<std::vector<uint8_t>>(nc);

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

void IndexIVF_HNSW::compute_s_c() {}

float IndexIVF_HNSW::fstdistfunc(uint8_t *code) {
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
