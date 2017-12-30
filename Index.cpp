//
// Created by dbaranchuk on 23.12.17.
//

#include "Index.h"

namespace ivfhnsw {
    Index::Index(size_t dim, size_t ncentroids, size_t bytes_per_code, size_t nbits_per_idx):
            d(dim), nc(ncentroids)
    {
        pq = new faiss::ProductQuantizer(dim, bytes_per_code, nbits_per_idx);
        norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);
        code_size = pq->code_size;

        norms.resize(65536);

        query_table.resize(pq->ksub * pq->M);
    }


    Index::~Index()
    {
        if (quantizer) delete quantizer;
        if (pq) delete pq;
        if (norm_pq) delete norm_pq;
    }


    void Index::buildQuantizer(const char *path_clusters, const char *path_info,
                                     const char *path_edges, int M, int efConstruction = 500) {
        if (exists_test(path_info) && exists_test(path_edges)) {
            quantizer = new hnswlib::HierarchicalNSW(path_info, path_clusters, path_edges);
            quantizer->ef_ = efConstruction;
            return;
        }
        quantizer = new hnswlib::HierarchicalNSW(d, nc, M, 2 * M, efConstruction);

        std::cout << "Constructing quantizer\n";
        int j1 = 0;
        std::ifstream input(path_clusters, ios::binary);

        float mass[d];
        readXvec<float>(input, mass, d);
        quantizer->addPoint(mass);

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
            quantizer->addPoint(mass);
        }
        input.close();
        quantizer->SaveInfo(path_info);
        quantizer->SaveEdges(path_edges);
    }

    
    void Index::assign(size_t n, const float *x, idx_t *labels, idx_t k) {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
            labels[i] = quantizer->searchKnn(const_cast<float *>(x + i * d), k).top().second;
    }


    void Index::compute_centroid_norms()
    {
        centroid_norms.resize(nc);
        for (int i = 0; i < nc; i++) {
            const float *centroid = quantizer->getDataByInternalId(i);
            centroid_norms[i] = faiss::fvec_norm_L2sqr(centroid, d);
        }
    }


    float Index::fstdistfunc(uint8_t *code)
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
}