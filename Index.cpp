//
// Created by dbaranchuk on 23.12.17.
//

#include "Index.h"

namespace ivfhnsw {
    void Index::buildCoarseQuantizer(const char *path_clusters, const char *path_info,
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


    void Index::assign(size_t n, const float *data, idx_t *idxs) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            idxs[i] = quantizer->searchKnn(const_cast<float *>(data + i * d), 1).top().second;

        }
    }
}