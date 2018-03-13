#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>

using namespace hnswlib;
using namespace ivfhnsw;

//===========================================
// IVF-HNSW + Grouping (+ Pruning) on DEEP1B
//===========================================
// Note: during construction process,
// we use <groups_per_iter> parameter.
// Set it based on the capacity of your RAM
//===========================================
int main(int argc, char **argv) {
    //===============
    // Parse Options 
    //===============
    Parser opt = Parser(argc, argv);

    //==================
    // Load Groundtruth 
    //==================
    std::cout << "Loading groundtruth from " << opt.path_gt << std::endl;
    std::vector<idx_t> massQA(opt.nq * opt.ngt);
    {
        std::ifstream gt_input(opt.path_gt, std::ios::binary);
        readXvec<idx_t>(gt_input, massQA.data(), opt.ngt, opt.nq);
    }
    //==============
    // Load Queries 
    //==============
    std::cout << "Loading queries from " << opt.path_q << std::endl;
    std::vector<float> massQ(opt.nq * opt.d);
    {
        std::ifstream query_input(opt.path_q, std::ios::binary);
        readXvecFvec<uint8_t>(query_input, massQ.data(), opt.d, opt.nq);
    }
    //==================
    // Initialize Index 
    //==================
    IndexIVF_HNSW_Grouping *index = new IndexIVF_HNSW_Grouping(opt.d, opt.nc, opt.code_size, 8, opt.nsubc);
    index->build_quantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);
    index->do_opq = opt.do_opq;

    //==========
    // Train PQ 
    //==========
    if (exists(opt.path_pq) && exists(opt.path_norm_pq)) {
        std::cout << "Loading Residual PQ codebook from " << opt.path_pq << std::endl;
        if (index->pq) delete index->pq;
        index->pq = faiss::read_ProductQuantizer(opt.path_pq);

        if (opt.do_opq){
            std::cout << "Loading Residual OPQ rotation matrix from " << opt.path_opq_matrix << std::endl;
            index->opq_matrix = dynamic_cast<faiss::LinearTransform *>(faiss::read_VectorTransform(opt.path_opq_matrix));
        }
        std::cout << "Loading Norm PQ codebook from " << opt.path_norm_pq << std::endl;
        if (index->norm_pq) delete index->norm_pq;
        index->norm_pq = faiss::read_ProductQuantizer(opt.path_norm_pq);
    }
    else {
        // Load learn set
        std::vector<float> trainvecs(opt.nt * opt.d);
        {
            std::ifstream learn_input(opt.path_learn, std::ios::binary);
            readXvecFvec<uint8_t>(learn_input, trainvecs.data(), opt.d, opt.nt);
        }
        // Set Random Subset of sub_nt trainvecs
        std::vector<float> trainvecs_rnd_subset(opt.nsubt * opt.d);
        random_subset(trainvecs.data(), trainvecs_rnd_subset.data(), opt.d, opt.nt, opt.nsubt);

        std::cout << "Training PQ codebooks" << std::endl;
        index->train_pq(opt.nsubt, trainvecs_rnd_subset.data());

        if (opt.do_opq){
            std::cout << "Saving Residual OPQ rotation matrix to " << opt.path_opq_matrix << std::endl;
            faiss::write_VectorTransform(index->opq_matrix, opt.path_opq_matrix);
        }
        std::cout << "Saving Residual PQ codebook to " << opt.path_pq << std::endl;
        faiss::write_ProductQuantizer(index->pq, opt.path_pq);

        std::cout << "Saving Norm PQ codebook to " << opt.path_norm_pq << std::endl;
        faiss::write_ProductQuantizer(index->norm_pq, opt.path_norm_pq);
    }

    //====================
    // Precompute indices 
    //====================
    if (!exists(opt.path_precomputed_idxs)){
        std::cout << "Precomputing indices" << std::endl;
        StopW stopw = StopW();

        std::ifstream input(opt.path_base, std::ios::binary);
        std::ofstream output(opt.path_precomputed_idxs, std::ios::binary);

        const uint32_t batch_size = 1000000;
        const size_t nbatches = opt.nb / batch_size;

        std::vector<float> batch(batch_size * opt.d);
        std::vector<idx_t> precomputed_idx(batch_size);

        index->quantizer->efSearch = 220;
        for (size_t i = 0; i < nbatches; i++) {
            if (i % 10 == 0) {
                std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                          << (100.*i) / nbatches << "%" << std::endl;
            }
            readXvecFvec<uint8_t>(input, batch.data(), opt.d, batch_size);
            index->assign(batch_size, batch.data(), precomputed_idx.data());

            output.write((char *) &batch_size, sizeof(uint32_t));
            output.write((char *) precomputed_idx.data(), batch_size * sizeof(idx_t));
        }
    }

    //=====================================
    // Construct IVF-HNSW + Grouping Index 
    //=====================================
    if (exists(opt.path_index)){
        // Load Index 
        std::cout << "Loading index from " << opt.path_index << std::endl;
        index->read(opt.path_index);
    } else {
        // Adding groups to index 
        std::cout << "Adding groups to index" << std::endl;
        StopW stopw = StopW();

        const size_t batch_size = 1000000;
        const size_t nbatches = opt.nb / batch_size;
        size_t groups_per_iter = 250000;

        std::vector<uint8_t> batch(batch_size * opt.d);
        std::vector<idx_t> idx_batch(batch_size);

        for (size_t ngroups_added = 0; ngroups_added < opt.nc; ngroups_added += groups_per_iter)
        {
            std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                      << ngroups_added << " / " << opt.nc << std::endl;

            std::vector<std::vector<uint8_t>> data(groups_per_iter);
            std::vector<std::vector<idx_t>> ids(groups_per_iter);

            // Iterate through the dataset extracting points from groups,
            // whose ids lie in [ngroups_added, ngroups_added + groups_per_iter)
            std::ifstream base_input(opt.path_base, std::ios::binary);
            std::ifstream idx_input(opt.path_precomputed_idxs, std::ios::binary);

            for (size_t b = 0; b < nbatches; b++) {
                readXvec<uint8_t>(base_input, batch.data(), opt.d, batch_size);
                readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);

                for (size_t i = 0; i < batch_size; i++) {
                    if (idx_batch[i] < ngroups_added ||
                        idx_batch[i] >= ngroups_added + groups_per_iter)
                        continue;

                    idx_t idx = idx_batch[i] % groups_per_iter;
                    for (size_t j = 0; j < opt.d; j++)
                        data[idx].push_back(batch[i * opt.d + j]);
                    ids[idx].push_back(b * batch_size + i);
                }
            }

            // If <opt.nc> is not a multiple of groups_per_iter, change <groups_per_iter> on the last iteration
            if (opt.nc - ngroups_added <= groups_per_iter)
                groups_per_iter = opt.nc - ngroups_added;

            size_t j = 0;
            #pragma omp parallel for
            for (size_t i = 0; i < groups_per_iter; i++) {
                #pragma omp critical
                {
                    if (j % 10000 == 0) {
                        std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                                  << (100. * (ngroups_added + j)) / opt.nc << "%" << std::endl;
                    }
                    j++;
                }
                const size_t group_size = ids[i].size();
                std::vector<float> group_data(group_size * opt.d);
                // Convert bytes to floats
                for (size_t k = 0; k < group_size * opt.d; k++)
                    group_data[k] = 1. *data[i][k];

                index->add_group(ngroups_added + i, group_size, group_data.data(), ids[i].data());
            }
        }
        // Computing centroid norms and inter-centroid distances
        std::cout << "Computing centroid norms"<< std::endl;
        index->compute_centroid_norms();
        std::cout << "Computing centroid dists"<< std::endl;
        index->compute_inter_centroid_dists();

        // Save index, pq and norm_pq 
        std::cout << "Saving index to " << opt.path_index << std::endl;
        index->write(opt.path_index);
    }
    // For correct search using OPQ encoding rotate points in the coarse quantizer
    if (opt.do_opq) {
        std::cout << "Rotating centroids"<< std::endl;
        index->rotate_quantizer();
    }
    //===================
    // Parse groundtruth
    //=================== 
    std::cout << "Parsing groundtruth" << std::endl;
    std::vector<std::priority_queue< std::pair<float, idx_t >>> answers;
    (std::vector<std::priority_queue< std::pair<float, idx_t >>>(opt.nq)).swap(answers);
    for (size_t i = 0; i < opt.nq; i++)
        answers[i].emplace(0.0f, massQA[opt.ngt*i]);

    //=======================
    // Set search parameters
    //=======================
    index->nprobe = opt.nprobe;
    index->max_codes = opt.max_codes;
    index->quantizer->efSearch = opt.efSearch;
    index->do_pruning = opt.do_pruning;

    //========
    // Search 
    //========
    size_t correct = 0;
    float distances[opt.k];
    long labels[opt.k];

    StopW stopw = StopW();
    for (size_t i = 0; i < opt.nq; i++) {
        index->search(opt.k, massQ.data() + i*opt.d, distances, labels);
        std::priority_queue<std::pair<float, idx_t >> gt(answers[i]);
        std::unordered_set<idx_t> g;

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        for (size_t j = 0; j < opt.k; j++)
            if (g.count(labels[j]) != 0) {
                correct++;
                break;
            }
    }
    //===================
    // Represent results 
    //===================
    const float time_us_per_query = stopw.getElapsedTimeMicro() / opt.nq;
    std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.nq << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;

    delete index;
    return 0;
}