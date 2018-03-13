#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_HNSW.h>
#include <ivf-hnsw/Parser.h>

using namespace hnswlib;
using namespace ivfhnsw;

//====================
// IVF-HNSW on SIFT1B
//====================
int main(int argc, char **argv)
{
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
    IndexIVF_HNSW *index = new IndexIVF_HNSW(opt.d, opt.nc, opt.code_size, 8);
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
            std::cout << "Loading OPQ rotation matrix from " << opt.path_opq_matrix << std::endl;
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

        std::cout << "Saving Residual PQ codebook to " << opt.path_pq << std::endl;
        faiss::write_ProductQuantizer(index->pq, opt.path_pq);

        if (opt.do_opq){
            std::cout << "Saving OPQ rotation matrix to " << opt.path_opq_matrix << std::endl;
            faiss::write_VectorTransform(index->opq_matrix, opt.path_opq_matrix);
        }
        std::cout << "Saving Norm PQ codebook to " << opt.path_norm_pq << std::endl;
        faiss::write_ProductQuantizer(index->norm_pq, opt.path_norm_pq);
    }

    /************************/
    /** Precompute indexes **/
    /************************/
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

    /******************************/
    /** Construct IVF-HNSW Index **/
    /******************************/
    if (exists(opt.path_index)){
        // Load Index
        std::cout << "Loading index from " << opt.path_index << std::endl;
        index->read(opt.path_index);
    } else {
        // Add elements
        StopW stopw = StopW();

        std::ifstream base_input(opt.path_base, std::ios::binary);
        std::ifstream idx_input(opt.path_precomputed_idxs, std::ios::binary);

        const size_t batch_size = 1000000;
        const size_t nbatches = opt.nb / batch_size;
        std::vector<float> batch(batch_size * opt.d);
        std::vector <idx_t> idx_batch(batch_size);
        std::vector <idx_t> ids_batch(batch_size);

        for (size_t b = 0; b < nbatches; b++) {
            if (b % 10 == 0) {
                std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] " << (100. * b) / nbatches << "%\n";
            }
            readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
            readXvecFvec<uint8_t>(base_input, batch.data(), opt.d, batch_size);

            for (size_t i = 0; i < batch_size; i++)
                ids_batch[i] = batch_size * b + i;

            index->add_batch(batch_size, batch.data(), ids_batch.data(), idx_batch.data());
        }

        // Computing Centroid Norms
        std::cout << "Computing centroid norms"<< std::endl;
        index->compute_centroid_norms();

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