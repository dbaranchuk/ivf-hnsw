#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_HNSW.h>
#include <ivf-hnsw/Parser.h>

using namespace hnswlib;
using namespace ivfhnsw;

/****************************/
/** Run IVF-HNSW on DEEP1B **/
/****************************/
int main(int argc, char **argv)
{
    /*******************/
    /** Parse Options **/
    /*******************/
    Parser opt = Parser(argc, argv);

    std::cout << opt.M << " " << opt.efConstruction << " " << opt.ngt << " " << opt.nq << std::endl;
    /**********************/
    /** Load Groundtruth **/
    /**********************/
    std::cout << "Loading groundtruth" << std::endl;
    std::vector<idx_t> massQA(opt.nq * opt.ngt);
    std::ifstream gt_input(opt.path_gt, ios::binary);
    readXvec<idx_t>(gt_input, massQA.data(), opt.ngt, opt.nq);
    gt_input.close();

    /******************/
    /** Load Queries **/
    /******************/
    std::cout << "Loading queries" << std::endl;
    std::vector<float> massQ(opt.nq * opt.d);
    std::ifstream query_input(opt.path_q, ios::binary);
    readXvec<float>(query_input, massQ.data(), opt.d, opt.nq);
    query_input.close();

    /**********************/
    /** Initialize Index **/
    /**********************/
    IndexIVF_HNSW *index = new IndexIVF_HNSW(opt.d, opt.nc, opt.code_size, 8);
    index->build_quantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);

    /********************/
    /** Load learn set **/
    /********************/
    std::ifstream learn_input(opt.path_learn, ios::binary);
    std::vector<float> trainvecs(opt.nt * opt.d);
    readXvec<float>(learn_input, trainvecs.data(), opt.d, opt.nt);
    learn_input.close();

    /** Set Random Subset of sub_nt trainvecs **/
    std::vector<float> trainvecs_rnd_subset(opt.nsubt * opt.d);
    random_subset(trainvecs.data(), trainvecs_rnd_subset.data(), opt.d, opt.nt, opt.nsubt);

    /**************/
    /** Train PQ **/
    /**************/
    if (exists_test(opt.path_pq) && exists_test(opt.path_norm_pq)) {
        std::cout << "Loading Residual PQ codebook from " << opt.path_pq << std::endl;
        index->pq = faiss::read_ProductQuantizer(opt.path_pq);
        std::cout << index->pq->d << " " << index->pq->code_size << " " << index->pq->dsub
                  << " " << index->pq->ksub << " " << index->pq->centroids[0] << std::endl;

        std::cout << "Loading Norm PQ codebook from " << opt.path_norm_pq << std::endl;
        index->norm_pq = faiss::read_ProductQuantizer(opt.path_norm_pq);
        std::cout << index->norm_pq->d << " " << index->norm_pq->code_size << " " << index->norm_pq->dsub
                  << " " << index->norm_pq->ksub << " " << index->norm_pq->centroids[0] << std::endl;
    }
    else {
        std::cout << "Training PQ codebooks" << std::endl;
        index->train_pq(opt.nsubt, trainvecs_rnd_subset.data());

        std::cout << "Saving Residual PQ codebook to " << opt.path_pq << std::endl;
        faiss::write_ProductQuantizer(index->pq, opt.path_pq);

        std::cout << "Saving Norm PQ codebook to " << opt.path_norm_pq << std::endl;
        faiss::write_ProductQuantizer(index->norm_pq, opt.path_norm_pq);
    }

    /************************/
    /** Precompute indexes **/
    /************************/
    if (!exists_test(opt.path_precomputed_idxs)){
        std::cout << "Precomputing indexes" << std::endl;
        StopW stopw = StopW();

        FILE *fout = fopen(opt.path_precomputed_idxs, "wb");
        std::ifstream input(opt.path_base, ios::binary);

        size_t batch_size = 1000000;
        size_t nbatch = opt.nb / batch_size;
        std::vector<float> batch(batch_size * opt.d);
        std::vector<idx_t> precomputed_idx(batch_size);

        index->quantizer->ef_ = 240;
        for (int i = 0; i < nbatch; i++) {
            if (i % 10 == 0) {
                std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] " << (100.*i) / nbatch << "%\n";
            }
            readXvecFvec<float>(input, batch.data(), opt.d, batch_size);
            index->assign(batch_size, batch.data(), precomputed_idx.data());

            fwrite((idx_t *) &batch_size, sizeof(idx_t), 1, fout);
            fwrite(precomputed_idx.data(), sizeof(idx_t), batch_size, fout);
        }
        input.close();
        fclose(fout);
    }

    /******************************/
    /** Construct IVF-HNSW Index **/
    /******************************/
    if (exists_test(opt.path_index)){
        /** Load Index **/
        std::cout << "Loading index from " << opt.path_index << std::endl;
        index->read(opt.path_index);
    } else {
        /** Add elements **/
        StopW stopw = StopW();

        std::ifstream base_input(opt.path_base, ios::binary);
        std::ifstream idx_input(opt.path_precomputed_idxs, ios::binary);

        size_t batch_size = 1000000;
        size_t nbatch = opt.nb / batch_size;
        std::vector<float> batch(batch_size * opt.d);
        std::vector <idx_t> idx_batch(batch_size);
        std::vector <idx_t> ids_batch(batch_size);

        for (int b = 0; b < nbatch; b++) {
            if (b % 10 == 0) {
                std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] " << (100. * b) / nbatch << "%\n";
            }
            readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
            readXvecFvec<float>(base_input, batch.data(), opt.d, batch_size);

            for (size_t i = 0; i < batch_size; i++)
                ids_batch[i] = batch_size * b + i;

            index->add_batch(batch_size, batch.data(), ids_batch.data(), idx_batch.data());
        }
        idx_input.close();
        base_input.close();

        /** Save index, pq and norm_pq **/
        std::cout << "Saving index to " << opt.path_index << std::endl;
        std::cout << "       pq to " << opt.path_pq << std::endl;
        std::cout << "       norm pq to " << opt.path_norm_pq << std::endl;

        /** Computing Centroid Norms **/
        std::cout << "Computing centroid norms"<< std::endl;
        index->compute_centroid_norms();
        index->write(opt.path_index);
    }

    /***********************/
    /** Parse groundtruth **/
    /***********************/
    std::cout << "Parsing groundtruth" << std::endl;
    std::vector<std::priority_queue< std::pair<float, idx_t >>> answers;
    (std::vector<std::priority_queue< std::pair<float, idx_t >>>(opt.nq)).swap(answers);
    for (int i = 0; i < opt.nq; i++)
        answers[i].emplace(0.0f, massQA[opt.ngt*i]);

    /***************************/
    /** Set search parameters **/
    /***************************/
    index->nprobe = opt.nprobe;
    index->max_codes = opt.max_codes;
    index->quantizer->ef_ = opt.efSearch;

    /************/
    /** Search **/
    /************/
    int correct = 0;
    float distances[opt.k];
    long labels[opt.k];

    StopW stopw = StopW();
    for (int i = 0; i < opt.nq; i++) {
        for (int j = 0; j < opt.k; j++){
            distances[j] = 0;
            labels[j] = 0;
        }

        index->search(opt.k, massQ.data() + i*opt.d, distances, labels);

        std::priority_queue<std::pair<float, idx_t >> gt(answers[i]);
        unordered_set<idx_t> g;

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        for (int j = 0; j < opt.k; j++)
            if (g.count(labels[j]) != 0) {
                correct++;
                break;
            }
    }
    /***********************/
    /** Represent results **/
    /***********************/
    float time_us_per_query = stopw.getElapsedTimeMicro() / opt.nq;
    std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.nq << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;

    delete index;
    return 0;
}