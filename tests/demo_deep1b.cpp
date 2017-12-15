#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>

#include <ivf-hnsw/IndexIVF_HNSW.h>
#include <ivf-hnsw/utils.h>

#include <ivf-hnsw/faiss/index_io.h>

#include <unordered_set>

#include "../Parser.h"

using namespace hnswlib;
using namespace ivfhnsw;

/**
 * Run IVF-HNSW / IVF-HNSW + Grouping (+Pruning) on DEEP1B
 */
int main(int argc, char **argv)
{
//    save_groups_sift("/home/dbaranchuk/data/groups/sift1B_groups.bvecs",
//                     "/home/dbaranchuk/data/bigann/bigann_base.bvecs",
//                     "/home/dbaranchuk/sift1B_precomputed_idxs_993127.ivecs",
//                     993127, 128, vecsize);
//    exit(0);
    /** Parse Options **/
    Parser opt = Parser(argc, argv);

    cout << "Loading GT\n";
    std::vector<idx_t> massQA(opt.qsize * opt.gtdim);
    std::ifstream gt_input(opt.path_gt, ios::binary);
    readXvec<idx_t>(gt_input, massQA.data(), opt.gtdim, opt.qsize);
    gt_input.close();

    cout << "Loading queries\n";
    std::vector<float> massQ(opt.qsize * opt.vecdim);
    std::ifstream query_input(opt.path_q, ios::binary);
    readXvec<float>(query_input, massQ.data(), opt.vecdim, opt.qsize);
    query_input.close();

    SpaceInterface<float> *l2space = new L2Space(opt.vecdim);

    /** Create Index **/
    //IndexIVF_HNSW_Grouping *index = new IndexIVF_HNSW_Grouping(opt.vecdim, opt.ncentroids, opt.M_PQ, 8, nsubcentroids);
    IndexIVF_HNSW *index = new IndexIVF_HNSW(opt.vecdim, opt.ncentroids, opt.M_PQ, 8);
    index->buildCoarseQuantizer(l2space, opt.path_centroids,
                                opt.path_info, opt.path_edges,
                                opt.M, opt.efConstruction);

    /** Train PQ **/
    std::ifstream learn_input(opt.path_learn, ios::binary);
    int nt = 1000000;//262144;
    int sub_nt = 131072;//262144;//65536;
    std::vector<float> trainvecs(nt * opt.vecdim);
    readXvec<float>(learn_input, trainvecs.data(), opt.vecdim, nt);
    learn_input.close();

    /** Set Random Subset of sub_nt trainvecs **/
    std::vector<float> trainvecs_rnd_subset(sub_nt * opt.vecdim);
    random_subset(trainvecs.data(), trainvecs_rnd_subset.data(), opt.vecdim, nt, sub_nt);

    /** Train PQ **/
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
        index->train_pq(sub_nt, trainvecs_rnd_subset.data());

        std::cout << "Saving Residual PQ codebook to " << opt.path_pq << std::endl;
        faiss::write_ProductQuantizer(index->pq, opt.path_pq);

        std::cout << "Saving Norm PQ codebook to " << opt.path_norm_pq << std::endl;
        faiss::write_ProductQuantizer(index->norm_pq, opt.path_norm_pq);
    }

    if (exists_test(opt.path_index)){
        /** Load Index **/
        std::cout << "Loading index from " << opt.path_index << std::endl;
        index->read(opt.path_index);
    } else {
        /** Add elements **/

//      index->add<float>(opt.path_groups, opt.path_idxs);
        index->add<float>(opt.vecsize, opt.path_data, opt.path_precomputed_idxs);

        /** Save index, pq and norm_pq **/
        std::cout << "Saving index to " << opt.path_index << std::endl;
        std::cout << "       pq to " << opt.path_pq << std::endl;
        std::cout << "       norm pq to " << opt.path_norm_pq << std::endl;

        /** Computing Centroid Norms **/
        std::cout << "Computing centroid norms"<< std::endl;
        index->compute_centroid_norms();
        index->write(opt.path_index);
    }
    index->compute_s_c();

    /** Parse groundtruth **/
    vector<std::priority_queue< std::pair<float, labeltype >>> answers;
    std::cout << "Parsing gt\n";
    (vector<std::priority_queue< std::pair<float, labeltype >>>(opt.qsize)).swap(answers);
    for (int i = 0; i < opt.qsize; i++)
        answers[i].emplace(0.0f, massQA[opt.gtdim*i]);

    /** Set search parameters **/
    index->max_codes = opt.max_codes;
    index->nprobe = opt.nprobes;
    index->quantizer->ef_ = opt.efSearch;

    /** Search **/
    int correct = 0;
    float distances[opt.k];
    long labels[opt.k];

    //StopW stopw = StopW();
    for (int i = 0; i < opt.qsize; i++) {
        for (int j = 0; j < opt.k; j++){
            distances[j] = 0;
            labels[j] = 0;
        }

        index->search(massQ.data() + i*opt.vecdim, opt.k, distances, labels);

        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;

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
    /**Represent results**/
    //float time_us_per_query = stopw.getElapsedTimeMicro() / opt.qsize;
    std::cout << "Recall@" << opt.k << ": " << 1.0f * correct / opt.qsize << std::endl;
    //std::cout << "Time per query: " << time_us_per_query << " us" << std::endl;
    //std::cout << "Average max_codes: " << index->average_max_codes / 10000 << std::endl;
    //std::cout << "Average reused q_s: " << (1.0 * index->counter_reused) / (index->counter_computed + index->counter_reused) << std::endl;
    //std::cout << "Average number of pruned points: " << (1.0 * index->filter_points) / 10000 << std::endl;

    //check_groupsizes(index, ncentroids);
    //std::cout << "Check precomputed idxs"<< std::endl;
    //check_precomputing(index, path_data, path_precomputed_idxs, vecdim, ncentroids, vecsize, gt_mistakes, gt_correct);

    delete index;
    delete l2space;
    return 0;
}