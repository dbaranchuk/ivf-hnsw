#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <unordered_set>

#include <ivf-hnsw/IndexIVF_HNSW_Grouping.h>
#include <ivf-hnsw/Parser.h>

using namespace hnswlib;
using namespace ivfhnsw;

/***************************************************/
/** Run IVF-HNSW + Grouping (+ Pruning) on DEEP1B **/
/***************************************************/
int main(int argc, char **argv)
{
    /*******************/
    /** Parse Options **/
    /*******************/
    Parser opt = Parser(argc, argv);

    /**********************/
    /** Load Groundtruth **/
    /**********************/
    std::cout << "Loading groundtruth from " << opt.path_gt << std::endl;
    std::vector<int> massQA(opt.nq * opt.ngt);
    std::ifstream gt_input(opt.path_gt, ios::binary);
    readXvec<int>(gt_input, massQA.data(), opt.ngt, opt.nq);
    gt_input.close();

    /******************/
    /** Load Queries **/
    /******************/
    std::cout << "Loading queries from " << opt.path_q << std::endl;
    std::vector<float> massQ(opt.nq * opt.d);
    std::ifstream query_input(opt.path_q, ios::binary);
    readXvecFvec<uint8_t>(query_input, massQ.data(), opt.d, opt.nq);
    query_input.close();

    /**********************/
    /** Initialize Index **/
    /**********************/
    IndexIVF_HNSW_Grouping *index = new IndexIVF_HNSW_Grouping(opt.d, opt.nc, opt.code_size, 8, opt.nsubc);
    index->build_quantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);

    /********************/
    /** Load learn set **/
    /********************/
    std::ifstream learn_input(opt.path_learn, ios::binary);
    std::vector<float> trainvecs(opt.nt * opt.d);
    readXvecFvec<uint8_t>(learn_input, trainvecs.data(), opt.d, opt.nt);
    learn_input.close();

    /** Set Random Subset of sub_nt trainvecs **/
    std::vector<float> trainvecs_rnd_subset(opt.nsubt * opt.d);
    random_subset(trainvecs.data(), trainvecs_rnd_subset.data(), opt.d, opt.nt, opt.nsubt);

    /**************/
    /** Train PQ **/
    /**************/
    if (exists(opt.path_pq) && exists(opt.path_norm_pq)) {
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
    /** Precompute indices **/
    /************************/
    if (!exists(opt.path_precomputed_idxs)){
        std::cout << "Precomputing indices" << std::endl;
        StopW stopw = StopW();

        std::ifstream input(opt.path_base, ios::binary);
        std::ofstream output(opt.path_precomputed_idxs, ios::binary);

        int batch_size = 1000000;
        int nbatches = opt.nb / batch_size;

        std::vector<float> batch(batch_size * opt.d);
        std::vector<idx_t> precomputed_idx(batch_size);

        index->quantizer->efSearch = 220;
        for (int i = 0; i < nbatches; i++) {
            if (i % 10 == 0) {
                std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                          << (100.*i) / nbatches << "%" << std::endl;
            }
            readXvecFvec<uint8_t>(input, batch.data(), opt.d, batch_size);
            index->assign(batch_size, batch.data(), precomputed_idx.data());

            output.write((char *) &batch_size, sizeof(int));
            output.write((char *) precomputed_idx.data(), batch_size * sizeof(idx_t));
        }
        input.close();
        output.close();
    }

    /*****************************************/
    /** Construct IVF-HNSW + Grouping Index **/
    /*****************************************/
    if (exists(opt.path_index)){
        /** Load Index **/
        std::cout << "Loading index from " << opt.path_index << std::endl;
        index->read(opt.path_index);
    } else {
        /** Adding groups to index **/
        std::cout << "Adding groups to index" << std::endl;
        StopW stopw = StopW();

        double baseline_average = 0.0;
        double modified_average = 0.0;

        int batch_size = 1000000;
        int nbatches = opt.nb / batch_size;
        int groups_per_iter = 100000;

        std::ofstream groups_output(opt.path_groups, ios::binary);
        std::ofstream ids_output(opt.path_idxs, ios::binary);

        std::vector<float> batch(batch_size * opt.d);
        std::vector<idx_t> idx_batch(batch_size);

        for (int ngroups_added = 0; ngroups_added < opt.nc; ngroups_added += groups_per_iter)
        {
            std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                      << ngroups_added << " / " << opt.nc << std::endl;

            if (opt.nc - ngroups_added <= groups_per_iter)
                groups_per_iter = opt.nc - ngroups_added;

            std::vector<std::vector<float>> data(groups_per_iter);
            std::vector<std::vector<idx_t>> ids(groups_per_iter);
            
            // Iterate through the dataset extracting points from groups,
            // whose ids lie in [ngroups_added, ngroups_added + groups_per_iter)
            std::ifstream base_input(opt.path_base, ios::binary);
            std::ifstream idx_input(opt.path_precomputed_idxs, ios::binary);

            for (int b = 0; b < nbatches; b++) {
                readXvecFvec<uint8_t>(base_input, batch.data(), opt.d, batch_size);
                readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);

                for (int i = 0; i < batch_size; i++) {
                    if (idx_batch[i] < ngroups_added ||
                        idx_batch[i] >= ngroups_added + groups_per_iter)
                        continue;

                    idx_t idx = idx_batch[i] % groups_per_iter;
                    for (int j = 0; j < opt.d; j++)
                        data[idx].push_back(batch[i * opt.d + j]);
                    ids[idx].push_back(b * batch_size + i);
                }
            }
            base_input.close();
            idx_input.close();

            int j1 = 0;
            #pragma omp parallel for reduction(+:baseline_average, modified_average)
            for (int i = 0; i < groups_per_iter; i++) {
                #pragma omp critical
                {
                    if (j1 % 10000 == 0) {
                        std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                                  << (100. * (ngroups_added+j1)) / 1000000 << "%" << std::endl;
                    }
                    j1++;
                }

                idx_t centroid_num = ngroups_added + i;
                int groupsize = ids[i].size();

                index->add_group(centroid_num, groupsize, data[i].data(), ids[i].data(), baseline_average, modified_average);
            }
        }

        std::cout << "[Baseline] Average Distance: " << baseline_average / opt.nb << std::endl;
        std::cout << "[Modified] Average Distance: " << modified_average / opt.nb << std::endl;

        /** Computing Centroid Norms **/
        std::cout << "Computing centroid norms"<< std::endl;
        index->compute_centroid_norms();
        std::cout << "Computing centroid dists"<< std::endl;
        index->compute_centroid_dists();

        /** Save index, pq and norm_pq **/
        std::cout << "Saving index to " << opt.path_index << std::endl;
        std::cout << "       pq to " << opt.path_pq << std::endl;
        std::cout << "       norm pq to " << opt.path_norm_pq << std::endl;
        index->write(opt.path_index);
    }

    /*****************************************/
    /** Construct IVF-HNSW + Grouping Index **/
    /*****************************************/
//    if (exists(opt.path_index)){
//        /** Load Index **/
//        std::cout << "Loading index from " << opt.path_index << std::endl;
//        index->read(opt.path_index);
//    } else {
//        /** Adding groups to index **/
//        std::cout << "Adding groups to index" << std::endl;
//        StopW stopw = StopW();
//
//        double baseline_average = 0.0;
//        double modified_average = 0.0;
//
//        std::ifstream input_groups(opt.path_groups, ios::binary);
//        std::ifstream input_idxs(opt.path_idxs, ios::binary);
//
//        int j1 = 0;
//#pragma omp parallel for reduction(+:baseline_average, modified_average)
//        for (int c = 0; c < opt.nc; c++) {
//            idx_t centroid_num;
//            int groupsize;
//            std::vector<float> data;
//            std::vector<uint8_t> pdata;
//            std::vector<idx_t> idxs;
//
//#pragma omp critical
//            {
//                int check_groupsize;
//                input_groups.read((char *) &groupsize, sizeof(int));
//                input_idxs.read((char *) &check_groupsize, sizeof(int));
//                if (check_groupsize != groupsize) {
//                    std::cout << "Wrong groupsizes: " << groupsize << " vs "
//                              << check_groupsize << std::endl;
//                    exit(1);
//                }
//
//                data.resize(groupsize * opt.d);
//                pdata.resize(groupsize * opt.d);
//                idxs.resize(groupsize);
//
//                input_groups.read((char *) pdata.data(), groupsize * opt.d * sizeof(uint8_t));
//                for (int i = 0; i < groupsize * opt.d; i++)
//                    data[i] = (1.0) * pdata[i];
//
//                input_idxs.read((char *) idxs.data(), groupsize * sizeof(idx_t));
//
//                if (j1 % 10000 == 0) {
//                    std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
//                              << (100. * j1) / 1000000 << "%" << std::endl;
//                }
//                centroid_num = j1++;
//            }
//            index->add_group(centroid_num, groupsize, data.data(), idxs.data(), baseline_average, modified_average);
//        }
//
//        std::cout << "[Baseline] Average Distance: " << baseline_average / opt.nb << std::endl;
//        std::cout << "[Modified] Average Distance: " << modified_average / opt.nb << std::endl;
//
//        input_groups.close();
//        input_idxs.close();
//
//        /** Computing Centroid Norms **/
//        std::cout << "Computing centroid norms"<< std::endl;
//        index->compute_centroid_norms();
//        std::cout << "Computing centroid dists"<< std::endl;
//        index->compute_centroid_dists();
//
//        /** Save index, pq and norm_pq **/
//        std::cout << "Saving index to " << opt.path_index << std::endl;
//        std::cout << "       pq to " << opt.path_pq << std::endl;
//        std::cout << "       norm pq to " << opt.path_norm_pq << std::endl;
//        index->write(opt.path_index);
//    }

    /** Parse groundtruth **/
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
    index->quantizer->efSearch = opt.efSearch;

    /************/
    /** Search **/
    /************/
    int correct = 0;
    float distances[opt.k];
    long labels[opt.k];

    StopW stopw = StopW();
    for (int i = 0; i < opt.nq; i++) {
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