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
    IndexIVF_HNSW_Grouping *index = new IndexIVF_HNSW_Grouping(opt.d, opt.nc, opt.code_size, 8, opt.nsubc);
    index->buildQuantizer(opt.path_centroids, opt.path_info, opt.path_edges, opt.M, opt.efConstruction);

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

        /** TODO **/
        //std::ofstream output(path_precomputed_idxs, ios::binary);
        size_t batch_size = 1000000;
        size_t nbatch = opt.nb / batch_size;
        std::vector<float> batch(batch_size * opt.d);
        std::vector<idx_t> precomputed_idx(batch_size);

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

    /****************************/
    /** TODO Precompute Groups **/
    /****************************/
    if (!exists_test(opt.path_groups)){ exit(0);}

    /******************************/
    /** Construct IVF-HNSW Index **/
    /******************************/
    if (exists_test(opt.path_index)){
        /** Load Index **/
        std::cout << "Loading index from " << opt.path_index << std::endl;
        index->read(opt.path_index);
    } else {
        /** Adding groups to index **/
        std::cout << "Adding groups to index" << std::endl;
        StopW stopw = StopW();

        double baseline_average = 0.0;
        double modified_average = 0.0;

        std::ifstream input_groups(opt.path_groups, ios::binary);
        std::ifstream input_idxs(opt.path_idxs, ios::binary);

        int j1 = 0;
        #pragma omp parallel for reduction(+:baseline_average, modified_average)
        for (int c = 0; c < opt.nc; c++) {
            idx_t centroid_num;
            int groupsize;
            std::vector<float> data;
            std::vector <idx_t> idxs;

            #pragma omp critical
            /** Read Original vectors from Group file**/
            {
                int check_groupsize;
                input_groups.read((char *) &groupsize, sizeof(int));
                input_idxs.read((char *) &check_groupsize, sizeof(int));
                if (check_groupsize != groupsize) {
                    std::cout << "Wrong groupsizes: " << groupsize << " vs "
                              << check_groupsize << std::endl;
                    exit(1);
                }

                data.resize(groupsize * opt.d);
                idxs.resize(groupsize);

                input_groups.read((char *) data.data(), groupsize * opt.d * sizeof(float));
                input_idxs.read((char *) idxs.data(), groupsize * sizeof(idx_t));

                if (j1 % 10000 == 0) {
                    std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                              << (100. * j1) / 1000000 << "%" << std::endl;
                }
                centroid_num = j1++;
            }
            index->add_group(centroid_num, groupsize, data.data(), idxs.data(), baseline_average, modified_average);
        }

        std::cout << "[Baseline] Average Distance: " << baseline_average / opt.nb << std::endl;
        std::cout << "[Modified] Average Distance: " << modified_average / opt.nb << std::endl;

        input_groups.close();
        input_idxs.close();

        /** Save index, pq and norm_pq **/
        std::cout << "Saving index to " << opt.path_index << std::endl;
        std::cout << "       pq to " << opt.path_pq << std::endl;
        std::cout << "       norm pq to " << opt.path_norm_pq << std::endl;

        /** Computing Centroid Norms **/
        std::cout << "Computing centroid norms"<< std::endl;
        index->compute_centroid_norms();
        index->compute_centroid_dists();
        index->write(opt.path_index);
    }

    /** Parse groundtruth **/
    std::cout << "Parsing groundtruth" << std::endl;
    std::vector<std::priority_queue< std::pair<float, idx_t >>> answers;
    (std::vector<std::priority_queue< std::pair<float, idx_t >>>(opt.nq)).swap(answers);
    for (int i = 0; i < opt.nq; i++)
        answers[i].emplace(0.0f, massQA[opt.gtd*i]);

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