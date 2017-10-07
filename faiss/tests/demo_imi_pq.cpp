/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

#include <sys/time.h>


#include "../IndexPQ.h"
#include "../IndexIVFPQ.h"
#include "../IndexFlat.h"
#include "../index_io.h"

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}


template<typename type>
void readXvec(FILE *fin, std::vector<float> &data, int d, int n)
{
    int dim, ret;
    type mass[d];

    for (int i = 0; i < n; i++) {
        ret = fread((int *) &dim, sizeof(int), 1, fin);
        if (ret == 0) printf("Huin9\n");

        if (dim != d) {
            printf("Wrong dim\n");
            exit(1);
        }
        ret = fread((type *) mass, sizeof(type), d, fin);
        for (size_t j = 0; j < d; j++) {
            data[i * d + j] = mass[j];
        }
    }
}

template<typename type>
void readIvec(FILE *fin, std::vector<int> &data, int d, int n)
{
    int dim, ret;
    type mass[d];

    for (int i = 0; i < n; i++) {
        ret = fread((int *) &dim, sizeof(int), 1, fin);
        if (ret == 0) printf("Huin9\n");

        if (dim != d) {
            printf("Wrong dim\n");
            exit(1);
        }
        ret = fread((type *) mass, sizeof(type), d, fin);
        for (size_t j = 0; j < d; j++) {
            data[i * d + j] = mass[j];
        }
    }
}

void readFvec(FILE *fin, std::vector<float> &data, int d, int n)
{
    int dim, ret;

    for (int i = 0; i < n; i++) {
        ret = fread((int *) &dim, sizeof(int), 1, fin);
        if (ret == 0) printf("Huin9\n");

        if (dim != d) {
            printf("Wrong dim\n");
            exit(1);
        }
        ret = fread((float *) (data.data() + i*d), sizeof(float), d, fin);
    }
}

void readLvec(FILE *fin, std::vector<long> &data, long d, int n)
{
    long dim, ret;

    for (int i = 0; i < n; i++) {
        ret = fread((long *) &dim, sizeof(long), 1, fin);
        if (dim != d) {
            printf("Wrong dim\n");
            exit(1);
        }
        ret = fread((long *)(data.data() + i*d), sizeof(long), d, fin);
        if (ret != d){
            printf("Buffer is overloaded\n");
            exit(1);
        }
    }
}

int main (int argc, char **argv) {
    if (argc != 3) {
        printf("You forgot parameters, bitch\n");
        exit(1);
    }
    std::string prefix = std::string(argv[1]);

    double t0 = elapsed();

    // dimension of the vectors to index
    int d = 96;
    int k = atoi(argv[2]);

    // size of the database we plan to index
    size_t nb = 1000 * 1000000; // 1000
    size_t batch_size = 1000000;//1000000; // # size of the blocks to add

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt = 20 * 1000000; // 100

    //---------------------------------------------------------------
    // Define the core quantizer
    // We choose a multiple inverted index for faster training with less data
    // and because it usually offers best accuracy/speed trade-offs
    //
    // We here assume that its lifespan of this coarse quantizer will cover the
    // lifespan of the inverted-file quantizer IndexIVFFlat below
    // With dynamic allocation, one may give the responsability to free the
    // quantizer to the inverted-file index (with attribute do_delete_quantizer)
    //
    // Note: a regular clustering algorithm would be defined as:
    //       faiss::IndexFlatL2 coarse_quantizer (d);
    //
    // Use nhash=2 subquantizers used to define the product coarse quantizer
    // Number of bits: we will have 2^nbits_coarse centroids per subquantizer
    //                 meaning (2^12)^nhash distinct inverted lists
    //
    // The parameter bytes_per_code is determined by the memory
    // constraint, the dataset will use nb * (bytes_per_code + 8)
    // bytes.
    //
    // The parameter nbits_subq is determined by the size of the dataset to index.
    //
    size_t nhash = 2;
    size_t nbits_subq = 14;
    size_t ncentroids = 1 << (nhash * nbits_subq);  // total # of centroids
    size_t bytes_per_code = 16;

    faiss::MultiIndexQuantizer coarse_quantizer(d, nhash, nbits_subq);

    printf("IMI (%ld,%ld): %ld virtual centroids (target: %ld base vectors)\n",
           nhash, nbits_subq, ncentroids, nb);

    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)
    faiss::MetricType metric = faiss::METRIC_L2; // can be METRIC_INNER_PRODUCT
    //faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids, bytes_per_code, 8);
    //index.quantizer_trains_alone = true;

    // define the number of probes. 2048 is for high-dim, overkill in practice
    // Use 4-1024 depending on the trade-off speed accuracy that you want
    //index.nprobe = 1024;


    // the index can be re-loaded later with
    //faiss::Index * idx = faiss::read_index((prefix + std::string("/faiss/final_deep_imi_PQ16.faissindex")).c_str());
    faiss::Index * idx = faiss::read_index((prefix + std::string("/faiss/trained_deep_imi_PQ16.faissindex")).c_str());
    //faiss::Index * idx = faiss::read_index((prefix + std::string("/faiss/final_imi_PQ32_compact.faissindex")).c_str());
    //faiss::Index * idx = faiss::read_index((prefix + std::string("/faiss/final_imi_PQ8.faissindex")).c_str());
    faiss::IndexIVFPQ index = *(dynamic_cast<faiss::IndexIVFPQ *>(idx));
    //faiss::IndexIVFPQCompact index = *(dynamic_cast<faiss::IndexIVFPQCompact *>(idx));
    //faiss::IndexIVFPQCompact index(_index);
    //faiss::write_index(&index, (prefix + std::string("/faiss/final_deep_imi_PQ16_compact.faissindex")).c_str());

    index.max_codes = 10000;
    index.nprobe = 1024;
    std::cout << index.max_codes << " " << index.nprobe << std::endl;
    std::cout << index.pq.M << " " << index.pq.code_size << std::endl;

    //FILE *fin = fopen((prefix + std::string("Bigann/deep1B_base.fvecs")).c_str(), "rb");
    FILE *fin = fopen("/home/arbabenko/Bigann/deep1B_base.fvecs", "rb");
    FILE *fout = fopen((prefix + std::string("/faiss/precomputed_deep_idxs.lvecs")).c_str(), "wb");

    std::vector<float> batch(batch_size * d);
    long *precomputed_idx = new long[batch_size];
    for (int i = 0; i < nb / batch_size; i++) {
        std::cout << "Batch number: " << i << std::endl;
        //readXvec<float>(fin, batch, d, batch_size);
        readFvec(fin, batch, d, batch_size);
        index.quantizer->assign(batch_size, batch.data(), precomputed_idx);

        fwrite((long *) &batch_size, sizeof(long), 1, fout);
        fwrite(precomputed_idx, sizeof(long), batch_size, fout);
    }
    delete precomputed_idx;
    fclose(fin);
    fclose(fout);
    return 0;
}
//    { // training.
//
//        // The distribution of the training vectors should be the same
//        // as the database vectors. It could be a sub-sample of the
//        // database vectors, if sampling is not biased.
//        //FILE *fin = fopen((prefix + std::string("Bigann/deep1B_learn.fvecs")).c_str(), "rb");
////        FILE *fin = fopen("/home/arbabenko/Bigann/deep1B_learn.fvecs", "rb");
////
////        printf("[%.3f s] Reading %ld vectors in %dD for training\n", elapsed() - t0, nt, d);
////
////        std::vector<float> trainvecs(nt * d);
////        readXvec<float>(fin, trainvecs, d, nt);
////        fclose(fin);
////
//////        index.pq = faiss::ProductQuantizer(d, bytes_per_code, 8);
//////        index.code_size = index.pq.code_size;
//////        index.verbose = true;
//////        index.train_residual(nt, trainvecs.data());
////        printf("[%.3f s] Training the index\n", elapsed() - t0);
////        index.train(nt, trainvecs.data());
////
////        faiss::write_index(&index, (prefix + std::string("/faiss/trained_deep_imi_PQ16.faissindex")).c_str());
//    }
//
//    size_t nq = 10000;
//    std::vector<float> queries(nq * d);
//    // Load Queries
//    //FILE *fin_q = fopen((prefix + std::string("/bigann/bigann_query.bvecs")).c_str(), "rb");
//    FILE *fin_q = fopen("/home/arbabenko/Bigann/deep1B_queries.fvecs", "rb");
//
//    printf ("[%.3f s] Reading %ld query vectors in %dD\n", elapsed() - t0, nq, d);
//
//    readXvec<float>(fin_q, queries, d, nq);
//    fclose(fin_q);
//
//    printf ("[%.3f s] Construct index\n", elapsed() - t0);
//    {
//        // Read Precomputed Indexes
//        printf ("[%.3f s] Building a dataset of %ld vectors to index\n", elapsed() - t0, nb);
//
//        //FILE *fin_base = fopen((prefix + std::string("/bigann/bigann_base.bvecs")).c_str(), "rb");
//        FILE *fin_base = fopen("/home/arbabenko/Bigann/deep1B_base.fvecs", "rb");
//        FILE *fin_idx = fopen((prefix + std::string("/faiss/precomputed_deep_idxs.lvecs")).c_str(), "rb");
//        std::vector<float> batch(batch_size * d);
//        std::vector<long> idx_batch(batch_size);
//        std::vector<long> ids(nb);
//
//        for (int b = 0; b < (nb / batch_size); b++) {
//            readLvec(fin_idx, idx_batch, batch_size, 1);
//            readXvec<float>(fin_base, batch, d, batch_size);
//            for (size_t i = 0; i < batch_size; i++)
//                ids[batch_size*b + i] = batch_size*b + i;
//
//            printf("[%.3f s] %.1f %c \n", elapsed() - t0, (100.*b)/(nb/batch_size), '%');
//
//            index.add_core_o(batch_size, batch.data(), ids.data() + batch_size*b, nullptr, idx_batch.data());
//        }
//        fclose(fin_base);
//        fclose(fin_idx);
//    }
//
//    // A few notes on the internal format of the index:
//    //
//    // - the positing lists for PQ codes are index.codes, which is a
//    //    std::vector < std::vector<uint8_t> >
//    //   if n is the length of posting list #i, codes[i] has length bytes_per_code * n
//    //
//    // - the corresponding ids are stored in index.ids
//    //
//    // - given a vector float *x, finding which k centroids are
//    //   closest to it (ie to find the nearest neighbors) can be done with
//    //
//    //   long *centroid_ids = new long[k];
//    //   float *distances = new float[k];
//    //   index.quantizer->search (1, x, k, dis, centroids_ids);
//
//    //std::string gt_path = prefix + std::string("/bigann/gnd/idx_1000M.ivecs");
//    std::string gt_path = "/home/arbabenko/Bigann/deep1B_groundtruth.ivecs";
//
//    FILE *fin_gt = fopen(gt_path.c_str(), "rb");
//    int gt_dim = 1;
//    std::vector<int> gt(nq * gt_dim);
//    readIvec<int>(fin_gt, gt, gt_dim, nq);
//    fclose(fin_gt);
//
////    std::string final_idx_path = prefix + std::string("/faiss/final_imi_PQ32.faissindex");
////    std::string final_idx_path = prefix + std::string("/faiss/final_deep_imi_PQ16.faissindex");
////    faiss::write_index(&index, final_idx_path.c_str());
////    faiss::write_index(&index, (prefix + std::string("/faiss/final_deep_imi_PQ16_backup.faissindex")).c_str());
//
//    { // searching the database
//
//
//        std::vector<faiss::Index::idx_t> nns (k * nq);
//        std::vector<float>               dis (k * nq);
//
//        printf ("[%.3f s] Searching the %d nearest neighbors of %ld vectors in the index\n", elapsed() - t0, k, nq);
//        int correct = 0;
//        for (int i = 0; i < nq; i++) {
//            //if (i < nq - 3) continue;
//            int answer = gt[i * gt_dim];
//
//            //printf ("query %2d: ", i);
//            index.search (1, queries.data()+d*i, k, dis.data()+k*i, nns.data()+k*i);
//            for (int j = 0; j < k; j++) {
//                //printf ("%7ld ", nns[j + i * k]);
//                if (answer == nns[j + i * k]){
//                    correct++;
//                    break;
//                }
//            }
//            //printf ("\n     dis: ");
//            //for (int j = 0; j < k; j++) {
//            //printf ("%7g ", dis[j + i * k]);
//            //}
//            //printf ("\n");
//        }
//        printf ("[%.3f s] Query results (vector ids, then distances):\n", elapsed() - t0);
//        printf("Recall@%d: %f\n",k, (1.*correct) / nq);
//    }
//    return 0;
//}
