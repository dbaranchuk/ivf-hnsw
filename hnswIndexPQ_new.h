#pragma once

#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>

#include "hnswIndexPQ.h"

using namespace std;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }
    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

namespace hnswlib {

    struct ModifiedIndex
	{
        Index *index;

		size_t d;             /** Vector Dimension **/
		size_t nc;            /** Number of Centroids **/
        size_t nsubc;         /** Number of Subcentroids **/
        size_t code_size;     /** PQ Code Size **/

        /** Query members **/
        size_t nprobe = 16;
        size_t max_codes = 10000;

        /** NEW **/
        std::vector < std::vector < std::vector<idx_t> > > ids;
        std::vector < std::vector < std::vector<uint8_t> > > codes;
        std::vector < std::vector < std::vector<uint8_t> > > norm_codes;

        std::vector < std::vector < idx_t > > nn_centroid_idxs;
        std::vector < float > alphas;

    public:
		ModifiedIndex(Index *trained_index, size_t nsubcentroids = 128):
                index(trained_index), nsubc(nsubcentroids)
		{
            nc = index->csize;
            d = index->d;
            code_size = index->code_size;

            codes.resize(nc);
            norm_codes.resize(nc);
            ids.resize(nc);
            alphas.resize(nc);
            nn_centroid_idxs.resize(nc);

            for (int i = 0; i < nc; i++){
                ids[i].resize(nsubc);
                codes[i].resize(nsubc);
                norm_codes[i].resize(nsubc);
                nn_centroid_idxs[i].resize(nsubc);
            }

            dis_table.resize(index->pq->ksub * index->pq->M);
            norms.resize(65536);
        }


		~ModifiedIndex() {}


        void add(const char *path_groups, const char *path_idxs)
        {
            StopW stopw = StopW();

            double baseline_average = 0.0;
            double modified_average = 0.0;

            std::ifstream input_groups(path_groups, ios::binary);
            std::ifstream input_idxs(path_idxs, ios::binary);

            /** Find NN centroids to source centroid **/
            std::cout << "Find NN centroids to source centroids\n";
            std::vector< std::vector<float> > centroid_vector_norms_L2sqr(nc);

            #pragma omp parallel for num_threads(16)
            for (int i = 0; i < nc; i++) {
                const float *centroid = (float *) index->quantizer->getDataByInternalId(i);
                std::priority_queue<std::pair<float, idx_t>> nn_centroids_raw = index->quantizer->searchKnn((void *) centroid, nsubc + 1);

                centroid_vector_norms_L2sqr[i].resize(nsubc);
                while (nn_centroids_raw.size() > 1) {
                    centroid_vector_norms_L2sqr[i][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().first;
                    nn_centroid_idxs[i][nn_centroids_raw.size() - 2] = nn_centroids_raw.top().second;
                    nn_centroids_raw.pop();
                }
            }

//            FILE *fout = fopen("/home/dbaranchuk/data/groups/nn_centroid_idxs.ivecs", "wb");
//            for(int i = 0; i < nc; i++) {
//                int size = nn_centroid_idxs[i].size();
//                fwrite(&size, sizeof(int), 1, fout);
//                fwrite(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fout);
//            }
//            fclose(fout);

            /** Adding groups to index **/
            std::cout << "Adding groups to index\n";
            int j1 = 0;
            #pragma omp parallel for reduction(+:baseline_average, modified_average) num_threads(16)
            for (int c = 0; c < nc; c++) {
                /** Read Original vectors from Group file**/
                idx_t centroid_num;
                int groupsize;
                std::vector<float> data;
                std::vector<idx_t> idxs;

                const float *centroid;

                #pragma omp critical
                {
                    input_groups.read((char *) &groupsize, sizeof(int));
                    input_idxs.read((char *) &groupsize, sizeof(int));

                    data.resize(groupsize * d);
                    idxs.resize(groupsize);

                    input_groups.read((char *) data.data(), groupsize * d * sizeof(float));
                    input_idxs.read((char *) idxs.data(), groupsize * sizeof(idx_t));

                    centroid_num = j1++;
                    centroid = (float *) index->quantizer->getDataByInternalId(centroid_num);

                    if (j1 % 10000 == 0)
                        std::cout << "[" << stopw.getElapsedTimeMicro() / 1000000 << "s] "
                                  << (100. * j1) / 1000000 << "%" << std::endl;
                }

                if (groupsize == 0)
                    continue;

                /** Pruning **/
                //index->quantizer->getNeighborsByHeuristicMerge(nn_centroids_before_heuristic, maxM);

                const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
                const float *centroid_vector_norms = centroid_vector_norms_L2sqr[centroid_num].data();

                /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
                std::vector<float> centroid_vectors(nsubc * d);
                for (int i = 0; i < nsubc; i++) {
                    float *neighbor_centroid = (float *) index->quantizer->getDataByInternalId(nn_centroids[i]);
                    sub_vectors(centroid_vectors.data() + i * d, neighbor_centroid, centroid);
                }

                /** Find alphas for vectors **/
                alphas[centroid_num] = compute_alpha(centroid_vectors.data(), data.data(), centroid,
                                                     centroid_vector_norms, groupsize);

                /** Compute final subcentroids **/
                std::vector<float> subcentroids(nsubc * d);
                for (int subc = 0; subc < nsubc; subc++) {
                    const float *centroid_vector = centroid_vectors.data() + subc * d;
                    float *subcentroid = subcentroids.data() + subc * d;

                    linear_op(subcentroid, centroid_vector, centroid, alphas[centroid_num]);
                }

                /** Find subcentroid idx **/
                std::vector<idx_t> subcentroid_idxs(groupsize);
                compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), groupsize);

                /** Compute Residuals **/
                std::vector<float> residuals(groupsize*d);
                compute_residuals(groupsize, residuals.data(), data.data(), subcentroids.data(), subcentroid_idxs.data());

                /** Compute Codes **/
                std::vector<uint8_t> xcodes(groupsize * code_size);
                index->pq->compute_codes(residuals.data(), xcodes.data(), groupsize);

                /** Decode Codes **/
                std::vector<float> decoded_residuals(groupsize*d);
                index->pq->decode(xcodes.data(), decoded_residuals.data(), groupsize);

                /** Reconstruct Data **/
                std::vector<float> reconstructed_x(groupsize*d);
                reconstruct(groupsize, reconstructed_x.data(), decoded_residuals.data(),
                            subcentroids.data(), subcentroid_idxs.data());

                /** Compute norms **/
                std::vector<float> norms(groupsize);
                faiss::fvec_norms_L2sqr (norms.data(), reconstructed_x.data(), d, groupsize);

                /** Compute norm codes **/
                std::vector<uint8_t > xnorm_codes(groupsize);
                index->norm_pq->compute_codes(norms.data(), xnorm_codes.data(), groupsize);

                /** Add codes **/
                for (int i = 0; i < groupsize; i++) {
                    const idx_t idx = idxs[i];
                    const idx_t subcentroid_idx = subcentroid_idxs[i];

                    ids[centroid_num][subcentroid_idx].push_back(idx);
                    norm_codes[centroid_num][subcentroid_idx].push_back(xnorm_codes[i]);
                    for (int j = 0; j < d; j++)
                        codes[centroid_num][subcentroid_idx].push_back(xcodes[i*code_size + j]);


                    const float *subcentroid = subcentroids.data() + subcentroid_idx*d;
                    const float *point = data.data() + i * d;
                    baseline_average += faiss::fvec_L2sqr(centroid, point, d);
                    modified_average += faiss::fvec_L2sqr(subcentroid, point, d);
                }
            }
            std::cout << "[Baseline] Average Distance: " << baseline_average / 1000000000 << std::endl;
            std::cout << "[Modified] Average Distance: " << modified_average / 1000000000 << std::endl;
            input_groups.close();
            input_idxs.close();
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<float, idx_t> const &a,
                                      std::pair<float, idx_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        void group_search(const float *x, const float q_c,
                          std::priority_queue <std::pair<float, idx_t> > &topResults, const int centroid_num)
        {
            const idx_t *nn_centroids = nn_centroid_idxs[centroid_num].data();
            const float *centroid = (float *) index->quantizer->getDataByInternalId(centroid_num);
            const std::vector< std::vector < idx_t > > &group_ids = ids[centroid_num];
            const std::vector< std::vector < uint8_t > > &group_codes = codes[centroid_num];
            const std::vector< std::vector < uint8_t > > &group_norm_codes = norm_codes[centroid_num];
            const float alpha = alphas[centroid_num];

            for (int subc = 0; subc < nsubc; subc++){
                const float *nn_centroid = (float *) index->quantizer->getDataByInternalId(nn_centroids[subc]);
                const float q_s = faiss::fvec_L2sqr(x, nn_centroid, d);

                std::vector<uint8_t> code = group_codes[subc];
                std::vector<uint8_t> norm_code = group_norm_codes[subc];
                int groupsize = norm_code.size();

                index->norm_pq->decode(norm_code.data(), norms.data(), groupsize);

                for (int i = 0; i < groupsize; i++){
                    float q_r = fstdistfunc(code.data() + i*code_size);
                    float dist = (alpha - 1) * q_c - alpha*q_s - 2*q_r + norms[i];
                    idx_t label = group_ids[subc][i];
                    topResults.emplace(std::make_pair(-dist, label));
                }
            }
        }

		void search(float *x, idx_t k, idx_t *results)
		{
            idx_t keys[nprobe];
            float q_c[nprobe];

            index->pq->compute_inner_prod_table(x, dis_table.data());
            //std::priority_queue<std::pair<float, idx_t>, std::vector<std::pair<float, idx_t>>, CompareByFirst> topResults;
            std::priority_queue<std::pair<float, idx_t>> topResults;

            auto coarse = index->quantizer->searchKnn(x, nprobe);

            for (int i = nprobe - 1; i >= 0; i--) {
                auto elem = coarse.top();
                q_c[i] = elem.first;
                keys[i] = elem.second;
                coarse.pop();
            }

            for (int i = 0; i < nprobe; i++){
                idx_t key = keys[i];
                group_search(x, q_c[i], topResults, key);
                if (topResults.size() > max_codes)
                    break;
            }

            for (int i = 0; i < k; i++) {
                results[i] = topResults.top().second;
                topResults.pop();
            }
		}

        void write(const char *path_index)
        {
            FILE *fout = fopen(path_index, "wb");

            fwrite(&d, sizeof(size_t), 1, fout);
            fwrite(&nc, sizeof(size_t), 1, fout);
            fwrite(&nsubc, sizeof(size_t), 1, fout);

            idx_t size;
            /** Save Vector Indexes per  **/
            for (size_t i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    size = ids[i][j].size();
                    fwrite(&size, sizeof(idx_t), 1, fout);
                    fwrite(ids[i][j].data(), sizeof(idx_t), size, fout);
                }

            /** Save PQ Codes **/
            for(int i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    size = codes[i][j].size();
                    fwrite(&size, sizeof(idx_t), 1, fout);
                    fwrite(codes[i][j].data(), sizeof(uint8_t), size, fout);
                }

            /** Save Norm Codes **/
            for(int i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    size = norm_codes[i][j].size();
                    fwrite(&size, sizeof(idx_t), 1, fout);
                    fwrite(norm_codes[i][j].data(), sizeof(uint8_t), size, fout);
                }

            /** Save NN Centroid Indexes **/
            for(int i = 0; i < nc; i++) {
                size = nn_centroid_idxs[i].size();
                fwrite(&size, sizeof(idx_t), 1, fout);
                fwrite(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fout);
            }

            /** Save Alphas **/
            fwrite(alphas.data(), sizeof(float), nc, fout);

            fclose(fout);
        }

        void read(const char *path_index)
        {
            FILE *fin = fopen(path_index, "rb");

            fread(&d, sizeof(size_t), 1, fin);
            fread(&nc, sizeof(size_t), 1, fin);
            fread(&nsubc, sizeof(size_t), 1, fin);

            ids.resize(nc);
            codes.resize(nc);
            norm_codes.resize(nc);
            alphas.resize(nc);
            nn_centroid_idxs.resize(nc);

            for (int i = 0; i < nc; i++){
                ids[i].resize(nsubc);
                codes[i].resize(nsubc);
                norm_codes[i].resize(nsubc);
            }

            idx_t size;
            for (size_t i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    fread(&size, sizeof(idx_t), 1, fin);
                    ids[i][j].reserve(size);
                    fread(ids[i][j].data(), sizeof(idx_t), size, fin);
                }

            for(size_t i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    fread(&size, sizeof(idx_t), 1, fin);
                    codes[i][j].reserve(size);
                    fread(codes[i][j].data(), sizeof(uint8_t), size, fin);
                }

            for(size_t i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    fread(&size, sizeof(idx_t), 1, fin);
                    norm_codes[i][j].reserve(size);
                    fread(norm_codes[i][j].data(), sizeof(uint8_t), size, fin);
                }


            for(int i = 0; i < nc; i++) {
                fread(&size, sizeof(idx_t), 1, fin);
                nn_centroid_idxs[i].reserve(size);
                fread(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fin);
            }

            fread(alphas.data(), sizeof(float), nc, fin);

            fclose(fin);
        }

//        void compute_centroid_norm_table()
//        {
//            c_norm_table.reserve(nc);
//            for (int i = 0; i < nc; i++){
//                const float *centroid = (float *)quantizer->getDataByInternalId(i);
//                faiss::fvec_norm_L2sqr(c_norm_table.data() + i, centroid, d);
//            }
//        }

//        void train_norm_pq(idx_t n, const float *x)
//        {
//            idx_t *assigned = new idx_t [n]; // assignement to coarse centroids
//            assign (n, x, assigned);
//
//            float *residuals = new float [n * d];
//            compute_residuals (n, x, residuals, assigned);
//
//            uint8_t * xcodes = new uint8_t [n * code_size];
//            pq->compute_codes (residuals, xcodes, n);
//
//            float *decoded_residuals = new float[n * d];
//            pq->decode(xcodes, decoded_residuals, n);
//
//            float *reconstructed_x = new float[n * d];
//            reconstruct(n, reconstructed_x, decoded_residuals, assigned);
//
//            float *trainset = new float[n];
//            faiss::fvec_norms_L2sqr (trainset, reconstructed_x, d, n);
//
//            norm_pq->verbose = true;
//            norm_pq->train (n, trainset);
//
//            delete assigned;
//            delete residuals;
//            delete xcodes;
//            delete decoded_residuals;
//            delete reconstructed_x;
//            delete trainset;
//        }
//
//        void train_residual_pq(idx_t n, const float *x)
//        {
//            idx_t *assigned = new idx_t [n];
//            assign (n, x, assigned);
//
//            std::vector<float> residuals(n * d);
//            compute_residuals (n, x, residuals, assigned);
//
//            printf ("Training %zdx%zd product quantizer on %ld vectors in %dD\n",
//                    pq->M, pq->ksub, n, d);
//            pq->verbose = true;
//            pq->train (n, residuals);
//
//            delete assigned;
//        }

	private:
        std::vector<float> dis_table;
        std::vector<float> norms;

        float fstdistfunc(uint8_t *code)
        {
            float result = 0.;
            int dim = code_size >> 2;
            int m = 0;
            for (int i = 0; i < dim; i++) {
                result += dis_table[index->pq->ksub * m + code[m]]; m++;
                result += dis_table[index->pq->ksub * m + code[m]]; m++;
                result += dis_table[index->pq->ksub * m + code[m]]; m++;
                result += dis_table[index->pq->ksub * m + code[m]]; m++;
            }
            return result;
        }
    public:
        void compute_residuals(size_t n, float *residuals, const float *points, const float *subcentroids, const idx_t *keys)
		{
            //#pragma omp parallel for num_threads(16)
            for (idx_t i = 0; i < n; i++) {
                const float *subcentroid = subcentroids + keys[i]*d;
                const float *point = points + i*d;
                for (int j = 0; j < d; j++) {
                    residuals[i*d + j] = point[j] - subcentroid[j];
                }
            }
		}

        void reconstruct(size_t n, float *x, const float *decoded_residuals, const float *subcentroids, const idx_t *keys)
        {
            //#pragma omp parallel for num_threads(16)
            for (idx_t i = 0; i < n; i++) {
                const float *subcentroid = subcentroids + keys[i]*d;
                const float *decoded_residual = decoded_residuals + i*d;
                for (int j = 0; j < d; j++)
                    x[i*d + j] = subcentroid[j] + decoded_residual[j];
            }
        }

        /** NEW **/
        void add_vectors(float *target, const float *x, const float *y)
        {
            for (int i = 0; i < d; i++)
                target[i] = x[i] + y[i];
        }

        void sub_vectors(float *target, const float *x, const float *y)
        {
            for (int i = 0; i < d; i++)
                target[i] = x[i] - y[i];
        }

        void normalize_vector(float *x)
        {
            float norm = sqrt(faiss::fvec_norm_L2sqr(x, d));
            for (int i = 0; i < d; i++)
                x[i] /= norm;
        }

        void linear_op(float *target, const float *x, const float *y, const float alpha)
        {
            for (int i = 0; i < d; i++)
                target[i] = x[i] * alpha + y[i];
        }

        void compute_subcentroid_idxs(idx_t *subcentroid_idxs, const float *subcentroids,
                                      const float *points, const int groupsize)
        {
            //#pragma omp parallel for num_threads(16)
            for (int i = 0; i < groupsize; i++) {
                std::priority_queue<std::pair<float, idx_t>> max_heap;
                for (int subc = 0; subc < nsubc; subc++) {
                    const float *subcentroid = subcentroids + subc * d;
                    const float *point = points + i * d;
                    float dist = faiss::fvec_L2sqr(subcentroid, point, d);
                    max_heap.emplace(std::make_pair(-dist, subc));
                }
                subcentroid_idxs[i] = max_heap.top().second;
            }

        }

        void compute_vectors(float *target, const float *x, const float *centroid, const int n)
        {
            //#pragma omp parallel for num_threads(16)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    target[i*d + j] = x[i*d + j] - centroid[j];
        }

        float compute_alpha(const float *centroid_vectors, const float *points,
                            const float *centroid, const float *centroid_vector_norms_L2sqr,
                            const int groupsize)
        {
            int counter_positive = 0;
            int counter_negative = 0;
            float positive_alpha = 0.0;
            float negative_alpha = 0.0;

            std::vector<float> point_vectors(groupsize*d);
            compute_vectors(point_vectors.data(), points, centroid, groupsize);

            for (int i = 0; i < groupsize; i++) {
                const float *point_vector = point_vectors.data() + i * d;

                std::priority_queue<std::pair<float, float>> max_heap;
                for (int subc = 0; subc < nsubc; subc++){
                    const float *centroid_vector = centroid_vectors + subc * d;
                    const float centroid_vector_norm_L2sqr = centroid_vector_norms_L2sqr[subc * d];

                    float alpha = faiss::fvec_inner_product (centroid_vector, point_vector, d);
                    alpha /= centroid_vector_norm_L2sqr;

                    std::vector<float> subcentroid(d);
                    linear_op(subcentroid.data(), centroid_vector, centroid, alpha);

                    float dist = faiss::fvec_L2sqr(point_vector, subcentroid.data(), d);
                    max_heap.emplace(std::make_pair(-dist, alpha));
                }
                float optim_alpha = max_heap.top().second;
                if (optim_alpha < 0) {
                    counter_negative++;
                    negative_alpha += optim_alpha;
                } else {
                    counter_positive++;
                    positive_alpha += optim_alpha;
                }
            }
            positive_alpha /= counter_positive;
            negative_alpha /= counter_negative;

//    if (counter_positive == 0)
//        std::cout << "Positive Alpha: every alpha is negative" << std::endl;
//    else
//        std::cout << "Positive Alpha: " << positive_alpha << std::endl;
//
//    if (counter_negative == 0)
//        std::cout << "Negative Alpha: every alphas is positive" << std::endl;
//    else
//        std::cout << "Negative Alpha: " << negative_alpha << std::endl;
            return (counter_positive > counter_negative) ? positive_alpha : negative_alpha;
        }
	};

}
