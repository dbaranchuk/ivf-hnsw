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

		size_t d;
		size_t nc;
        size_t nsubc;
        size_t code_size;

        /** Query members **/
        size_t nprobe = 16;
        size_t max_codes = 10000;

        /** NEW **/
        std::vector < std::vector < std::vector<idx_t> > > ids;
        std::vector < std::vector < std::vector<uint8_t> > > codes;
        std::vector < std::vector < std::vector<uint8_t> > > norm_codes;

        std::vector < std::vector < idx_t > > nn_centroid_idxs;
        std::vector < float > alphas;

        //std::vector < float > c_norm_table;

    public:
		ModifiedIndex(Index *trained_index, size_t nsubcentroids = 128):
                index(trained_index), nsubc(nsubcentroids)
		{
            nc = index->csize;
            d = index->d;
            code_size = index->code_size;

            codes.reserve(nc);
            norm_codes.reserve(nc);
            ids.reserve(nc);
            alphas.reserve(nc);
            nn_centroid_idxs.reserve(nc);

            for (int i = 0; i < nc; i++){
                ids[i].reserve(nsubc);
                codes[i].reserve(nsubc);
                norm_codes[i].reserve(nsubc);
                nn_centroid_idxs[i].reserve(nsubc);
            }
        }


		~ModifiedIndex() {}


        void add(const char *path_groups, const char *path_idxs)
        {
            StopW stopw = StopW();

            std::ifstream input_groups(path_groups, ios::binary);
            std::ifstream input_idxs(path_idxs, ios::binary);

            double baseline_average = 0.0;
            double modified_average = 0.0;

            /** Find NN centroids to source centroid **/
            std::vector<std::priority_queue<std::pair<float, idx_t>>> nn_centroids_raw(nc);

            std::cout << "Find NN centroids to source centroids\n";
            #pragma omp parallel for num_threads(16)
            for (int i = 0; i < nc; i++) {
                const float *centroid = (float *) index->quantizer->getDataByInternalId(i);
                nn_centroids_raw[i] = index->quantizer->searchKnn((void *) centroid, nsubc + 1);
            }

            int j1 = 0;
            //#pragma omp parallel for reduction(+:baseline_average, modified_average) num_threads(16)
            for (int c = 0; c < nc; c++) {
                /** Read Original vectors from Group file**/
                idx_t centroid_num;
                int groupsize;
                std::vector<float> data;
                std::vector<idx_t> idxs;

                const float *centroid;

                //#pragma omp critical
                {
                    input_groups.read((char *) &groupsize, sizeof(int));
                    input_idxs.read((char *) &groupsize, sizeof(int));

                    data.reserve(groupsize * d);
                    idxs.reserve(groupsize);

                    input_groups.read((char *) data.data(), groupsize * d * sizeof(float));
                    input_idxs.read((char *) idxs.data(), groupsize * sizeof(idx_t));

                    centroid_num = j1++;
                    centroid = (float *) index->quantizer->getDataByInternalId(centroid_num);

                    if (j1 % 10000 == 0)
                        std::cout << (100. * j1)/1000000 << "%" << std::endl;
                }

                if (groupsize == 0)
                    continue;

                /** Remove source centroid from consideration **/
                std::vector<idx_t> &nn_centroids = nn_centroid_idxs[centroid_num];
                while (nn_centroids_raw[centroid_num].size() > 1) {
                    nn_centroids[nn_centroids_raw[centroid_num].size() - 1] = nn_centroids_raw[centroid_num].top().second;
                    nn_centroids_raw[centroid_num].pop();
                }

                /** Pruning **/
                //index->quantizer->getNeighborsByHeuristicMerge(nn_centroids_before_heuristic, maxM);

                /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
                std::vector<float> normalized_centroid_vectors(nc * d);
                for (int i = 0; i < nsubc; i++) {
                    float *neighbor_centroid = (float *) index->quantizer->getDataByInternalId(nn_centroids[i]);
                    sub_vectors(normalized_centroid_vectors.data() + i * d, neighbor_centroid, centroid);

                    /** Normalize them **/
                    normalize_vector(normalized_centroid_vectors.data() + i * d);
                }

                /** Find alphas for vectors **/
                alphas[centroid_num] = compute_alpha(normalized_centroid_vectors.data(), data.data(), centroid, groupsize);

                /** Compute final subcentroids **/
                std::vector<float> subcentroids(nsubc * d);
                for (int subc = 0; subc < nsubc; subc++) {
                    const float *centroid_vector = normalized_centroid_vectors.data() + subc * d;
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
                index->pq->compute_codes(point_vectors.data(), xcodes.data(), groupsize);

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
                index->norm_pq->compute_codes(norms.data(), xnorm_codes.data());

                /** Add codes **/
                for (int i = 0; i < groupsize; i++) {
                    const idx_t idx = idxs[i];
                    const idx_t subcentroid_idx = subcentroid_idxs[i];

                    ids[centroid_num][subcentroid_idx].push_back(idx);
                    norm_codes[centroid_num][subcentroid_idx].push_back(xnorm_code);
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
            std::cout << "Time(s): " << stopw.getElapsedTimeMicro() / 1000000 << std::endl;
            input_groups.close();
            input_idxs.close();
        }


        struct CompareByFirst {
            constexpr bool operator()(std::pair<float, idx_t> const &a,
                                      std::pair<float, idx_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };
//		void search (float *x, idx_t k, idx_t *results)
//		{
//            idx_t keys[nprobe];
//            float q_c[nprobe];
//            if (!norms)
//                norms = new float[65536];
//            if (!dis_table)
//                dis_table = new float [pq->ksub * pq->M];
//
//            pq->compute_inner_prod_table(x, dis_table);
//            std::priority_queue<std::pair<float, idx_t>, std::vector<std::pair<float, idx_t>>, CompareByFirst> topResults;
//            //std::priority_queue<std::pair<float, idx_t>> topResults;
//
//            auto coarse = quantizer->searchKnn(x, nprobe);
//
//            for (int i = nprobe - 1; i >= 0; i--) {
//                auto elem = coarse.top();
//                q_c[i] = elem.first;
//                keys[i] = elem.second;
//                coarse.pop();
//            }
//
//            for (int i = 0; i < nprobe; i++){
//                idx_t key = keys[i];
//                std::vector<uint8_t> code = codes[key];
//                std::vector<uint8_t> norm_code = norm_codes[key];
//                float term1 = q_c[i] - c_norm_table[key];
//                int ncodes = norm_code.size();
//
//                norm_pq->decode(norm_code.data(), norms, ncodes);
//
//                for (int j = 0; j < ncodes; j++){
//                    float q_r = fstdistfunc(code.data() + j*code_size);
//                    float dist = term1 - 2*q_r + norms[j];
//                    idx_t label = ids[key][j];
//                    topResults.emplace(std::make_pair(-dist, label));
//                }
//                if (topResults.size() > max_codes)
//                    break;
//            }
//
//            for (int i = 0; i < k; i++) {
//                results[i] = topResults.top().second;
//                topResults.pop();
//            }
//		}

        void write(const char *path_index)
        {
            FILE *fout = fopen(path_index, "wb");

            fwrite(&d, sizeof(size_t), 1, fout);
            fwrite(&nc, sizeof(size_t), 1, fout);
            fwrite(&nsubc, sizeof(size_t), 1, fout);

            idx_t size;
            for (size_t i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    size = ids[i][j].size();
                    fwrite(&size, sizeof(idx_t), 1, fout);
                    fwrite(ids[i][j].data(), sizeof(idx_t), size, fout);
                }


            for(int i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    size = codes[i][j].size();
                    fwrite(&size, sizeof(idx_t), 1, fout);
                    fwrite(codes[i][j].data(), sizeof(uint8_t), size, fout);
                }


            for(int i = 0; i < nc; i++)
                for (size_t j = 0; j < nsubc; j++) {
                    size = norm_codes[i][j].size();
                    fwrite(&size, sizeof(idx_t), 1, fout);
                    fwrite(norm_codes[i][j].data(), sizeof(uint8_t), size, fout);
                }

            for(int i = 0; i < nc; i++) {
                size = nn_centroid_idxs[i].size();
                fwrite(&size, sizeof(idx_t), 1, fout);
                fwrite(nn_centroid_idxs[i].data(), sizeof(idx_t), size, fout);
            }

            fwrite(alphas.data(), sizeof(float), nc, fout);

            fclose(fout);
        }

        void read(const char *path_index)
        {
            FILE *fin = fopen(path_index, "rb");

            fread(&d, sizeof(size_t), 1, fin);
            fread(&nc, sizeof(size_t), 1, fin);
            fread(&nsubc, sizeof(size_t), 1, fin);

            ids.reserve(nc);
            codes.reserve(nc);
            norm_codes.reserve(nc);
            alphas.reserve(nc);
            nn_centroid_idxs.reserve(nc);

            for (int i = 0; i < nc; i++){
                ids[i].reserve(nsubc);
                codes[i].reserve(nsubc);
                norm_codes[i].reserve(nsubc);
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


	private:
        float *dis_table;
        float *norms;

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
//        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys)
//        {
//            for (idx_t i = 0; i < n; i++) {
//                float *centroid = (float *) quantizer->getDataByInternalId(keys[i]);
//                for (int j = 0; j < d; j++)
//                    x[i*d + j] = centroid[j] + decoded_residuals[i*d + j];
//            }
//        }
//
//		void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys)
//		{
//            for (idx_t i = 0; i < n; i++) {
//                float *centroid = (float *) quantizer->getDataByInternalId(keys[i]);
//                for (int j = 0; j < d; j++) {
//                    residuals[i*d + j] = x[i*d + j] - centroid[j];
//                }
//            }
//		}

        void compute_residuals(size_t n, float *residuals, const float *points, const float *subcentroids, const idx_t *keys)
		{
            #pragma omp parallel for num_threads(16)
            for (idx_t i = 0; i < n; i++) {
                const float *subcentroid = subcentroids + keys[i]*d;
                const float *point = points + i*d;
                for (int j = 0; j < d; j++) {
                    residuals[i*d + j] = point[j] - centroid[j];
                }
            }
		}

        void reconstruct(size_t n, float *x, const float *decoded_residuals, const float *subcentroids, const idx_t *keys)
        {
            #pragma omp parallel for num_threads(16)
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
            #pragma omp parallel for num_threads(16)
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

        float compute_alpha(const float *centroid_vectors, const float *points,
                            const float *centroid, const int groupsize)
        {
            int counter_positive = 0;
            int counter_negative = 0;
            float positive_alpha = 0.0;
            float negative_alpha = 0.0;

            std::vector<float> point_vectors(groupsize*d);
            #pragma omp parallel for num_threads(16)
            for (int i = 0; i < groupsize; i++) {
                float *point_vector = point_vectors.data() + i * d;
                sub_vectors(point_vector, points.data() + i * d, centroid);
            }
            for (int i = 0; i < groupsize; i++) {
                const float *point_vector = point_vectors.data() + i * d;

                std::priority_queue<std::pair<float, float>> max_heap;
                for (int subc = 0; subc < nsubc; subc++){
                    const float *centroid_vector = centroid_vectors + subc * d;
                    float alpha = faiss::fvec_inner_product (centroid_vector, point_vector, d);

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
