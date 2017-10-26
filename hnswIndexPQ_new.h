#pragma once

#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>

#include "L2space.h"
#include "hnswalg.h"
#include <faiss/ProductQuantizer.h>
#include <faiss/utils.h>
#include <faiss/index_io.h>

using namespace std;

typedef unsigned int idx_t;
typedef unsigned char uint8_t;

template <typename format>
void readXvec(std::ifstream &input, format *mass, const int d, const int n = 1)
{
    int in = 0;
    for (int i = 0; i < n; i++) {
        input.read((char *) &in, sizeof(int));
        if (in != d) {
            std::cout << "file error\n";
            exit(1);
        }
        input.read((char *)(mass+i*d), in * sizeof(format));
    }
}

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


        std::vector < float > c_norm_table;

    public:
		ModifiedIndex(Index *trained_index, size_t nsubcentroids = 128):
                index(trained_index), nsubc(nsubcentroids)
		{
            nc = index->csize;
            d = index->d;
            code_size = index->pq->code_size;

            codes.reserve(nc);
            norm_codes.reserve(nc);
            ids.reserve(nc);
            alphas.reserve(nc);
            nn_centroid_idxs.reserve(nc);

            for (int i = 0; i < nc; i++){
                ids[i].reserve(nsubc);
                nn_centroids_idxs[i].reserve(nsubc);
            }
        }


		~ModifiedIndex() {}


        void add(const char *path_groups, const char *path_precomputed_idxs)
        {
            StopW stopw = StopW();

            int total_groupsizes = 0;
            const char *path_groups = "/home/dbaranchuk/data/groups/groups999973.dat";

            std::ifstream input(path_groups, ios::binary);

            double baseline_average = 0.0;
            double modified_average = 0.0;

            int j1 = 0;
//#pragma omp parallel for num_threads(16)
            for (int c = 0; c < nc; c++) {
                /** Read Original vectors from Group file**/
                idx_t centroid_num;
                int groupsize;
                std::vector<float> data;
                const float *centroid;

//#pragma omp critical
                {
                    input.read((char *) &groupsize, sizeof(int));
                    data.reserve(groupsize * vecdim);
                    input.read((char *) data.data(), groupsize * vecdim * sizeof(float));

                    centroid_num = j1++;
                    centroid = (float *) quantizer->getDataByInternalId(centroid_num);

                    if (j1 % 100000 == 0)
                        std::cout << j1 << std::endl;

                    total_groupsizes += groupsize;
                    if (groupsize == 0)
                        continue;
                }

                /** Find NN centroids to source centroid **/
                auto nn_centroids_raw = quantizer->searchKnn((void *) centroid, nsubc + 1);

                /** Remove source centroid from consideration **/
                std::vector<idx_t> &nn_centroids = nn_centroid_idxs[centroid_num];
                while (nn_centroids_raw.size() > 1) {
                    nn_centroids[nn_centroids_raw.size() - 1] = nn_centroids_raw.top().second;
                    nn_centroids_raw.pop();
                }

                /** Pruning **/
                //quantizer->getNeighborsByHeuristicMerge(nn_centroids_before_heuristic, maxM);

                /** Compute centroid-neighbor_centroid and centroid-group_point vectors **/
                std::vector<float> normalized_centroid_vectors(nc * d);
                std::vector<float> point_vectors(groupsize * d);

                for (int i = 0; i < nsubc; i++) {
                    float *neighbor_centroid = (float *) quantizer->getDataByInternalId(nn_centroids[i]);
                    sub_vectors(normalized_centroid_vectors.data() + i * d, neighbor_centroid, centroid);

                    /** Normalize them **/
                    normalize_vector(normalized_centroid_vectors.data() + i * d);
                }

                for (int i = 0; i < groupsize; i++) {
                    sub_vectors(point_vectors.data() + i * vecdim, data.data() + i * d, centroid);
                    baseline_average += faiss::fvec_norm_L2sqr(point_vectors.data() + i * d, d);
                }

                /** Find alphas for vectors **/
                compute_alpha(normalized_centroid_vectors.data(), point_vectors.data(), centroid, centroid_num, groupsize);

                /** Compute final subcentroids **/
                std::vector<float> subcentroids(nsubc * d);
                for (int subc = 0; subc < nsubc; subc++) {
                    const float *centroid_vector = normalized_centroid_vectors.data() + subc * d;
                    float *subcentroid = subcentroids.data() + subc * d;

                    linear_op(subcentroid, centroid_vector, centroid, alphas[centroid_num]);
                }

                for (int i = 0; i < groupsize; i++) {
                    const float *point = points + i * d;
                    const float *residual = point_vectors.data() + i*d;
                    const idx_t idx = idxs[i];

                    /** Find subcentroid idx **/
                    std::priority_queue<std::pair<float, idx_t>> max_heap;

                    for (int subc = 0; subc < nsubc; subc++) {
                        const float *subcentroid = subcentroids + subc * d;
                        float dist = faiss::fvec_L2sqr(subcentroid, point, d);
                        max_heap.emplace(std::make_pair(-dist, subc));
                    }
                    idx_t subcentroid_idx = max_heap.top().second;
                    const float *subcentroid = subcentroids + subcentroid_idx * d;

                    ids[centroid_num][subcentroid_idx].push_back(idx);
                    modified_average += -max_heap.top().first;

                    /** Compute code **/
                    uint8_t xcode[code_size];
                    pq->compute_code(residual, xcode);
                    for (int j = 0; j < d; j++)
                        codes[centroid_num][subcentroid_idx].push_back(xcode[j]);

                    /** Compute norm code **/
                    float decoded_residuals[d];
                    pq->decode(xcodes, decoded_residuals);

                    float reconstructed_x[d];
                    reconstruct(reconstructed_x, decoded_residuals, subcentroid);

                    float norm = faiss::fvec_norm_L2sqr (reconstructed_x, d);
                    uint8_t xnorm_code;
                    norm_pq->compute_code(norm, xnorm_code);
                    norm_codes[centroid_num][subcentroid_idx].push_back(xnorm_code);
                }
            }
            std::cout << "[Baseline] Average Distance: " << baseline_average / total_groupsizes << std::endl;
            std::cout << "[Modified] Average Distance: " << modified_average / total_groupsizes << std::endl;

            std::cout << "Build Time(s): " << stopw.getElapsedTimeMicro() / 1000000 << std::endl;
            input.close();
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
            fwrite(&nprobe, sizeof(size_t), 1, fout);
            fwrite(&max_codes, sizeof(size_t), 1, fout);

            size_t size;
            for (size_t i = 0; i < csize; i++) {
                size = ids[i].size();
                fwrite(&size, sizeof(size_t), 1, fout);
                fwrite(ids[i].data(), sizeof(idx_t), size, fout);
            }

            for(int i = 0; i < csize; i++) {
                size = codes[i].size();
                fwrite(&size, sizeof(size_t), 1, fout);
                fwrite(codes[i].data(), sizeof(uint8_t), size, fout);
            }

            for(int i = 0; i < csize; i++) {
                size = norm_codes[i].size();
                fwrite(&size, sizeof(size_t), 1, fout);
                fwrite(norm_codes[i].data(), sizeof(uint8_t), size, fout);
            }
            fclose(fout);
        }

        void read(const char *path_index)
        {
            FILE *fin = fopen(path_index, "rb");

            fread(&d, sizeof(size_t), 1, fin);
            fread(&nc, sizeof(size_t), 1, fin);
            fread(&nsubc, sizeof(size_t), 1, fin);
            fread(&nprobe, sizeof(size_t), 1, fin);
            fread(&max_codes, sizeof(size_t), 1, fin);

            ids = std::vector<std::vector<idx_t>>(csize);
            codes = std::vector<std::vector<uint8_t>>(csize);
            norm_codes = std::vector<std::vector<uint8_t>>(csize);

            size_t size;
            for (size_t i = 0; i < csize; i++) {
                fread(&size, sizeof(size_t), 1, fin);
                ids[i].resize(size);
                fread(ids[i].data(), sizeof(idx_t), size, fin);
            }

            for(size_t i = 0; i < csize; i++){
                fread(&size, sizeof(size_t), 1, fin);
                codes[i].resize(size);
                fread(codes[i].data(), sizeof(uint8_t), size, fin);
            }

            for(size_t i = 0; i < csize; i++){
                fread(&size, sizeof(size_t), 1, fin);
                norm_codes[i].resize(size);
                fread(norm_codes[i].data(), sizeof(uint8_t), size, fin);
            }
            fclose(fin);
        }

        void compute_centroid_norm_table()
        {
            c_norm_table.reserve(nc);
            for (int i = 0; i < nc; i++){
                const float *centroid = (float *)quantizer->getDataByInternalId(i);
                faiss::fvec_norm_L2sqr(c_norm_table.data() + i, centroid, d);
            }
        }


	private:
        float *dis_table;
        float *norms;

        float fstdistfunc(uint8_t *code)
        {
            float result = 0.;
            int dim = code_size >> 2;
            int m = 0;
            for (int i = 0; i < dim; i++) {
                result += dis_table[pq->ksub * m + code[m]]; m++;
                result += dis_table[pq->ksub * m + code[m]]; m++;
                result += dis_table[pq->ksub * m + code[m]]; m++;
                result += dis_table[pq->ksub * m + code[m]]; m++;
            }
            return result;
        }
    public:
        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys)
        {
            for (idx_t i = 0; i < n; i++) {
                float *centroid = (float *) quantizer->getDataByInternalId(keys[i]);
                for (int j = 0; j < d; j++)
                    x[i*d + j] = centroid[j] + decoded_residuals[i*d + j];
            }
        }

		void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys)
		{
            for (idx_t i = 0; i < n; i++) {
                float *centroid = (float *) quantizer->getDataByInternalId(keys[i]);
                for (int j = 0; j < d; j++) {
                    residuals[i*d + j] = x[i*d + j] - centroid[j];
                }
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

        void compute_alpha(const float *centroid_vectors, const float *point_vectors,
                            const float *centroid, const int centroid_num, const int groupsize)
        {
            int counter_positive = 0;
            int counter_negative = 0;
            float positive_alpha = 0.0;
            float negative_alpha = 0.0;

            for (int i = 0; i < groupsize; i++) {
                const float *point_vector = point_vectors + i * d;
                std::priority_queue<std::pair<float, float>> max_heap;

                for (int subc = 0; c < nsubc; subc++){
                    const float *centroid_vector = centroid_vectors + subc * d;
                    float alpha = faiss::fvec_inner_product (centroid_vector, point_vector, d);

                    std::vector<float> subcentroid(d);

                    linear_op(subcentroid.data(), centroid_vector, centroid, alpha);

                    float dist = faiss::fvec_L2sqr(point_vector, subcentroid.data(), vecdim);
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
            alphas[centroid_num] = (counter_positive > counter_negative) ? positive_alpha : negative_alpha;
        }
	};

}
