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

    void read_pq(const char *path, faiss::ProductQuantizer *_pq)
    {
        if (!_pq) {
            std::cout << "PQ object does not exists" << std::endl;
            return;
        }
        FILE *fin = fopen(path, "rb");

        fread(&_pq->d, sizeof(size_t), 1, fin);
        fread(&_pq->M, sizeof(size_t), 1, fin);
        fread(&_pq->nbits, sizeof(size_t), 1, fin);
        _pq->set_derived_values ();

        size_t size;
        fread (&size, sizeof(size_t), 1, fin);
        _pq->centroids.resize(size);

        float *centroids = _pq->centroids.data();
        fread(centroids, sizeof(float), size, fin);

        std::cout << _pq->d << " " << _pq->M << " " << _pq->nbits << " " << _pq->byte_per_idx << " " << _pq->dsub << " "
                  << _pq->code_size << " " << _pq->ksub << " " << size << " " << centroids[0] << std::endl;
        fclose(fin);
    }

    void write_pq(const char *path, faiss::ProductQuantizer *_pq)
    {
        if (!_pq){
            std::cout << "PQ object does not exist" << std::endl;
            return;
        }
        FILE *fout = fopen(path, "wb");

        fwrite(&_pq->d, sizeof(size_t), 1, fout);
        fwrite(&_pq->M, sizeof(size_t), 1, fout);
        fwrite(&_pq->nbits, sizeof(size_t), 1, fout);

        size_t size = _pq->centroids.size();
        fwrite (&size, sizeof(size_t), 1, fout);

        float *centroids = _pq->centroids.data();
        fwrite(centroids, sizeof(float), size, fout);

        std::cout << _pq->d << " " << _pq->M << " " << _pq->nbits << " " << _pq->byte_per_idx << " " << _pq->dsub << " "
                  << _pq->code_size << " " << _pq->ksub << " " << size << " " << centroids[0] << std::endl;
        fclose(fout);
    }


    struct Index
	{
		size_t d;
		size_t csize;
        size_t code_size;

        /** Query members **/
        size_t nprobe = 16;
        size_t max_codes = 10000;

		faiss::ProductQuantizer *norm_pq;
        faiss::ProductQuantizer *pq;

        std::vector < std::vector<idx_t> > ids;
        std::vector < std::vector<uint8_t> > codes;
        std::vector < std::vector<uint8_t> > norm_codes;

        size_t *c_size_table = NULL;
        float *c_var_table = NULL;
        float *c_norm_table = NULL;
		HierarchicalNSW<float, float> *quantizer;

    public:
		Index(size_t dim, size_t ncentroids,
			  size_t bytes_per_code, size_t nbits_per_idx):
				d(dim), csize(ncentroids)
		{
            codes.reserve(ncentroids);
            norm_codes.reserve(ncentroids);
            ids.reserve(ncentroids);

            pq = new faiss::ProductQuantizer(dim, bytes_per_code, nbits_per_idx);
            norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);

            code_size = pq->code_size;
        }


		~Index() {
            if (dis_table)
                delete dis_table;

            if (norms)
                delete norms;

            if (c_norm_table)
                delete c_norm_table;

//            if (c_var_table)
//                delete c_var_table;
//
//            if (c_size_table)
//                delete c_size_table;

            delete pq;
            delete norm_pq;
            delete quantizer;
		}

		void buildQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                            const char *path_info, const char *path_edges, int efSearch)
		{
            if (exists_test(path_info) && exists_test(path_edges)) {
                quantizer = new HierarchicalNSW<float, float>(l2space, path_info, path_clusters, path_edges);
                quantizer->ef_ = efSearch;
                return;
            }
            quantizer = new HierarchicalNSW<float, float>(l2space, {{csize, {16, 32}}}, 500);
            quantizer->ef_ = efSearch;

			std::cout << "Constructing quantizer\n";
			int j1 = 0;
			std::ifstream input(path_clusters, ios::binary);

			float mass[d];
			readXvec<float>(input, mass, d);
			quantizer->addPoint((void *) (mass), j1);

			size_t report_every = 100000;
		#pragma omp parallel for num_threads(16)
			for (int i = 1; i < csize; i++) {
				float mass[d];
		#pragma omp critical
				{
					readXvec<float>(input, mass, d);
					if (++j1 % report_every == 0)
						std::cout << j1 / (0.01 * csize) << " %\n";
				}
				quantizer->addPoint((void *) (mass), (size_t) j1);
			}
			input.close();
			quantizer->SaveInfo(path_info);
			quantizer->SaveEdges(path_edges);
		}


		void assign(size_t n, const float *data, idx_t *idxs)
		{
		    #pragma omp parallel for num_threads(24)
			for (int i = 0; i < n; i++)
				idxs[i] = quantizer->searchKnn(const_cast<float *>(data + i*d), 1).top().second;
		}


		void add(idx_t n, float * x, const idx_t *xids, const idx_t *idx)
		{
            float *residuals = new float [n * d];
            compute_residuals(n, x, residuals, idx);

            uint8_t *xcodes = new uint8_t [n * code_size];
			pq->compute_codes (residuals, xcodes, n);

            float *decoded_residuals = new float[n * d];
            pq->decode(xcodes, decoded_residuals, n);

            float *reconstructed_x = new float[n * d];
            reconstruct(n, reconstructed_x, decoded_residuals, idx);

            float *norms = new float[n];
            faiss::fvec_norms_L2sqr (norms, reconstructed_x, d, n);

            uint8_t *xnorm_codes = new uint8_t[n];
            norm_pq->compute_codes(norms, xnorm_codes, n);

			for (size_t i = 0; i < n; i++) {
				idx_t key = idx[i];
				idx_t id = xids[i];
				ids[key].push_back(id);
				uint8_t *code = xcodes + i * code_size;
				for (size_t j = 0; j < code_size; j++)
					codes[key].push_back(code[j]);

				norm_codes[key].push_back(xnorm_codes[i]);
			}

            delete residuals;
            delete xcodes;
            delete decoded_residuals;
            delete reconstructed_x;

			delete norms;
			delete xnorm_codes;
		}

        struct CompareByFirst {
            constexpr bool operator()(std::pair<float, idx_t> const &a,
                                      std::pair<float, idx_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };

		void search (float *x, idx_t k, idx_t *results)
		{
            idx_t keys[nprobe];
            float q_c[nprobe];
            if (!norms)
                norms = new float[65536];
            if (!dis_table)
                dis_table = new float [pq->ksub * pq->M];

            pq->compute_inner_prod_table(x, dis_table);
            std::priority_queue<std::pair<float, idx_t>, std::vector<std::pair<float, idx_t>>, CompareByFirst> topResults;
            //std::priority_queue<std::pair<float, idx_t>> topResults;

            auto coarse = quantizer->searchKnn(x, nprobe);

            for (int i = nprobe - 1; i >= 0; i--) {
                auto elem = coarse.top();
                q_c[i] = elem.first;
                keys[i] = elem.second;
                coarse.pop();
            }

            for (int i = 0; i < nprobe; i++){
                idx_t key = keys[i];
                std::vector<uint8_t> code = codes[key];
                std::vector<uint8_t> norm_code = norm_codes[key];
                float term1 = q_c[i] - c_norm_table[key];
                int ncodes = norm_code.size();

                norm_pq->decode(norm_code.data(), norms, ncodes);

                for (int j = 0; j < ncodes; j++){
                    float q_r = fstdistfunc(code.data() + j*code_size);
                    float dist = term1 - 2*q_r + norms[j];
                    idx_t label = ids[key][j];
                    topResults.emplace(std::make_pair(-dist, label));
                }
                if (topResults.size() > max_codes)
                    break;
            }

            for (int i = 0; i < k; i++) {
                results[i] = topResults.top().second;
                topResults.pop();
            }
//            if (topResults.size() < k) {
//                for (int j = topResults.size(); j < k; j++)
//                    topResults.emplace(std::make_pair(std::numeric_limits<float>::max(), 0));
//                std::cout << "Ignored query" << std:: endl;
//            }
		}

        void train_norm_pq(idx_t n, const float *x)
        {
            idx_t *assigned = new idx_t [n]; // assignement to coarse centroids
            assign (n, x, assigned);

            float *residuals = new float [n * d];
            compute_residuals (n, x, residuals, assigned);

            uint8_t * xcodes = new uint8_t [n * code_size];
            pq->compute_codes (residuals, xcodes, n);

            float *decoded_residuals = new float[n * d];
            pq->decode(xcodes, decoded_residuals, n);

            float *reconstructed_x = new float[n * d];
            reconstruct(n, reconstructed_x, decoded_residuals, assigned);

            float *trainset = new float[n];
            faiss::fvec_norms_L2sqr (trainset, reconstructed_x, d, n);

            norm_pq->verbose = true;
            norm_pq->train (n, trainset);

            delete assigned;
            delete residuals;
            delete xcodes;
            delete decoded_residuals;
            delete reconstructed_x;
            delete trainset;
        }

        void train_residual_pq(idx_t n, const float *x)
        {
            idx_t *assigned = new idx_t [n];
            assign (n, x, assigned);

            float *residuals = new float [n * d];
            compute_residuals (n, x, residuals, assigned);

            printf ("Training %zdx%zd product quantizer on %ld vectors in %dD\n",
                    pq->M, pq->ksub, n, d);
            pq->verbose = true;
            pq->train (n, residuals);

            delete assigned;
            delete residuals;
        }


        void precompute_idx(size_t n, const char *path_data, const char *fo_name)
        {
            if (exists_test(fo_name))
                return;

            std::cout << "Precomputing indexes" << std::endl;
            size_t batch_size = 1000000;
            FILE *fout = fopen(fo_name, "wb");

            std::ifstream input(path_data, ios::binary);

            float *batch = new float[batch_size * d];
            idx_t *precomputed_idx = new idx_t[batch_size];
            for (int i = 0; i < n / batch_size; i++) {
                std::cout << "Batch number: " << i+1 << " of " << n / batch_size << std::endl;

                readXvec(input, batch, d, batch_size);
                assign(batch_size, batch, precomputed_idx);

                fwrite((idx_t *) &batch_size, sizeof(idx_t), 1, fout);
                fwrite(precomputed_idx, sizeof(idx_t), batch_size, fout);
            }
            delete precomputed_idx;
            delete batch;

            input.close();
            fclose(fout);
        }


        void write(const char *path_index)
        {
            FILE *fout = fopen(path_index, "wb");

            fwrite(&d, sizeof(size_t), 1, fout);
            fwrite(&csize, sizeof(size_t), 1, fout);
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

//            fwrite(c_norm_table, sizeof(float), csize, fout);
//            fwrite(c_size_table, sizeof(size_t), csize, fout);
//            fwrite(c_var_table, sizeof(float), csize, fout);

            fclose(fout);
        }

        void read(const char *path_index)
        {
            FILE *fin = fopen(path_index, "rb");

            fread(&d, sizeof(size_t), 1, fin);
            fread(&csize, sizeof(size_t), 1, fin);
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

            //fread(c_norm_table, sizeof(float), csize, fin);
            //fread(c_size_table, sizeof(size_t), csize, fin);
            //fread(c_var_table, sizeof(float), csize, fin);

            fclose(fin);
        }

        void update_centroids(const char *path_data, const char *path_precomputed_idxs,
                              const char *path_updated_centroids)
        {
            if (!c_size_table){
                std::cout << "Precompute centroid size table first\n";
                return;
            }
            float *c_e_table = new float[csize*d];

            for (int i = 0; i < csize*d; i++)
                c_e_table[i] = 0.0;

            size_t batch_size = 1000000;
            std::ifstream base_input(path_data, ios::binary);
            std::ifstream idx_input(path_precomputed_idxs, ios::binary);
            std::vector<float> batch(batch_size * d);
            std::vector<idx_t> idx_batch(batch_size);

            for (int b = 0; b < 1000; b++) {
                readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
                readXvec<float>(base_input, batch.data(), d, batch_size);

                #pragma omp parallel for num_threads(16)
                for (int i = 0; i < batch_size; i++) {
                    idx_t key = idx_batch[i];
                    add_vector(batch.data() + i*d, c_e_table + key*d);
                }
            }

            for (int i = 0; i < csize; i++) {
                size_t size = c_size_table[i];
                for (int j = 0; j < d; j++)
                    c_e_table[i * d + j] /= size;
            }

            idx_input.close();
            base_input.close();

            FILE* fout = fopen(path_updated_centroids, "wb");
            for (int i = 0; i < csize; i++) {
                fwrite((int *) &d, sizeof(int), 1, fout);
                fwrite((c_e_table + i*d), sizeof(float), d, fout);
            }
            fclose(fout);
            delete c_e_table;
        }

        void compute_centroid_norm_table()
        {
            c_norm_table = new float[csize];
            for (int i = 0; i < csize; i++){
                float *c = (float *)quantizer->getDataByInternalId(i);
                faiss::fvec_norms_L2sqr (c_norm_table+i, c, d, 1);
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

        void add_vector(const float *x, float *y)
        {
            for (int i = 0; i < d; i++)
                y[i] += x[i];
        }
	};

}
