#pragma once

#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>
#include <limits>


#include "L2space.h"
#include "brutoforce.h"
#include "hnswalg.h"
#include <faiss/ProductQuantizer.h>
#include <faiss/utils.h>

typedef unsigned int idx_t;
typedef unsigned char uint8_t;

inline bool exists_test(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

template <typename format>
static void readXvec(std::ifstream &input, format *mass, const int d, const int n = 1)
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

	struct Index
	{
		int d;
		size_t csize;

		faiss::ProductQuantizer norm_pq;

		bool by_residual = true;
        bool verbose = true;

		size_t code_size;
		std::vector < std::vector<uint8_t> > codes;
        std::vector < std::vector<idx_t> > ids;

		/** Query members **/
		size_t nprobe = 1024;
        size_t max_codes = 10000;

        float *q_norm_table;
        float *c_norm_table;

		HierarchicalNSW<float, float> *quantizer;
		faiss::ProductQuantizer pq;


		Index(size_t dim, size_t ncentroids,
			  size_t bytes_per_code, size_t nbits_per_idx):
				d(dim), csize(ncentroids),
                pq (dim, bytes_per_code, nbits_per_idx),
                norm_pq (1, 1, nbits_per_idx)
		{
            codes.reserve(ncentroids);
            ids.reserve(ncentroids);
        }


		~Index() {
			delete quantizer;
            if (q_norm_table)
                delete q_norm_table;
            if (c_norm_table)
                delete c_norm_table;
		}

		void buildQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                            const char *path_info, const char *path_edges)
		{
            if (exists_test(path_info) && exists_test(path_edges)) {
                quantizer = new HierarchicalNSW<float, float>(l2space, path_info, path_clusters, path_edges);
                quantizer->ef_ = 60;
                return;
            }

            quantizer = new HierarchicalNSW<float, float>(l2space, {{csize, {16, 32}}}, 240);
            quantizer->ef_ = 60;

			std::cout << "Constructing quantizer\n";
			int j1 = 0;
			std::ifstream input(path_clusters, ios::binary);

			float mass[d];
			readXvec<float>(input, mass, d);
			quantizer->addPoint((void *) (mass), j1);

			size_t report_every = 100000;
		#pragma omp parallel for num_threads(32)
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


		void assign(size_t n, float *data, idx_t *precomputed_idx)
		{
		#pragma omp parallel for num_threads(32)
			for (int i = 0; i < n; i++)
				precomputed_idx[i] = quantizer->searchKnn((data + i*d), 1).top().second;
		}


		void add(idx_t n, const float * x, const idx_t *xids, const idx_t *precomputed_idx)
		{
			const idx_t * idx = precomputed_idx;

			uint8_t * xcodes = new uint8_t [n * code_size];
			uint8_t *norm_codes = new uint8_t[n];

			const float *to_encode = nullptr;
			float *norm_to_encode = new float[n];

			if (by_residual) {
				float *residuals = new float [n * d];
				for (size_t i = 0; i < n; i++)
					compute_residual(x + i * d, residuals + i * d, idx[i]);

				to_encode = residuals;
			} else {
				to_encode = x;
			}
			pq.compute_codes (to_encode, xcodes, n);

            faiss::fvec_norms_L2sqr (norm_to_encode, x, d, n);
			norm_pq.compute_codes(norm_to_encode, norm_codes, n);

			for (size_t i = 0; i < n; i++) {
				idx_t key = idx[i];
				idx_t id = xids[i];
				ids[key].push_back(id);
				uint8_t *code = xcodes + i * code_size;
				for (size_t j = 0; j < code_size; j++)
					codes[key].push_back (code[j]);
				codes[key].push_back(norm_codes[i]);
			}
			delete xcodes;
			delete norm_to_encode;
			delete norm_codes;
		}

		void search (float *x, idx_t k, idx_t *results)
		{
            idx_t keys[nprobe];
            float q_c[nprobe];

            float * dis_tables = new float [pq.ksub * pq.M];
            pq.compute_distance_tables (1, x, dis_tables);

            std::priority_queue<std::pair<float, idx_t>> topResults;
            auto coarse = quantizer->searchKnn(x, nprobe);

            for (int i = nprobe - 1; i >= 0; i--) {
                auto elem = coarse.top();
                q_c[i] = elem.first;
                keys[i] = elem.second;
                coarse.pop();
            }
            for (int i = 0; i < nprobe; i--){
                idx_t key = keys[i];
                std::vector<uint8_t> code = codes[key];
                float c = c_norm_table[key];
                float q_r = 0.;

                int ncodes = code.size()/(code_size+1);

                for (int j = 0; j < ncodes; j++){
                    for (int m = 0; m < code_size; m++)
                        q_r += dis_tables[pq.ksub * m + code[j*(code_size + 1) + m]];

                    float norm;
                    norm_pq.decode(code.data()+j*(code_size+1) + code_size, &norm);
                    float dist = q_c[i] - c - 2*q_r + norm;
                    idx_t label = ids[key][j];
                    topResults.emplace(std::make_pair(dist, label));
                }
                if (topResults.size() > max_codes)
                    break;
            }

            while (topResults.size() > k)
                topResults.pop();

            if (topResults.size() < k) {
                for (int j = topResults.size(); j < k; j++)
                    topResults.emplace(std::make_pair(std::numeric_limits<float>::max(), 0));
                std::cout << "Ignored query" << std:: endl;
            }
            for (int j = k-1; j >= 0; j--) {
                results[j] = topResults.top().second;
                topResults.pop();
            }
		}

        void compute_query_norm_table(float *massQ, size_t qsize)
        {
            q_norm_table = new float[qsize];
            faiss::fvec_norms_L2sqr (q_norm_table, massQ, d, qsize);
        }

        void compute_centroid_norm_table()
        {
            c_norm_table = new float[csize];
            for (int i = 0; i < csize; i++){
                float *c = (float *)quantizer->getDataByInternalId(i);
                faiss::fvec_norms_L2sqr (c_norm_table+i, c, d, 1);
            }
        }

        void train_norm_pq(idx_t n, float *x)
        {
            idx_t *assigned = new idx_t [n]; // assignement to coarse centroids
            this->assign (n, x, assigned);
            float *residuals = new float [n * d];
            for (idx_t i = 0; i < n; i++)
                compute_residual (x + i * d, residuals+i*d, assigned[i]);

            uint8_t * xcodes = new uint8_t [n * code_size];
            pq.compute_codes (residuals, xcodes, n);
            pq.decode(xcodes, residuals, n);

            float *decoded_x = new float[n*d];
            for (idx_t i = 0; i < n; i++) {
                float *centroid = (float *) quantizer->getDataByInternalId(assigned[i]);
                for (int j = 0; j < d; j++)
                    decoded_x[i*d + j] = centroid[j] + residuals[i*d + j];

            }
            delete residuals;
            delete assigned;
            delete xcodes;

            float *trainset = new float[n];
            faiss::fvec_norms_L2sqr (trainset, decoded_x, d, n);

            norm_pq.verbose = verbose;
            norm_pq.train (n, trainset);

            delete trainset;
        }

        void train_residual_pq(idx_t n, float *x)
        {
            const float *trainset;
            float *residuals;
            idx_t * assigned;

            if (by_residual) {
                if(verbose) printf("computing residuals\n");
                assigned = new idx_t [n]; // assignement to coarse centroids
                this->assign (n, x, assigned);

                residuals = new float [n * d];
                for (idx_t i = 0; i < n; i++)
                    this->compute_residual (x + i * d, residuals+i*d, assigned[i]);

                trainset = residuals;
            } else {
                trainset = x;
            }
            if (verbose)
                printf ("training %zdx%zd product quantizer on %ld vectors in %dD\n",
                        pq.M, pq.ksub, n, d);
            pq.verbose = verbose;
            pq.train (n, trainset);

            //if (by_residual) {
            //    precompute_table ();
            //}
            delete assigned;
            delete residuals;
        }


        void precompute_idx(const char *path_data, const char *fo_name)
        {
            if (exists_test(fo_name))
                return;

            size_t batch_size = 1000000;
            FILE *fout = fopen(fo_name, "wb");

            std::ifstream input(path_data, ios::binary);

            float *batch = new float[batch_size * vecdim];
            idx_t *precomputed_idx = new idx_t[batch_size];
            for (int i = 0; i < vecsize / batch_size; i++) {
                std::cout << "Batch number: " << i+1 << " of " << vecsize / batch_size << std::endl;

                readXvec(input, batch, vecdim, batch_size);
                index->assign(batch_size, batch, precomputed_idx);

                fwrite((idx_t *) &batch_size, sizeof(idx_t), 1, fout);
                fwrite(precomputed_idx, sizeof(idx_t), batch_size, fout);
            }
            delete precomputed_idx;
            delete batch;

            input.close();
            fclose(fout);
        }

	private:
		void compute_residual(const float *x, float *residual, idx_t key)
		{
			float *centroid = (float *) quantizer->getDataByInternalId(key);
			for (int i = 0; i < d; i++){
				residual[i] = x[i] - centroid[i];
			}
		}

//		float compute_norm(const float *x)
//		{
//			float result = 0.;
//			int dim = d >> 2;
//
//			for (int i = 0; i < dim; i++){
//				result += (*x)*(*x); x++;
//				result += (*x)*(*x); x++;
//				result += (*x)*(*x); x++;
//				result += (*x)*(*x); x++;
//			}
//			return result;
//		}

//		float fstdistfunc(const size_t q_idx, const uint8_t *y)
//		{
//			float res = 0.;
//			int ksub = pq.ksub;
//			int dim = code_size >> 3;
//
//			int n = 0;
//			for (int i = 0; i < dim; ++i) {
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
//			}
//			return res;
//		};
	};

}
