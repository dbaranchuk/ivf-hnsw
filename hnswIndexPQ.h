#pragma once

#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>



#include "L2space.h"
#include "brutoforce.h"
#include "hnswalg.h"
#include <faiss/ProductQuantizer.h>

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

		/** Query members **/
		size_t nprobe = 16;
		float *dis_tables;

        std::vector<idx_t> thresholds;

		HierarchicalNSW<float, float> *quantizer;
		faiss::ProductQuantizer pq;


		Index(size_t dim, size_t ncentroids,
			  size_t bytes_per_code, size_t nbits_per_idx):
				d(dim), csize(ncentroids), pq (dim, bytes_per_code, nbits_per_idx)
		{
            for (int i = 0; i < csize; i++)
                thresholds.push_back(0);
        }


		~Index() {
			delete quantizer;
			if (dis_tables)
				delete dis_tables;
		}

		void buildQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                            const char *path_info, const char *path_edges)
		{
            if (exists_test(path_info) && exists_test(path_edges)) {
                quantizer = new HierarchicalNSW<float, float>(l2space, path_info, path_clusters, path_edges);
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
			//int j1 = 0;
			//std::ifstream input(path_base, ios::binary);


			//vtype *ma
			//readXvec<vtype>(input, mass, d);
			//precomputed_idx[0] = quantizer->searchKnn(data, 1).top().second;

			//size_t report_every = 1000000;
		#pragma omp parallel for num_threads(32)
			for (int i = 0; i < n; i++) {
			//	vtype mass[d];
		//#pragma omp critical
			//	{
			//		readXvec<vtype>(input, mass, d);
			//		if (++j1 % report_every == 0)
			//			std::cout << j1 / (0.01 * vecsize) << " %\n";
			//	}
				precomputed_idx[i] = quantizer->searchKnn((data + i*d), 1).top().second;
			}

			//input.close();

			//Fill thresholds
			//count number of elements per cluster
			for(int i = 0; i < n; i++){
				thresholds[precomputed_idx[i]]++;
			}
			//for (int i = 1; i < csize; i++)
			//	thresholds[i] += thresholds[i-1];

			//if (thresholds.back() != vecsize){
			//	std::cout << "Something Wrong\n";
			//	exit(1);
			//}
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

			for (size_t i = 0; i < n; i++)
				norm_to_encode[i]  = compute_norm(x + i * d);
			norm_pq.compute_codes(norm_to_encode, norm_codes, n);

			for (size_t i = 0; i < n; i++) {
				idx_t key = idx[i];
				//idx_t id = xids[i];
				//ids[key].push_back(id);
				uint8_t *code = xcodes + i * code_size;
				for (size_t j = 0; j < code_size; j++)
					codes[key].push_back (code[j]);
				codes[key].push_back(norm_codes[i]);
			}

			delete xcodes;
			delete norm_to_encode;
			delete norm_codes;
		}

		void search (size_t nx, const float *x, idx_t k,
					 idx_t *results) const
		{
			float *x_residual = new float[nx*nprobe*d];
			idx_t *centroids = new idx_t[nx*nprobe];

			for (int i = 0; i < nx; i++) {
				auto coarse = quantizer->searchKnn(x+i*d, nprobe);
				// add from the end because coarse is a max_heap
				for (int j = nprobe - 1; j >= 0; j--)
				{
					auto elem = coarse.top();
					compute_residual(x+i*d, x_residual + (nprobe*i + j)*d, elem.second);
					centroids[nprobe*i + j] = elem.second;
					coarse.pop();
				}
			}

			compute_query_tables(x_residual, nx*nprobe);

			for (int i = 0; i < nx; i++){
				std::priority_queue<std::pair<float, idx_t>> topResults;
				for (int j = 0; j < nprobe; j++){
					size_t left_border = thresholds[centroids[i*nprobe + j]];
					size_t right_border = thresholds[centroids[i*nprobe + j]+1];
					for (int id = left_border; id < right_border; id++){
						float dist = fstdistfunc(i, codes[id]);
						topResults.insert({dist, id});
					}
				}
				while (topResults.size() > k)
					topResults.pop();
				for (int j = k-1; j >= 0; j--) {
					results[i * k + j] = topResults.top().second;
					topResults.pop();
				}
			}


			delete centroids;
			delete x_residual;
		}

		void compute_query_tables(float *massQ, size_t qsize)
		{
			int ksub = 256;
			dis_tables = new float[qsize*ksub*code_size];
			pq.compute_distance_tables(qsize, massQ, dis_tables);
		}


        void train_residual(idx_t n, const float *x)
        {
            const float *trainset;
            float *residuals;
            idx_t * assign;

            if (by_residual) {
                if(verbose) printf("computing residuals\n");
                assign = new idx_t [n]; // assignement to coarse centroids
                this->assign (n, x, assign);

                residuals = new float [n * d];
                for (idx_t i = 0; i < n; i++)
                    this->compute_residual (x + i * d, residuals+i*d, assign[i]);

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
            delete assign;
            delete residuals;
        }

	private:
		void compute_residual(const float *x, float *residual, idx_t key)
		{
			float *centroid = quantizer->getDataByInternalId(key);
			for (int i = 0; i < d; i++){
				residual[i] = x[i] - centroid[i];
			}
		}

		float compute_norm(const float *x)
		{
			float result = 0.;
			int dim = d >> 2;

			for (int i = 0; i < dim; i++){
				result += (*x)*(*x); x++;
				result += (*x)*(*x); x++;
				result += (*x)*(*x); x++;
				result += (*x)*(*x); x++;
			}
			return result;
		}

		float fstdistfunc(const size_t q_idx, const uint8_t *y)
		{
			float res = 0.;
			int ksub = pq.ksub;
			int dim = code_size >> 3;

			int n = 0;
			for (int i = 0; i < dim; ++i) {
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
				res += dis_tables[ksub * (code_size * q_idx + n) + y[n]]; ++n;
			}
			return res;
		};
	};

}
