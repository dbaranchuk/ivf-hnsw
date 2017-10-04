#pragma once

#include <fstream>
#include <cstdio>
#include <vector>
#include <queue>



#include "L2space.h"
#include "brutoforce.h"
#include "hnswalg.h"
#include <faiss/ProductQuantizer.h>

typedef unsigned int labeltype;
typedef unsigned int idx_t;
typedef unsigned char uint8_t;

template <typename format>
static void readXvec(std::ifstream &input, format *mass, const int d)
{
	int in = 0;
	input.read((char *) &in, sizeof(int));
	if (in != d) {
		std::cout << "file error\n";
		exit(1);
	}
	input.read((char *) mass, in * sizeof(format));
}

namespace hnswlib {

	template<typename dist_t, typename vtype>
	class Index
	{
		float *clusters;
		int d;
		size_t size;

		std::vector<unsigned char> data;
		std::vector<unsigned int> thresholds;

		std::vector<float> norms;
		faiss::ProductQuantizer norm_pq;

		bool by_residual = true;

		size_t nprobe = 16;
		size_t code_size;
		std::vector < std::vector<uint8_t> > codes;
	public:
		HierarchicalNSW<dist_t, vtype> *quantizer;
		faiss::ProductQuantizer pq;


		Index(SpaceInterface<dist_t> *l2space, size_t dim, size_t ncentroids,
			  size_t bytes_per_code, size_t nbits_per_idx):
				d(dim), size(ncentroids), pq (dim, bytes_per_code, nbits_per_idx)
		{
			quantizer = new HierarchicalNSW<dist_t, vtype>(l2space, {size, {16, 32}}, 240);
		}


		~Index() { delete quantizer; }

		void loadQuantizer(const char *path_info, const char *path_edges) {};

		void buildQuantizer(const char *path_clusters, const char *path_info,
							const char *path_edges)
		{
			cout << "Constructing quantizer\n";
			int j1 = 0;
			std::ifstream input(path_clusters, ios::binary);

			dist_t mass[d];
			readXvec<dist_t>(input, mass, d);
			quantizer->addPoint((void *) (mass), j1);

			size_t report_every = 1000000;
		#pragma omp parallel for num_threads(32)
			for (int i = 1; i < size; i++) {
				dist_t mass[d];
		#pragma omp critical
				{
					readXvec<dist_t>(input, mass, d);
					if (++j1 % report_every == 0)
						std::cout << j1 / (0.01 * size) << " %\n";
				}
				quantizer->addPoint((void *) (mass), (size_t) j1);
			}
			input.close();
			cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
			appr_alg->SaveInfo(path_info);
			appr_alg->SaveEdges(path_edges);
		}


		void assign(const char *path_base, unsigned int *precomputed_idx, size_t vecsize)
		{
			cout << "Assigning base elements\n";
			int j1 = 0;
			std::ifstream input(path_clusters, ios::binary);

			dist_t mass[d];
			std::priority_queue <std::pair<dist_t, labeltype >> result;
			readXvec<d_type>(input, mass, d);
			precomputed_idx[j1] = quantizer->searchKnn(mass, 1).second;

			size_t report_every = 10000000;
		#pragma omp parallel for num_threads(32)
			for (int i = 1; i < size; i++) {
				dist_t mass[d];
				std::priority_queue <std::pair<dist_t, labeltype >> result;
		#pragma omp critical
				{
					readXvec<dist_t>(input, mass, d);
					if (++j1 % report_every == 0)
						std::cout << j1 / (0.01 * vecsize) << " %\n";
				}
				precomputed_idx[j1] = quantizer->searchKnn(mass, 1).second;
			}
			input.close();
		}


		void add(idx_t n, const float * x, const unsigned int *xids, const unsigned int *precomputed_idx)
		{
			const long * idx = precomputed_idx;

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
			norm_pq.compute_code(norm_to_encode, norm_codes, n);

			for (size_t i = 0; i < n; i++) {
				idx_t key = idx[i];

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

		void search (const float *x, idx_t k,
					 float *distances, idx_t *labels) const
		{
			idx_t * idx = new labeltype [nprobe];
			dist_t * coarse_dis = new dist_t [nprobe];

			std::priority_queue <std::pair<dist_t, idx_t>> coarse = quantizer->searchKnn(x, nprobe);
			for (int i = 0; i < nprobe; i++){
				coarse_dis[i] = coarse.top().first;
				idx[i] = coarse.top().second;
				coarse.pop();
			}

			search_preassigned (1, x, k, idx, coarse_dis,
								distances, labels, false);

			delete coarse_dis;
			delete idx;
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

		void search_preassigned (const float *x, idx_t k,
								 float *coarse_dis, idx_t *idx,
								 float *distances, idx_t *labels)
		{
			for (int i = 0; i < nprobe; i++){

			}
		}
	};

}
