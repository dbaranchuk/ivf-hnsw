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
#include <faiss/Heap.h>

using namespace std;

typedef unsigned int idx_t;
typedef unsigned char uint8_t;

template <typename format>
void readXvec(std::ifstream &input, format *data, const int d, const int n = 1)
{
    int in = 0;
    for (int i = 0; i < n; i++) {
        input.read((char *) &in, sizeof(int));
        if (in != d) {
            std::cout << "file error\n";
            exit(1);
        }
        input.read((char *)(data + i*d), in * sizeof(format));
    }
}

template <typename format>
void readXvecFvec(std::ifstream &input, float *data, const int d, const int n = 1)
{
    int in = 0;
    format mass[d];

    for (int i = 0; i < n; i++) {
        input.read((char *) &in, sizeof(int));
        if (in != d) {
            std::cout << "file error\n";
            exit(1);
        }
        input.read((char *)mass, in * sizeof(format));
        for (int j = 0; j < d; j++)
            data[i*d + j] = (1.0)*mass[j];
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
		size_t nc;
        size_t code_size;

        /** Query members **/
        size_t nprobe = 16;
        size_t max_codes = 10000;

		faiss::ProductQuantizer *norm_pq;
        faiss::ProductQuantizer *pq;

        std::vector < std::vector<idx_t> > ids;
        std::vector < std::vector<uint8_t> > codes;
        std::vector < std::vector<uint8_t> > norm_codes;

        std::vector < float > centroid_norms;
		HierarchicalNSW<float, float> *quantizer;

    public:
		Index(size_t dim, size_t ncentroids,
			  size_t bytes_per_code, size_t nbits_per_idx);


		~Index();

		void buildQuantizer(SpaceInterface<float> *l2space, const char *path_clusters,
                            const char *path_info, const char *path_edges, int efSearch);


		void assign(size_t n, const float *data, idx_t *idxs);


		void add(idx_t n, float * x, const idx_t *xids, const idx_t *idx);

        struct CompareByFirst {
            constexpr bool operator()(std::pair<float, idx_t> const &a,
                                      std::pair<float, idx_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        double average_max_codes = 0;

		void search (float *x, idx_t k, float *distances, long *labels);

        void train_norm_pq(idx_t n, const float *x);

        void train_residual_pq(idx_t n, const float *x);


        void precompute_idx(size_t n, const char *path_data, const char *fo_name);


        void write(const char *path_index);

        void read(const char *path_index);

        void compute_centroid_norms();

        void compute_graphic(float *x, const idx_t *groundtruth, size_t gt_dim, size_t qsize);

        void compute_s_c() {}

    private:
        std::vector < float > dis_table;
        std::vector < float > norms;

        float fstdistfunc(uint8_t *code);
    public:
        void reconstruct(size_t n, float *x, const float *decoded_residuals, const idx_t *keys);

		void compute_residuals(size_t n, const float *x, float *residuals, const idx_t *keys);
	};

}
