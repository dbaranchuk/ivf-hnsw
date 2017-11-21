#include <queue>
#include <limits>
#include <cmath>
#include <chrono>

#include <faiss/ProductQuantizer.h>

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