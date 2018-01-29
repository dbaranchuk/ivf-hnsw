#ifndef IVF_HNSW_LIB_UTILS_H
#define IVF_HNSW_LIB_UTILS_H

#include <queue>
#include <limits>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include <faiss/utils.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

#define EPS 0.00001

namespace ivfhnsw {

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


    template<typename T>
    void read_value(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename T>
    void read_vector(std::istream &in, std::vector<T> &vec) {
        uint32_t size;
        in.read((char *) &size, sizeof(uint32_t));
        vec.resize(size);
        in.read((char *) vec.data(), size * sizeof(T));
    }

    template<typename T>
    void read_vector2(std::istream &in, std::vector<T> &vec) {
        size_t size;
        in.read((char *) &size, sizeof(size_t));
        vec.resize(size);
        in.read((char *) vec.data(), size * sizeof(T));
    }

    template<typename T>
    void write_value(std::ostream &out, const T &val) {
        out.write((char *) &val, sizeof(T));
    }

    template<typename T>
    void write_vector(std::ostream &out, std::vector<T> &vec) {
        uint32_t size = vec.size();
        out.write((char *) &size, sizeof(uint32_t));
        out.write((char *) vec.data(), size * sizeof(T));
    }


    template<typename T>
    void readXvec(std::ifstream &in, T *data, const size_t d, const size_t n = 1) {
        uint32_t dim = 0;
        for (size_t i = 0; i < n; i++) {
            in.read((char *) &dim, sizeof(uint32_t));
            if (dim != d) {
                std::cout << "file error\n";
                exit(1);
            }
            in.read((char *) (data + i * dim), dim * sizeof(T));
        }
    }


    template<typename T>
    void writeXvec(std::ofstream &out, T *data, const size_t d, const size_t n = 1) {
        uint32_t dim = d;
        for (size_t i = 0; i < n; i++) {
            out.write((char *) &dim, sizeof(uint32_t));
            out.write((char *) (data + i * dim), dim * sizeof(T));
        }
    }


    template<typename T>
    void readXvecFvec(std::ifstream &in, float *data, const size_t d, const size_t n = 1) {
        uint32_t dim = 0;
        T mass[dim];

        for (size_t i = 0; i < n; i++) {
            in.read((char *) &dim, sizeof(uint32_t));
            if (dim != d) {
                std::cout << "file error\n";
                exit(1);
            }
            in.read((char *) mass, dim * sizeof(T));
            for (size_t j = 0; j < d; j++)
                data[i * dim + j] = (1.0) * mass[j];
        }
    }

    inline bool exists(const std::string& name) {
        std::ifstream f(name.c_str());
        return f.good();
    }

    void random_subset(const float *x, float *x_out, size_t d, size_t nx, size_t sub_nx);

    float fvec_L2sqr(const float *x, const float *y, size_t d);
}
#endif //IVF_HNSW_LIB_UTILS_H
