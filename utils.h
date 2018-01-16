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
    void readXval(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }


    template<typename T>
    void writeXval(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }


    template<typename T>
    void readXvec(std::ifstream &in, T *data, const int d, const int n = 1) {
        int D = 0;
        for (int i = 0; i < n; i++) {
            in.read((char *) &D, sizeof(int));
            if (D != d) {
                std::cout << "file error\n";
                exit(1);
            }
            in.read((char *) (data + i * d), d * sizeof(T));
        }
    }


    template<typename T>
    void writeXvec(std::ofstream &out, T *data, const int d, const int n = 1) {
        for (int i = 0; i < n; i++) {
            out.write((char *) &d, sizeof(int));
            out.write((char *) (data + i * d), d * sizeof(T));
        }
    }


    template<typename T>
    void readXvecFvec(std::ifstream &in, float *data, const int d, const int n = 1) {
        int D = 0;
        T mass[d];

        for (int i = 0; i < n; i++) {
            in.read((char *) &D, sizeof(int));
            if (D != d) {
                std::cout << "file error\n";
                exit(1);
            }
            in.read((char *) mass, d * sizeof(T));
            for (int j = 0; j < d; j++)
                data[i * d + j] = (1.0) * mass[j];
        }
    }

    inline bool exists(const std::string& name) {
        std::ifstream f(name.c_str());
        return f.good();
    }

    void random_subset(const float *x, float *x_out, int d, int nx, int sub_nx);

    float fvec_L2sqr(const float *x, const float *y, size_t d);
}
#endif //IVF_HNSW_LIB_UTILS_H
