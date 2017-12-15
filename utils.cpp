//
// Created by dbaranchuk on 15.12.17.
//

#include "utils.h"

template<typename format>
void readXvec(std::ifstream &input, format *data, const int d, const int n = 1) {
    int in = 0;
    for (int i = 0; i < n; i++) {
        input.read((char *) &in, sizeof(int));
        if (in != d) {
            std::cout << "file error\n";
            exit(1);
        }
        input.read((char *) (data + i * d), in * sizeof(format));
    }
}

template<typename format>
void readXvecFvec(std::ifstream &input, float *data, const int d, const int n = 1) {
    int in = 0;
    format mass[d];

    for (int i = 0; i < n; i++) {
        input.read((char *) &in, sizeof(int));
        if (in != d) {
            std::cout << "file error\n";
            exit(1);
        }
        input.read((char *) mass, in * sizeof(format));
        for (int j = 0; j < d; j++)
            data[i * d + j] = (1.0) * mass[j];
    }
}

void random_subset(const float *x, float *x_out, int d, int nx, int sub_nx) {
    int seed = 1234;
    std::vector<int> perm(nx);
    faiss::rand_perm(perm.data(), nx, seed);

    for (int i = 0; i < sub_nx; i++)
        memcpy(x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
}
