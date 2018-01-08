
#include "utils.h"

namespace ivfhnsw {

    void random_subset(const float *x, float *x_out, int d, int nx, int sub_nx) {
        int seed = 1234;
        std::vector<int> perm(nx);
        faiss::rand_perm(perm.data(), nx, seed);

        for (int i = 0; i < sub_nx; i++)
            memcpy(x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
    }


    float fvec_L2sqr(const float *x, const float *y, size_t d) {
        float PORTABLE_ALIGN32 TmpRes[8];
        #ifdef USE_AVX
        size_t qty16 = d >> 4;

        const float *pEnd1 = x + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (x < pEnd1) {
            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return (res);
        #else
        size_t qty16 = d >> 4;

        const float *pEnd1 = x + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (x < pEnd1) {
            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(x);
            x += 4;
            v2 = _mm_loadu_ps(y);
            y += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return (res);
        #endif
    }
}
























void save_groups(const char *path_groups, const char *path_idxs,
                 const char *path_data, const char *path_precomputed_idxs,
                 const int d, const int nb)
{
    const int ncentroids = 999973;
    const int batch_size = 1000000;
    const int nbatches = nb / batch_size;

    std::vector<std::vector<float>> data(groups_per_iter);
    std::vector<std::vector<idx_t>> idxs(groups_per_iter);

    std::ifstream base_input(path_data, ios::binary);
    std::ifstream idx_input(path_precomputed_idxs, ios::binary);
    std::vector<float> batch(batch_size * d);
    std::vector<idx_t> idx_batch(batch_size);

    FILE *groups_output = fopen(path_groups, "wb");
    FILE *idxs_output = fopen(path_idxs, "wb");

    // Set range of groups we can afford to add per iteration
    int ngroups_added = 0;
    int groups_per_iter = 100000;
    while (ngroups_added < ncentroids) {
        std::cout << "Groups " << ngroups_added << " / " << ncentroids << std::endl;
        if (ncentroids-group_idxs <= groups_per_iter)
            groups_per_iter = ncentroids - ngroups_added;

        // Iterate through the dataset extracting points from groups in [groups_idxs, group_idxs + groups_per_iter)
        for (int b = 0; b < nbatches; b++) {
            readXvec<idx_t>(idx_input, idx_batch.data(), batch_size, 1);
            readXvec<float>(base_input, batch.data(), d, batch_size);

            for (size_t i = 0; i < batch_size; i++) {
                if (idx_batch[i] < ngroups_added ||
                    idx_batch[i] >= ngroups_added + groups_per_iter)
                    continue;

                idx_t idx = idx_batch[i] % groups_per_iter;
                for (int j = 0; j < d; j++)
                    data[idx].push_back(batch[i * d + j]);
                idxs[idx].push_back(b * batch_size + i);
            }

            if (b % 10 == 0) printf("%.1f %c \n", (100. * b) / nbatch, '%');
        }
        // Save collected groups and point idxs to files
        for (int i = 0; i < groups_per_iter; i++) {
            int groupsize = data[i].size() / d;

            fwrite(&groupsize, sizeof(int), 1, groups_output);
            fwrite(data[i].data(), sizeof(float), data[i].size(), groups_output);

            fwrite(&groupsize, sizeof(int), 1, idxs_output);
            fwrite(idxs[i].data(), sizeof(idx_t), idxs[i].size(), idxs_output);
        }
        ngroups_added += groups_per_iter;
    }
    base_input.close();
    idx_input.close();

    fclose(groups_output);
    fclose(idxs_output);
}