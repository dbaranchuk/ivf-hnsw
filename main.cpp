#include <iostream>
#include <cstring>

//void sift_test1B();
//void deep_test10M();
void sift_test1B_PQ();

int main(int argc, char **argv) {
    const int subset_size_milllions = 10;
    const size_t vecsize = subset_size_milllions * 1000000;
    const size_t qsize = 10000;
    const size_t vecdim = 128;

    const int efConstruction = 240;
    const int M_PQ = 16;
    const int k = 1;

    const map<size_t, size_t> M_map = {{vecsize, 32}};//{{50000000, 32}, {100000000, 24}, {150000000, 16}, {800000000, 8}, {900000000, 6}, {1000000000, 4}};
    const vector<size_t> elements_per_level = {vecsize};//{947368422, 50000000, 2500000, 125000, 6250, 312, 16};

    char path_index[1024];
    char path_edges[1024];
    char path_info[1024];
    char path_gt[1024];
    const char *path_q = NULL;
    const char *path_data = NULL;
    const char *path_codebooks = NULL;
    const char *path_tables = NULL;

    sprintf(path_edges, "/sata2/dbaranchuk/bigann/base1B_M%d/sift%dm_ef_%d_edges.ivecs", M_PQ, subset_size_milllions, efConstruction);
    sprintf(path_info, "/sata2/dbaranchuk/bigann/base1B_M%d/sift%dm_ef_%d_info.bin", M_PQ, subset_size_milllions, efConstruction);

    sprintf(path_index, "/sata2/dbaranchuk/bigann/base1B_M%d/sift%dm_ef_%d.bin", M_PQ, subset_size_milllions, efConstruction);
    sprintf(path_gt,"/sata2/dbaranchuk/bigann/gnd/idx_%dM.ivecs", subset_size_milllions);
    //sprintf(path_gt,"/sata2/dbaranchuk/bigann/base1B_M%d/idx_%dM_pq.ivecs", M_PQ, subset_size_milllions);

    int d;
    int n, k, ret;

    for (int i = 1 ; i < argc ; i++) {
        char *a = argv[i];
        if (!strcmp (a, "-data") && i+1 < argc) {
            path_data = argv[++i];
        }
        else if (!strcmp (a, "-info") && i+1 < argc) {
            fi_name = argv[++i];
            fmt_in = FMT_BVEC;
        }
        else if (!strcmp (a, "-o") && i+1 < argc) {
            fo_name = argv[++i];
        }
        else if (!strcmp (a, "-n") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &n);
            assert (ret);
        }
        else if (!strcmp (a, "-k") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &k);
            assert (ret);
        }
        else if (!strcmp (a, "-d") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &d);
            assert (ret);
        }
        else if (!strcmp (a, "-rnd")) {
            rnd = true;
        }
    }
    assert (fi_name && fo_name);

    //sift_test1B();
    sift_test1B_PQ();
    //deep_test10M();
    return 0;  
};