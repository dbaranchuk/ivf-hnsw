#include <iostream>
#include <cstring>
#include <cassert>

//void sift_test1B();
//void deep_test10M();
void sift_test1B_PQ(const char *, const char *, const char *, const char *,
                    const char *, const char *, const char *, const int, const int);

int main(int argc, char **argv) {
    const int subset_size_milllions = 10;
    const size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize = 10000;
    size_t vecdim = 128;
    size_t M = 16;

    int efConstruction = 240;
    const int M_PQ = 16;

    const char *path_gt = NULL;
    const char *path_q = NULL;
    const char *path_data = NULL;
    const char *path_codebooks = NULL;
    const char *path_tables = NULL;
    const char *path_edges = NULL;
    const char *path_info = NULL;

    int k, ret, ep;

    for (int i = 1 ; i < argc ; i++) {
        char *a = argv[i];
        if (!strcmp (a, "-path_codebooks") && i+1 < argc) {
            path_codebooks = argv[++i];
        }
        else if (!strcmp (a, "-path_tables") && i+1 < argc) {
            path_tables = argv[++i];
        }
        else if (!strcmp (a, "-path_data") && i+1 < argc) {
            path_data = argv[++i];
        }
        else if (!strcmp (a, "-path_info") && i+1 < argc) {
            path_info = argv[++i];
        }
        else if (!strcmp (a, "-path_edges") && i+1 < argc) {
            path_edges = argv[++i];
        }
        else if (!strcmp (a, "-path_q") && i+1 < argc) {
            path_q = argv[++i];
        }
        else if (!strcmp (a, "-path_gt") && i+1 < argc) {
            path_gt = argv[++i];
        }
        //else if (!strcmp (a, "-enterpoint") && i+1 < argc) {
        //    ret = sscanf (argv[++i], "%d", &ep);
        //    assert (ret);
        //}
        else if (!strcmp (a, "-k") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &k);
            assert (ret);
        }
        else if (!strcmp (a, "-efConstruction") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &efConstruction);
            assert (ret);
        }
        else if (!strcmp (a, "-d") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &vecdim);
            assert (ret);
        }
    }
    //assert(argc == 18);
    //sift_test1B();
    sift_test1B_PQ(path_codebooks, path_tables, path_data, path_info, path_edges, path_q, path_gt, k, qsize);
    return 0;  
};