#include <iostream>
#include <cstring>
#include <cassert>

//void sift_test1B();
//void deep_test10M();
template<typename dist_t>
void sift_test1B_PQ(const char *, const char *, const char *, const char *,
                    const char *, const char *, const char *,
                    const int, const int, const int, const int, const int, const int);

int main(int argc, char **argv) {
    size_t vecsize = 10000000;
    size_t qsize = 10000;
    size_t vecdim = 128;
    size_t M = 16;
    size_t efConstruction = 60;

    const char *path_gt = NULL;
    const char *path_q = NULL;
    const char *path_data = NULL;
    const char *path_codebooks = NULL;
    const char *path_tables = NULL;
    const char *path_edges = NULL;
    const char *path_info = NULL;

    const char *l2space_type = NULL; //{int, float}
    int k = 1, ret, ep;

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
        else if (!strcmp (a, "-n") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &vecsize);
            assert (ret);
        }
        else if (!strcmp (a, "-nq") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &qsize);
            assert (ret);
        }
        else if (!strcmp (a, "-M") && i+1 < argc) {
            ret = sscanf (argv[++i], "%d", &M);
            assert (ret);
        }
        else if (!strcmp (a, "-l2space") && i+1 < argc) {
            l2space_type = argv[++i];
        }
    }
    //assert(argc == 18);
    //sift_test1B();

    if (!path_q) path_q = "/sata2/dbaranchuk/bigann/bigann_query.bvecs";
    if (!path_data) path_data = "/sata2/dbaranchuk/bigann/base1B_M16/bigann_base_pq.bvecs";
    if (!path_codebooks) path_codebooks = "/sata2/dbaranchuk/bigann/base1B_M16/codebooks.fvecs";
    if (!path_tables) path_tables = "/sata2/dbaranchuk/bigann/base1B_M16/distance_tables.dat";

    if ((!path_codebooks && path_tables) || (!path_codebooks && !path_tables)) {
        std::cerr << "Enter path_codebooks and path_tables to use PQ" << std::endl;
        exit(1);
    }

    if (!strcmp (l2space_type, "int")) {
        if (path_codebooks || path_tables) {
            std::cerr << "Use l2space_type = float for PQ" << std::endl;
            exit(1);
        }
        //sift_test1B_PQ<int>(path_codebooks, path_tables, path_data, path_info, path_edges, path_q, path_gt,
                            //k, vecsize, qsize, vecdim, efConstruction, M);
    } else if (!strcmp (l2space_type, "float")) {
        sift_test1B_PQ<float>(path_codebooks, path_tables, path_data, path_info, path_edges, path_q, path_gt, k, vecsize, qsize, vecdim, efConstruction, M);
    }

    return 0;  
};