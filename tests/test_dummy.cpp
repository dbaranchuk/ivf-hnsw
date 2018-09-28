#include <iostream>
#include "IndexIVF_HNSW.h"


int main(int argc, char **argv) {
    ivfhnsw::IndexIVF_HNSW* index = new ivfhnsw::IndexIVF_HNSW(4,4,4,4);
    delete index;
    std::cout << "OK" << std::endl;
    return 0;
}
