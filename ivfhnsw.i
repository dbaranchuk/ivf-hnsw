%module ivfhnsw
%{
#define SWIG_FILE_WITH_INIT
#include "IndexIVF_HNSW.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, float* IN_ARRAY1) {(size_t n, const float* x)};
%apply (unsigned int* ARGOUT_ARRAY1, int DIM1) {(ivfhnsw::IndexIVF_HNSW::idx_t *labels, size_t k)};

%ignore assign(size_t, const float*, idx_t*);

%include "IndexIVF_HNSW.h"
