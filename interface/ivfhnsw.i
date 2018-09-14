%module ivfhnsw
%{
#define SWIG_FILE_WITH_INIT
#include "IndexIVF_HNSW.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float *x, size_t n, size_t d)};
%apply (unsigned int* ARGOUT_ARRAY1, int DIM1) {(ivfhnsw::IndexIVF_HNSW::idx_t *labels, size_t k)};
%apply (long* ARGOUT_ARRAY1, int DIM1) {(long *labels, size_t k)};
%apply (float* ARGOUT_ARRAY1, int DIM1) {(float* distances, size_t k_)};


/*
Wrapper for IndexIVF_HNSW::assign
*/
%rename (assign) assign_numpy;
%exception assign_numpy {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%extend ivfhnsw::IndexIVF_HNSW {
void assign_numpy(const float *x, size_t n, size_t d, idx_t *labels, size_t k) {
    if (d != $self->d) {
        PyErr_Format(PyExc_ValueError,
                     "Query vectors must be of length d=%d, got %d",
                     $self->d, d);
        return;
    }
    return $self->assign(n, x, labels, k);
}
}
%ignore assign;


/*
Wrapper for IndexIVF_HNSW::search
*/
%exception _search {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%extend ivfhnsw::IndexIVF_HNSW {
void _search(const float *x, size_t n, size_t d, float* distances, size_t k_, long *labels, size_t k) {
    if (d != $self->d) {
        PyErr_Format(PyExc_ValueError,
                     "Query vectors must be of length d=%d, got %d",
                     $self->d, d);
        return;
    }
    if (k != k_) {
        PyErr_Format(PyExc_ValueError,
                     "Output sizes must be the same, got %d and %d",
                     k_, k);
        return;
    }
    $self->search(k, x, distances, labels);
}
}
%ignore search;

%include "IndexIVF_HNSW.h"

%pythoncode %{
import functools

cls = IndexIVF_HNSW

@functools.wraps(cls._search)
def search_wrapper(self, x, k):
    """
    Query n vectors of dimension d to the index.

    Return at most k vectors. If there are not enough results for the query,
    the result array is padded with -1s.
    """
    return self._search(x, k, k)

cls.search = search_wrapper
%}

