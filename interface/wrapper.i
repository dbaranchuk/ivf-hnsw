%module wrapper
%{
#define SWIG_FILE_WITH_INIT
#include "IndexIVF_HNSW.h"
#include "IndexIVF_HNSW_Grouping.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {(const float *x, size_t d)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float *x, size_t n, size_t d)};
%apply (unsigned int* ARGOUT_ARRAY1, int DIM1) {(ivfhnsw::IndexIVF_HNSW::idx_t *labels, size_t k)};
%apply (unsigned int* IN_ARRAY1, int DIM1) {(const ivfhnsw::IndexIVF_HNSW::idx_t *xids, size_t n1)};
%apply (unsigned int* IN_ARRAY1, int DIM1) {(const ivfhnsw::IndexIVF_HNSW::idx_t *precomputed_idx, size_t n2)};
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
Wrapper for IndexIVF_HNSW::train_pq
*/
%exception train_pq {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%extend ivfhnsw::IndexIVF_HNSW {
void train_pq(const float *x, size_t n, size_t d) {
    if (d != $self->d) {
        PyErr_Format(PyExc_ValueError,
                     "Query vectors must be of length d=%d, got %d",
                     $self->d, d);
        return;
    }
    return $self->train_pq(n, x);
}
}
%ignore train_pq;


/*
Wrapper for IndexIVF_HNSW::search
*/
%exception search {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%extend ivfhnsw::IndexIVF_HNSW {
void search(const float *x, size_t d, float* distances, size_t k_, long *labels, size_t k) {
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


/*
Wrapper for IndexIVF_HNSW::add_batch
*/
%exception add_batch {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%extend ivfhnsw::IndexIVF_HNSW {
void add_batch(const float *x, size_t n, size_t d, const idx_t* xids, size_t n1, const idx_t *precomputed_idx, size_t n2) {
    if (d != $self->d) {
        PyErr_Format(PyExc_ValueError,
                     "Query vectors must be of length d=%d, got %d",
                     $self->d, d);
        return;
    }
    if (!(n == n1 && n == n2)) {
        PyErr_Format(PyExc_ValueError,
                     "Arrays must have the same first dimention size, got %d, %d, %d",
                     n, n1, n2);
        return;
    }
    $self->add_batch(n, x, xids, precomputed_idx);
}
}
%ignore add_batch;

%include "IndexIVF_HNSW.h"
%include "IndexIVF_HNSW_Grouping.h"

