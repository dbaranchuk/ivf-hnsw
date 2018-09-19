def test_import_ivfhnsw():
    import ivfhnsw


def test_lowlevel_constructor_and_destructor_wrappers():
    from ivfhnsw import _wrapper
    i = _wrapper.new_IndexIVF_HNSW(4,4,4,4)
    _wrapper.delete_IndexIVF_HNSW(i)


def test_pipeline():
    from ivfhnsw import Index
    index = Index(4,4,4,4)
    index.build_quantizer('', '', '')
    index.assign([[5,5,5,5]], 2)
    distances, labels = index.search([[1,2,3,4],
                                      [6,2,3,2]], 3)
    assert distances.shape[0] == 3
    assert labels.shape[0] == 3
