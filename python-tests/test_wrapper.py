def test_wrapper():
    from ivfhnsw import _wrapper
    i = _wrapper.new_IndexIVF_HNSW(4,4,4,4)
    _wrapper.delete_IndexIVF_HNSW(i)
