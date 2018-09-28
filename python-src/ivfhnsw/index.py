from .wrapper import IndexIVF_HNSW

class Index(IndexIVF_HNSW):
    def search(self, x, k):
        """
        Query n vectors of dimension d to the index.

        Return at most k vectors. If there are not enough results for the query,
        the result array is padded with -1s.
        """
        return super().search(x, k, k)

