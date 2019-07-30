
class LSH(object):
    """
    Base Class for all LSH schemes
    """
    def __init__(self, name):
        self.__name__ = name

    def insert(self, *x):
        """
        Insert an item into LSH.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def query(self, *x):
        """
        Query a data point from LSH to get its near neighbors.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
