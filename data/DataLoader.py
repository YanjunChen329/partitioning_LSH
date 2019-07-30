
class DataLoader(object):
    """
    Base Class for DataLoader
    """
    def __init__(self, dataset_name):
        self.__name__ = dataset_name

    def get_size(self):
        """
        Get the number of data points in the DataLoader.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_item(self, id):
        """
        Get the data point item identified by ID.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
