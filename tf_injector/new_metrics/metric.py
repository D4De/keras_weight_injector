from abc import abstractmethod, ABCMeta

class Metric(metaclass=ABCMeta):

    def __init__(self):
        return
    
    @abstractmethod
    def chiSono(self):
        pass
