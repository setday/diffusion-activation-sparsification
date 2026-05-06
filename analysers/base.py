from abc import ABC, abstractmethod


class AbstractAnalyser(ABC):
    """
    Base class for model analysers. This class defines the interface for all model analysers and provides common functionality that can be shared across different types of analysers.
    """
    
    @abstractmethod
    def add_analytical_point(self, *args, **kwargs):
        """
        Add an analytical point to the analyser. This method should be implemented by subclasses to define how analytical points are added and what information they contain.
        """
        pass
    
    @abstractmethod
    def make_report(self, *args, **kwargs):
        """
        Generate a report based on the analytical points collected by the analyser. This method should be implemented by subclasses to define how the report is generated and what information it contains.
        """
        pass
