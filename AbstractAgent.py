import abc


class AbstractAgent:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def observe(self):
        return

    @abc.abstractmethod
    def training(self):
        return

    @abc.abstractmethod
    def evaluate(self):
        return
