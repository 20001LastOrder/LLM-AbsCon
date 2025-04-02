import networkx as nx
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from loguru import logger


class Abstractor(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    partial_model: nx.DiGraph = nx.DiGraph()
    candidate_count: int = 0

    def __call__(self, models: list[nx.DiGraph], **kwargs) -> nx.DiGraph:
        return self.abstract(models)

    def abstract(self, models: list[nx.DiGraph], **kwargs) -> nx.DiGraph:
        for model in models:
            self.add_concrete_model

        return self.get_partial_model

    @abstractmethod
    def add_concrete_model(self, model: nx.DiGraph, **kwargs):
        pass

    def get_partial_model(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node, data in self.partial_model.nodes(data=True):
            for key in data:
                data = data.copy()
                if "weight" in key:
                    data[key] = data[key] / self.candidate_count
            graph.add_node(node, **data)

        for source, target, data in self.partial_model.edges(data=True):
            for key in data:
                data = data.copy()
                if "weight" in key:
                    data[key] = data[key] / self.candidate_count
            graph.add_edge(source, target, **data)

        return graph


class Concretizer(ABC):
    @abstractmethod
    def concretize(self, relations: dict[tuple, tuple], nodes: list[str]) -> nx.DiGraph:
        pass
