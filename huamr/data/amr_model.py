from dataclasses import dataclass


@dataclass
class AMRModel:
    id: str
    sentence: str
    amr_graph: str
