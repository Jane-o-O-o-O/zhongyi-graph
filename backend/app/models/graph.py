from typing import Literal

from pydantic import BaseModel, Field


NodeLabel = Literal[
    "Symptom",
    "Syndrome",
    "Treatment",
    "Formula",
    "Prescription",
    "Herb",
    "Dose",
    "Dosage",
    "Function",
    "Indication",
    "Channel",
    "Property",
    "Flavor",
    "Alias",
    "Source",
    "DistributionArea",
    "Meridian",
    "Category",
    "TextSource",
    "Evidence",
    "ExternalSource",
]


class GraphNode(BaseModel):
    id: str
    label: NodeLabel
    name: str
    description: str = ""
    properties: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    display: str
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceCard(BaseModel):
    id: str
    title: str
    source: str
    snippet: str
    source_type: Literal["local", "external"]
    location: str = ""
