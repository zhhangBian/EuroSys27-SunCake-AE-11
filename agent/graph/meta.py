from __future__ import annotations

from typing import List, Union, Dict
from dataclasses import dataclass
from uuid import UUID


@dataclass
class LLMCallMetadata:
    model: str
    max_new_tokens: int
    temperature: float


@dataclass
class LLMTextChunk:
    text: str
    metadata: dict

    @staticmethod
    def from_text(text: str) -> LLMTextChunk:
        return LLMTextChunk(text=text, metadata={})


@dataclass
class LLMTextChunkChain:
    chunks: List[LLMTextChunk]

    @staticmethod
    def from_single_text(text: str) -> LLMTextChunkChain:
        return LLMTextChunkChain(chunks=[LLMTextChunk.from_text(text)])

    def to_text(self) -> str:
        return "".join([chunk.text for chunk in self.chunks])

@dataclass
class TransferDataItem:
    data: Union[Dict[UUID, LLMTextChunkChain], LLMTextChunkChain]

    def __getitem__(self, node_uuid: UUID) -> LLMTextChunkChain:
        return self.data[node_uuid] if isinstance(self.data, dict) else self.data

    def __setitem__(self, node_uuid: UUID, chain: LLMTextChunkChain):
        if isinstance(self.data, dict):
            self.data[node_uuid] = chain
        else:
            self.data = chain

    def to_text(self) -> str:
        assert isinstance(self.data, dict) or \
            isinstance(self.data, LLMTextChunkChain), \
            "The data must be a map or a single chain."
        if isinstance(self.data, dict):
            return "".join([chain.to_text() for chain in self.data.values()])
        else:
            return self.data.to_text()
