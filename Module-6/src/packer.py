from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from src.tokenizer import PAD_TOKEN_ID, EOD_TOKEN_ID
from src.shards import Document


@dataclass
class PackedSequence:
    input_ids: List[int]
    sample_ids: List[str]
    spans: List[Dict[str, Any]]  # [{"doc_id": str, "start_idx": int, "end_idx": int}]
    utilization: float
    policy: str
    padding_count: int


class SequencePacker:
    """
    Sequence packing engine supporting 5 sequence packing policies.
    """

    def __init__(self, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len

    def pack(self, documents: List[Document], policy: str = "structure-preserving") -> List[PackedSequence]:
        valid_policies = {"pad-only", "concat-chop", "greedy", "best-fit", "structure-preserving"}
        if policy not in valid_policies:
            raise ValueError(f"Invalid packing policy '{policy}'. Must be one of {valid_policies}")

        if policy == "pad-only":
            return self._pack_pad_only(documents)
        elif policy == "concat-chop":
            return self._pack_concat_chop(documents)
        elif policy == "greedy":
            return self._pack_greedy(documents)
        elif policy == "best-fit":
            return self._pack_best_fit(documents)
        elif policy == "structure-preserving":
            return self._pack_structure_preserving(documents)
        else:
            return self._pack_structure_preserving(documents)

    def _pack_pad_only(self, documents: List[Document]) -> List[PackedSequence]:
        packed_list = []
        for doc in documents:
            tokens = doc.tokens[: self.max_seq_len]
            padding_needed = self.max_seq_len - len(tokens)
            padded_tokens = tokens + [PAD_TOKEN_ID] * padding_needed

            utilization = len(tokens) / float(self.max_seq_len)
            spans = [{"doc_id": doc.doc_id, "start_idx": 0, "end_idx": len(tokens)}]

            packed_list.append(
                PackedSequence(
                    input_ids=padded_tokens,
                    sample_ids=[doc.doc_id],
                    spans=spans,
                    utilization=utilization,
                    policy="pad-only",
                    padding_count=padding_needed,
                )
            )
        return packed_list

    def _pack_concat_chop(self, documents: List[Document]) -> List[PackedSequence]:
        stream_tokens = []
        stream_doc_map = []  # (doc_id, tok)

        for doc in documents:
            for tok in doc.tokens:
                stream_tokens.append(tok)
                stream_doc_map.append(doc.doc_id)

        packed_list = []
        for i in range(0, len(stream_tokens), self.max_seq_len):
            chunk = stream_tokens[i : i + self.max_seq_len]
            chunk_doc_map = stream_doc_map[i : i + self.max_seq_len]

            padding_needed = self.max_seq_len - len(chunk)
            padded_chunk = chunk + [PAD_TOKEN_ID] * padding_needed

            sample_ids = list(dict.fromkeys(chunk_doc_map))
            
            # Compute spans
            spans = []
            if chunk_doc_map:
                curr_doc = chunk_doc_map[0]
                start_idx = 0
                for idx, doc_id in enumerate(chunk_doc_map):
                    if doc_id != curr_doc:
                        spans.append({"doc_id": curr_doc, "start_idx": start_idx, "end_idx": idx})
                        curr_doc = doc_id
                        start_idx = idx
                spans.append({"doc_id": curr_doc, "start_idx": start_idx, "end_idx": len(chunk_doc_map)})

            utilization = len(chunk) / float(self.max_seq_len)
            packed_list.append(
                PackedSequence(
                    input_ids=padded_chunk,
                    sample_ids=sample_ids,
                    spans=spans,
                    utilization=utilization,
                    policy="concat-chop",
                    padding_count=padding_needed,
                )
            )
        return packed_list

    def _pack_greedy(self, documents: List[Document]) -> List[PackedSequence]:
        packed_list = []
        curr_tokens = []
        curr_sample_ids = []
        curr_spans = []

        for doc in documents:
            doc_toks = doc.tokens[: self.max_seq_len]
            if len(curr_tokens) + len(doc_toks) <= self.max_seq_len:
                start_idx = len(curr_tokens)
                curr_tokens.extend(doc_toks)
                curr_sample_ids.append(doc.doc_id)
                curr_spans.append({"doc_id": doc.doc_id, "start_idx": start_idx, "end_idx": len(curr_tokens)})
            else:
                if curr_tokens:
                    padding_needed = self.max_seq_len - len(curr_tokens)
                    padded = curr_tokens + [PAD_TOKEN_ID] * padding_needed
                    packed_list.append(
                        PackedSequence(
                            input_ids=padded,
                            sample_ids=curr_sample_ids,
                            spans=curr_spans,
                            utilization=len(curr_tokens) / float(self.max_seq_len),
                            policy="greedy",
                            padding_count=padding_needed,
                        )
                    )
                curr_tokens = list(doc_toks)
                curr_sample_ids = [doc.doc_id]
                curr_spans = [{"doc_id": doc.doc_id, "start_idx": 0, "end_idx": len(doc_toks)}]

        if curr_tokens:
            padding_needed = self.max_seq_len - len(curr_tokens)
            padded = curr_tokens + [PAD_TOKEN_ID] * padding_needed
            packed_list.append(
                PackedSequence(
                    input_ids=padded,
                    sample_ids=curr_sample_ids,
                    spans=curr_spans,
                    utilization=len(curr_tokens) / float(self.max_seq_len),
                    policy="greedy",
                    padding_count=padding_needed,
                )
            )
        return packed_list

    def _pack_best_fit(self, documents: List[Document]) -> List[PackedSequence]:
        # Bin-packing best-fit allocation
        bins: List[Dict[str, Any]] = []

        for doc in documents:
            doc_toks = doc.tokens[: self.max_seq_len]
            req_len = len(doc_toks)

            best_bin_idx = -1
            min_remaining = self.max_seq_len + 1

            for idx, b in enumerate(bins):
                rem = self.max_seq_len - len(b["tokens"])
                if rem >= req_len and rem < min_remaining:
                    min_remaining = rem
                    best_bin_idx = idx

            if best_bin_idx != -1:
                b = bins[best_bin_idx]
                start_idx = len(b["tokens"])
                b["tokens"].extend(doc_toks)
                b["sample_ids"].append(doc.doc_id)
                b["spans"].append({"doc_id": doc.doc_id, "start_idx": start_idx, "end_idx": len(b["tokens"])})
            else:
                bins.append({
                    "tokens": list(doc_toks),
                    "sample_ids": [doc.doc_id],
                    "spans": [{"doc_id": doc.doc_id, "start_idx": 0, "end_idx": req_len}],
                })

        packed_list = []
        for b in bins:
            toks = b["tokens"]
            padding_needed = self.max_seq_len - len(toks)
            padded = toks + [PAD_TOKEN_ID] * padding_needed
            packed_list.append(
                PackedSequence(
                    input_ids=padded,
                    sample_ids=b["sample_ids"],
                    spans=b["spans"],
                    utilization=len(toks) / float(self.max_seq_len),
                    policy="best-fit",
                    padding_count=padding_needed,
                )
            )
        return packed_list

    def _pack_structure_preserving(self, documents: List[Document]) -> List[PackedSequence]:
        # Structure-preserving: packs full documents with EOD tokens, strictly forbidding cross-doc truncation
        packed_list = []
        curr_tokens = []
        curr_sample_ids = []
        curr_spans = []

        for doc in documents:
            doc_toks = list(doc.tokens)
            if doc_toks and doc_toks[-1] != EOD_TOKEN_ID:
                doc_toks.append(EOD_TOKEN_ID)

            if len(doc_toks) > self.max_seq_len:
                # Truncate single oversized doc with EOD preserved
                doc_toks = doc_toks[: self.max_seq_len - 1] + [EOD_TOKEN_ID]

            if len(curr_tokens) + len(doc_toks) <= self.max_seq_len:
                start_idx = len(curr_tokens)
                curr_tokens.extend(doc_toks)
                curr_sample_ids.append(doc.doc_id)
                curr_spans.append({"doc_id": doc.doc_id, "start_idx": start_idx, "end_idx": len(curr_tokens)})
            else:
                if curr_tokens:
                    padding_needed = self.max_seq_len - len(curr_tokens)
                    padded = curr_tokens + [PAD_TOKEN_ID] * padding_needed
                    packed_list.append(
                        PackedSequence(
                            input_ids=padded,
                            sample_ids=curr_sample_ids,
                            spans=curr_spans,
                            utilization=len(curr_tokens) / float(self.max_seq_len),
                            policy="structure-preserving",
                            padding_count=padding_needed,
                        )
                    )
                curr_tokens = list(doc_toks)
                curr_sample_ids = [doc.doc_id]
                curr_spans = [{"doc_id": doc.doc_id, "start_idx": 0, "end_idx": len(doc_toks)}]

        if curr_tokens:
            padding_needed = self.max_seq_len - len(curr_tokens)
            padded = curr_tokens + [PAD_TOKEN_ID] * padding_needed
            packed_list.append(
                PackedSequence(
                    input_ids=padded,
                    sample_ids=curr_sample_ids,
                    spans=curr_spans,
                    utilization=len(curr_tokens) / float(self.max_seq_len),
                    policy="structure-preserving",
                    padding_count=padding_needed,
                )
            )
        return packed_list
