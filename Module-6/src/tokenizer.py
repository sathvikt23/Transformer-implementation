import hashlib
from typing import List, Dict, Tuple

# Special token constants
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
EOD_TOKEN_ID = 3  # End of Document token
UNK_TOKEN_ID = 4

SPECIAL_TOKENS = {
    "<PAD>": PAD_TOKEN_ID,
    "<BOS>": BOS_TOKEN_ID,
    "<EOS>": EOS_TOKEN_ID,
    "<EOD>": EOD_TOKEN_ID,
    "<UNK>": UNK_TOKEN_ID,
}

# Subword BPE Merges Table (Ordered by priority)
BPE_MERGES_PAIRS = [
    (b"t", b"h"), (b"e", b"r"), (b"i", b"n"), (b"a", b"n"), (b"o", b"n"),
    (b"a", b"t"), (b"e", b"n"), (b"o", b"r"), (b"e", b"s"), (b"e", b"d"),
    (b"o", b"u"), (b"i", b"t"), (b"t", b"o"), (b"a", b"r"), (b"s", b"t"),
    (b"th", b"e"), (b"in", b"g"), (b"a", b"nd"), (b"t", b"ion"), (b"f", b"or"),
    (b"c", b"ode"), (b"d", b"ata"), (b"t", b"rain"), (b"s", b"hard"), (b"m", b"od"),
    (b"i", b"ndic"), (b"r", b"eason"), (b"t", b"ext"), (b"p", b"ack"), (b"l", b"ean"),
]


class FrozenTokenizer:
    """
    Frozen, deterministic Byte-Pair Encoding (BPE) subword tokenizer.
    Enforces deterministic subword encoding, exact decoding, and immutable version hashing.
    """

    VERSION = "v5.0.0-bpe-frozen"

    def __init__(self):
        self.special_tokens = SPECIAL_TOKENS
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self._byte_offset = len(self.special_tokens)

        # Build BPE Vocabulary & Merge Rules
        self.bpe_ranks: Dict[Tuple[int, int], int] = {}
        self.token_to_bytes: Dict[int, bytes] = {}
        self.bytes_to_token: Dict[bytes, int] = {}

        # 1. Base Byte Tokens (IDs 5 to 260 for bytes 0..255)
        for b in range(256):
            tok_id = self._byte_offset + b
            b_val = bytes([b])
            self.token_to_bytes[tok_id] = b_val
            self.bytes_to_token[b_val] = tok_id

        # 2. Add Subword BPE Merges
        next_id = self._byte_offset + 256
        for rank, (p1, p2) in enumerate(BPE_MERGES_PAIRS):
            merged_bytes = p1 + p2
            if p1 in self.bytes_to_token and p2 in self.bytes_to_token:
                id1 = self.bytes_to_token[p1]
                id2 = self.bytes_to_token[p2]
                if merged_bytes not in self.bytes_to_token:
                    self.bytes_to_token[merged_bytes] = next_id
                    self.token_to_bytes[next_id] = merged_bytes
                    self.bpe_ranks[(id1, id2)] = rank
                    next_id += 1

        # Compute deterministic content hash of frozen BPE tokenizer definition
        vocab_representation = (
            f"VERSION:{self.VERSION};"
            f"SPECIAL:{sorted(self.special_tokens.items())};"
            f"MERGES:{[(p1.decode('latin1'), p2.decode('latin1')) for p1, p2 in BPE_MERGES_PAIRS]}"
        )
        self._hash = hashlib.sha256(vocab_representation.encode("utf-8")).hexdigest()

    @property
    def hash(self) -> str:
        """Returns the frozen deterministic content hash of the BPE tokenizer."""
        return self._hash

    def _bpe_encode_bytes(self, raw_bytes: bytes) -> List[int]:
        if not raw_bytes:
            return []

        tokens = [self._byte_offset + b for b in raw_bytes]
        if len(tokens) < 2:
            return tokens

        while len(tokens) >= 2:
            # Find best merge pair with minimum BPE rank
            min_rank = float("inf")
            best_pair = None
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair
                    best_idx = i

            if best_pair is None:
                break

            # Replace pair with merged BPE token ID
            merged_bytes = self.token_to_bytes[best_pair[0]] + self.token_to_bytes[best_pair[1]]
            merged_id = self.bytes_to_token[merged_bytes]

            new_tokens = tokens[:best_idx] + [merged_id] + tokens[best_idx + 2:]
            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_eod: bool = False) -> List[int]:
        """
        Deterministically encodes text into BPE subword token IDs.
        If add_eod is True, appends EOD_TOKEN_ID at the end.
        """
        raw_bytes = text.encode("utf-8")
        tokens = self._bpe_encode_bytes(raw_bytes)
        if add_eod:
            tokens.append(EOD_TOKEN_ID)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of BPE token IDs back into string text.
        Ignores special tokens and invalid byte sequences during text decoding.
        """
        byte_list = bytearray()
        for tok in tokens:
            if tok in self.token_to_bytes:
                byte_list.extend(self.token_to_bytes[tok])
        return byte_list.decode("utf-8", errors="ignore")


_GLOBAL_TOKENIZER = FrozenTokenizer()


def get_global_tokenizer() -> FrozenTokenizer:
    return _GLOBAL_TOKENIZER
