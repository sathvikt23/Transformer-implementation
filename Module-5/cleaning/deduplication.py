import hashlib
import re
from typing import Set

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


class DeduplicationEngine:
    """
    Stage 2: Exact & Near Deduplication Engine.
    - Exact Dedup: MD5 hash tracking.
    - Near Dedup: 128-permutation MinHash LSH at 0.90 Jaccard similarity threshold.
    """

    def __init__(self, num_perm: int = 128, jaccard_threshold: float = 0.90):
        self.num_perm = num_perm
        self.jaccard_threshold = jaccard_threshold
        self.seen_exact_hashes: Set[str] = set()

        if HAS_DATASKETCH:
            self.lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=self.num_perm)
            self.doc_count = 0
        else:
            self.lsh = None

    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        words = re.findall(r"\w+", text.lower())
        if len(words) < k:
            return set(words)
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

    def is_exact_duplicate(self, text: str) -> bool:
        """Returns True if the document exact hash has been seen before."""
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if doc_hash in self.seen_exact_hashes:
            return True
        self.seen_exact_hashes.add(doc_hash)
        return False

    def is_near_duplicate(self, text: str) -> bool:
        """Returns True if near-duplicate found via MinHash LSH."""
        if not HAS_DATASKETCH or self.lsh is None:
            return False  # Skip near-dedup if datasketch library is missing

        shingles = self._get_shingles(text)
        if not shingles:
            return False

        minhash = MinHash(num_perm=self.num_perm)
        for s in shingles:
            minhash.update(s.encode("utf-8"))

        result = self.lsh.query(minhash)
        if len(result) > 0:
            return True

        self.doc_count += 1
        self.lsh.insert(f"doc_{self.doc_count}", minhash)
        return False
