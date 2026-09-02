from src.tokenizer import get_global_tokenizer, PAD_TOKEN_ID
from src.shards import Document
from src.packer import SequencePacker
from src.batch import BatchBuilder


def test_packing_loss_mask_and_attention_isolation():
    tokenizer = get_global_tokenizer()
    packer = SequencePacker(max_seq_len=128)
    batch_builder = BatchBuilder(seq_len=128)

    doc1 = Document(doc_id="doc_A", text="Doc A text", tokens=tokenizer.encode("Doc A text"))
    doc2 = Document(doc_id="doc_B", text="Doc B text sample", tokens=tokenizer.encode("Doc B text sample"))

    # Test Structure-Preserving Packing Policy
    packed_seqs = packer.pack([doc1, doc2], policy="structure-preserving")
    assert len(packed_seqs) > 0
    packed = packed_seqs[0]

    mb = batch_builder.build_microbatch("mb_test_01", packed)

    # 1. Invariant: Loss mask NEVER covers padding tokens
    for tok, mask_val in zip(mb.input_ids, mb.loss_mask):
        if tok == PAD_TOKEN_ID:
            assert mask_val == 0.0, "Loss mask must be 0.0 for padding tokens"
        else:
            assert mask_val == 1.0, "Loss mask must be 1.0 for non-padding content tokens"

    # 2. Invariant: No cross-document attention leakage under structure-preserving policy
    spans = mb.spans
    if len(spans) >= 2:
        span_A = spans[0]
        span_B = spans[1]

        # Token in Doc B (span_B["start_idx"]) should NOT attend to token in Doc A (span_A["start_idx"])
        idx_A = span_A["start_idx"]
        idx_B = span_B["start_idx"]

        attn_val = mb.attention_mask[idx_B][idx_A]
        assert attn_val == 0, f"Cross-document attention leakage detected! Token at index {idx_B} in {span_B['doc_id']} attended to token at index {idx_A} in {span_A['doc_id']}"
