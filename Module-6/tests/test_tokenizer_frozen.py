from src.tokenizer import FrozenTokenizer, get_global_tokenizer


def test_tokenizer_frozen_determinism():
    t1 = FrozenTokenizer()
    t2 = FrozenTokenizer()
    t3 = get_global_tokenizer()

    assert t1.hash == t2.hash == t3.hash, "Tokenizer hashes must be identical across instances"

    sample_text = "The quick brown fox jumps over 123 lazy dogs! Deterministic tokenizer test."
    tokens1 = t1.encode(sample_text)
    tokens2 = t2.encode(sample_text)
    tokens3 = t3.encode(sample_text)

    assert tokens1 == tokens2 == tokens3, "Encoding same text must produce identical token IDs"

    decoded1 = t1.decode(tokens1)
    assert decoded1 == sample_text, "Decoding tokens must reconstruct exact original text"
