from pathlib import Path
import numpy as np
import pandas as pd


def make_text_bin_df(n: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    pos_Pool = ["produk sangat bagus mantap cepat puas sekali", "layanan ramah barang asli berkualitas tinggi"]
    neg_Pool = ["barang rusak parah kecewa berat pengiriman lambat", "respon buruk sekali penipuan zonk"]
    y = np.array([i % 2 for i in range(n)], dtype=int)
    texts = [
        f"{pos_Pool[i % 2] if y[i] == 1 else neg_Pool[i % 2]} #{i}"
        for i in range(n)
    ]
    tr = pd.DataFrame({"id": range(n), "text": texts, "num_meta": rng.normal(size=n), "cat_meta": ["A" if i % 3 == 0 else "B" for i in range(n)], "y": y})
    te_texts = [f"{pos_Pool[i % 2]} test {i}" if i % 2 == 1 else f"{neg_Pool[i % 2]} test {i}" for i in range(12)]
    te = pd.DataFrame({"id": range(100, 112), "text": te_texts, "num_meta": rng.normal(size=12), "cat_meta": ["A" if i % 2 == 0 else "C" for i in range(12)]})
    return tr, te


def create_tiny_hf_model_dir(base_dir: Path) -> Path:
    """Create a tiny 100% offline BERT model and WordPiece tokenizer on disk for fast CPU unit tests."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    model_dir = base_dir / "tiny_bert"
    model_dir.mkdir(parents=True, exist_ok=True)

    vocab_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "produk", "sangat", "bagus", "mantap", "cepat", "puas", "sekali",
        "layanan", "ramah", "barang", "asli", "berkualitas", "tinggi",
        "rusak", "parah", "kecewa", "berat", "pengiriman", "lambat",
        "respon", "buruk", "penipuan", "zonk", "test", "tidak", "yang",
        "col", "val", "a", "b", "c", "0", "1", "2",
    ]
    while len(vocab_tokens) < 64:
        vocab_tokens.append(f"tok{len(vocab_tokens)}")
    (model_dir / "vocab.txt").write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tok = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(str(model_dir))

    cfg = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    model = BertForSequenceClassification(cfg)
    model.save_pretrained(str(model_dir))
    return model_dir
