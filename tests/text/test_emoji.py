import json
from pathlib import Path
from utils.text.preprocessing import DEFAULT_ID_SLANG_MAP, load_slang_map

ROOT = Path(__file__).resolve().parents[2]
EMOJI_MAP = ROOT / "assets/dictionaries/emoji_map.json"
AMBIGUOUS = {"💀", "😭", "🙏", "🔥", "💸", "💩", "🤡"}


def test_emoji_map_file_shape():
    data = json.loads(EMOJI_MAP.read_text(encoding="utf-8"))
    assert len(data) == 10
    assert data["👍"] == "bagus" and data["❤"] == "suka"
    assert not (set(data) & AMBIGUOUS)


def test_emoji_values_are_not_slang_keys():  # spec test 15
    values = set(json.loads(EMOJI_MAP.read_text(encoding="utf-8")).values())
    slang = load_slang_map(ROOT / "assets/dictionaries/slang_id.json")
    assert not (values & set(slang)) and not (values & set(DEFAULT_ID_SLANG_MAP))
