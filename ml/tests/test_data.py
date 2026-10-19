import pandas as pd

from smishing import LABELS
from smishing.data import deduplicate, load_splits, split, template_key


def test_template_key_masks_links_and_numbers():
    a = "URGENT: your parcel is held. Pay 1.99 GBP at bit.do/xK2a to release it"
    b = "URGENT: your parcel is held. Pay 2.49 GBP at bit.do/Qm9z to release it"
    assert template_key(a) == template_key(b)


def test_template_key_masks_full_urls_and_bare_domains():
    assert template_key("Log in at https://secure-bank.example/verify?id=7") == template_key(
        "Log in at www.other-site.net/x"
    )
    assert template_key("visit bankofamerica.com now") == template_key("visit paypal.com now")


def test_template_key_keeps_different_messages_apart():
    assert template_key("See you at 7 tonight") != template_key("Your account is locked")


def test_deduplicate_drops_exact_copies_and_conflicts():
    df = pd.DataFrame(
        {
            "text": ["hi there", "hi there", "win a prize", "win a prize", "call me"],
            "label": ["ham", "ham", "spam", "smishing", "ham"],
        }
    )
    out, report = deduplicate(df)

    assert sorted(out["text"]) == ["call me", "hi there"]
    assert report.exact_duplicates_dropped == 1
    assert report.conflicting_texts == 1
    assert report.conflicting_rows_dropped == 2
    assert report.rows_out == 2


def _synthetic(n_templates: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n_templates):
        label = LABELS[i % 3]
        word = f"template{chr(97 + i % 26)}{chr(97 + i // 26)}"
        for variant in range(3):
            text = f"{word} message pay {variant} at bit.do/v{variant}"
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def test_split_never_puts_one_template_in_two_splits():
    splits = split(_synthetic())
    keys = {name: set(part["text"].map(template_key)) for name, part in splits.items()}

    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    assert sum(len(p) for p in splits.values()) == len(_synthetic())


def test_committed_splits_have_no_leakage():
    splits = load_splits()
    keys = {name: set(part["text"].map(template_key)) for name, part in splits.items()}
    texts = {name: set(part["text"]) for name, part in splits.items()}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not keys[a] & keys[b], f"template keys shared between {a} and {b}"
        assert not texts[a] & texts[b], f"identical texts shared between {a} and {b}"
    for name, part in splits.items():
        assert set(part["label"]) == set(LABELS), f"{name} is missing a class"
