"""tests/test_pipeline.py — Unit + integration tests for NewsLens."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessor import clean_text
from src.data_loader  import _apply_label_map, _combine_text, LABEL_MAP
import pandas as pd


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_removes_urls(self):
        r = clean_text("Visit https://example.com for more")
        assert "http" not in r and "example" not in r

    def test_removes_html_tags(self):
        r = clean_text("<b>Breaking</b> news today")
        assert "<b>" not in r and "breaking" in r

    def test_removes_wire_tags(self):
        r = clean_text("AP - The stock market crashed")
        assert not r.strip().startswith("ap -")

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string_none(self):
        assert clean_text(None) == ""

    def test_non_string_int(self):
        assert clean_text(123) == ""

    def test_preserves_numbers(self):
        r = clean_text("The 4G network covers 99% of users")
        assert "4g" in r and "99" in r

    def test_normalises_whitespace(self):
        r = clean_text("hello   world\n\ttab")
        assert "  " not in r and "\n" not in r


class TestDataLoader:
    def _df(self):
        return pd.DataFrame({
            "Class Index": [1,2,3,4,99],
            "Title":       ["World","Sports","Business","Tech","Unknown"],
            "Description": ["A","B","C","D","E"],
        })

    def test_label_map_applied(self):
        df = _apply_label_map(self._df(), LABEL_MAP)
        assert df[df["Class Index"]==1]["label"].values[0] == "World"
        assert df[df["Class Index"]==4]["label"].values[0] == "Sci/Tech"

    def test_unmapped_labels_dropped(self):
        df = _apply_label_map(self._df(), LABEL_MAP)
        assert 99 not in df["Class Index"].values

    def test_combine_text(self):
        df = pd.DataFrame({"Title":["Fed raises rates"],
                           "Description":["The Federal Reserve raised rates by 0.25%."]})
        r  = _combine_text(df)
        assert "text" in r.columns
        assert "fed raises rates" in r["text"].values[0].lower()


class TestModelIntegration:
    @pytest.fixture(scope="class")
    def model(self):
        mp = ROOT/"models"/"best_model.joblib"
        if not mp.exists():
            pytest.skip("Run train.py first.")
        import joblib
        return joblib.load(str(mp))

    def test_valid_class(self, model):
        valid = {"World","Sports","Business","Sci/Tech"}
        assert model.predict(["Apple reports record quarterly revenue"])[0] in valid

    def test_proba_sums_to_one(self, model):
        p = model.predict_proba(["Premier League season kicks off"])
        assert abs(p.sum(axis=1)[0] - 1.0) < 1e-5

    def test_proba_shape(self, model):
        p = model.predict_proba(["text one","text two","text three"])
        assert p.shape == (3, 4)

    def test_short_input(self, model):
        assert len(model.predict(["a"])) == 1

    def test_deterministic(self, model):
        t = ["NASA launches Mars mission"]
        assert model.predict(t)[0] == model.predict(t)[0]

    def test_proba_all_positive(self, model):
        p = model.predict_proba(["Scientists discover new planet"])
        assert (p >= 0).all()

    def test_batch_consistency(self, model):
        texts = ["Gold prices surge","Athletes compete in Paris","Tech stocks rally"]
        singles = [model.predict([t])[0] for t in texts]
        batch   = list(model.predict(texts))
        assert singles == batch
