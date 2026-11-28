import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from mumblebee.pronunciation import (
    filter_and_validate_fixes,
    kokoro_ipa_to_standard,
    phonetic_skeleton,
    run_pronunciation_audit,
    standard_ipa_to_kokoro,
)
from mumblebee.tts_kokoro import apply_pronunciations


class TestPronunciationAudit(unittest.TestCase):
    def test_ipa_roundtrip_and_vocab_validation(self):
        self.assertEqual(kokoro_ipa_to_standard("lˈOz ˈAnʤəlz"), "ˈloʊz ˈeɪndʒəlz")
        self.assertEqual(kokoro_ipa_to_standard("tˈʊɹɪŋ"), "ˈtʊɹɪŋ")

        vocab = set("lˌɔs ˈænʤələsɡɑɹOAIWYʧɚəɪiːu")
        norm, missing = standard_ipa_to_kokoro("lˌɔs ˈændʒələs", vocab)
        self.assertEqual(norm, "lˌɔs ˈænʤələs")
        self.assertEqual(missing, [])

        # British ɒ and ASCII g/r normalize cleanly into American Kokoro symbols
        norm2, missing2 = standard_ipa_to_kokoro("ˈgeɪlɒr", vocab)
        self.assertEqual(norm2, "ˈɡAlɑɹ")
        self.assertEqual(missing2, [])

        # Invalid character caught
        _, bad_missing = standard_ipa_to_kokoro("ˈɡAl@n", vocab)
        self.assertEqual(bad_missing, ["@"])

    def test_filter_rejects_noops_skips_and_invalid_chars(self):
        vocab = set("lˌɔs ˈænʤələsɡAlnəbɛɹtuː")
        cand_map = {
            "Gehlen": {"kokoro_ipa": "ʤˈAlən"},
            "Baer": {"kokoro_ipa": "bˈɛɹ"},
            "tabu": {"kokoro_ipa": "tˈæbu"},
            "ForAllTokens": {"kokoro_ipa": "fˌɔɹˈɔltˈOkᵊnz"},
        }
        raw = [
            {"term": "Gehlen", "type": "ipa", "fix": "ˈgeɪlən", "reason": "German hard G"},
            {"term": "Baer", "type": "ipa", "fix": "ˈbɛɹ", "reason": "Stress placement only"},
            {"term": "tabu", "type": "ipa", "fix": "təˈbuː", "reason": "Actually current is fine. Skipping."},
            {"term": "ForAllTokens", "type": "respell", "fix": "For All Tokens", "reason": "CamelCase"},
            {"term": "Gehlen", "type": "ipa", "fix": "ˈɡAlən", "reason": "Duplicate"},
        ]
        out = filter_and_validate_fixes(raw, cand_map, vocab)
        self.assertEqual(set(out.keys()), {"Gehlen", "ForAllTokens"})
        self.assertEqual(out["Gehlen"]["ipa"], "ˈɡAlən")
        self.assertEqual(out["ForAllTokens"]["respell"], "For All Tokens")
        self.assertEqual(phonetic_skeleton("bˈɛɹ"), phonetic_skeleton("ˈbɛɹ"))

    def test_apply_pronunciations_supports_ipa_and_respell(self):
        table = {
            "Los Angeles": {"ipa": "lˌɔs ˈænʤələs"},
            "ForAllTokens": {"respell": "For All Tokens"},
        }
        res = apply_pronunciations("In Los Angeles we run ForAllTokens.", table)
        self.assertEqual(res, "In [Los Angeles](/lˌɔs ˈænʤələs/) we run For All Tokens.")
        # Idempotent
        self.assertEqual(apply_pronunciations(res, table), res)

    def test_audit_timeout_gracefully_returns_partial(self):
        fake_cands = [
            {"term": "Gehlen", "category": "freq_name", "count": 6, "kokoro_ipa": "ʤˈAlən", "std_ipa": "ˈdʒeɪlən", "snippet": "Gehlen wrote..."}
        ]
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(
                json.dumps({"vocab": {"ɡ": 1, "A": 2, "l": 3, "ə": 4, "n": 5, "ˈ": 6}}),
                encoding="utf-8",
            )
            fake_kokoro = types.SimpleNamespace(KPipeline=mock.MagicMock(return_value=mock.MagicMock()))
            fake_hf = types.SimpleNamespace(hf_hub_download=mock.MagicMock(return_value=str(cfg_path)))
            with mock.patch.dict(sys.modules, {"kokoro": fake_kokoro, "huggingface_hub": fake_hf}), \
                 mock.patch("mumblebee.pronunciation.extract_candidates", return_value=fake_cands), \
                 mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
                overrides, stats = run_pronunciation_audit(
                    "Gehlen wrote...",
                    "Demo",
                    "http://127.0.0.1:8080/v1/chat/completions",
                    "mock",
                    timeout_s=10.0,
                )
        self.assertEqual(overrides, {})
        self.assertTrue(stats["timed_out"])


if __name__ == "__main__":
    unittest.main()
