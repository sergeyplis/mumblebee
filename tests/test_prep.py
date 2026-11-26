import unittest

from book2opus.prep import (
    _restore_pdf_paragraphs,
    postprocess_llm_chunk,
    split_section_into_chunks,
    stitch_chunks,
    validate_cleaned_chunk,
)


class TestPrepPipeline(unittest.TestCase):
    def test_restore_pdf_paragraphs(self):
        lines = [
            "This is a full wrapped line of text that continues across the page width cleanly",
            "and ends on a short line.",
            "Next paragraph begins with a capital letter and continues across the full width",
            "of the printed column without breaking.",
        ]
        restored = _restore_pdf_paragraphs(lines)
        self.assertIn("", restored)
        joined = "\n".join(restored)
        self.assertIn("and ends on a short line.\n\nNext paragraph begins", joined)

    def test_chunker_never_splits_mid_sentence_across_header_and_table(self):
        filler = " ".join(["word"] * 120) + "."
        sec_lines = [
            filler,
            "",
            "Against the view that human erect bipedalism has been made possible by the",
            "retention of a neotenous fetal relationship, especially the gluteus",
            "\fASHLEY MONTAGU",
            "23",
            "TABLE IV",
            "Neotenous Functional Traits",
            "maximus (the large muscle of the buttock), render it unlikely.",
            "",
            "A new paragraph starts cleanly here with plenty of words to form a sentence.",
        ]
        chunks = split_section_into_chunks(
            sec_lines,
            target_words=100,
            max_words=400,
            running_headers=["ASHLEY MONTAGU"],
            is_pdf=False,
        )
        # gluteus and maximus must stay in the same chunk!
        gluteus_chunks = [c for c in chunks if "gluteus" in c]
        self.assertEqual(len(gluteus_chunks), 1)
        self.assertIn("maximus", gluteus_chunks[0])

    def test_validator_catches_real_defects(self):
        raw = "Some input text.\n\nAnother paragraph.\n\nThird paragraph.\n\nFourth paragraph."
        # Leaked running header + Fig caption + letter-spaced name
        bad = (
            "12\nASHLEY MONTAGU\n\n"
            "Fig. 1. Showing skull growth.\n\n"
            "A n n a H e m p s t e a d B r a n c h"
        )
        issues = validate_cleaned_chunk(bad, raw, ["ASHLEY MONTAGU", "Growing Young"], "stop")
        joined = " | ".join(issues)
        self.assertIn("leaked running headers", joined)
        self.assertIn("leaked figure/table captions", joined)
        self.assertIn("letter-spaced text", joined)

        # Author signature in Title Case should NOT trigger false-positive header leak
        ok_sig = "Good prose here.\n\nAshley Montagu"
        self.assertEqual(validate_cleaned_chunk(ok_sig, raw, ["ASHLEY MONTAGU"], "stop"), [])

    def test_postprocess_and_stitch(self):
        raw = "Here is the cleaned text:\n\nExample in the adult dog (Figure 9).\n\nSecond half."
        cleaned = postprocess_llm_chunk(raw)
        self.assertNotIn("Here is the cleaned text", cleaned)
        self.assertNotIn("(Figure 9)", cleaned)
        self.assertIn("adult dog.", cleaned)

        stitched = stitch_chunks(["The dog's body is\n\nalso horizontal while humans are vertical."])
        self.assertEqual(stitched, "The dog's body is also horizontal while humans are vertical.")


if __name__ == "__main__":
    unittest.main()
