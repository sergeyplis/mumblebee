import unittest

from book2opus.tts_kokoro import (
    build_atempo_chain,
    build_narration_segments,
    parse_chaptered_book,
    slugify_chapter,
    strip_duplicate_chapter_heading,
)


class TestChapterAndAudioUtils(unittest.TestCase):
    def test_parse_chaptered_book(self):
        sample = (
            "<<<BOOK_TITLE: Growing Young>>>\n"
            "<<<BOOK_AUTHOR: Ashley Montagu>>>\n\n"
            "<<<CHAPTER: Preface>>>\n\n"
            "First paragraph.\n\n"
            "<<<CHAPTER: Chapter 1: Neoteny>>>\n\n"
            "Body of chapter one.\n"
        )
        meta, chapters = parse_chaptered_book(sample)
        self.assertEqual(meta, {"title": "Growing Young", "author": "Ashley Montagu"})
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0], ("Preface", "First paragraph."))
        self.assertEqual(chapters[1], ("Chapter 1: Neoteny", "Body of chapter one."))

    def test_slugify_chapter(self):
        self.assertEqual(slugify_chapter("Chapter 1: Neoteny & Evolution?"), "Chapter_1_Neoteny_Evolution")
        self.assertLessEqual(len(slugify_chapter("A" * 100, max_len=32)), 32)

    def test_atempo_chain(self):
        self.assertIsNone(build_atempo_chain(1.0))
        self.assertEqual(build_atempo_chain(1.3), "atempo=1.3")
        self.assertEqual(build_atempo_chain(2.5), "atempo=2.0,atempo=1.25")

    def test_strip_duplicate_chapter_heading(self):
        self.assertEqual(
            strip_duplicate_chapter_heading("Preface", "Preface\nSometimes one has to say difficult things."),
            "Sometimes one has to say difficult things.",
        )
        self.assertEqual(
            strip_duplicate_chapter_heading(
                "Chapter 1: Neoteny and Evolution",
                "Neoteny and Evolution\n\nIt is the possibility of escaping.",
            ),
            "It is the possibility of escaping.",
        )
        self.assertEqual(
            strip_duplicate_chapter_heading(
                "Communication from a Flying Saucer",
                "Communication\nfrom a\nFlying Saucer\n\nThe paperback edition.",
            ),
            "The paperback edition.",
        )

    def test_build_narration_segments_pauses(self):
        body = "A Short Subheading\n\nFirst full paragraph of prose.\n\nSecond full paragraph of prose."
        segs = build_narration_segments(
            body,
            chapter_title="Preface",
            chapter_pause=1.2,
            heading_pause=0.7,
            paragraph_pause=0.45,
        )
        self.assertEqual(segs[0], ("Preface.", 1.2))
        self.assertEqual(segs[1], ("A Short Subheading.", 0.7))
        self.assertEqual(segs[2], ("First full paragraph of prose.", 0.45))
        self.assertEqual(segs[3], ("Second full paragraph of prose.", 0.0))


if __name__ == "__main__":
    unittest.main()
