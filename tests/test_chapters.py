import unittest

from book2opus.tts_kokoro import build_atempo_chain, parse_chaptered_book, slugify_chapter


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


if __name__ == "__main__":
    unittest.main()
