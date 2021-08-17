"""File parser utility class."""
from typing import Iterator, List, Tuple

import tarfile
from html.parser import HTMLParser


class FileParser:
    @staticmethod
    def parse(filepath: str) -> Iterator[str]:
        """A generator from file yielding a line. Currently works with `.tar.gz`
            and `.tsv` file extensions.

        Args:
            filepath: Path to file.

        Yields:
            Single line in a file.
        """
        if filepath.endswith(".tar.gz"):
            return FileParser._parse_tar(filepath)
        elif filepath.endswith(".tsv") or filepath.endswith(".txt"):
            return FileParser._parse_text_file(filepath)
        elif filepath.endswith(".cbor"):
            return FileParser._parse_cbor(filepath)
        elif filepath.endswith(".trecweb"):
            return FileParser._parse_trecweb(filepath)
        else:
            raise ValueError("File type not supported")

    @staticmethod
    def _parse_tar(filepath: str) -> Iterator[str]:
        """Iterates through all files inside tar.gz file line by line.

        Args:
            filepath: Path to file.

        Yields:
            Single line in a file.
        """
        with tarfile.open(filepath, mode="r|gz") as tar:
            for member in tar:
                for line in tar.extractfile(member):
                    yield line.decode().strip()

    @staticmethod
    def _parse_text_file(filepath: str) -> Iterator[str]:
        """Iterates through a text file line by line.

        Args:
            filepath: Path to file.

        Yields:
            Single line in a file.
        """
        with open(filepath, mode="r") as f:
            for line in f:
                yield line.strip()

    @staticmethod
    def _parse_trecweb(filepath: str) -> Iterator[Tuple[str]]:
        """Iterates through a trecweb file passage by passage.

        Args:
            filepath: Path to file.

        Yields:
            A tuple of doc_id_passage_id, document title, and the corresponding
                passage.
        """
        parser = TrecwebParser()
        for line in FileParser._parse_text_file(filepath):
            parser.feed(line)
            if parser.next_passage:
                yield parser.passage


class TrecwebParser(HTMLParser):
    def __init__(self, *, convert_charrefs: bool = True) -> None:
        """Parser for TRECWEB files. Inherits from HTMLParser due to the
        compatible structure.

        Args:
            convert_charrefs (optional): If convert_charrefs is True, all
                character references are automatically converted to the
                corresponding Unicode characters. Defaults to True.
        """
        super().__init__(convert_charrefs=convert_charrefs)
        self._next_passage = False
        self._doc_count = 0
        self._accepted_tags = [
            "DOC",
            "DOCNO",
            "DOCHDR",
            "URL",
            "HTML",
            "TITLE",
            "BODY",
            "passage",
        ]
        self.reset()

    @property
    def doc_count(self):
        return self._doc_count

    @property
    def next_passage(self):
        return self._next_passage

    @property
    def passage(self):
        self._next_passage = False
        passage = (
            f"{self._doc_id}-{self._passage_id}",
            self._title,
            self._passage,
        )
        self._passage = ""
        return passage

    def reset(self):
        """Reset all attributes. Mostly used when starting to parse a new
        document
        """
        self._doc_id = None
        self._passage_id = None
        self._title = ""
        self._passage = ""
        super().reset()

    def parse_starttag(self, i: int) -> int:
        """Ignore "<" if it is not a part of an accepted tag.

        Args:
            i: Location at which to test in the text provided to the feed
            method.

        Returns:
            Next location the check.
        """
        if not any(
            self.rawdata.startswith(tag, i + 1) for tag in self._accepted_tags
        ):
            self.handle_data("<")
            return i + 1
        return super().parse_starttag(i)

    def parse_endtag(self, i: int) -> int:
        """Ignore "</" if it is not a part of an accepted tag.

        Args:
            i: Location at which to test in the text provided to the feed
            method.

        Returns:
            Next location the check.
        """
        if not any(
            self.rawdata.startswith(tag, i + 2) for tag in self._accepted_tags
        ):
            self.handle_data("</")
            return i + 2
        return super().parse_endtag(i)

    def parse_comment(self, i: int) -> int:
        """Ignore "<!--" if it is not a part of an accepted tag.

        Args:
            i: Location at which to test in the text provided to the feed
            method.

        Returns:
            Next location the check.
        """
        self.handle_data("<!--")
        return i + 4

    def parse_pi(self, i: int) -> int:
        """Ignore "<?" if it is not a part of an accepted tag.

        Args:
            i: Location at which to test in the text provided to the feed
            method.

        Returns:
            Next location the check.
        """
        self.handle_data("<?")
        return i + 2

    def parse_html_declaration(self, i: int) -> int:
        """Ignore "<!" if it is not a part of an accepted tag.

        Args:
            i: Location at which to test in the text provided to the feed
            method.

        Returns:
            Next location the check.
        """
        self.handle_data("<!")
        return i + 2

    def handle_starttag(self, tag: str, attrs: List[Tuple]) -> None:
        """Handles opening tag.

        Args:
            tag: Lowercased tag name.
            attrs: List of tuples containing all element attributes and their
                values.
        """

        if tag == "passage":
            # Each attribute is a tuple of key-value pairs
            # There should be only id as a single attribute
            if len(attrs) != 1:
                print(
                    f"Additional unknown attributes encountered in <{tag}>",
                    attrs,
                )
            self._passage_id = attrs[0][1]
        if tag == "doc":
            self._doc_count = 0

    def handle_endtag(self, tag: str) -> None:
        """Handles closing tag.

        Args:
            tag: Lowercased tag name.
        """
        if tag == "passage":
            self._next_passage = True
        if tag == "doc":
            self._doc_count += 1
            self.reset()

    def handle_data(self, data: str) -> None:
        """Handles text between opening and closing tags.

        Args:
            data: Text between opening and closing tags.
        """
        if self.lasttag == "passage":
            self._passage += data
        elif self.lasttag == "title":
            self._title += data
        elif self.lasttag == "docno":
            self._doc_id = data
