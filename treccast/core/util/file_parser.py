"""File parser utility class."""

from typing import Iterator
import tarfile


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
        elif filepath.endswith(".tsv"):
            return FileParser._parse_tsv(filepath)
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
    def _parse_tsv(filepath: str) -> Iterator[str]:
        """Iterates through a TSV file line by line.

        Args:
            filepath: Path to file.

        Yields:
            Single line in a file.
        """
        with open(filepath, mode="r") as f:
            for line in f:
                yield line.strip()
