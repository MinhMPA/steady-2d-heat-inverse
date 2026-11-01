"""Read-only access to a saved full-field conductivity calculation."""

from posterior_store import RunStore


class PosteriorSamples:
    """A saved calculation handle; physical summaries are added separately."""

    @classmethod
    def open(cls, output):
        """Open saved evidence for inspection, requiring only NumPy.

        Parameters
        ----------
        output : path-like
            Existing complete or interrupted calculation directory.

        Returns
        -------
        PosteriorSamples
            Read-only handle with copied metadata and a resolved output path.
        """
        result = cls.__new__(cls)
        result._store = RunStore.open(output)
        return result

    @property
    def output(self):
        """Resolved directory holding this saved physical calculation."""
        return self._store.path

    @property
    def metadata(self):
        """Independent copy of the saved controls, starts and requested total."""
        return self._store.metadata

    def close(self):
        """Release the saved calculation handle; repeated calls are harmless."""
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
