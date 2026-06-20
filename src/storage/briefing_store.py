"""
src/storage/briefing_store.py
==============================
Persistent JSON storage for Briefing objects.

Each day's briefing is stored as a separate JSON file:
    data/briefings/YYYY-MM-DD.json

Design decisions:
  - One file per day: easy to inspect, back up, and version.
  - Pydantic's .model_dump_json() handles serialization cleanly.
  - Date-based file naming makes retrieval O(1) — no index needed.
  - atomic_write: write to .tmp file first, then rename. This prevents
    a partial file being left on disk if the process is killed mid-write.

Storage operations:
  - save(briefing)             → write to disk
  - load(date_str)             → load from disk, return Briefing or None
  - load_latest()              → find the most recent briefing
  - list_dates()               → list all stored briefing dates
  - exists(date_str)           → check if a briefing for a date exists
  - delete(date_str)           → remove a briefing file

Usage:
    from src.storage.briefing_store import BriefingStore
    store = BriefingStore(settings.data_dir)
    store.save(briefing)
    latest = store.load_latest()
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.exceptions import StorageError
from src.logger import get_logger

if TYPE_CHECKING:
    from src.models import Briefing

log = get_logger(__name__)

_BRIEFING_DIR_NAME = "briefings"


class BriefingStore:
    """
    File-system storage for daily Briefing objects.

    Briefings are stored as JSON files under:
        <data_dir>/briefings/YYYY-MM-DD.json

    Thread-safety: The class is not thread-safe. It is intended to be
    used in a single-threaded async context (one orchestrator, one
    pipeline run at a time).
    """

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.briefings_dir = self.data_dir / _BRIEFING_DIR_NAME
        self.briefings_dir.mkdir(parents=True, exist_ok=True)
        log.debug("BriefingStore initialized at %s", self.briefings_dir)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save(self, briefing: "Briefing") -> Path:
        """
        Serialize and save a Briefing to disk.

        Uses an atomic write: write to a temp file, then rename to final path.
        This guarantees that the file on disk is always complete (never partial).

        Parameters
        ----------
        briefing:
            The Briefing object to persist.

        Returns
        -------
        Path
            Absolute path to the saved file.

        Raises
        ------
        StorageError
            If the file cannot be written.
        """
        file_path = self._path_for_date(briefing.date)
        tmp_path = file_path.with_suffix(".tmp")

        try:
            json_str = briefing.model_dump_json(indent=2)
            tmp_path.write_text(json_str, encoding="utf-8")
            # Atomic rename
            tmp_path.replace(file_path)
        except OSError as e:
            # Clean up temp file if rename failed
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise StorageError(
                f"Failed to save briefing: {e}",
                path=str(file_path),
                operation="write",
            ) from e

        log.debug("Briefing saved: %s", file_path)
        return file_path

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load(self, date_str: str) -> "Briefing | None":
        """
        Load a briefing for the given date string (YYYY-MM-DD).

        Returns None if no briefing exists for that date.

        Raises
        ------
        StorageError
            If the file exists but cannot be parsed.
        """
        file_path = self._path_for_date(date_str)
        if not file_path.exists():
            return None

        try:
            from src.models import Briefing
            text = file_path.read_text(encoding="utf-8")
            data = json.loads(text)
            return Briefing.model_validate(data)
        except json.JSONDecodeError as e:
            raise StorageError(
                f"Corrupted briefing file: {e}",
                path=str(file_path),
                operation="read",
            ) from e
        except Exception as e:
            raise StorageError(
                f"Failed to load briefing: {e}",
                path=str(file_path),
                operation="read",
            ) from e

    def load_latest(self) -> "Briefing | None":
        """
        Load the most recently generated briefing.

        Returns None if no briefings are stored yet.
        """
        dates = self.list_dates()
        if not dates:
            return None
        latest_date = sorted(dates)[-1]  # ISO date strings sort lexicographically
        return self.load(latest_date)

    def load_range(
        self,
        start_date: str,
        end_date: str,
    ) -> list["Briefing"]:
        """
        Load all briefings between start_date and end_date (inclusive).

        Parameters
        ----------
        start_date, end_date:
            Date strings in YYYY-MM-DD format.

        Returns
        -------
        list[Briefing]
            Briefings in chronological order (oldest first).
        """
        dates = self.list_dates()
        in_range = [d for d in dates if start_date <= d <= end_date]
        result = []
        for d in sorted(in_range):
            briefing = self.load(d)
            if briefing is not None:
                result.append(briefing)
        return result

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def exists(self, date_str: str) -> bool:
        """Return True if a briefing for the given date is stored."""
        return self._path_for_date(date_str).exists()

    def list_dates(self) -> list[str]:
        """
        Return a sorted list of all dates for which briefings are stored.

        Returns date strings (YYYY-MM-DD), oldest first.
        """
        dates = []
        for f in self.briefings_dir.glob("????-??-??.json"):
            dates.append(f.stem)  # stem strips .json
        return sorted(dates)

    def count(self) -> int:
        """Return the total number of stored briefings."""
        return len(list(self.briefings_dir.glob("????-??-??.json")))

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete(self, date_str: str) -> bool:
        """
        Delete the briefing for the given date.

        Returns True if deleted, False if it didn't exist.

        Raises
        ------
        StorageError
            If the file exists but cannot be deleted.
        """
        file_path = self._path_for_date(date_str)
        if not file_path.exists():
            return False
        try:
            file_path.unlink()
            log.debug("Briefing deleted: %s", file_path)
            return True
        except OSError as e:
            raise StorageError(
                f"Failed to delete briefing: {e}",
                path=str(file_path),
                operation="delete",
            ) from e

    def cleanup_old(self, keep_days: int = 30) -> int:
        """
        Delete briefings older than ``keep_days`` days.

        Returns the number of files deleted.
        """
        cutoff = date.today().isoformat()
        # Build cutoff by going back keep_days
        from datetime import timedelta
        cutoff_date = (date.today() - timedelta(days=keep_days)).isoformat()

        deleted = 0
        for date_str in self.list_dates():
            if date_str < cutoff_date:
                self.delete(date_str)
                deleted += 1

        if deleted:
            log.info("Cleaned up %d old briefings (keep_days=%d)", deleted, keep_days)
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for_date(self, date_str: str) -> Path:
        """Return the file path for a given date string."""
        # Validate format loosely (YYYY-MM-DD)
        if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
            raise StorageError(
                f"Invalid date format: {date_str!r} (expected YYYY-MM-DD)",
                path="",
                operation="path",
            )
        return self.briefings_dir / f"{date_str}.json"
