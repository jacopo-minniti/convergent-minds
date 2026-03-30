from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from convminds.data.io import _require_pandas


@dataclass(frozen=True)
class TokenEvent:
    text: str
    onset: float
    duration: float | None = None
    metadata: dict[str, object] | None = None


def load_events_tsv(
    path: str | Path,
    *,
    text_columns: Sequence[str] = ("word", "trial_type", "token", "text"),
    onset_column: str = "onset",
    duration_column: str = "duration",
) -> list[TokenEvent]:
    pd = _require_pandas()
    df = pd.read_csv(Path(path).expanduser(), sep="\t")

    text_column = None
    for candidate in text_columns:
        if candidate in df.columns:
            text_column = candidate
            break
    if text_column is None:
        raise ValueError(f"None of the text columns {text_columns} found in {path}.")
    if onset_column not in df.columns:
        raise ValueError(f"Expected onset column '{onset_column}' in {path}.")

    events: list[TokenEvent] = []
    for _, row in df.iterrows():
        text = str(row[text_column]).strip()
        if not text or text.lower() == "nan":
            continue
        onset = float(row[onset_column])
        duration = None
        if duration_column in df.columns:
            value = row[duration_column]
            if value == value:
                duration = float(value)
        metadata = {key: row[key] for key in df.columns if key not in {text_column, onset_column, duration_column}}
        events.append(TokenEvent(text=text, onset=onset, duration=duration, metadata=metadata))
    return events
