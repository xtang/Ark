"""Database models using dataclasses for the podcast generator."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class Generation:
    """Main record for each video generation."""

    id: Optional[int] = None
    topic_key: str = ""
    topic_name: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Output paths
    dialogue_json_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None


@dataclass
class DialogueRequest:
    """Record for Gemini dialogue generation requests."""

    id: Optional[int] = None
    generation_id: int = 0
    prompt: str = ""
    response_raw: str = ""
    dialogue_json: str = ""  # Parsed dialogue as JSON string
    references: str = ""  # JSON array of reference sources
    summary: str = ""
    word_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: Optional[str] = None

    def get_dialogue(self) -> list[dict]:
        """Parse dialogue JSON to list."""
        if not self.dialogue_json:
            return []
        return json.loads(self.dialogue_json)

    def get_references(self) -> list[str]:
        """Parse references JSON to list."""
        if not self.references:
            return []
        return json.loads(self.references)


@dataclass
class AudioRequest:
    """Record for ElevenLabs TTS requests."""

    id: Optional[int] = None
    generation_id: int = 0
    dialogue_count: int = 0  # Number of dialogue lines
    audio_path: str = ""
    duration_seconds: float = 0.0
    voice_segments_json: str = ""  # JSON string of timing data
    created_at: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: Optional[str] = None

    def get_voice_segments(self) -> list[dict]:
        """Parse voice segments JSON to list."""
        if not self.voice_segments_json:
            return []
        return json.loads(self.voice_segments_json)


@dataclass
class ImageRequest:
    """Record for Gemini image generation requests."""

    id: Optional[int] = None
    generation_id: int = 0
    prompt: str = ""
    image_index: int = 0  # Which image in the sequence (0-3)
    image_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class VideoOutput:
    """Record for final video output."""

    id: Optional[int] = None
    generation_id: int = 0
    video_path: str = ""
    duration_seconds: float = 0.0
    resolution: str = ""
    file_size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: Optional[str] = None
