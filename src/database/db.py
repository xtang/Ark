"""SQLite database connection and CRUD operations."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Generation, DialogueRequest, AudioRequest, ImageRequest, VideoOutput


class Database:
    """SQLite database manager for the podcast generator."""

    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Main generations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_key TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                dialogue_json_path TEXT,
                audio_path TEXT,
                video_path TEXT
            )
        """)

        # Dialogue requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                response_raw TEXT,
                dialogue_json TEXT,
                references_json TEXT,
                summary TEXT,
                word_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success INTEGER DEFAULT 0,
                error_message TEXT,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)

        # Audio requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                dialogue_count INTEGER DEFAULT 0,
                audio_path TEXT,
                duration_seconds REAL DEFAULT 0,
                voice_segments_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success INTEGER DEFAULT 0,
                error_message TEXT,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)

        # Image requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                image_index INTEGER DEFAULT 0,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success INTEGER DEFAULT 0,
                error_message TEXT,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)

        # Video outputs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                video_path TEXT,
                duration_seconds REAL DEFAULT 0,
                resolution TEXT,
                file_size_bytes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success INTEGER DEFAULT 0,
                error_message TEXT,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)

        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    # ==================== Generation CRUD ====================

    def create_generation(self, topic_key: str, topic_name: str) -> Generation:
        """Create a new generation record."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO generations (topic_key, topic_name, status)
            VALUES (?, ?, 'pending')
            """,
            (topic_key, topic_name),
        )
        self.conn.commit()

        gen = Generation(
            id=cursor.lastrowid,
            topic_key=topic_key,
            topic_name=topic_name,
            status="pending",
        )
        return gen

    def update_generation_status(
        self,
        gen_id: int,
        status: str,
        error_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Update generation status and optional fields."""
        updates = ["status = ?"]
        values = [status]

        if error_message:
            updates.append("error_message = ?")
            values.append(error_message)

        if status == "completed":
            updates.append("completed_at = ?")
            values.append(datetime.now().isoformat())

        for key, value in kwargs.items():
            if key in ("dialogue_json_path", "audio_path", "video_path"):
                updates.append(f"{key} = ?")
                values.append(value)

        values.append(gen_id)

        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE generations SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        self.conn.commit()

    def get_generation(self, gen_id: int) -> Optional[Generation]:
        """Get a generation by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM generations WHERE id = ?", (gen_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return Generation(
            id=row["id"],
            topic_key=row["topic_key"],
            topic_name=row["topic_name"],
            status=row["status"],
            error_message=row["error_message"],
            dialogue_json_path=row["dialogue_json_path"],
            audio_path=row["audio_path"],
            video_path=row["video_path"],
        )

    def get_recent_generations(self, limit: int = 10) -> list[Generation]:
        """Get recent generations ordered by creation time."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM generations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()

        return [
            Generation(
                id=row["id"],
                topic_key=row["topic_key"],
                topic_name=row["topic_name"],
                status=row["status"],
                error_message=row["error_message"],
                dialogue_json_path=row["dialogue_json_path"],
                audio_path=row["audio_path"],
                video_path=row["video_path"],
            )
            for row in rows
        ]

    # ==================== Dialogue Request CRUD ====================

    def create_dialogue_request(self, generation_id: int, prompt: str) -> DialogueRequest:
        """Create a new dialogue request record."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO dialogue_requests (generation_id, prompt) VALUES (?, ?)",
            (generation_id, prompt),
        )
        self.conn.commit()

        return DialogueRequest(id=cursor.lastrowid, generation_id=generation_id, prompt=prompt)

    def update_dialogue_request(
        self,
        req_id: int,
        response_raw: str,
        dialogue: list[dict],
        references: list[str],
        summary: str,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Update dialogue request with response data."""
        word_count = sum(len(d.get("text", "")) for d in dialogue)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE dialogue_requests SET
                response_raw = ?,
                dialogue_json = ?,
                references_json = ?,
                summary = ?,
                word_count = ?,
                success = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                response_raw,
                json.dumps(dialogue, ensure_ascii=False),
                json.dumps(references, ensure_ascii=False),
                summary,
                word_count,
                1 if success else 0,
                error_message,
                req_id,
            ),
        )
        self.conn.commit()

    # ==================== Audio Request CRUD ====================

    def create_audio_request(self, generation_id: int, dialogue_count: int) -> AudioRequest:
        """Create a new audio request record."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO audio_requests (generation_id, dialogue_count) VALUES (?, ?)",
            (generation_id, dialogue_count),
        )
        self.conn.commit()

        return AudioRequest(
            id=cursor.lastrowid, generation_id=generation_id, dialogue_count=dialogue_count
        )

    def update_audio_request(
        self,
        req_id: int,
        audio_path: str,
        duration_seconds: float,
        voice_segments: list[dict],
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Update audio request with response data."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE audio_requests SET
                audio_path = ?,
                duration_seconds = ?,
                voice_segments_json = ?,
                success = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                audio_path,
                duration_seconds,
                json.dumps(voice_segments, ensure_ascii=False),
                1 if success else 0,
                error_message,
                req_id,
            ),
        )
        self.conn.commit()

    # ==================== Image Request CRUD ====================

    def create_image_request(
        self, generation_id: int, prompt: str, image_index: int
    ) -> ImageRequest:
        """Create a new image request record."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO image_requests (generation_id, prompt, image_index)
            VALUES (?, ?, ?)
            """,
            (generation_id, prompt, image_index),
        )
        self.conn.commit()

        return ImageRequest(
            id=cursor.lastrowid,
            generation_id=generation_id,
            prompt=prompt,
            image_index=image_index,
        )

    def update_image_request(
        self,
        req_id: int,
        image_path: str,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Update image request with result."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE image_requests SET
                image_path = ?,
                success = ?,
                error_message = ?
            WHERE id = ?
            """,
            (image_path, 1 if success else 0, error_message, req_id),
        )
        self.conn.commit()

    def get_image_requests(self, generation_id: int) -> list[ImageRequest]:
        """Get all image requests for a generation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM image_requests WHERE generation_id = ? ORDER BY image_index",
            (generation_id,),
        )
        rows = cursor.fetchall()

        return [
            ImageRequest(
                id=row["id"],
                generation_id=row["generation_id"],
                prompt=row["prompt"],
                image_index=row["image_index"],
                image_path=row["image_path"],
                success=bool(row["success"]),
                error_message=row["error_message"],
            )
            for row in rows
        ]

    # ==================== Video Output CRUD ====================

    def create_video_output(
        self,
        generation_id: int,
        video_path: str,
        duration_seconds: float,
        resolution: str,
        file_size_bytes: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> VideoOutput:
        """Create a video output record."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO video_outputs
            (generation_id, video_path, duration_seconds, resolution, file_size_bytes, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generation_id,
                video_path,
                duration_seconds,
                resolution,
                file_size_bytes,
                1 if success else 0,
                error_message,
            ),
        )
        self.conn.commit()

        return VideoOutput(
            id=cursor.lastrowid,
            generation_id=generation_id,
            video_path=video_path,
            duration_seconds=duration_seconds,
            resolution=resolution,
            file_size_bytes=file_size_bytes,
            success=success,
            error_message=error_message,
        )
