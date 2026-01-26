"""Audio generator using ElevenLabs Text-to-Dialogue API."""

import base64
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from ..database import Database


class AudioGenerator:
    """Generate podcast audio using ElevenLabs Text-to-Dialogue API with timestamps."""

    API_URL = "https://api.elevenlabs.io/v1/text-to-dialogue/with-timestamps"

    def __init__(self, config: dict[str, Any], db: Database):
        """
        Initialize the audio generator.

        Args:
            config: Application configuration.
            db: Database instance.
        """
        self.config = config
        self.db = db

        load_dotenv()
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")

        # Build speaker -> voice_id mapping
        speakers = config.get("dialogue", {}).get("speakers", [])
        self.voice_map = {s["name"]: s["voice_id"] for s in speakers}

    def _apply_speed_effect(self, input_path: Path, output_path: Path, speed_ratio: float) -> None:
        """Apply speed up effect using FFmpeg atempo filter."""
        # atempo filter range is 0.5 to 2.0
        # For higher speeds, we'd need to chain filters, but 1.0-2.0 is expected range here.
        speed_ratio = max(0.5, min(2.0, speed_ratio))
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-filter:a", f"atempo={speed_ratio}",
            "-vn", # Disable video just in case
            str(output_path)
        ]
        
        # Suppress output unless error
        subprocess.run(cmd, check=True, capture_output=True)

    def generate(
        self,
        generation_id: int,
        dialogue: list[dict],
        output_dir: Path,
    ) -> tuple[str, float, list[dict]]:
        """
        Generate audio from dialogue.

        Args:
            generation_id: Database generation ID.
            dialogue: List of dialogue lines with speaker and text.
            output_dir: Directory to save output files.

        Returns:
            Tuple of (audio_path, duration_seconds, voice_segments).

        Raises:
            Exception: If generation fails.
        """
        # Create DB record
        req = self.db.create_audio_request(generation_id, len(dialogue))

        try:
            # Build API inputs
            inputs = []
            for line in dialogue:
                speaker = line.get("speaker")
                text = line.get("text", "")

                voice_id = self.voice_map.get(speaker)
                if not voice_id:
                    raise ValueError(f"Unknown speaker: {speaker}")

                inputs.append({
                    "voice_id": voice_id,
                    "text": text,
                })

            # Call API
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
            }

            payload = {"inputs": inputs}

            response = requests.post(self.API_URL, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            # Decode audio
            audio_base64 = data.get("audio_base64", "")
            audio_bytes = base64.b64decode(audio_base64)

            # Save audio file
            output_dir.mkdir(parents=True, exist_ok=True)
            original_audio_path = output_dir / f"audio_{generation_id}_original.mp3"
            audio_path = output_dir / f"audio_{generation_id}.mp3"
            
            with open(original_audio_path, "wb") as f:
                f.write(audio_bytes)

            # Extract timing data
            voice_segments = data.get("voice_segments", [])

            # Apply speed up optimization if configured
            speed_ratio = self.config.get("audio", {}).get("speed_ratio", 1.0)
            
            # Check if speedup is requested and meaningful (> 1% diff)
            if abs(speed_ratio - 1.0) > 0.01:
                print(f"  ⚡ Applying {speed_ratio}x speed up...")
                try:
                    self._apply_speed_effect(original_audio_path, audio_path, speed_ratio)
                    
                    # Update timestamps
                    scale_factor = 1.0 / speed_ratio
                    for seg in voice_segments:
                        if "start_time_seconds" in seg:
                            seg["start_time_seconds"] *= scale_factor
                        if "end_time_seconds" in seg:
                            seg["end_time_seconds"] *= scale_factor
                            
                except Exception as e:
                    print(f"  ⚠️ Audio speed up failed, using original: {e}")
                    # Fallback to original
                    shutil.copy(original_audio_path, audio_path)
            else:
                # Just copy original to final path
                shutil.copy(original_audio_path, audio_path)

            # Calculate duration from last segment (which is now scaled)
            duration_seconds = 0.0
            if voice_segments:
                # Recalculate based on scaled segments
                # Note: actual file duration should match this roughly
                duration_seconds = max(seg.get("end_time_seconds", 0) for seg in voice_segments)

            # Update DB
            self.db.update_audio_request(
                req_id=req.id,
                audio_path=str(audio_path),
                duration_seconds=duration_seconds,
                voice_segments=voice_segments,
                success=True,
            )

            self.db.update_generation_status(
                generation_id,
                status="audio_complete",
                audio_path=str(audio_path),
            )

            return str(audio_path), duration_seconds, voice_segments

        except Exception as e:
            self.db.update_audio_request(
                req_id=req.id,
                audio_path="",
                duration_seconds=0,
                voice_segments=[],
                success=False,
                error_message=str(e),
            )
            self.db.update_generation_status(
                generation_id,
                status="failed",
                error_message=f"Audio generation failed: {e}",
            )
            raise
