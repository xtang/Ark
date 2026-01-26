"""Video generator using FFmpeg with subtitles, animations, and fade effects."""

import json
import os
import shutil
import random
import subprocess
from pathlib import Path
from typing import Any

from ..database import Database


class VideoGenerator:
    """Generate podcast videos with subtitles and animations using FFmpeg."""

    # Font settings for Chinese subtitles
    FONT_FILE = "/System/Library/Fonts/PingFang.ttc"  # macOS Chinese font
    FALLBACK_FONT = "Arial"

    def __init__(self, config: dict[str, Any], db: Database):
        """
        Initialize the video generator.

        Args:
            config: Application configuration.
            db: Database instance.
        """
        self.config = config
        self.db = db

        self.resolution = config.get("output", {}).get("video_resolution", "1920x1080")
        self.video_format = config.get("output", {}).get("video_format", "mp4")

        # Animation settings
        self.fade_duration = 0.3  # seconds for fade in/out (shorter = faster first frame)
        self.transition_duration = 0.5  # seconds for image transitions
        self.subtitle_margin = 20  # pixels from bottom (lower = closer to bottom)
        self.subtitle_font_size = config.get("output", {}).get("subtitle_font_size", 24)
        
        # Motion settings
        self.enable_motion = config.get("video", {}).get("motion_effect", True)

    def _get_background_music(self) -> str | None:
        """Get a random background music file from assets."""
        # Look in project/assets/music
        # Assuming src/generators/video.py -> src/generators -> src -> project root
        project_root = Path(__file__).parent.parent.parent
        music_dir = project_root / "assets" / "music"
        
        if not music_dir.exists():
            return None
            
        music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
        if not music_files:
            return None
            
        return str(random.choice(music_files))

    def _calculate_image_durations(
        self,
        audio_duration: float,
        voice_segments: list[dict],
        num_images: int,
    ) -> list[float]:
        """
        Calculate how long each image should be displayed.
        Uses voice segments for more accurate timing when available.
        """
        if num_images <= 0:
            return []

        if voice_segments and len(voice_segments) >= num_images:
            # Map images to dialogue segments
            durations = []
            segments_per_image = len(voice_segments) / num_images

            for i in range(num_images):
                start_idx = int(i * segments_per_image)
                end_idx = int((i + 1) * segments_per_image)

                if end_idx >= len(voice_segments):
                    end_idx = len(voice_segments) - 1

                start_time = voice_segments[start_idx].get("start_time_seconds", 0)
                end_time = voice_segments[end_idx].get("end_time_seconds", audio_duration)

                duration = end_time - start_time
                durations.append(max(0.5, duration))  # Minimum 0.5s per image

            # Extend the last image by 2 seconds to match audio padding
            if durations:
                durations[-1] += 2.0

            return durations

        # Fallback to equal distribution if no segments
        base_duration = (audio_duration + 2.0) / num_images  # Include padding in calculation
        return [base_duration] * num_images

    def _create_subtitle_file(
        self,
        dialogue: list[dict],
        voice_segments: list[dict],
        output_dir: Path,
    ) -> str:
        """Create SRT subtitle file from dialogue and timing data."""
        srt_path = output_dir / "subtitles.srt"

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, (line, segment) in enumerate(zip(dialogue, voice_segments)):
                start_time = segment.get("start_time_seconds", i * 5)
                end_time = segment.get("end_time_seconds", start_time + 5)

                # Delay first subtitle to account for video fade in
                if i == 0:
                    start_time = max(start_time, self.fade_duration)

                text = line.get("text", "")
                # Remove mood/background markers like [笑声], [惊讶] etc.
                import re
                text = re.sub(r'\[.*?\]', '', text).strip()

                # Format time as HH:MM:SS,mmm
                start_str = self._format_srt_time(start_time)
                end_str = self._format_srt_time(end_time)

                # Write SRT entry (text only, no speaker, no markers)
                f.write(f"{i + 1}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n\n")

        return str(srt_path)

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _build_ffmpeg_command(
        self,
        image_paths: list[str],
        durations: list[float],
        audio_path: str,
        output_path: str,
        subtitle_path: str | None = None,
        audio_duration: float = 0,
        music_path: str | None = None,
    ) -> list[str]:
        """Build FFmpeg command with animations, subtitles, music and fade effects."""
        width, height = self.resolution.split("x")
        width_int, height_int = int(width), int(height)

        # Build input arguments
        inputs = []
        filter_parts = []

        # Calculate total duration for fade out timing
        total_duration = sum(durations)


        # Build video filter chain (Ken Burns or Scale)
        for i, (img_path, duration) in enumerate(zip(image_paths, durations)):
            # Add extra time for transitions
            input_duration = duration + self.transition_duration
            inputs.extend(["-loop", "1", "-t", str(input_duration), "-i", img_path])

            if self.enable_motion:
                # Ken Burns effect: Zoom in 1.0 -> 1.15
                # Using 24fps for cinematic feel and performance
                # z='min(zoom+0.0015,1.5)' is too fast for 5s, let's use time-based
                # ZOOM_SPEED = 0.0005 per frame roughly 1.0 -> 1.15 in 300 frames (12s)
                # Ensure even numbers for scale to avoid libx264 errors
                filter_chain = (
                    f"[{i}:v]scale=1920:-2,zoompan=z='min(zoom+0.0005,1.15)':d=700:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={self.resolution}:fps=24,"
                    f"setsar=1"
                )
            else:
                # Simple scale and pad
                filter_chain = (
                    f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                    f"setsar=1"
                )

            # Add crossfade for all except last image
            if i < len(image_paths) - 1:
                filter_chain += f",fade=t=out:st={duration - self.transition_duration}:d={self.transition_duration}"

            # NOTE: Fade-in for first image removed to ensure WeChat can show thumbnail
            # Previously had fade=t=in:st=0 which caused first frame to be black

            filter_chain += f"[v{i}]"
            filter_parts.append(filter_chain)

        # Add audio input (Voice)
        inputs.extend(["-i", audio_path])
        voice_input_idx = len(image_paths)
        
        # Add Music input if exists
        music_input_idx = None
        if music_path:
            inputs.extend(["-i", music_path])
            music_input_idx = voice_input_idx + 1

        # Concat all video streams with xfade transitions
        if len(image_paths) > 1:
            concat_parts = []
            current_stream = "[v0]"
            offset = durations[0] - self.transition_duration

            for i in range(1, len(image_paths)):
                next_stream = f"[v{i}]"
                out_stream = f"[xf{i}]" if i < len(image_paths) - 1 else "[vconcat]"

                concat_parts.append(
                    f"{current_stream}{next_stream}xfade=transition=fade:duration={self.transition_duration}:offset={offset:.2f}{out_stream}"
                )
                current_stream = out_stream
                offset += durations[i] - self.transition_duration
        else:
            concat_parts = ["[v0]copy[vconcat]"]

        filter_parts.extend(concat_parts)

        # Add final fade out at video end
        # Add 2 seconds padding at the end so audio doesn't cut off abruptly
        fade_out_start = audio_duration + 2.0 - self.fade_duration
        filter_parts.append(
            f"[vconcat]fade=t=out:st={fade_out_start}:d={self.fade_duration}[vfaded]"
        )

        # Add subtitles if available
        if subtitle_path and os.path.exists(subtitle_path):
            # Escape special characters in path for FFmpeg
            escaped_path = subtitle_path.replace(":", "\\:").replace("'", "\\'")
            filter_parts.append(
                f"[vfaded]subtitles='{escaped_path}':force_style='FontSize={self.subtitle_font_size},PrimaryColour=&HFFFFFF&,"
                f"OutlineColour=&H000000&,Outline=2,MarginV={self.subtitle_margin}'[outv]"
            )
        else:
            filter_parts.append("[vfaded]copy[outv]")

        # Audio mixing
        # Voice: fade in/out
        voice_filter = (
            f"[{voice_input_idx}:a]apad=pad_dur=2,"
            f"afade=t=in:st=0:d={self.fade_duration},"
            f"afade=t=out:st={audio_duration + 2.0 - self.fade_duration}:d={self.fade_duration}[voice_a]"
        )
        filter_parts.append(voice_filter)
        
        final_audio = "[voice_a]"
        
        if music_input_idx is not None:
            # Music: loop, lower volume (15%), fade out
            # aloop=loop=-1:size=2e+09 loops indefinitely
            music_filter = (
                f"[{music_input_idx}:a]aloop=loop=-1:size=2e+09,"
                f"volume=0.15,"
                f"afade=t=out:st={audio_duration + 2.0 - self.fade_duration}:d={self.fade_duration}[music_a]"
            )
            filter_parts.append(music_filter)
            
            # Mix voice and music
            # duration=first means stop when voice stops (roughly) but we controlled duration via fade out
            filter_parts.append(f"[voice_a][music_a]amix=inputs=2:duration=first:dropout_transition=2[outa]")
        else:
            filter_parts.append(f"[voice_a]acopy[outa]")

        filter_complex = ";".join(filter_parts)

        # Build full command
        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # Enables quick preview/thumbnail in WeChat
            output_path,
        ]

        return cmd

    def generate(
        self,
        generation_id: int,
        image_paths: list[str],
        audio_path: str,
        audio_duration: float,
        voice_segments: list[dict],
        output_dir: Path,
        dialogue: list[dict] | None = None,
    ) -> str:
        """
        Generate video from images and audio with animations and subtitles.

        Args:
            generation_id: Database generation ID.
            image_paths: List of image file paths.
            audio_path: Path to audio file.
            audio_duration: Audio duration in seconds.
            voice_segments: Voice segment timing data.
            output_dir: Directory to save output.
            dialogue: Optional dialogue for subtitles.

        Returns:
            Path to generated video file.
            Exception: If generation fails.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"podcast_{generation_id}.{self.video_format}"

        # Create DB record
        req = self.db.create_video_output(generation_id, "", 0, self.resolution, 0, False)
        
        try:
            # Set cover image (save first image as cover.jpg)
            if image_paths and len(image_paths) > 0:
                cover_path = output_dir / "cover.jpg"
                try:
                    import shutil
                    shutil.copy(image_paths[0], cover_path)
                    print(f"🖼️ Cover image saved to: {cover_path}")
                except Exception as e:
                    print(f"⚠️ Failed to create cover image: {e}")

            # Calculate durations
            durations = self._calculate_image_durations(
                audio_duration, voice_segments, len(image_paths)
            )

            # Create subtitles if dialogue provided
            subtitle_path = None
            if dialogue and voice_segments and len(dialogue) == len(voice_segments):
                subtitle_path = self._create_subtitle_file(
                    dialogue, voice_segments, output_dir
                )

            # Get background music
            music_path = self._get_background_music()
            if music_path:
                print(f"🎵 Adding background music: {Path(music_path).name}")

            # Build and run FFmpeg command
            cmd = self._build_ffmpeg_command(
                image_paths,
                durations,
                audio_path,
                str(output_path),
                subtitle_path,
                audio_duration,
                music_path,
            )

            # For debugging: save command
            cmd_file = output_dir / "ffmpeg_cmd.txt"
            with open(cmd_file, "w") as f:
                f.write(" ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Get file size
            file_size = os.path.getsize(output_path)

            # Save to DB
            self.db.create_video_output(
                generation_id=generation_id,
                video_path=str(output_path),
                duration_seconds=audio_duration,
                resolution=self.resolution,
                file_size_bytes=file_size,
                success=True,
            )

            self.db.update_generation_status(
                generation_id,
                status="completed",
                video_path=str(output_path),
            )

            return str(output_path)

        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg failed: {e.stderr}"
            self.db.create_video_output(
                generation_id=generation_id,
                video_path="",
                duration_seconds=0,
                resolution=self.resolution,
                file_size_bytes=0,
                success=False,
                error_message=error_msg,
            )
            self.db.update_generation_status(
                generation_id,
                status="failed",
                error_message=error_msg,
            )
            raise RuntimeError(error_msg)

        except Exception as e:
            self.db.create_video_output(
                generation_id=generation_id,
                video_path="",
                duration_seconds=0,
                resolution=self.resolution,
                file_size_bytes=0,
                success=False,
                error_message=str(e),
            )
            self.db.update_generation_status(
                generation_id,
                status="failed",
                error_message=f"Video generation failed: {e}",
            )
            raise
