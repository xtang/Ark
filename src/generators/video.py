"""Video generator module acting as controller."""

import os
from pathlib import Path
from typing import Any

from ..database import Database
from .veo import VeoGenerator
from .renderer import VideoRenderer

class VideoGenerator:
    """
    Main Video Generator Controller.
    Orchestrates VeoGenerator (AI video) and VideoRenderer (FFmpeg synthesis).
    """

    def __init__(self, config: dict[str, Any], db: Database):
        """
        Initialize the video generator controller.

        Args:
            config: Application configuration.
            db: Database instance.
        """
        self.config = config
        self.db = db
        
        self.resolution = config.get("output", {}).get("video_resolution", "1920x1080")
        self.video_format = config.get("output", {}).get("video_format", "mp4")
        
        # Initialize sub-components
        self.veo_gen = VeoGenerator(config)
        self.renderer = VideoRenderer(config)

    def _prepare_static_visuals(
        self,
        image_paths: list[str],
        audio_duration: float,
        voice_segments: list[dict],
        output_dir: Path,
        title: str | None,
        cover_image_path: str | None,
    ) -> tuple[list[str], list[float]]:
        """Prepare static image assets and durations."""
        # Prepare Cover
        cover_source_image = None
        title_overlay = None

        if cover_image_path and os.path.exists(cover_image_path):
            cover_source_image = cover_image_path
        elif image_paths and len(image_paths) > 0:
            cover_source_image = image_paths[0]
            title_overlay = title

        # Calculate image durations
        durations = self.renderer.calculate_image_durations(
            audio_duration, voice_segments, len(image_paths)
        )

        final_image_paths = list(image_paths)

        if cover_source_image:
            cover_path = output_dir / "cover.jpg"
            self.renderer.create_cover_with_title(cover_source_image, cover_path, title_overlay)
            print(f"🖼️ Cover image saved to: {cover_path}")
            
            # Insert cover at start
            COVER_DURATION = 1.0
            if durations:
                durations[0] = max(0.5, durations[0] - COVER_DURATION)
                durations.insert(0, COVER_DURATION)
                final_image_paths = [str(cover_path)] + final_image_paths
            else:
                durations = [COVER_DURATION]
                final_image_paths = [str(cover_path)]
                
        return final_image_paths, durations

    def _prepare_veo_visuals(
        self,
        generation_id: int,
        title: str | None,
        output_dir: Path,
    ) -> str:
        """Prepare Veo video asset."""
        print("🎥 Using Veo Loop Mode...")
        veo_prompt = "A high quality, cinematic video background." 
        if title:
            veo_prompt = f"Cinematic background for podcast about {title}, professional studio setting, 4k, highly detailed, subtle motion."
        
        veo_path = output_dir / f"veo_bg_{generation_id}.mp4"
        
        if not veo_path.exists():
                self.veo_gen.generate_clip(
                    prompt=veo_prompt,
                    output_path=veo_path,
                )
        else:
            print(f"   Using existing Veo background: {veo_path}")
        
        return str(veo_path)

    def generate(
        self,
        generation_id: int,
        image_paths: list[str],
        audio_path: str,
        audio_duration: float,
        voice_segments: list[dict],
        output_dir: Path,
        dialogue: list[dict] | None = None,
        title: str | None = None,
        cover_image_path: str | None = None,
    ) -> str:
        """
        Generate video from images/video and audio.

        Args:
            generation_id: Database generation ID.
            image_paths: List of image file paths (used for static mode or fallback).
            audio_path: Path to audio file.
            audio_duration: Audio duration in seconds.
            voice_segments: Voice segment timing data.
            output_dir: Directory to save output.
            dialogue: Optional dialogue for subtitles.
            title: Optional title for cover/prompt.
            cover_image_path: Optional path to dedicated AI-generated cover image.

        Returns:
            Path to generated video file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"podcast_{generation_id}.{self.video_format}"

        # Create DB record (Start)
        self.db.create_video_output(generation_id, "", 0, self.resolution, 0, False)

        try:
            # 1. Determine Video Mode
            video_mode = self.config.get("video", {}).get("mode", "static_images")
            
            # Asset Containers
            final_image_paths = []
            durations = []
            video_background_path = None

            # 2. Prepare Visual Assets based on Mode
            if video_mode == "veo_loop":
                video_background_path = self._prepare_veo_visuals(
                    generation_id, title, output_dir
                )
            elif video_mode == "static_images":
                final_image_paths, durations = self._prepare_static_visuals(
                    image_paths, audio_duration, voice_segments, output_dir, title, cover_image_path
                )
            else:
                # Default fallback or mixed mode (future)
                # For now fall back to static if provided, else maybe error or empty
                if image_paths:
                    final_image_paths, durations = self._prepare_static_visuals(
                        image_paths, audio_duration, voice_segments, output_dir, title, cover_image_path
                    )

            # 3. Prepare Subtitles
            subtitle_path = None
            if dialogue and voice_segments and len(dialogue) == len(voice_segments):
                subtitle_path = self.renderer.create_subtitle_file(
                    dialogue, voice_segments, output_dir
                )

            # 4. Get Music
            music_path = self.renderer.get_background_music()
            if music_path:
                print(f"🎵 Adding background music: {Path(music_path).name}")

            # 5. Render Final Video
            print("🎬 Rendering final video with FFmpeg...")
            self.renderer.render_video(
                output_path=output_path,
                image_paths=final_image_paths,
                audio_path=audio_path,
                audio_duration=audio_duration,
                durations=durations,
                subtitle_path=subtitle_path,
                music_path=music_path,
                video_background_path=video_background_path
            )

            # 6. Update DB (Success)
            file_size = os.path.getsize(output_path)
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

        except Exception as e:
            error_msg = str(e)
            if hasattr(e, "stderr"): # subprocess error
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
                error_message=f"Video generation failed: {error_msg}",
            )
            raise
