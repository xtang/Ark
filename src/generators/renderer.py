"""Video rendering module using FFmpeg."""

import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..database import Database

class VideoRenderer:
    """Render podcast videos with subtitles and animations using FFmpeg."""

    # Font settings for Chinese subtitles
    FONT_FILE = "/System/Library/Fonts/PingFang.ttc"  # macOS Chinese font
    FALLBACK_FONT = "Arial"

    def __init__(self, config: dict[str, Any]):
        """Initialize the video renderer."""
        self.config = config
        
        self.resolution = config.get("output", {}).get("video_resolution", "1920x1080")
        self.subtitle_font_size = config.get("output", {}).get("subtitle_font_size", 24)
        
        # Animation settings
        self.fade_duration = 0.3
        self.transition_duration = 0.5
        self.subtitle_margin = 20
        self.enable_motion = config.get("video", {}).get("motion_effect", True)

    def create_cover_with_title(
        self,
        source_image: str,
        output_path: Path,
        title: str | None = None,
    ) -> None:
        """Create a cover image with title text overlay."""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
        except ImportError:
            # Fallback to simple copy if Pillow not available
            shutil.copy(source_image, output_path)
            return

        # Open source image
        img = Image.open(source_image).convert("RGBA")
        
        if not title:
            # No title, just save the image
            img.convert("RGB").save(output_path, "JPEG", quality=95)
            return

        width, height = img.size

        # Create gradient overlay for text readability (bottom to middle)
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw gradient
        gradient_height = height // 2
        for y in range(gradient_height):
            alpha = int(200 * (1 - y / gradient_height))
            draw.rectangle(
                [(0, height - gradient_height + y), (width, height - gradient_height + y + 1)],
                fill=(0, 0, 0, alpha)
            )

        # Composite the gradient overlay
        img = Image.alpha_composite(img, overlay)

        # Load font
        font_size = max(48, width // 15)
        font = None
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "C:\\Windows\\Fonts\\msyh.ttc",  # Windows
        ]
        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    continue
        
        if font is None:
            font = ImageFont.load_default()

        # Draw title text
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = height - height // 4 - text_height // 2

        # Draw text shadow
        shadow_offset = 3
        draw.text((x + shadow_offset, y + shadow_offset), title, font=font, fill=(0, 0, 0, 200))
        
        # Draw main text
        draw.text((x, y), title, font=font, fill=(255, 255, 255, 255))

        # Save as JPEG
        img.convert("RGB").save(output_path, "JPEG", quality=95)

    def get_background_music(self) -> str | None:
        """Get a random background music file from assets."""
        project_root = Path(__file__).parent.parent.parent
        music_dir = project_root / "assets" / "music"
        
        if not music_dir.exists():
            return None
            
        music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
        if not music_files:
            return None
            
        return str(random.choice(music_files))

    def calculate_image_durations(
        self,
        audio_duration: float,
        voice_segments: list[dict],
        num_images: int,
    ) -> list[float]:
        """Calculate how long each image should be displayed."""
        if num_images <= 0:
            return []

        if voice_segments and len(voice_segments) >= num_images:
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
                durations.append(max(0.5, duration))

            if durations:
                durations[-1] += 2.0  # Padding matching audio

            return durations

        base_duration = (audio_duration + 2.0) / num_images
        return [base_duration] * num_images

    def create_subtitle_file(
        self,
        dialogue: list[dict],
        voice_segments: list[dict],
        output_dir: Path,
    ) -> str:
        """Create SRT subtitle file."""
        srt_path = output_dir / "subtitles.srt"

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, (line, segment) in enumerate(zip(dialogue, voice_segments)):
                start_time = segment.get("start_time_seconds", i * 5)
                end_time = segment.get("end_time_seconds", start_time + 5)

                if i == 0:
                    start_time = max(start_time, self.fade_duration)

                text = line.get("text", "")
                import re
                text = re.sub(r'\[.*?\]', '', text).strip()

                start_str = self._format_srt_time(start_time)
                end_str = self._format_srt_time(end_time)

                f.write(f"{i + 1}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n\n")

        return str(srt_path)

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def build_ffmpeg_command(
        self,
        image_paths: list[str],
        durations: list[float],
        audio_path: str,
        output_path: str,
        subtitle_path: str | None = None,
        audio_duration: float = 0,
        music_path: str | None = None,
        video_background_path: str | None = None,
        video_intro_path: str | None = None,
        cover_path: str | None = None,
        cover_duration: float = 0,
        enable_transitions: bool = True,
    ) -> list[str]:
        """Build FFmpeg command."""
        width, height = self.resolution.split("x")
        
        inputs = []
        filter_parts = []
        concat_nodes = []
        
        # Helper to process an image input
        def add_image_input(path, duration, label):
            inputs.extend(["-loop", "1", "-t", str(duration), "-i", path])
            idx = len(inputs) // 2 - 1  # -loop 1 -t D -i P -> 2 args before -i? No, inputs list. 
            # inputs list grows: ["-loop", "1", "-t", "...", "-i", "..."] -> 6 items per image.
            # But we are extending list. Let's track index manually.
            return idx

        input_counter = 0
        def get_next_input_idx():
            nonlocal input_counter
            idx = input_counter
            input_counter += 1
            return idx

        # --- Visual Inputs ---
        
        # 1. Video Background (Loops, overrides everything)
        if video_background_path:
            inputs.extend(["-stream_loop", "-1", "-i", video_background_path])
            idx = get_next_input_idx()
            filter_parts.append(
                f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1[vconcat]"
            )
            
        else:
            # Sequence: [Cover] -> [Intro Video] -> [Slideshow Images]
            
            # A. Cover Image
            if cover_path and cover_duration > 0:
                inputs.extend(["-loop", "1", "-t", str(cover_duration), "-i", cover_path])
                idx = get_next_input_idx()
                
                # Simple scale (no motion) for cover usually
                filter_parts.append(
                    f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                    f"setsar=1[v_cover]"
                )
                concat_nodes.append("[v_cover]")

            # B. Intro Video
            if video_intro_path:
                inputs.extend(["-i", video_intro_path])
                idx = get_next_input_idx()
                filter_parts.append(
                    f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                    f"setsar=1[v_intro]"
                )
                concat_nodes.append("[v_intro]")

            # C. Slideshow Images
            slideshow_nodes = []
            for i, (img_path, duration) in enumerate(zip(image_paths, durations)):
                # If transitions enable, we need extra overlap time. If not, just duration.
                # However, for hard cuts, duration is exact.
                # For crossfades, input length = duration + transition.
                
                input_duration = duration
                if enable_transitions and i < len(image_paths) - 1:
                     input_duration += self.transition_duration

                inputs.extend(["-loop", "1", "-t", str(input_duration), "-i", img_path])
                idx = get_next_input_idx()

                node_name = f"v_img_{i}"
                if self.enable_motion and enable_transitions:
                    # Ken Burns only if transitions enabled/requested? Or always?
                    # User said "不需要渐入效果" (no fade in). Might imply static images or just hard cuts.
                    # Let's keep motion if configured, but do hard cuts.
                    filter_parts.append(
                        f"[{idx}:v]scale=1920:-2,zoompan=z='min(zoom+0.0005,1.15)':d=700:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={self.resolution}:fps=24,"
                        f"setsar=1[{node_name}]"
                    )
                else:
                    filter_parts.append(
                        f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                        f"setsar=1[{node_name}]"
                    )
                slideshow_nodes.append(f"[{node_name}]")

            # Combine Slideshow Images
            if slideshow_nodes:
                if enable_transitions and len(slideshow_nodes) > 1:
                    # XFADE Logic
                    current = slideshow_nodes[0]
                    offset = durations[0] - self.transition_duration
                    for i in range(1, len(slideshow_nodes)):
                        next_node = slideshow_nodes[i]
                        out_node = f"[v_slide_out_{i}]"
                        filter_parts.append(
                            f"{current}{next_node}xfade=transition=fade:duration={self.transition_duration}:offset={offset:.2f}{out_node}"
                        )
                        current = out_node
                        offset += durations[i] - self.transition_duration
                    concat_nodes.append(current)
                else:
                    # HARD CUTS (Concat) or Single Image
                    # If multiple images, we need to concat them first or just add to main concat list?
                    # We can add them all to main concat list directly.
                    concat_nodes.extend(slideshow_nodes)

            # Final Concat of All Parts
            if concat_nodes:
                if len(concat_nodes) > 1:
                    filter_parts.append(f"{''.join(concat_nodes)}concat=n={len(concat_nodes)}:v=1:a=0[vconcat]")
                else:
                    filter_parts.append(f"{concat_nodes[0]}copy[vconcat]")
            else:
                 raise RuntimeError("No visual inputs provided.")

        # Fade Out of Final Video (Visual)
        # Calculate Fade Start based on total expected duration or audio duration?
        # User wants valid fade out.
        fade_out_start = audio_duration + 2.0 - self.fade_duration
        filter_parts.append(
            f"[vconcat]fade=t=out:st={fade_out_start}:d={self.fade_duration}[vfaded]"
        )

        # Subtitles
        if subtitle_path and os.path.exists(subtitle_path):
            escaped_path = subtitle_path.replace(":", "\\:").replace("'", "\\'")
            filter_parts.append(
                f"[vfaded]subtitles='{escaped_path}':force_style='FontSize={self.subtitle_font_size},PrimaryColour=&HFFFFFF&,"
                f"OutlineColour=&H000000&,Outline=2,MarginV={self.subtitle_margin}'[outv]"
            )
        else:
            filter_parts.append("[vfaded]copy[outv]")

        # --- Audio Inputs ---
        audio_idx = input_counter
        inputs.extend(["-i", audio_path])
        input_counter += 1
        
        # Audio Fades: "声音开头和结尾需要有渐入和渐出"
        voice_filter = (
            f"[{audio_idx}:a]apad=pad_dur=2,"
            f"afade=t=in:st=0:d={self.fade_duration}," 
            f"afade=t=out:st={audio_duration + 2.0 - 2.0}:d=2.0[voice_a]"
        )
        filter_parts.append(voice_filter)
        
        # Music
        music_filter = None
        if music_path:
            music_idx = input_counter
            inputs.extend(["-i", music_path])
            input_counter += 1
            music_filter = (
                f"[{music_idx}:a]aloop=loop=-1:size=2e+09,"
                f"volume=0.1,"
                f"afade=t=in:st=0:d={self.fade_duration}," # Also fade in music
                f"afade=t=out:st={audio_duration + 2.0 - 2.0}:d=2.0[music_a]"
            )
            filter_parts.append(music_filter)
            filter_parts.append(f"[voice_a][music_a]amix=inputs=2:duration=first:dropout_transition=2[outa]")
        else:
             filter_parts.append(f"[voice_a]acopy[outa]")

        filter_complex = ";".join(filter_parts)

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
            "-movflags", "+faststart",
            output_path,
        ]

        return cmd

    def render_video(
        self,
        output_path: Path,
        image_paths: list[str],
        audio_path: str,
        audio_duration: float,
        durations: list[float],
        subtitle_path: str | None = None,
        music_path: str | None = None,
        video_background_path: str | None = None,
        video_intro_path: str | None = None,
        cover_path: str | None = None,
        cover_duration: float = 0,
        enable_transitions: bool = True,
    ) -> None:
        """Execute FFmpeg command to render video."""
        cmd = self.build_ffmpeg_command(
            image_paths,
            durations,
            audio_path,
            str(output_path),
            subtitle_path,
            audio_duration,
            music_path,
            video_background_path,
            video_intro_path,
            cover_path,
            cover_duration,
            enable_transitions,
        )

        # Save command for debugging
        cmd_file = output_path.parent / "ffmpeg_cmd.txt"
        with open(cmd_file, "w") as f:
            f.write(" ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
