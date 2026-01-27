"""
Unified Podcast Generation Workflow.
Handles the orchestration of Dialogue, Audio, Image/Veo, and Video generation.
"""

from pathlib import Path
from typing import Any, Callable

from .database import Database
from .generators import DialogueGenerator, AudioGenerator, ImageGenerator, VideoGenerator


class PodcastWorkflow:
    """
    Orchestrator for the podcast generation pipeline.
    Used by both CLI and TUI to ensure consistent behavior.
    """

    def __init__(self, config: dict[str, Any], db: Database, logger: Callable[[str], None] = print):
        """
        Initialize the workflow.

        Args:
            config: Application configuration dictionary.
            db: Database instance.
            logger: Callback function for logging messages (default: print).
        """
        self.config = config
        self.db = db
        self.log = logger

    def run(
        self,
        generation_id: int,
        topic_key: str,
        topic_name: str,
        output_dir: Path,
        stock_code: str | None = None,
        language: str = "CN",
    ) -> str:
        """
        Run the full generation pipeline.

        Args:
            generation_id: DB ID for this generation.
            topic_key: Key of the topic (e.g., 'morning_news', 'custom').
            topic_name: Display name of the topic.
            output_dir: Root directory for outputs.
            stock_code: Optional stock code for stock_talk topic.
            language: Language code (CN, EN, JP).

        Returns:
            Path to the final generated video.
        """
        gen_output_dir = output_dir / f"gen_{generation_id}"
        
        # --- Step 1: Dialogue ---
        self.log(f"📝 Step 1/4: Dialogue Generation ({language})...")
        dialogue_gen = DialogueGenerator(self.config, self.db)
        dialogue, references, summary, title = dialogue_gen.generate(
            generation_id, topic_key, topic_name, gen_output_dir,
            stock_code=stock_code,
            language=language
        )
        self.log(f"  ✓ Dialogue complete. Title: {title}")

        # --- Step 2: Audio ---
        self.log("🔊 Step 2/4: Audio Generation...")
        audio_gen = AudioGenerator(self.config, self.db)
        audio_path, duration, voice_segments = audio_gen.generate(
            generation_id, dialogue, gen_output_dir
        )
        self.log(f"  ✓ Audio complete: {duration:.1f}s")

        # --- Step 3: Visuals (Static Images or Skip for Veo) ---
        video_mode = self.config.get("video", {}).get("mode", "static_images")
        image_paths = []
        cover_path = None

        if video_mode == "veo_loop":
            self.log("🖼️ Step 3/4: Visuals... (Skipping Image Generation for Veo Loop)")
        else:
            self.log("🖼️ Step 3/4: Image Generation...")
            image_gen = ImageGenerator(self.config, self.db)
            image_paths = image_gen.generate(generation_id, dialogue, summary, gen_output_dir, language=language)
            self.log(f"  ✓ Generated {len(image_paths)} images")

            # Generate dedicated cover
            self.log("🎨 Generating Cover Art...")
            cover_path = image_gen.generate_cover(generation_id, title, summary, gen_output_dir, language=language)
            if cover_path:
                self.log(f"  ✓ Cover art generated: {Path(cover_path).name}")

        # --- Step 4: Video ---
        self.log("🎬 Step 4/4: Video Generation...")
        video_gen = VideoGenerator(self.config, self.db)
        video_path = video_gen.generate(
            generation_id, image_paths, audio_path, duration, voice_segments, gen_output_dir,
            dialogue=dialogue,
            title=title,  # Pass title for cover generation or Veo prompt
            cover_image_path=cover_path
        )
        
        self.log(f"✅ Video Generated Successfully: {video_path}")
        return video_path
