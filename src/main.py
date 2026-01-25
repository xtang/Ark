"""Main entry point for the AI Podcast Generator."""

import argparse
import sys
from pathlib import Path

from .config import load_config, get_topic_name
from .database import Database
from .generators import DialogueGenerator, AudioGenerator, ImageGenerator, VideoGenerator
from .tui import PodcastGeneratorApp


def run_cli(topic_key: str, config_path: str | None = None) -> None:
    """Run generation pipeline via CLI (non-interactive)."""
    config = load_config(config_path)
    db = Database(config["database"]["path"])
    output_dir = Path(config["output"]["directory"])

    try:
        topic_name = get_topic_name(config, topic_key)
        print(f"🚀 开始生成: {topic_name}")

        # Create generation record
        generation = db.create_generation(topic_key, topic_name)
        gen_output_dir = output_dir / f"gen_{generation.id}"

        # Step 1: Dialogue
        print("📝 Step 1/4: 生成对话内容...")
        dialogue_gen = DialogueGenerator(config, db)
        dialogue, references, summary = dialogue_gen.generate(
            generation.id, topic_key, topic_name, gen_output_dir
        )
        print(f"  ✓ 完成，共 {len(dialogue)} 句对话")

        # Step 2: Audio
        print("🔊 Step 2/4: 生成语音...")
        audio_gen = AudioGenerator(config, db)
        audio_path, duration, voice_segments = audio_gen.generate(
            generation.id, dialogue, gen_output_dir
        )
        print(f"  ✓ 完成，时长 {duration:.1f} 秒")

        # Step 3: Images
        print("🖼️ Step 3/4: 生成图片...")
        image_gen = ImageGenerator(config, db)
        image_paths = image_gen.generate(generation.id, dialogue, summary, gen_output_dir)
        print(f"  ✓ 完成，共 {len(image_paths)} 张图片")

        # Step 4: Video
        print("🎬 Step 4/4: 生成视频...")
        video_gen = VideoGenerator(config, db)
        video_path = video_gen.generate(
            generation.id, image_paths, audio_path, duration, voice_segments, gen_output_dir,
            dialogue=dialogue  # Pass dialogue for subtitles
        )
        print(f"  ✓ 完成!")

        print(f"\n✅ 视频已生成: {video_path}")
        print(f"📄 摘要: {summary}")
        print("📚 参考资料:")
        for ref in references[:5]:
            print(f"   • {ref}")

    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        sys.exit(1)
    finally:
        db.close()


def run_tui(config_path: str | None = None) -> None:
    """Run the interactive TUI."""
    app = PodcastGeneratorApp(config_path)
    app.run()


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="AI Podcast Generator - Generate short podcast videos using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive TUI
  uv run python -m src.main

  # Generate for a specific topic via CLI
  uv run python -m src.main --topic life_tips

Available topics:
  life_tips  - 生活常识 (Daily life knowledge)
  health     - 健康保养 (Health & wellness)
  history    - 历史野史 (Historical stories)
  curiosity  - 猎奇故事 (Curiosity & mysteries)
        """,
    )

    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        choices=["life_tips", "health", "history", "curiosity"],
        help="Topic to generate (runs in CLI mode)",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config file (default: config/default_config.yaml)",
    )

    args = parser.parse_args()

    if args.topic:
        # CLI mode
        run_cli(args.topic, args.config)
    else:
        # TUI mode
        run_tui(args.config)


if __name__ == "__main__":
    main()
