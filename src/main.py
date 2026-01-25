"""Main entry point for the AI Podcast Generator."""

import argparse
import json
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
        import traceback
        print(f"\n❌ 生成失败: {e}")
        print("🔍 错误详情:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


def show_history(config_path: str | None = None, limit: int = 10) -> None:
    """Show recent generation history."""
    config = load_config(config_path)
    db = Database(config["database"]["path"])

    try:
        generations = db.get_recent_generations(limit)

        if not generations:
            print("📭 暂无生成记录")
            return

        print(f"\n📋 最近 {len(generations)} 条生成记录:\n")
        print(f"{'ID':<5} {'状态':<12} {'主题':<12} {'视频路径'}")
        print("-" * 80)

        for gen in generations:
            status_icon = "✅" if gen.status == "completed" else ("❌" if gen.status == "failed" else "⏳")
            video_path = gen.video_path or "-"
            if len(video_path) > 40:
                video_path = "..." + video_path[-37:]
            print(f"{gen.id:<5} {status_icon} {gen.status:<10} {gen.topic_name:<12} {video_path}")

    finally:
        db.close()


def show_session(session_id: int, config_path: str | None = None) -> None:
    """Show detailed info for a specific generation session."""
    config = load_config(config_path)
    db = Database(config["database"]["path"])

    try:
        gen = db.get_generation(session_id)
        if not gen:
            print(f"❌ 找不到 ID 为 {session_id} 的生成记录")
            return

        print(f"\n{'='*60}")
        print(f"📋 Generation #{gen.id} - {gen.topic_name}")
        print(f"{'='*60}")
        print(f"状态: {gen.status}")
        print(f"主题: {gen.topic_key} ({gen.topic_name})")
        if gen.error_message:
            print(f"错误: {gen.error_message}")

        # Dialogue Request
        print(f"\n{'─'*60}")
        print("📝 [Stage 1] Dialogue Generation (Gemini)")
        print(f"{'─'*60}")
        dialogue_req = db.get_dialogue_request(session_id)
        if dialogue_req:
            print(f"Word Count: {dialogue_req.word_count}")
            print(f"Summary: {dialogue_req.summary}")
            print(f"Success: {'✅' if dialogue_req.success else '❌'}")
            print(f"\n[Prompt Preview]:")
            print(dialogue_req.prompt[:500] + "..." if len(dialogue_req.prompt) > 500 else dialogue_req.prompt)
            if dialogue_req.dialogue_json:
                dialogue = json.loads(dialogue_req.dialogue_json)
                print(f"\n[Dialogue] ({len(dialogue)} lines):")
                for i, line in enumerate(dialogue[:3]):
                    print(f"  {line.get('speaker', '?')}: {line.get('text', '')[:50]}...")
                if len(dialogue) > 3:
                    print(f"  ... 还有 {len(dialogue) - 3} 行")
        else:
            print("  (无记录)")

        # Audio Request
        print(f"\n{'─'*60}")
        print("🔊 [Stage 2] Audio Generation (ElevenLabs)")
        print(f"{'─'*60}")
        audio_req = db.get_audio_request(session_id)
        if audio_req:
            print(f"Dialogue Count: {audio_req.dialogue_count}")
            print(f"Duration: {audio_req.duration_seconds:.1f}s")
            print(f"Audio Path: {audio_req.audio_path}")
            print(f"Success: {'✅' if audio_req.success else '❌'}")
            if audio_req.voice_segments_json:
                segments = json.loads(audio_req.voice_segments_json)
                print(f"Voice Segments: {len(segments)}")
        else:
            print("  (无记录)")

        # Image Requests
        print(f"\n{'─'*60}")
        print("🖼️ [Stage 3] Image Generation (Gemini)")
        print(f"{'─'*60}")
        image_reqs = db.get_image_requests(session_id)
        if image_reqs:
            print(f"Images Generated: {len(image_reqs)}")
            for img in image_reqs:
                status = "✅" if img.success else "❌"
                print(f"  [{img.image_index}] {status} {img.image_path or '(failed)'}")
                print(f"      Prompt: {img.prompt[:80]}...")
        else:
            print("  (无记录)")

        # Video Output
        print(f"\n{'─'*60}")
        print("🎬 [Stage 4] Video Output (FFmpeg)")
        print(f"{'─'*60}")
        video_out = db.get_video_output(session_id)
        if video_out:
            print(f"Video Path: {video_out.video_path}")
            print(f"Duration: {video_out.duration_seconds:.1f}s")
            print(f"Resolution: {video_out.resolution}")
            print(f"File Size: {video_out.file_size_bytes / 1024 / 1024:.2f} MB")
            print(f"Success: {'✅' if video_out.success else '❌'}")
        else:
            print("  (无记录)")

        print(f"\n{'='*60}\n")

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

  # Show generation history
  uv run python -m src.main --history

  # Show details for a specific session
  uv run python -m src.main --show 5

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

    parser.add_argument(
        "--history",
        "-H",
        action="store_true",
        help="Show recent generation history",
    )

    parser.add_argument(
        "--show",
        "-s",
        type=int,
        metavar="ID",
        help="Show detailed info for a specific generation session",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Number of history records to show (default: 10)",
    )

    args = parser.parse_args()

    if args.history:
        show_history(args.config, args.limit)
    elif args.show:
        show_session(args.show, args.config)
    elif args.topic:
        run_cli(args.topic, args.config)
    else:
        run_tui(args.config)


if __name__ == "__main__":
    main()
