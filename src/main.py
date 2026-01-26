"""Main entry point for the AI Podcast Generator."""

import argparse
import json
import sys
import traceback
from pathlib import Path

from .config import load_config, get_topic_name
from .database import Database
from .generators import DialogueGenerator, AudioGenerator, ImageGenerator, VideoGenerator
from .tui import PodcastGeneratorApp


def run_cli(topic_key: str, config_path: str | None = None, stock_code: str | None = None) -> None:
    """Run generation pipeline via CLI (non-interactive)."""
    config = load_config(config_path)
    db = Database(config["database"]["path"])
    output_dir = Path(config["output"]["directory"])

    try:
        topic_name = get_topic_name(config, topic_key)
        # For stock_talk, append stock code to display name
        if topic_key == "stock_talk" and stock_code:
            topic_name = f"{topic_name} - {stock_code}"
        print(f"🚀 开始生成: {topic_name}")

        # Create generation record
        generation = db.create_generation(topic_key, topic_name)
        gen_output_dir = output_dir / f"gen_{generation.id}"

        # Step 1: Dialogue
        print("📝 Step 1/4: 生成对话内容...")
        dialogue_gen = DialogueGenerator(config, db)
        dialogue, references, summary, title = dialogue_gen.generate(
            generation.id, topic_key, topic_name, gen_output_dir,
            stock_code=stock_code
        )
        print(f"  ✓ 完成，共 {len(dialogue)} 句对话")
        print(f"  📌 标题: {title}")

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
            dialogue=dialogue,
            title=title  # Pass title for cover generation
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


def resume_cli(gen_id: int, config_path: str | None = None) -> None:
    """Resume a failed generation from the last successful stage."""
    config = load_config(config_path)
    db = Database(config["database"]["path"])
    output_dir = Path(config["output"]["directory"])

    try:
        gen = db.get_generation(gen_id)
        if not gen:
            print(f"❌ 找不到 ID 为 {gen_id} 的生成记录")
            sys.exit(1)
            
        print(f"🔄 恢复生成: #{gen.id} {gen.topic_name} (状态: {gen.status})")
        gen_output_dir = output_dir / f"gen_{gen.id}"
        gen_output_dir.mkdir(parents=True, exist_ok=True)

        # Reconstruct generation state
        dialogue = []
        references = []
        summary = ""
        audio_path = ""
        duration = 0.0
        voice_segments = []
        image_paths = []

        # --- Check Step 1: Dialogue ---
        # We need to know if dialogue was completed.
        # Check explicit status or query dialogue_request
        dialogue_req = db.get_dialogue_request(gen.id)
        
        if dialogue_req and dialogue_req.success and gen.dialogue_json_path:
            print(f"📝 Step 1/4: 对话内容已生成 (跳过)")
            dialogue = dialogue_req.get_dialogue()
            references = dialogue_req.get_references()
            summary = dialogue_req.summary
            # Try to get title from saved JSON
            title = ""
            if gen.dialogue_json_path and Path(gen.dialogue_json_path).exists():
                with open(gen.dialogue_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", summary[:12] if summary else "")
        else:
            print("📝 Step 1/4: 重新生成对话内容...")
            dialogue_gen = DialogueGenerator(config, db)
            dialogue, references, summary, title = dialogue_gen.generate(
                gen.id, gen.topic_key, gen.topic_name, gen_output_dir
            )
            print(f"  ✓ 完成，共 {len(dialogue)} 句对话")
        
        # --- Check Step 2: Audio ---
        audio_req = db.get_audio_request(gen.id)
        # Check if audio file exists
        audio_exists = audio_req and audio_req.audio_path and Path(audio_req.audio_path).exists()
        
        if audio_req and audio_req.success and audio_exists:
            print(f"🔊 Step 2/4: 语音已生成 (跳过)")
            audio_path = audio_req.audio_path
            duration = audio_req.duration_seconds
            voice_segments = audio_req.get_voice_segments()
        else:
            print("🔊 Step 2/4: 重新生成语音...")
            audio_gen = AudioGenerator(config, db)
            audio_path, duration, voice_segments = audio_gen.generate(
                gen.id, dialogue, gen_output_dir
            )
            print(f"  ✓ 完成，时长 {duration:.1f} 秒")

        # --- Check Step 3: Images ---
        # Check generation status for 'images_complete' or 'audio_complete' vs others
        # Ideally check status flag. 'images_complete' means all images done.
        # But if it failed midway, we re-run all images for simplicity (idempotency depends on prompt logic but safe to overwrite)
        
        # Check DB images
        image_reqs = db.get_image_requests(gen.id)
        # Use any successful images that exist on disk
        successful_images = [img for img in image_reqs if img.success and Path(img.image_path).exists()]
        
        # If we have any successful images, use them (don't regenerate due to rate limits)
        if successful_images:
            print(f"🖼️ Step 3/4: 使用已有图片 (共 {len(successful_images)} 张)")
            image_paths = [img.image_path for img in successful_images]
        else:
            print("🖼️ Step 3/4: 生成图片...")
            image_gen = ImageGenerator(config, db)
            image_paths = image_gen.generate(gen.id, dialogue, summary, gen_output_dir)
            print(f"  ✓ 完成，共 {len(image_paths)} 张图片")

        # --- Step 4: Video ---
        video_out = db.get_video_output(gen.id)
        video_exists = video_out and video_out.video_path and Path(video_out.video_path).exists()
        
        if video_exists and video_out.success:
             print(f"🎬 Step 4/4: 视频已生成 (跳过)")
             video_path = video_out.video_path
        else:
            print("🎬 Step 4/4: 生成视频...")
            video_gen = VideoGenerator(config, db)
            video_path = video_gen.generate(
                gen.id, image_paths, audio_path, duration, voice_segments, gen_output_dir,
                dialogue=dialogue,
                title=title
            )
            print(f"  ✓ 完成!")

        print(f"\n✅ 视频已恢复/生成: {video_path}")
        
    except Exception as e:
        import traceback
        print(f"\n❌ 恢复生成失败: {e}")
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
        print(f"{'ID':<5} {'状态':<15} {'主题':<12} {'视频路径'}")
        print("-" * 80)

        for gen in generations:
            status_icon = "✅" if gen.status == "completed" else ("❌" if gen.status == "failed" else "⏳")
            video_path = gen.video_path or "-"
            if len(video_path) > 40:
                video_path = "..." + video_path[-37:]
            print(f"{gen.id:<5} {status_icon} {gen.status:<14} {gen.topic_name:<12} {video_path}")

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

  # Resume a failed generation
  uv run python -m src.main --resume 14

  # Show details for a specific session
  uv run python -m src.main --show 5

Available topics:
  life_tips   - 生活常识 (Daily life knowledge)
  health      - 健康保养 (Health & wellness)
  history     - 历史野史 (Historical stories)
  curiosity   - 猎奇故事 (Curiosity & mysteries)
  stock_talk  - 股票公司杂谈 (Stock company talk, requires --stock)
        """,
    )

    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        choices=["life_tips", "health", "history", "curiosity", "stock_talk"],
        help="Topic to generate (runs in CLI mode)",
    )

    parser.add_argument(
        "--stock",
        "-S",
        type=str,
        metavar="CODE",
        help="Stock code for stock_talk topic (e.g., AAPL, 600519, 00700.HK)",
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
        "--resume",
        "-r",
        type=int,
        metavar="ID",
        help="Resume a failed generation from last successful stage",
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

    # Validate stock_talk requires --stock
    if args.topic == "stock_talk" and not args.stock:
        parser.error("--stock CODE is required when using --topic stock_talk")

    if args.history:
        show_history(args.config, args.limit)
    elif args.resume:
        resume_cli(args.resume, args.config)
    elif args.show:
        show_session(args.show, args.config)
    elif args.topic:
        run_cli(args.topic, args.config, stock_code=args.stock)
    else:
        run_tui(args.config)


if __name__ == "__main__":
    main()
