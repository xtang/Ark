"""Terminal User Interface for the Podcast Generator."""

from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Button,
    Static,
    Label,
    Log,
    LoadingIndicator,
    ListView,
    ListItem,
)
from textual.binding import Binding
from textual import work

from ..config import load_config, get_topic_name
from ..database import Database
from ..generators import DialogueGenerator, AudioGenerator, ImageGenerator, VideoGenerator


class TopicSelector(Static):
    """Widget for selecting a topic."""

    def __init__(self, topics: dict[str, str]):
        super().__init__()
        self.topics = topics

    def compose(self) -> ComposeResult:
        yield Label("📖 选择主题 / Select Topic:", id="topic-label")
        with Vertical(id="topic-buttons"):
            for key, name in self.topics.items():
                yield Button(f"{name}", id=f"topic-{key}", variant="primary")


class ProgressPanel(Static):
    """Widget showing generation progress."""

    def compose(self) -> ComposeResult:
        yield Label("⏳ 生成进度 / Progress:", id="progress-label")
        yield Static("等待开始...", id="progress-status")
        yield Log(id="progress-log", highlight=True, max_lines=100)


class ResultPanel(Static):
    """Widget showing generation results."""

    def compose(self) -> ComposeResult:
        yield Label("✅ 生成结果 / Results:", id="result-label")
        yield Static("暂无结果", id="result-content")


class PodcastGeneratorApp(App):
    """Main TUI application for the Podcast Generator."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
    }

    #left-panel {
        height: 100%;
        border: solid green;
        padding: 1;
    }

    #right-panel {
        height: 100%;
        border: solid blue;
        padding: 1;
    }

    TopicSelector {
        height: auto;
        margin-bottom: 1;
    }

    #topic-buttons {
        height: auto;
    }

    #topic-buttons Button {
        width: 100%;
        margin-bottom: 1;
    }

    ProgressPanel {
        height: 1fr;
    }

    #progress-log {
        height: 1fr;
        border: solid gray;
        margin-top: 1;
    }

    ResultPanel {
        height: auto;
        min-height: 10;
    }

    #result-content {
        height: auto;
        padding: 1;
        border: solid gray;
    }

    #history-section {
        height: auto;
        margin-top: 1;
    }

    .generating {
        background: $primary-darken-2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.config = load_config(config_path)
        self.db = Database(self.config["database"]["path"])
        self.current_generation_id: int | None = None
        self.is_generating = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="left-panel"):
            yield TopicSelector(self.config.get("topics", {}))
            yield Static("", id="history-section")

        with Container(id="right-panel"):
            yield ProgressPanel()
            yield ResultPanel()

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "🎙️ AI Podcast Generator"
        self.sub_title = "Powered by Gemini + ElevenLabs"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id and button_id.startswith("topic-"):
            if self.is_generating:
                self._log("⚠️ 正在生成中，请等待...")
                return

            topic_key = button_id.replace("topic-", "")
            self._start_generation(topic_key)

    def _log(self, message: str) -> None:
        """Add message to progress log."""
        log = self.query_one("#progress-log", Log)
        log.write_line(message)

    def _update_status(self, status: str) -> None:
        """Update progress status."""
        status_widget = self.query_one("#progress-status", Static)
        status_widget.update(status)

    def _update_result(self, content: str) -> None:
        """Update result panel."""
        result_widget = self.query_one("#result-content", Static)
        result_widget.update(content)

    @work(thread=True)
    def _start_generation(self, topic_key: str) -> None:
        """Start the generation pipeline in a background thread."""
        self.is_generating = True
        topic_name = get_topic_name(self.config, topic_key)
        output_dir = Path(self.config["output"]["directory"])

        self.call_from_thread(self._update_status, f"🚀 开始生成: {topic_name}")
        self.call_from_thread(self._log, f"━━━ 新任务: {topic_name} ━━━")

        try:
            # Create generation record
            generation = self.db.create_generation(topic_key, topic_name)
            self.current_generation_id = generation.id
            gen_output_dir = output_dir / f"gen_{generation.id}"

            # Step 1: Generate dialogue
            self.call_from_thread(self._update_status, "📝 Step 1/4: 生成对话内容...")
            self.call_from_thread(self._log, "📝 正在调用 Gemini AI 生成对话...")

            dialogue_gen = DialogueGenerator(self.config, self.db)
            dialogue, references, summary, title = dialogue_gen.generate(
                generation.id, topic_key, topic_name, gen_output_dir
            )

            self.call_from_thread(self._log, f"✓ 对话生成完成，共 {len(dialogue)} 句")
            self.call_from_thread(self._log, f"  主题: {summary}")
            self.call_from_thread(self._log, f"  标题: {title}")

            # Step 2: Generate audio
            self.call_from_thread(self._update_status, "🔊 Step 2/4: 生成语音...")
            self.call_from_thread(self._log, "🔊 正在调用 ElevenLabs 生成语音...")

            audio_gen = AudioGenerator(self.config, self.db)
            audio_path, duration, voice_segments = audio_gen.generate(
                generation.id, dialogue, gen_output_dir
            )

            self.call_from_thread(self._log, f"✓ 语音生成完成，时长: {duration:.1f}s")

            # Step 3: Generate images
            self.call_from_thread(self._update_status, "🖼️ Step 3/4: 生成图片...")
            self.call_from_thread(self._log, "🖼️ 正在调用 Gemini 生成配图...")

            image_gen = ImageGenerator(self.config, self.db)
            image_paths = image_gen.generate(
                generation.id, dialogue, summary, gen_output_dir
            )

            self.call_from_thread(self._log, f"✓ 图片生成完成，共 {len(image_paths)} 张")

            # Step 4: Generate video
            self.call_from_thread(self._update_status, "🎬 Step 4/4: 生成视频...")
            self.call_from_thread(self._log, "🎬 正在使用 FFmpeg 合成视频...")

            video_gen = VideoGenerator(self.config, self.db)
            video_path = video_gen.generate(
                generation.id,
                image_paths,
                audio_path,
                duration,
                voice_segments,
                gen_output_dir,
                dialogue=dialogue,
                title=title,  # Pass title for cover generation
            )

            # Success!
            self.call_from_thread(self._update_status, "✅ 生成完成!")
            self.call_from_thread(self._log, f"✅ 视频已保存: {video_path}")

            result_text = f"""🎉 生成成功!

📄 主题: {topic_name}
📝 摘要: {summary}
⏱️ 时长: {duration:.1f} 秒
🎬 视频: {video_path}

📚 参考资料:
{chr(10).join(f"  • {ref}" for ref in references[:3])}
"""
            self.call_from_thread(self._update_result, result_text)

        except Exception as e:
            self.call_from_thread(self._update_status, f"❌ 生成失败")
            self.call_from_thread(self._log, f"❌ 错误: {e}")
            self.call_from_thread(self._update_result, f"生成失败: {e}")

        finally:
            self.is_generating = False

    def action_refresh(self) -> None:
        """Refresh the app state."""
        self._log("🔄 刷新中...")

    def action_quit(self) -> None:
        """Quit the application."""
        self.db.close()
        self.exit()
