"""Terminal User Interface for the Podcast Generator."""

from pathlib import Path
from typing import Any
import subprocess
import platform

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Button,
    Static,
    Label,
    Log,
    ListView,
    ListItem,
    ContentSwitcher,
    RadioSet,
    RadioButton,
    Input,
)
from textual.screen import ModalScreen
from textual.binding import Binding
from textual import work
from textual.reactive import reactive

from ..config import load_config, get_topic_name
from ..database import Database
from ..generators import DialogueGenerator, AudioGenerator, ImageGenerator, VideoGenerator


class NewGenerationModal(ModalScreen[dict]):
    """Modal for selecting a topic for new generation."""

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Cancel"),
        Binding("up", "focus_previous", "Up"),
        Binding("down", "focus_next", "Down"),
    ]

    def __init__(self, topics: dict[str, str]):
        super().__init__()
        self.topics = topics
        self.selected_language = "CN"

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Container(id="modal-dialog"):
            yield Label("Select Language / 选择语言", classes="modal-section-title")
            with RadioSet(id="language-radio"):
                yield RadioButton("中文 (Chinese)", value=True, id="lang-CN")
                yield RadioButton("English", id="lang-EN")
                yield RadioButton("日本語 (Japanese)", id="lang-JP")
            
            yield Label("Custom Topic / 自定义主题", classes="modal-section-title")
            yield Input(placeholder="Enter custom topic... / 输入自定义主题", id="custom-topic-input")
            yield Button("Start Custom / 开始自定义", id="btn-start-custom", variant="success", disabled=True)
            
            yield Label("Preset Topics / 预设主题", classes="modal-section-title")
            with Vertical():
                for key, name in self.topics.items():
                    yield Button(f"{name}", id=f"topic-{key}", classes="topic-button", variant="primary")
            yield Button("Cancel / 取消", id="cancel", classes="topic-button", variant="error")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed.id == "lang-EN":
            self.selected_language = "EN"
        elif event.pressed.id == "lang-JP":
            self.selected_language = "JP"
        else:
            self.selected_language = "CN"

    def on_input_changed(self, event: Input.Changed) -> None:
        # Enable start button only if input is not empty
        self.query_one("#btn-start-custom", Button).disabled = not event.value.strip()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.dismiss({
                "topic": "custom",
                "custom_topic_name": event.value.strip(),
                "language": self.selected_language
            })

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "btn-start-custom":
            custom_topic = self.query_one("#custom-topic-input", Input).value.strip()
            if custom_topic:
                 self.dismiss({
                    "topic": "custom",
                    "custom_topic_name": custom_topic,
                    "language": self.selected_language
                })
        elif event.button.id and event.button.id.startswith("topic-"):
            topic_key = event.button.id.replace("topic-", "")
            self.dismiss({"topic": topic_key, "language": self.selected_language})


class SessionListItem(ListItem):
    """List item for a generation session."""

    def __init__(self, generation: Any):
        super().__init__()
        self.generation = generation
        self.gen_id = generation.id

    def compose(self) -> ComposeResult:
        icon = "✅" if self.generation.status == "completed" else "❌" if self.generation.status == "failed" else "⏳"
        label = f"#{self.generation.id} {icon} {self.generation.topic_name}"
        yield Label(label)


class Dashboard(Container):
    """Main dashboard view (default when no session active)."""

    def compose(self) -> ComposeResult:
        yield Label("Podcast Generator Dashboard", classes="title")
        
        with Container(classes="stats-container"):
            with Container(classes="stat-card"):
                yield Label("Total Gen", classes="stat-label")
                yield Label("0", id="stat-total", classes="stat-value")
            
            with Container(classes="stat-card"):
                yield Label("Success Rate", classes="stat-label")
                yield Label("0%", id="stat-success", classes="stat-value")

        yield Button("Start New Generation", id="btn-new-gen", variant="success")

    def update_stats(self, total: int, success_rate: float) -> None:
        self.query_one("#stat-total", Label).update(str(total))
        self.query_one("#stat-success", Label).update(f"{success_rate:.1f}%")


class SessionView(Container):
    """Detailed view for a specific session."""

    current_gen_id: int | None = None

    def compose(self) -> ComposeResult:
        with Container(id="session-header"):
            yield Label("Select a session...", id="session-title")
            yield Label("Idle", id="session-status")

        yield Log(id="session-log", highlight=True)

        with Container(id="session-actions"):
            yield Button("Retry Generation", id="btn-retry", disabled=True)
            yield Button("Open Folder", id="btn-open-folder", disabled=True)

    def set_session(self, gen: Any) -> None:
        self.current_gen_id = gen.id
        self.query_one("#session-title", Label).update(f"#{gen.id} - {gen.topic_name}")
        self.query_one("#session-status", Label).update(gen.status)
        
        self.query_one("#session-log", Log).clear()
        self.query_one("#session-log", Log).write_line(f"Topic: {gen.topic_name}")
        self.query_one("#session-log", Log).write_line(f"Status: {gen.status}")
        if gen.video_path:
             self.query_one("#session-log", Log).write_line(f"Video: {gen.video_path}")
        
        # Load logs/details from DB if possible using show_session logic re-implementation
        # For now, just show basic info + allow retry if failed/incomplete
        # self.query_one("#btn-retry").disabled = (gen.status == "completed")
        self.query_one("#btn-retry").disabled = False # Always allow retry for now
        self.query_one("#btn-open-folder").disabled = False


    def log(self, message: str) -> None:
        self.query_one("#session-log", Log).write_line(message)


class PodcastGeneratorApp(App):
    """Main TUI application."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("n", "new_generation", "New Gen"),
    ]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.config = load_config(config_path)
        self.db = Database(self.config["database"]["path"])
        self.is_generating = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Horizontal():
            with Container(id="sidebar"):
                yield Label("History / 历史", id="sidebar-header")
                yield ListView(id="session-list")
            
            with ContentSwitcher(id="main-content", initial="dashboard"):
                yield Dashboard(id="dashboard")
                yield SessionView(id="session-view")
        
        yield Footer()

    def on_mount(self) -> None:
        self.title = "🎙️ AI Podcast Generator"
        self.sub_title = "IRC Style Interface"
        self.refresh_history()

    def refresh_history(self) -> None:
        """Reload history list and update stats."""
        generations = self.db.get_recent_generations(limit=50)
        
        # Update List
        list_view = self.query_one("#session-list", ListView)
        list_view.clear()
        for gen in generations:
            list_view.append(SessionListItem(gen))
            
        # Update Stats
        total = len(generations)
        success = sum(1 for g in generations if g.status == "completed")
        rate = (success / total * 100) if total > 0 else 0
        
        self.query_one(Dashboard).update_stats(total, rate)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, SessionListItem):
            # Switch to session view
            self.query_one("#main-content", ContentSwitcher).current = "session-view"
            self.query_one(SessionView).set_session(event.item.generation)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-new-gen":
            self.action_new_generation()
        elif event.button.id == "btn-retry":
            session_view = self.query_one(SessionView)
            if session_view.current_gen_id:
                # Logic to retry... logic similar to resume_cli
                # For now just start generation again with same topic?
                # Ideally we should resume. But for simplicity let's treat as "New Generation"
                # Or implement full resume logic. 
                # Let's just log for now:
                session_view.log("Resuming not fully implemented in TUI yet. Please use CLI --resume ID")
        elif event.button.id == "btn-open-folder":
            session_view = self.query_one(SessionView)
            if session_view.current_gen_id:
                gen_dir = Path(self.config["output"]["directory"]) / f"gen_{session_view.current_gen_id}"
                if gen_dir.exists():
                     try:
                        if platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", str(gen_dir)])
                        elif platform.system() == "Windows":
                            subprocess.run(["explorer", str(gen_dir)])
                        else:  # Linux
                            subprocess.run(["xdg-open", str(gen_dir)])
                        session_view.log(f"📂 Opened folder: {gen_dir}")
                     except Exception as e:
                        session_view.log(f"❌ Failed to open folder: {e}")
                else:
                    session_view.log(f"❌ Folder not found: {gen_dir}")

    def action_new_generation(self) -> None:
        if self.is_generating:
            self.notify("⚠️ Generation in progress...", severity="warning")
            return
            
        def handle_topic(result: dict | None) -> None:
            if result and result.get("topic"):
                self._start_generation(
                    result["topic"], 
                    result.get("language", "CN"),
                    custom_topic_name=result.get("custom_topic_name")
                )

        self.push_screen(NewGenerationModal(self.config.get("topics", {})), handle_topic)

    @work(thread=True)
    def _start_generation(self, topic_key: str, language: str = "CN", custom_topic_name: str | None = None) -> None:
        self.is_generating = True
        
        if topic_key == "custom" and custom_topic_name:
            topic_name = custom_topic_name
        else:
            topic_name = get_topic_name(self.config, topic_key)
            
        output_dir = Path(self.config["output"]["directory"])
        
        # Initialize thread-local DB connection
        db = Database(self.config["database"]["path"])
        
        try:
            # Create new record
            generation = db.create_generation(topic_key, topic_name)
            gen_output_dir = output_dir / f"gen_{generation.id}"
            
            # Switch to view and update
            def init_view():
                self.refresh_history()
                # Find the new item (crudely by reloading) or just force set
                # We need to switch view manually
                self.query_one("#main-content", ContentSwitcher).current = "session-view"
                self.query_one(SessionView).set_session(generation)
                
            self.call_from_thread(init_view)
            
            session_view = self.query_one(SessionView)
            
            def log(msg):
                self.call_from_thread(session_view.log, msg)

            log(f"🚀 Starting generation: {topic_name} [{language}]")
            
            # Step 1
            log("📝 Step 1/4: Dialogue Generation...")
            dialogue_gen = DialogueGenerator(self.config, db)
            
            # Use stock code if applicable (not handled in TUI simple flow yet, assume None) 
            stock_code = None 
            if topic_key == "stock_talk":
                # Hack: hardcode or prompt? For now let's hardcode 'AAPL' for demo or skip
                # Real implementation needs a Input Modal
                stock_code = "AAPL" 
                log(f"ℹ️ Auto-selected stock: {stock_code}")

            dialogue, references, summary, title = dialogue_gen.generate(
                generation.id, topic_key, topic_name, gen_output_dir, stock_code=stock_code, language=language
            )
            log(f"✓ Dialogue complete. Title: {title}")

            # Step 2
            log("🔊 Step 2/4: Audio Generation...")
            audio_gen = AudioGenerator(self.config, db)
            audio_path, duration, voice_segments = audio_gen.generate(
                generation.id, dialogue, gen_output_dir
            )
            log(f"✓ Audio complete: {duration:.1f}s")
            
            # Step 3
            log("🖼️ Step 3/4: Image Generation...")
            image_gen = ImageGenerator(self.config, db)
            image_paths = image_gen.generate(generation.id, dialogue, summary, gen_output_dir, language=language)
            
            # Generate Cover
            log("🎨 Generating Cover Art...")
            cover_path = image_gen.generate_cover(generation.id, title, summary, gen_output_dir, language=language)
            if cover_path:
                log(f"✓ Cover art generated")  
            
            log(f"✓ Images complete: {len(image_paths)}")

            # Step 4
            log("🎬 Step 4/4: Video Generation...")
            video_gen = VideoGenerator(self.config, db)
            video_path = video_gen.generate(
                generation.id, image_paths, audio_path, duration, voice_segments, gen_output_dir,
                dialogue=dialogue, title=title, cover_image_path=cover_path
            )
            
            log(f"✅ Video Generated: {video_path}")
            
        except Exception as e:
            # We can't log to session view easily if session view isn't set up yet, 
            # but usually it dies after init_view.
            # Try logging if session_view exists
            try:
                self.call_from_thread(session_view.log, f"❌ Error: {e}")
            except:
                pass
            print(f"Error in generation thread: {e}") # Fallback
            
        finally:
            self.is_generating = False
            if db:
                db.close()
            self.call_from_thread(self.refresh_history)

    def action_refresh(self) -> None:
        self.refresh_history()

    def action_quit(self) -> None:
        self.db.close()
        self.exit()
