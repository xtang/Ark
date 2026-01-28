"""Dialogue generator using Gemini AI."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from ..database import Database
from ..config import load_prompts, get_topic_config


class DialogueGenerator:
    """Generate podcast dialogue content using Gemini AI."""

    def __init__(self, config: dict[str, Any], db: Database):
        """
        Initialize the dialogue generator.

        Args:
            config: Application configuration.
            db: Database instance.
        """
        self.config = config
        self.db = db

        # Load prompt templates from config
        self.prompts = load_prompts()

        load_dotenv()
        api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY not found in environment")

        self.client = genai.Client(
            vertexai=True,
            api_key=api_key,
        )
        self.model = "gemini-3-flash-preview"

        # Initialize Grounding Tool
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

    def _build_prompt(
        self,
        topic_name: str,
        history: list[str],
        topic_key: str | None = None,
        stock_code: str | None = None,
        language: str = "CN",
    ) -> str:
        """Build the prompt for dialogue generation."""
        # Get topic specific config
        topic_conf = get_topic_config(self.config, topic_key) if topic_key else {}

        # 1. Resolve Speakers
        # Priority: Topic Config -> Global Config
        speakers_source = topic_conf.get("speakers")
        if not speakers_source:
            speakers_source = self.config.get("dialogue", {}).get("speakers", [])
        
        # Determine speakers based on language
        if isinstance(speakers_source, dict):
            # Pick by language, fallback to CN, then fallback to first available
            speakers = speakers_source.get(language, speakers_source.get("CN"))
            if not speakers and speakers_source:
                 # Fallback to first available value if specific and default missing
                 speakers = next(iter(speakers_source.values()))
        else:
            # List format or backward compatibility
            speakers = speakers_source

        if not speakers:
             speakers = []

        # Build speaker names list for strict matching
        speaker_names = [s['name'] for s in speakers]
        speakers_desc = "、".join(
            f"{s['name']}（{s['role']}）" for s in speakers
        )
        # Add explicit instruction about using exact names
        speakers_desc += f"\n     **【重要】JSON 中的 speaker 字段只能填写：{speaker_names}，禁止使用其他名字！**"

        # Build JSON example with actual speaker names
        if len(speaker_names) == 1:
            # Single speaker (monologue)
            speakers_json_example = f'{{"speaker": "{speaker_names[0]}", "text": "对话内容"}}'
        else:
            # Multiple speakers
            examples = [f'{{"speaker": "{name}", "text": "对话内容"}}' for name in speaker_names[:2]]
            speakers_json_example = ",\n      ".join(examples)

        # 2. Resolve Word Count
        word_count = topic_conf.get("word_count")
        if not word_count:
            word_count = self.config.get("dialogue", {}).get("target_word_count", 180)
        
        history_text = "\n".join(f"- {h}" for h in history) if history else "（无）"
        
        # 3. Resolve Language Instructions
        languages_config = self.prompts.get("languages", {})
        lang_config = languages_config.get(language, {})
        
        lang_instr = lang_config.get("instruction", "请全程使用中文进行对话。")
        culture_instr = lang_config.get("culture", "Target Audience: Chinese.")

        # 4. Resolve Prompt Template
        template_key = topic_conf.get("prompt_template")
        
        # Fallback/Legacy logic if not specified in config
        if not template_key:
            if topic_key == "daily_china_finance":
                template_key = "daily_china_finance"
            elif stock_code or topic_key == "stock_talk":
                template_key = "stock_talk"
            else:
                template_key = "default"

        template = self.prompts.get(template_key, "")
        if not template:
            # Fallback if key exists but template missing, or some other error
             template = self.prompts.get("default", "")

        # 5. Format Prompt
        # Handle variations in available keys for formatting
        today = datetime.now()
        current_date = today.strftime("%Y年%m月%d日")
        current_date_search = today.strftime("%Y-%m-%d")  # For search queries

        format_args = {
            "topic": topic_name,
            "word_count": word_count,
            "speakers_desc": speakers_desc,
            "speakers_json_example": speakers_json_example,
            "history": history_text,
            "language_instruction": lang_instr,
            "culture_instruction": culture_instr,
            "stock_code": stock_code or "",
            "current_date": current_date,
            "current_date_search": current_date_search,
        }

        return template.format(**format_args)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from AI response, handling markdown code blocks."""
        # Try to find JSON in code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        return json.loads(json_str)

    def generate(
        self,
        generation_id: int,
        topic_key: str,
        topic_name: str,
        output_dir: Path,
        stock_code: str | None = None,
        language: str = "CN",
    ) -> tuple[list[dict], list[str], str, str]:
        """
        Generate dialogue content for a topic.

        Args:
            generation_id: Database generation ID.
            topic_key: Topic key (e.g., 'life_tips').
            topic_name: Topic display name.
            output_dir: Directory to save output files.
            stock_code: Optional stock code for stock_talk topic.
            language: Output language code ("CN" or "EN").

        Returns:
            Tuple of (dialogue list, references list, summary, title).

        Raises:
            Exception: If generation fails.
        """

        # Fetch recent history to avoid repetition
        history = self.db.get_topic_summary_history(topic_key, limit=5)
        prompt = self._build_prompt(topic_name, history, topic_key=topic_key, stock_code=stock_code, language=language)

        # Create DB record
        req = self.db.create_dialogue_request(generation_id, prompt)

        try:
            # Configure generation
            topic_conf = get_topic_config(self.config, topic_key)
            
            # Resolve Model
            model_name = topic_conf.get("model", self.model)
            
            # Resolve Tools (Search)
            use_search = topic_conf.get("use_search", False)
            
            # Legacy fallback checks
            is_legacy_special = stock_code or topic_key in ["stock_talk", "daily_china_finance"]
            
            if "model" not in topic_conf and is_legacy_special:
                 model_name = "gemini-3-pro-preview"
            
            if "use_search" not in topic_conf and is_legacy_special:
                 use_search = True

            tools = [self.grounding_tool] if use_search else []

            gen_config = types.GenerateContentConfig(
                temperature=0.8,
                top_p=0.95,
                max_output_tokens=4096,
                tools=tools,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
            )

            # Build content
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            # Generate
            response_text = ""
            grounding_chunks = []

            if tools:
                # Use non-streaming for tools to easily access grounding metadata
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                if response.text:
                    response_text = response.text

                # Extract grounding metadata
                if response.candidates and response.candidates[0].grounding_metadata:
                    metadata = response.candidates[0].grounding_metadata
                    if metadata.grounding_chunks:
                        grounding_chunks = metadata.grounding_chunks

            else:
                # Use streaming for normal chat
                for chunk in self.client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                ):
                    if chunk.text:
                        response_text += chunk.text

            # Parse response
            if not response_text or not response_text.strip():
                raise ValueError(f"Empty response from LLM. Model: {model_name}")

            # Debug: print first 500 chars of response if no JSON found
            try:
                data = self._extract_json(response_text)
            except ValueError as e:
                print(f"DEBUG: Response text (first 1000 chars):\n{response_text[:1000]}")
                raise
            dialogue = data.get("dialogue", [])
            references = data.get("references", [])
            summary = data.get("summary", "")
            title = data.get("title", summary[:12] if summary else "")  # Fallback to summary prefix

            # Append grounding references
            for chunk in grounding_chunks:
                if chunk.web and chunk.web.uri:
                     title_text = chunk.web.title or "Web Source"
                     references.append(f"[{title_text}]({chunk.web.uri})")

            # Validate dialogue structure
            for i, line in enumerate(dialogue):
                if "speaker" not in line or "text" not in line:
                    raise ValueError(f"Invalid dialogue line {i}: missing speaker or text")

            # Save dialogue JSON
            output_dir.mkdir(parents=True, exist_ok=True)
            dialogue_path = output_dir / f"dialogue_{generation_id}.json"
            with open(dialogue_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Update DB
            self.db.update_dialogue_request(
                req_id=req.id,
                response_raw=response_text,
                dialogue=dialogue,
                references=references,
                summary=summary,
                success=True,
            )

            self.db.update_generation_status(
                generation_id,
                status="dialogue_complete",
                dialogue_json_path=str(dialogue_path),
            )

            return dialogue, references, summary, title

        except Exception as e:
            self.db.update_dialogue_request(
                req_id=req.id,
                response_raw="",
                dialogue=[],
                references=[],
                summary="",
                success=False,
                error_message=str(e),
            )
            self.db.update_generation_status(
                generation_id,
                status="failed",
                error_message=f"Dialogue generation failed: {e}",
            )
            raise
