"""Dialogue generator using Gemini AI."""

import json
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from ..database import Database
from ..config import load_prompts


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
        stock_code: str | None = None,
        language: str = "CN",
    ) -> str:
        """Build the prompt for dialogue generation."""
        speakers_config = self.config.get("dialogue", {}).get("speakers", [])
        
        # Determine speakers based on language
        if isinstance(speakers_config, dict):
            # Pick by language, fallback to CN, then fallback to first available
            speakers = speakers_config.get(language, speakers_config.get("CN"))
            if not speakers and speakers_config:
                 # Fallback to first available value if specific and default missing
                 speakers = next(iter(speakers_config.values()))
        else:
            # Backward compatibility for list
            speakers = speakers_config

        if not speakers:
             speakers = []

        speakers_desc = "、".join(
            f"{s['name']}（{s['role']}）" for s in speakers
        )

        word_count = self.config.get("dialogue", {}).get("target_word_count", 180)
        history_text = "\n".join(f"- {h}" for h in history) if history else "（无）"
        
        # Get language and culture instruction
        languages_config = self.prompts.get("languages", {})
        lang_config = languages_config.get(language, {})
        
        # Fallback to defaults if language not found
        lang_instr = lang_config.get("instruction", "请全程使用中文进行对话。")
        culture_instr = lang_config.get("culture", "Target Audience: Chinese.")

        # Use stock-specific prompt if stock_code is provided
        if stock_code:
            template = self.prompts.get("stock_talk", "")
            return template.format(
                stock_code=stock_code,
                word_count=word_count,
                speakers_desc=speakers_desc,
                history=history_text,
                language_instruction=lang_instr,
                culture_instruction=culture_instr,
            )

        template = self.prompts.get("default", "")
        return template.format(
            topic=topic_name,
            word_count=word_count,
            speakers_desc=speakers_desc,
            history=history_text,
            language_instruction=lang_instr,
            culture_instruction=culture_instr,
        )

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
        prompt = self._build_prompt(topic_name, history, stock_code=stock_code, language=language)

        # Create DB record
        req = self.db.create_dialogue_request(generation_id, prompt)

        try:
            # Configure generation
            tools = []
            model_name = self.model
            if stock_code or topic_key == "stock_talk":
                tools = [self.grounding_tool]
                model_name = "gemini-3-pro-preview"

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
            data = self._extract_json(response_text)
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
