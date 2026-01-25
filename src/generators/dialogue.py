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


class DialogueGenerator:
    """Generate podcast dialogue content using Gemini AI."""

    PROMPT_TEMPLATE = """你是一个专业的播客内容生成器。请生成一段关于「{topic}」的双人对话脚本。

## 要求
1. **内容真实可查**：所有信息必须是真实的，观众可以在网上找到相关参考资料
2. **对话时长**：总字数约{word_count}字，适合1分钟的正常语速朗读
3. **对话者**：{speakers_desc}
4. **风格**：生动有趣，适合播客收听，可以包含情感标记如 [笑声]、[惊讶] 等
5. **结构**：有开场、主体内容、结尾
6. **避免重复**：请避开以下最近已经讨论过的内容：
{history}

## 输出格式
请严格按照以下JSON格式输出，不要包含任何其他文字：
```json
{{
  "dialogue": [
    {{"speaker": "角色名", "text": "对话内容"}},
    {{"speaker": "角色名", "text": "对话内容"}}
  ],
  "references": ["参考来源1", "参考来源2"],
  "summary": "一句话概括这段对话的主题"
}}
```"""

    def __init__(self, config: dict[str, Any], db: Database):
        """
        Initialize the dialogue generator.

        Args:
            config: Application configuration.
            db: Database instance.
        """
        self.config = config
        self.db = db

        load_dotenv()
        api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY not found in environment")

        self.client = genai.Client(
            vertexai=True,
            api_key=api_key,
        )
        self.model = "gemini-3-flash-preview"

    def _build_prompt(self, topic_name: str, history: list[str]) -> str:
        """Build the prompt for dialogue generation."""
        speakers = self.config.get("dialogue", {}).get("speakers", [])
        speakers_desc = "、".join(
            f"{s['name']}（{s['role']}）" for s in speakers
        )

        word_count = self.config.get("dialogue", {}).get("target_word_count", 180)

        history_text = "\n".join(f"- {h}" for h in history) if history else "（无）"

        return self.PROMPT_TEMPLATE.format(
            topic=topic_name,
            word_count=word_count,
            speakers_desc=speakers_desc,
            history=history_text,
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
    ) -> tuple[list[dict], list[str], str]:
        """
        Generate dialogue content for a topic.

        Args:
            generation_id: Database generation ID.
            topic_key: Topic key (e.g., 'life_tips').
            topic_name: Topic display name.
            output_dir: Directory to save output files.

        Returns:
            Tuple of (dialogue list, references list, summary).

        Raises:
            Exception: If generation fails.
        """
        """
        # Fetch recent history to avoid repetition
        history = self.db.get_topic_summary_history(topic_key, limit=5)
        prompt = self._build_prompt(topic_name, history)

        # Create DB record
        req = self.db.create_dialogue_request(generation_id, prompt)

        try:
            # Configure generation
            gen_config = types.GenerateContentConfig(
                temperature=0.8,
                top_p=0.95,
                max_output_tokens=4096,
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
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
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

            return dialogue, references, summary

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
