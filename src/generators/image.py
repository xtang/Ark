"""Image generator using Gemini AI."""

import base64
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from ..database import Database


class ImageGenerator:
    """Generate podcast images using Gemini AI."""

    SCENE_PROMPT_TEMPLATE = """分析以下播客对话内容，提取{count}个最适合可视化的关键场景或概念。
每个场景应该对应对话中的不同部分，确保视觉内容与对话进度同步。

对话内容：
{dialogue_text}

对话主题：{summary}

请为每个场景生成一个详细的英文图片生成提示词（prompt），风格：{style}。
注意：
1. 每张图片的prompt应该具体描述场景中的人物、物体、环境
2. 保持视觉风格一致，使用写实摄影风格（realistic photography）
3. 如果场景涉及人物但未指定国籍，默认使用中国人（Chinese people）
4. 适合作为播客/短视频的配图

输出格式（JSON数组）：
```json
[
  {{"scene": "场景描述", "prompt": "English image generation prompt, realistic photography style, Chinese people if nationality not specified, high quality, 4K"}},
  ...
]
```"""

    def __init__(self, config: dict[str, Any], db: Database):
        """
        Initialize the image generator.

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
        self.text_model = "gemini-3-flash-preview"
        self.image_model = "gemini-2.5-flash-image"

        # Dynamic image count settings
        self.count_per_lines = config.get("images", {}).get("count_per_lines", 2)
        self.min_count = config.get("images", {}).get("min_count", 3)
        self.max_count = config.get("images", {}).get("max_count", 10)
        self.aspect_ratio = config.get("images", {}).get("aspect_ratio", "16:9")
        self.style = config.get("images", {}).get("style", "realistic illustration")

    def _calculate_image_count(self, dialogue_length: int) -> int:
        """Calculate optimal number of images based on dialogue length."""
        # 1 image per N dialogue lines
        count = max(1, dialogue_length // self.count_per_lines)
        # Clamp to min/max
        return max(self.min_count, min(self.max_count, count))

    def _extract_scenes(self, dialogue: list[dict], summary: str, image_count: int) -> list[dict]:
        """Extract key scenes from dialogue for image generation."""
        import json
        import re

        # Build dialogue text
        dialogue_text = "\n".join(
            f"{line['speaker']}: {line['text']}" for line in dialogue
        )

        prompt = self.SCENE_PROMPT_TEMPLATE.format(
            count=image_count,
            dialogue_text=dialogue_text,
            summary=summary,
            style=self.style,
        )

        gen_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        response_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.text_model,
            contents=contents,
            config=gen_config,
        ):
            if chunk.text:
                response_text += chunk.text

        # Extract JSON
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match = re.search(r"\[[\s\S]*\]", response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in scene extraction response")

        return json.loads(json_str)

    def _generate_image(self, prompt: str, output_path: Path) -> bool:
        """Generate a single image using Gemini."""
        gen_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            image_config=types.ImageConfig(
                aspect_ratio=self.aspect_ratio,
                image_size="1K",
                output_mime_type="image/png",
            ),
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        response = self.client.models.generate_content(
            model=self.image_model,
            contents=contents,
            config=gen_config,
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image_data = part.inline_data.data
                if isinstance(image_data, str):
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = image_data

                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                return True

        return False

    def generate(
        self,
        generation_id: int,
        dialogue: list[dict],
        summary: str,
        output_dir: Path,
    ) -> list[str]:
        """
        Generate images for dialogue content.

        Args:
            generation_id: Database generation ID.
            dialogue: List of dialogue lines.
            summary: Dialogue summary.
            output_dir: Directory to save output files.

        Returns:
            List of generated image paths.

        Raises:
            Exception: If generation fails.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []

        try:
            # Calculate dynamic image count
            image_count = self._calculate_image_count(len(dialogue))

            # Extract scenes
            scenes = self._extract_scenes(dialogue, summary, image_count)

            # Generate each image
            for i, scene in enumerate(scenes[:image_count]):
                prompt = scene.get("prompt", "")
                if not prompt:
                    continue

                # Create DB record
                req = self.db.create_image_request(generation_id, prompt, i)

                image_path = output_dir / f"image_{generation_id}_{i}.png"

                try:
                    success = self._generate_image(prompt, image_path)

                    if success:
                        image_paths.append(str(image_path))
                        self.db.update_image_request(
                            req_id=req.id,
                            image_path=str(image_path),
                            success=True,
                        )
                    else:
                        self.db.update_image_request(
                            req_id=req.id,
                            image_path="",
                            success=False,
                            error_message="No image data in response",
                        )

                except Exception as e:
                    self.db.update_image_request(
                        req_id=req.id,
                        image_path="",
                        success=False,
                        error_message=str(e),
                    )

            if not image_paths:
                raise ValueError("No images were generated successfully")

            self.db.update_generation_status(generation_id, status="images_complete")

            return image_paths

        except Exception as e:
            self.db.update_generation_status(
                generation_id,
                status="failed",
                error_message=f"Image generation failed: {e}",
            )
            raise
