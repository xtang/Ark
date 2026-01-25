"""Image generator using Gemini AI with retry logic."""

import base64
import os
import time
import traceback
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from ..database import Database


class ImageGenerator:
    """Generate podcast images using Gemini AI with retry support."""

    SCENE_PROMPT_TEMPLATE = """分析以下播客对话内容，你必须提取【恰好{count}个】最适合可视化的关键场景。
这非常重要：你必须返回恰好 {count} 个场景，不能多也不能少！

对话内容：
{dialogue_text}

对话主题：{summary}

请为每个场景生成一个详细的英文图片生成提示词（prompt），风格：{style}。
要求：
1. 必须返回恰好 {count} 个场景（JSON数组长度必须是 {count}）
2. 每张图片的prompt应该具体描述场景中的人物、物体、环境
3. 保持视觉风格一致，使用写实摄影风格（realistic photography）
4. 如果场景涉及人物但未指定国籍，默认使用中国人（Chinese people）
5. 场景应均匀分布在对话的不同部分

输出格式（JSON数组，必须有 {count} 个元素）：
```json
[
  {{"scene": "场景1描述", "prompt": "English prompt for scene 1, realistic photography, Chinese people, 4K"}},
  {{"scene": "场景2描述", "prompt": "English prompt for scene 2, realistic photography, Chinese people, 4K"}},
  ... (共 {count} 个)
]
```"""

    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

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

    def _generate_image_with_retry(
        self,
        prompt: str,
        output_path: Path,
        req_id: int,
    ) -> tuple[bool, str, int]:
        """
        Generate a single image with retry logic.

        Returns:
            Tuple of (success, error_message, retry_count)
        """
        last_error = ""
        retry_count = 0

        for attempt in range(self.MAX_RETRIES):
            start_time = time.time()
            try:
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

                duration = time.time() - start_time

                # Check for blocked content or no candidates
                if not response.candidates:
                    last_error = f"No candidates in response (attempt {attempt + 1})"
                    retry_count = attempt + 1
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY_SECONDS)
                    continue

                candidate = response.candidates[0]

                # Check finish reason
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason)
                    if 'SAFETY' in finish_reason or 'BLOCKED' in finish_reason:
                        last_error = f"Content blocked: {finish_reason} (attempt {attempt + 1})"
                        retry_count = attempt + 1
                        if attempt < self.MAX_RETRIES - 1:
                            time.sleep(self.RETRY_DELAY_SECONDS)
                        continue

                # Extract image from response
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            image_data = part.inline_data.data
                            if isinstance(image_data, str):
                                image_bytes = base64.b64decode(image_data)
                            else:
                                image_bytes = image_data

                            with open(output_path, "wb") as f:
                                f.write(image_bytes)

                            # Update with timing info
                            self.db.update_image_request(
                                req_id=req_id,
                                image_path=str(output_path),
                                success=True,
                                duration_seconds=duration,
                                retry_count=attempt,
                            )
                            return True, "", attempt

                last_error = f"No image data in response (attempt {attempt + 1})"
                retry_count = attempt + 1

            except Exception as e:
                duration = time.time() - start_time
                last_error = f"Error (attempt {attempt + 1}): {str(e)}\n{traceback.format_exc()}"
                retry_count = attempt + 1

            # Wait before retry
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY_SECONDS)

        # All retries failed
        self.db.update_image_request(
            req_id=req_id,
            image_path="",
            success=False,
            error_message=last_error,
            retry_count=retry_count,
        )
        return False, last_error, retry_count

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
        total_start_time = time.time()

        try:
            # Calculate dynamic image count
            image_count = self._calculate_image_count(len(dialogue))

            # Extract scenes
            scenes = self._extract_scenes(dialogue, summary, image_count)

            # Generate each image with retry
            for i, scene in enumerate(scenes[:image_count]):
                prompt = scene.get("prompt", "")
                if not prompt:
                    continue

                # Create DB record
                req = self.db.create_image_request(generation_id, prompt, i)

                image_path = output_dir / f"image_{generation_id}_{i}.png"

                success, error_msg, retries = self._generate_image_with_retry(
                    prompt, image_path, req.id
                )

                if success:
                    image_paths.append(str(image_path))
                    # Rate limit: wait 10s between successful generations
                    if i < len(scenes[:image_count]) - 1:  # Don't wait after the last one
                        print("⏳ Waiting 10s for rate limit...")
                        time.sleep(10)
                else:
                    # Log failure details
                    print(f"⚠️ Image {i} failed after {retries} retries: {error_msg[:100]}...")

            # Update generation with timing
            total_duration = time.time() - total_start_time
            self.db.update_generation_timing(
                generation_id,
                image_duration_seconds=total_duration,
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
