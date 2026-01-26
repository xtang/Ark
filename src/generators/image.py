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

    SCENE_PROMPT_TEMPLATE = """You are an expert visual storyteller. Analyze the podcast dialogue below and extract【exactly {count}】key scenes that are most visually impactful.
This is CRITICAL: You must return exactly {count} scenes.

Dialogue Context:
{dialogue_text}

Topic Summary:
{summary}

Task:
Generate a detailed English image prompt for {count} distinct scenes.
For each scene, consider the cultural likely context (e.g., if talking about Chinese history, specify "ancient China style"; if modern tech, "futuristic modern lab"). DO NOT default to any specific ethnicity unless the context implies it.

Requirements:
1. Return a JSON array with exactly {count} objects.
2. Prompts must be highly detailed, describing:
   - Subject (who/what)
   - Action (what is happening)
   - Environment (lighting, background, weather)
   - Mood/Atmosphere
   - Photographic Style: {style}, cinematic lighting, 8k resolution, highly detailed
3. Ensure visual variety across scenes.

Output Format (JSON Array):
```json
[
  {{"scene": "Brief description of scene 1", "prompt": "Detailed English prompt for scene 1, including subject, action, lighting, style"}},
  {{"scene": "Brief description of scene 2", "prompt": "Detailed English prompt for scene 2"}},
  ...
]
```"""


    COVER_PROMPT_TEMPLATE = """You are an expert graphical designer for podcast covers.
Title: "{title}"
Topic Summary: "{summary}"

Task:
Create a high-impact, minimalist, and professional podcast cover art prompt.

Requirements:
1. **Style**: {style}. Must look premium and eye-catching on small screens (like Spotify/Apple Podcasts).
2. **Typography**: The image MUST include the title "{title}" integrated into the design. The text should be bold, legible, and artistic.
3. **Composition**: Center-weighted or balanced. Text should be the focal point or seamlessly integrated with the imagery.
4. **Elements**: Use symbolic or metaphorical imagery representing the topic. Avoid clutter.
5. **Lighting**: Dramatic, studio quality, or soft natural light depending on the mood.

Return ONLY the English image prompt descriptions.
"""


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
        model_name: str | None = None,
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
                    model=model_name or self.image_model,
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

    def generate_cover(
        self,
        generation_id: int,
        title: str,
        summary: str,
        output_dir: Path,
    ) -> str | None:
        """
        Generate a dedicated cover image for the podcast.

        Args:
            generation_id: Database generation ID.
            title: Podcast title.
            summary: Podcast summary.
            output_dir: Directory to save output.

        Returns:
            Path to generated cover image, or None if failed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cover_path = output_dir / f"cover_{generation_id}_raw.png"
        
        try:
            # Create Prompt
            prompt_text = self.COVER_PROMPT_TEMPLATE.format(
                title=title,
                summary=summary,
                style=self.style,
            )

            # Generate Prompt using Text Model first to refine it (Optional, but let's stick to direct prompt for now 
            # or use the template as the prompt directly if it's descriptive enough. 
            # Actually the template asks for a prompt *description*. Let's do a quick text generation step to get the actual image prompt.)
            
            # Step 1: Generate the image prompt description
            gen_config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]
            
            image_prompt = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.text_model,
                contents=contents,
                config=gen_config,
            ):
                if chunk.text:
                    image_prompt += chunk.text
            
            image_prompt = image_prompt.strip()
            print(f"🎨 Cover Art Prompt: {image_prompt[:100]}...")

            # Step 2: Generate the Image
            # Create a dummy request ID for tracking? Or just pass 999
            # Since we didn't add a database method for 'cover_request', we'll just log it.
            

            success, error_msg, _ = self._generate_image_with_retry(
                image_prompt,
                cover_path,
                req_id=0, # 0 means not tracked individually in image_requests table for now
                model_name="gemini-3-pro-image-preview",
            )


            if success:
                print(f"✅ Cover art generated: {cover_path}")
                return str(cover_path)
            else:
                print(f"❌ Cover art generation failed: {error_msg}")
                return None

        except Exception as e:
            print(f"❌ Error generating cover: {e}")
            return None

