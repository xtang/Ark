"""Veo video generator module."""

import os
import time
from pathlib import Path
from typing import Any

class VeoGenerator:
    """Generate video clips using Google Vertex AI Veo model."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Veo generator."""
        self.config = config

    def generate_clip(
        self,
        prompt: str,
        output_path: Path,
    ) -> str:
        """
        Generate a video clip using Google Vertex AI Veo model.
        
        Args:
            prompt: Text prompt for video generation.
            output_path: Path to save the generated video.
            
        Returns:
            Path to the generated video file.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package is required for Veo generation. "
                "Please install it with: uv add google-genai"
            )

        veo_config = self.config.get("video", {}).get("veo", {})
        project_id = veo_config.get("project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = veo_config.get("location", "us-central1")
        model_name = veo_config.get("model", "veo-3.1-fast-generate-001")
        
        # Veo Config Parameters
        duration_seconds = veo_config.get("duration_seconds", 4)
        resolution = veo_config.get("resolution", "720p")
        aspect_ratio = veo_config.get("aspect_ratio", "16:9")
        
        if not project_id:
            raise ValueError(
                "Google Cloud Project ID is required for Veo generation (Vertex AI mode). "
                "Set it in config['video']['veo']['project_id'] or GOOGLE_CLOUD_PROJECT env var."
            )

        print(f"🎥 Initializing Veo generation (Model: {model_name})...")
        print(f"   Settings: {resolution}, {duration_seconds}s, {aspect_ratio}")
        print(f"   Prompt: {prompt[:100]}...")

        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError
            
            # Verify credentials exist before trying to create client
            try:
                google.auth.default()
            except DefaultCredentialsError:
                raise RuntimeError(
                    "❌ Google Cloud Credentials not found.\n"
                    "Veo requires Application Default Credentials (ADC).\n"
                    "To fix this:\n"
                    "1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install\n"
                    "2. Run: gcloud auth application-default login\n"
                    "3. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable to a service account key."
                )

            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
        except ImportError:
             raise ImportError("google-auth package is missing.")

        source = types.GenerateVideosSource(
            prompt=prompt,
        )

        config = types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            number_of_videos=1,
            generate_audio=False,
            duration_seconds=duration_seconds,
            person_generation="allow_all",
            resolution=resolution,
        )
        
        # Generate the video generation request
        operation = client.models.generate_videos(
            model=model_name, source=source, config=config
        )

        # Waiting for the video(s) to be generated
        print("   Waiting for video generation...")
        while not operation.done:
            print("   ... generating (check again in 10s)")
            time.sleep(10)
            operation = client.operations.get(operation)

        response = operation.result
        if not response:
            raise RuntimeError("Veo generation returned no response.")

        generated_videos = response.generated_videos
        if not generated_videos:
            raise RuntimeError("Veo generation returned no videos.")

        # Save the first video
        video = generated_videos[0].video
        if not video:
            raise RuntimeError("Veo generation returned a response, but the video object is missing.")

        if video.video_bytes:
            with open(output_path, "wb") as f:
                f.write(video.video_bytes)
        elif video.uri:
            raise RuntimeError(f"Veo returned a remote URI ({video.uri}) but no video content bytes. Remote video downloading is not yet implemented.")
        else:
            raise RuntimeError("Veo returned a video object with no content (no bytes, no URI).")
            
        print(f"   ✓ Video generated: {output_path}")
        return str(output_path)
