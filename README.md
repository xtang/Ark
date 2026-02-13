# AI Podcast Generator

Reclaim Our Parents' Attention

## Quick Start

### Prerequisites

- [ffmpeg](https://ffmpeg.org/) must be installed with [libass](https://github.com/libass/libass) support and available on your `PATH`.

```bash
# macOS
brew tap homebrew-ffmpeg/ffmpeg
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass

# Ubuntu / Debian
sudo apt install ffmpeg libass-dev
```

### 1. Installation

This project uses `uv` for package management.

```bash
# Install dependencies
uv sync
```

### 2. Environment Setup

Configure your API keys (Gemini, ElevenLabs, etc.):

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Usage

Run the interactive Terminal UI:

```bash
uv run python -m src.main
```
