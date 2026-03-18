---
name: index-tts
description: Generate high-quality speech from text using the IndexTTS2 model. Use this skill when the user wants to synthesize audio from text, optionally providing a reference audio for voice cloning and a lexicon for pronunciation control.
---

# Index TTS

## Overview

This skill allows you to generate speech from text using the IndexTTS2 model present in the current project. It supports voice cloning from a reference audio, custom pronunciation using a lexicon, and advanced emotion/generation control.

## Workflow

When the user wants to generate TTS, you MUST follow these steps:

1.  **Check for Required Information**:
    Check if the user has provided the following information in their request:
    *   **Reference Audio**: Path to the audio file for voice cloning (e.g., `examples/voice_01.wav`).
    *   **Text**: The text content to synthesize.
    *   **Lexicon** (Optional): Path to a pronunciation glossary YAML file.
    *   **Output Path** (Optional): Where to save the result (default to `output.wav` if not specified).
    *   **Emotion/Advanced Params** (Optional): Check if user wants specific emotion control or generation parameters.

2.  **Ask for Missing Information**:
    If any required information (Reference Audio or Text) is missing, or if you want to confirm optional parameters, you MUST use the `AskUserQuestion` tool to ask the user.
    
    Example questions:
    *   "Please provide the path to the reference audio file you'd like to use for voice cloning."
    *   "What text would you like to synthesize?"
    *   "Do you have a specific lexicon file for pronunciation? (Optional)"
    *   "Where should I save the generated audio file?"

3.  **Execute Generation**:
    Once you have all the necessary information, use the `scripts/generate_tts.py` script to generate the audio.

    **Optimization Note**: If the user wants to generate multiple audio files or complains about the slow startup time (model loading), use the **Interactive Mode**.

## Usage

To generate TTS, use the `scripts/generate_tts.py` script.

### Arguments

**Basic:**
- `--interactive`: (Optional) Run in interactive mode to keep the model loaded for multiple generations.
- `--ref_audio`: (Required in non-interactive) Path to the reference audio file (wav, mp3, etc.) to clone the voice from.
- `--text`: (Required in non-interactive) The text to synthesize (max 1000 characters).
- `--output_path`: (Required in non-interactive) Path where the generated audio will be saved.
- `--lexicon`: (Optional) Path to a YAML file containing a glossary for pronunciation.
- `--model_dir`: (Optional) Path to the model directory (default: `checkpoints`).
- `--device`: (Optional) Device to use (`cuda` or `cpu`).

**Emotion Control:**
- `--emo_ref_audio`: Path to emotion reference audio (for "Use Emotion Reference Audio" mode).
- `--emo_text`: Emotion description text (for "Use Emotion Description Text Control" mode).
- `--emo_weight`: Emotion weight (0.0-1.0, default 0.65).
- `--emo_random`: Enable random emotion sampling.
- `--emo_vec`: Comma-separated 8 floats for emotion vector (Joy, Anger, Sorrow, Fear, Disgust, Depression, Surprise, Calm).

**Advanced Generation:**
- `--do_sample` / `--no_sample`: Enable/Disable sampling (default: True).
- `--temperature`: Sampling temperature (default: 0.8).
- `--top_p`: Top-p sampling (default: 0.8).
- `--top_k`: Top-k sampling (default: 30).
- `--num_beams`: Number of beams (default: 3).
- `--repetition_penalty`: Repetition penalty (default: 10.0).
- `--length_penalty`: Length penalty (default: 0.0).
- `--max_mel_tokens`: Max mel tokens (default: 1500).
- `--max_text_tokens_per_segment`: Max text tokens per segment (default: 120).

### Example (Basic)

```bash
uv run scripts/generate_tts.py \
  --ref_audio "examples/voice_01.wav" \
  --text "Hello, this is a test." \
  --output_path "output.wav" \
  --lexicon "lexicon.yaml"
```

### Example (Advanced Emotion Control)

```bash
uv run scripts/generate_tts.py \
  --ref_audio "examples/voice_01.wav" \
  --text "I am so happy today!" \
  --output_path "happy.wav" \
  --emo_vec "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0" \
  --emo_weight 0.8
```

### Example (Interactive Mode)

To start the interactive session:

```bash
uv run scripts/generate_tts.py --interactive
```

In interactive mode, you can set advanced parameters via command line flags when launching, and then interactively provide text and audio paths.
