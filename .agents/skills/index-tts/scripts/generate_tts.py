import argparse
import os
import sys
import yaml
import torch
import time

# Add project root to sys.path to ensure indextts can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from indextts.infer_v2 import IndexTTS2

def load_lexicon(lexicon_path):
    if not lexicon_path or not os.path.exists(lexicon_path):
        return None
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            lexicon = yaml.safe_load(f)
            if isinstance(lexicon, dict):
                return lexicon
            else:
                print(f"Warning: Lexicon file {lexicon_path} is not a valid dictionary. Ignoring.")
                return None
    except Exception as e:
        print(f"Error loading lexicon: {e}")
        return None

def get_input(prompt, default=None):
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def interactive_mode(tts, args):
    print("\n--- Interactive Mode ---")
    print("Type 'exit' or 'quit' at any prompt to stop.")
    
    current_ref_audio = args.ref_audio
    current_lexicon = args.lexicon
    output_count = 1

    # Load initial lexicon if provided
    if current_lexicon:
        lexicon = load_lexicon(current_lexicon)
        if lexicon:
            print(f"Loading lexicon from {current_lexicon}...")
            tts.normalizer.load_glossary(lexicon)
    
    # Pre-calculate emotion vector if provided
    emo_vec = None
    if args.emo_vec:
        try:
            vec_values = [float(x) for x in args.emo_vec.split(',')]
            if len(vec_values) == 8:
                emo_vec = tts.normalize_emo_vec(vec_values, apply_bias=True)
            else:
                print("Warning: --emo_vec must contain exactly 8 comma-separated numbers. Ignoring.")
        except ValueError:
            print("Warning: Invalid format for --emo_vec. Ignoring.")

    while True:
        try:
            # 1. Reference Audio
            ref_audio = get_input("Reference Audio Path", current_ref_audio)
            if ref_audio.lower() in ['exit', 'quit']: break
            if not os.path.exists(ref_audio):
                print(f"Error: File not found: {ref_audio}")
                continue
            current_ref_audio = ref_audio

            # 2. Text
            text = get_input("Text to Synthesize")
            if text.lower() in ['exit', 'quit']: break
            if not text:
                print("Error: Text cannot be empty.")
                continue
            
            if len(text) > 1000:
                print(f"Error: Text length ({len(text)}) exceeds the limit of 1000 characters.")
                continue

            # 3. Output Path
            default_output = f"output_{output_count}.wav"
            output_path = get_input("Output Path", default_output)
            if output_path.lower() in ['exit', 'quit']: break

            # 4. Lexicon (Optional Update)
            change_lexicon = get_input("Change Lexicon? (y/n)", "n")
            if change_lexicon.lower() == 'y':
                lexicon_path = get_input("New Lexicon Path", current_lexicon)
                if lexicon_path and os.path.exists(lexicon_path):
                    lexicon = load_lexicon(lexicon_path)
                    if lexicon:
                        print(f"Loading lexicon from {lexicon_path}...")
                        tts.normalizer.load_glossary(lexicon)
                        current_lexicon = lexicon_path
                elif lexicon_path:
                     print(f"Warning: Lexicon file not found: {lexicon_path}")

            # Generate
            print(f"Synthesizing...")
            start_time = time.time()
            
            # Use advanced parameters from args
            tts.infer(
                spk_audio_prompt=current_ref_audio,
                text=text,
                output_path=output_path,
                emo_audio_prompt=args.emo_ref_audio,
                emo_alpha=args.emo_weight,
                emo_vector=emo_vec,
                use_emo_text=bool(args.emo_text),
                emo_text=args.emo_text,
                use_random=args.emo_random,
                max_text_tokens_per_segment=args.max_text_tokens_per_segment,
                do_sample=args.do_sample,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                length_penalty=args.length_penalty,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                max_mel_tokens=args.max_mel_tokens
            )
            elapsed = time.time() - start_time
            print(f"Successfully generated audio at {output_path} in {elapsed:.2f}s")
            
            if output_path == default_output:
                output_count += 1
            
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio using IndexTTS2")
    parser.add_argument("--config", help="Path to a YAML configuration file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--ref_audio", help="Path to reference audio file (optional in interactive)")
    parser.add_argument("--text", help="Text to synthesize (optional in interactive)")
    parser.add_argument("--lexicon", help="Path to lexicon YAML file (optional)")
    parser.add_argument("--output_path", help="Path to save output audio (optional in interactive)")
    parser.add_argument("--model_dir", default="checkpoints", help="Path to model directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

    # Emotion Control Arguments
    parser.add_argument("--emo_ref_audio", help="Path to emotion reference audio")
    parser.add_argument("--emo_text", help="Emotion description text")
    parser.add_argument("--emo_weight", type=float, default=0.65, help="Emotion weight (0.0-1.0)")
    parser.add_argument("--emo_random", action="store_true", help="Enable random emotion sampling")
    parser.add_argument("--emo_vec", help="Comma-separated 8 floats for emotion vector (Joy, Anger, Sorrow, Fear, Disgust, Depression, Surprise, Calm)")

    # Advanced Generation Parameters
    parser.add_argument("--do_sample", action="store_true", default=True, help="Enable sampling (default: True)")
    parser.add_argument("--no_sample", action="store_false", dest="do_sample", help="Disable sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k sampling")
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for beam search")
    parser.add_argument("--repetition_penalty", type=float, default=10.0, help="Repetition penalty")
    parser.add_argument("--length_penalty", type=float, default=0.0, help="Length penalty")
    parser.add_argument("--max_mel_tokens", type=int, default=1500, help="Max mel tokens")
    parser.add_argument("--max_text_tokens_per_segment", type=int, default=120, help="Max text tokens per segment")

    args = parser.parse_args()

    # Load from config if provided
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    for key, value in config.items():
                        # Only set if the argument exists and was not provided on command line
                        if hasattr(args, key) and getattr(args, key) is None:
                            setattr(args, key, value)
                else:
                    print(f"Warning: Config file {args.config} is not a valid dictionary.")
        else:
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)

    if not args.interactive and (not args.ref_audio or not args.text or not args.output_path):
        parser.error("--ref_audio, --text, and --output_path are required unless --interactive is specified.")

    print(f"Initializing IndexTTS2 with model_dir={args.model_dir} device={args.device}...")
    tts = IndexTTS2(model_dir=args.model_dir, device=args.device)

    if args.interactive:
        interactive_mode(tts, args)
    else:
        # Check text length limit
        if len(args.text) > 1000:
            print(f"Error: Text length ({len(args.text)}) exceeds the limit of 1000 characters.")
            sys.exit(1)

        # Load lexicon if provided
        if args.lexicon:
            lexicon = load_lexicon(args.lexicon)
            if lexicon:
                print(f"Loading lexicon from {args.lexicon}...")
                tts.normalizer.load_glossary(lexicon)
        
        # Pre-calculate emotion vector if provided
        emo_vec = None
        if args.emo_vec:
            try:
                vec_values = [float(x) for x in args.emo_vec.split(',')]
                if len(vec_values) == 8:
                    emo_vec = tts.normalize_emo_vec(vec_values, apply_bias=True)
                else:
                    print("Warning: --emo_vec must contain exactly 8 comma-separated numbers. Ignoring.")
            except ValueError:
                print("Warning: Invalid format for --emo_vec. Ignoring.")

        print(f"Processing reference audio: {args.ref_audio}")
        try:
            print(f"Synthesizing text: {args.text}")
            tts.infer(
                spk_audio_prompt=args.ref_audio,
                text=args.text,
                output_path=args.output_path,
                emo_audio_prompt=args.emo_ref_audio,
                emo_alpha=args.emo_weight,
                emo_vector=emo_vec,
                use_emo_text=bool(args.emo_text),
                emo_text=args.emo_text,
                use_random=args.emo_random,
                max_text_tokens_per_segment=args.max_text_tokens_per_segment,
                do_sample=args.do_sample,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                length_penalty=args.length_penalty,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                max_mel_tokens=args.max_mel_tokens
            )
            print(f"Successfully generated audio at {args.output_path}")
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
