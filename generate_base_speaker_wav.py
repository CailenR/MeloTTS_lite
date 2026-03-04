#!/usr/bin/env python3
"""
Generate a short WAV of the MeloTTS default (base) speaker for use with
openvoice_v2/build_reference_se.py to create base_se.json.

  python generate_base_speaker_wav.py --output base_speaker.wav

Then (from openvoice_v2, with checkpoints_v2/converter):
  python build_reference_se.py --reference_wav ../MeloTTS_lite/base_speaker.wav --output base_se.json --no_binary
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="base_speaker.wav", help="Output WAV path.")
    parser.add_argument("--model_path", default="./model_en.onnx", help="MeloTTS ONNX model.")
    parser.add_argument("--lexicon", default="./lexicon.txt")
    parser.add_argument("--tokens", default="./tokens.txt")
    args = parser.parse_args()

    from tts_onnx import TTS

    tts = TTS(model_path=args.model_path, lexicon_path=args.lexicon, tokens_path=args.tokens)
    text = (
        "This is a sample of the base speaker voice. "
        "Use this file with OpenVoice v2 build_reference_se to create base_se.json for tone conversion."
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tts.synthesize(text, output_path=args.output, speed=1.0)
    print("Wrote", args.output)
    print("Next: from openvoice_v2 run build_reference_se.py with this file to create base_se.json.")


if __name__ == "__main__":
    main()
