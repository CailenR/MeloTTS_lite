#!/usr/bin/env python3
"""
Standalone TTS class for ONNX inference without PyTorch.

Dependencies: onnxruntime, numpy, soundfile
"""

import re
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf


def _tokenize_english(text: str) -> List[str]:
    """Tokenize English text into words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text)


class Lexicon:
    """Grapheme-to-phoneme lookup using lexicon and token files."""

    def __init__(self, lexicon_filename: str, tokens_filename: str):
        tokens = {}
        with open(tokens_filename, encoding="utf-8") as f:
            for line in f:
                s, i = line.split()
                tokens[s] = int(i)
        # Map "v" to "V" token ID (same as post_replace_ph in MeloTTS, only for English models)
        if tokens.get("V") == 14 and "v" in tokens:
            tokens["v"] = tokens["V"]

        lexicon = {}
        with open(lexicon_filename, encoding="utf-8") as f:
            for line in f:
                splits = line.split()
                word_or_phrase = splits[0]
                phone_tone_list = splits[1:]
                assert len(phone_tone_list) & 1 == 0, len(phone_tone_list)
                phones = phone_tone_list[: len(phone_tone_list) // 2]
                phones = [tokens[p] for p in phones]
                tones = phone_tone_list[len(phone_tone_list) // 2 :]
                tones = [int(t) for t in tones]
                lexicon[word_or_phrase] = (phones, tones)

        self.lexicon = lexicon

        punctuation = ["!", "?", "…", ",", ".", "'", "-"]
        for p in punctuation:
            i = tokens[p]
            tone = 0
            self.lexicon[p] = ([i], [tone])
        self.lexicon[" "] = ([tokens["_"]], [0])

    def _convert(self, text: str, lowercase: bool = True) -> Tuple[List[int], List[int]]:
        phones = []
        tones = []

        if text == "，":
            text = ","
        elif text == "。":
            text = "."
        elif text == "！":
            text = "!"
        elif text == "？":
            text = "?"

        lookup = text.lower() if lowercase and len(text) > 1 else text
        if lookup not in self.lexicon:
            if len(text) > 1:
                for w in text:
                    p, t = self._convert(w, lowercase=False)
                    if p:
                        phones += p
                        tones += t
            return phones, tones

        phones, tones = self.lexicon[lookup]
        return list(phones), list(tones)

    def convert(self, text_list: Iterable[str], lowercase: bool = True) -> Tuple[List[int], List[int]]:
        phones = []
        tones = []
        for text in text_list:
            lookup = text.lower() if lowercase and len(text) > 1 else text
            p, t = self._convert(lookup, lowercase=False)
            phones += p
            tones += t
        return phones, tones


class OnnxModel:
    """ONNX inference for MeloTTS. Uses numpy only, no PyTorch."""

    def __init__(self, filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.model = ort.InferenceSession(
            filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.model.get_modelmeta().custom_metadata_map
        self.bert_dim = int(meta["bert_dim"])
        self.ja_bert_dim = int(meta["ja_bert_dim"])
        self.add_blank = int(meta["add_blank"])
        self.sample_rate = int(meta["sample_rate"])
        self.speaker_id = int(meta["speaker_id"])
        self.lang_id = int(meta["lang_id"])

    def __call__(
        self,
        phones: np.ndarray,
        tones: np.ndarray,
        speed: float = 1.0,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
    ) -> np.ndarray:
        """
        Run inference.

        Args:
            phones: 1-D int64 array of phone token IDs
            tones: 1-D int64 array of tone IDs
            speed: length_scale (1.0 = normal, >1 = faster, <1 = slower)
            noise_scale: inference noise scale
            noise_scale_w: inference noise scale for duration

        Returns:
            1-D float32 audio array
        """
        x = np.expand_dims(phones, axis=0)
        tones_arr = np.expand_dims(tones, axis=0)

        sid = np.array([self.speaker_id], dtype=np.int64)
        noise_scale_arr = np.array([noise_scale], dtype=np.float32)
        length_scale = np.array([speed], dtype=np.float32)
        noise_scale_w_arr = np.array([noise_scale_w], dtype=np.float32)
        x_lengths = np.array([x.shape[-1]], dtype=np.int64)

        y = self.model.run(
            ["y"],
            {
                "x": x,
                "x_lengths": x_lengths,
                "tones": tones_arr,
                "sid": sid,
                "noise_scale": noise_scale_arr,
                "noise_scale_w": noise_scale_w_arr,
                "length_scale": length_scale,
            },
        )[0][0][0]
        return y


class TTS:
    """Text-to-speech using ONNX model. No PyTorch dependency."""

    def __init__(
        self,
        model_path: str = "./model_en.onnx",
        lexicon_path: str = "./lexicon.txt",
        tokens_path: str = "./tokens.txt",
    ):
        self.lexicon = Lexicon(lexicon_path, tokens_path)
        self.model = OnnxModel(model_path)

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        speed: float = 1.0,
    ):
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            output_path: If set, write WAV here and return path. If None, return (audio, sample_rate).
            speed: Playback speed (1.0 = normal, >1 = faster, <1 = slower)

        Returns:
            If output_path is set: path string. Else: (audio: np.ndarray, sample_rate: int).
        """
        tokens = _tokenize_english(text)
        phones, tones = self.lexicon.convert(tokens)

        if self.model.add_blank:
            new_phones = [0] * (2 * len(phones) + 1)
            new_tones = [0] * (2 * len(tones) + 1)
            new_phones[1::2] = phones
            new_tones[1::2] = tones
            phones = new_phones
            tones = new_tones

        phones_arr = np.array(phones, dtype=np.int64)
        tones_arr = np.array(tones, dtype=np.int64)

        audio = self.model(phones_arr, tones_arr, speed=speed)
        sr = self.model.sample_rate
        if output_path is not None:
            sf.write(output_path, audio, sr)
            return output_path
        return audio, sr


if __name__ == "__main__":
    tts = TTS(model_path="./model_en.onnx")
    tts.synthesize(
        "Complete line-up checks. When ready for takeoff, commence engine run-up.",
        output_path="./outputs/onnx/test.wav",
        speed=1.0,
    )
    print("Done.")
    # Example: return audio without writing
    # audio, sr = tts.synthesize("Hello world.", output_path=None)
    # print(audio.shape, sr)
