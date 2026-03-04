# MeloTTS C# ONNX

.NET 8 TTS application using the MeloTTS ONNX model. No PyTorch dependency.

## Requirements

- .NET 8 SDK
- `model_en.onnx`, `lexicon.txt`, `tokens.txt` (from the MeloTTS_lite project root)

## Build

```bash
dotnet build
```

## Run

Paths are relative to the current working directory. Run from the MeloTTS_lite root so assets are found:

```bash
cd ..
dotnet run --project csharp/MeloTTS.Cli -- --text "Hello world" --output output.wav
```

Or copy `model_en.onnx`, `lexicon.txt`, and `tokens.txt` to `csharp/MeloTTS.Cli/bin/Debug/net8.0/` and run from there.

### Options

| Option | Default | Description |
|-------|---------|-------------|
| `--text` | (required) | Text to synthesize |
| `--output` | `output.wav` | Output WAV path |
| `--speed` | `1.0` | Playback speed (>1 faster, <1 slower) |
| `--model` | `model_en.onnx` | ONNX model path |
| `--lexicon` | `lexicon.txt` | Lexicon path |
| `--tokens` | `tokens.txt` | Tokens path |

### Example

```bash
dotnet run --project csharp/MeloTTS.Cli -- --text "Complete line-up checks." --output outputs/onnx/test.wav --speed 1.2
```

### Tone conversion (OpenVoice v2)

Tone conversion is applied **after** synthesis: text → TTS audio → (optional) tone convert → output WAV.  
When `--converter`, `--src-se`, and `--tgt-se` are provided, the synthesized audio is converted to the target speaker tone before writing.

```bash
dotnet run --project csharp/MeloTTS.Cli -- --text "Hello world" --output output.wav --converter converter.onnx --src-se base_se.json --tgt-se target_se.json [--config converter_config.json] [--tau 0.3]
```

| Option | Description |
|--------|-------------|
| `--converter` | Path to converter ONNX (e.g. from Python export). With `--src-se` and `--tgt-se`, runs tone conversion after TTS. |
| `--src-se` | Source speaker SE (base/MeloTTS voice): `.json` or `.bin` (+ `.bin.json`) |
| `--tgt-se` | Target speaker SE (clone voice) |
| `--config` | Optional converter config JSON (else `{onnx}_config.json`) |
| `--tau` | Conversion strength (default 0.3) |

## Library Usage

### TTS

```csharp
using MeloTTS.Onnx;

var tts = new TTS(
    modelPath: "model_en.onnx",
    lexiconPath: "lexicon.txt",
    tokensPath: "tokens.txt");

tts.Synthesize("Hello world", "output.wav", speed: 1.0f);
```

### Tone converter

```csharp
using MeloTTS.Onnx;

// Option 1: reuse a converter instance
using var converter = new OnnxToneConverter("converter.onnx", configPath: "converter_config.json");
float[] converted = converter.Convert(audioSamples, srcSePath: "base_se.json", tgtSePath: "target_se.json", inputSampleRate: 22050, tau: 0.3f);

// Option 2: one-shot with SE file paths
var (convertedAudio, outSampleRate) = OnnxToneConverter.ApplyToneConversion(
    audioSamples, sampleRate: 22050, srcSePath: "base_se.json", tgtSePath: "target_se.json",
    converterOnnxPath: "converter.onnx", converterConfigPath: "converter_config.json", tau: 0.3f);
```
