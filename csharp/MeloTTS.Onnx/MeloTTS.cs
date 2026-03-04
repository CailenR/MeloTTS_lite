using NAudio.Wave;
using System;
using System.IO;

namespace MeloTTS.Onnx;

/// <summary>
/// Pipeline that owns a TTS and an OnnxToneConverter. Initialize once, then call Run with text and target SE path.
/// Disposes the converter when disposed.
/// </summary>
public class MeloTTSEngine : IDisposable
{
    private readonly TTS _tts;
    private readonly OnnxToneConverter _converter;
    private readonly string _srcSePath;
    private string _tgtSePath;
    private bool _disposed;

    /// <summary>Source (base) speaker embedding path used for conversion.</summary>
    public string SrcSePath => _srcSePath;

    /// <summary>Output sample rate after conversion (from converter config).</summary>
    public int SampleRate => _converter.SamplingRate;

    /// <summary>
    /// Build pipeline from model/config paths. Creates and owns both TTS and converter.
    /// </summary>
    /// <param name="modelPath">TTS ONNX model path (e.g. model_en.onnx).</param>
    /// <param name="lexiconPath">Lexicon path.</param>
    /// <param name="tokensPath">Tokens path.</param>
    /// <param name="converterOnnxPath">Converter ONNX path (e.g. converter.onnx).</param>
    /// <param name="converterConfigPath">Converter config (e.g. converter_config.json).</param>
    /// <param name="srcSePath">Source/base speaker SE path (default base_se.json).</param>
    public MeloTTSEngine(
        TTS? tts = null,
        string? converterOnnxPath = "converter.onnx",
        string? converterConfigPath = "converter_config.json",
        string? srcSePath = "base_se.json",
        string? tgtSePath = "target_se.json",
        string? lexicon = "lexicon.txt",
        string? tokens = "tokens.txt"
        )
    {
        _tts = tts ?? new TTS(lexiconPath:lexicon, tokensPath:tokens);

        _converter = new OnnxToneConverter(converterOnnxPath ?? "", converterConfigPath ?? "");
        _srcSePath = srcSePath ?? "base_se.json";
        _tgtSePath = tgtSePath ?? "target_se.json";
    }

    /// <summary>
    /// Synthesize text to audio, then apply tone conversion to the target speaker.
    /// </summary>
    /// <param name="text">Text to synthesize.</param>
    /// <param name="tgtSePath">Target speaker embedding path (.json or .bin + .bin.json).</param>
    /// <param name="speed">TTS playback speed (1.0 = normal).</param>
    /// <param name="tau">Tone conversion strength (default 0.3).</param>
    /// <returns>(converted audio, output sample rate).</returns>
    public (float[] Audio, int SampleRate) SynthesizeToBuffer(
        string text,
        string target_voice_path,
        float speed = 1.0f,
        float tau = 0.3f,
        bool convertTone=true
        )
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var (audio, sampleRate) = _tts.SynthesizeToBuffer(text, speed);
        if (convertTone) {
            if (target_voice_path != null)
            {
                // use their provided voice
                var converted = _converter.Convert(audio, _srcSePath, target_voice_path, sampleRate, tau);
                return (converted, sampleRate);
            }
            else
            {
                // use our default voice
                var converted = _converter.Convert(audio, _srcSePath, _tgtSePath, sampleRate, tau);
                return (converted, sampleRate);
            }
        }
        return (audio, sampleRate);

    }

    /// <summary>
    /// Synthesize text to audio, then apply tone conversion to the target speaker.
    /// </summary>
    /// <param name="text">Text to synthesize.</param>
    /// <param name="tgtSePath">Target speaker embedding path (.json or .bin + .bin.json).</param>
    /// <param name="speed">TTS playback speed (1.0 = normal).</param>
    /// <param name="tau">Tone conversion strength (default 0.3).</param>
    /// <returns>(converted audio, output sample rate).</returns>
    public string Synthesize(
        string text,
        string outputPath,
        string target_voice_path,
        float speed = 1.0f,
        float tau = 0.3f,
        bool convertTone=true
        )
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (convertTone)
        {
            var (audio, sampleRate) = _tts.SynthesizeToBuffer(text, speed);
            float[] outAudio = audio;
            int outSampleRate = sampleRate;
            if (target_voice_path != null)
            {
                // use their provided voice
                outAudio = _converter.Convert(audio, _srcSePath, target_voice_path, sampleRate, tau);
            }
            else
            {
                // use our default voice
                outAudio = _converter.Convert(audio, _srcSePath, _tgtSePath, sampleRate, tau);
            }
            var dir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);
            var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
            using (var writer = new WaveFileWriter(outputPath, format))
                writer.WriteSamples(outAudio, 0, outAudio.Length);
            return outputPath;
        }
            //synth text and return
            _tts.Synthesize(text, outputPath, speed);
            return outputPath;
    }


    public void Dispose()
    {
        if (_disposed) return;
        _converter.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
