using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using NAudio.Wave;

namespace MeloTTS.Onnx;

/// <summary>
/// Text-to-speech using ONNX model. No PyTorch dependency.
/// </summary>
public class TTS
{
    private readonly Lexicon _lexicon;
    private readonly OnnxModel _model;

    public TTS(
        string modelPath = "./model_en.onnx",
        string lexiconPath = "./lexicon.txt",
        string tokensPath = "./tokens.txt")
    {
        _lexicon = new Lexicon(lexiconPath, tokensPath);
        _model = new OnnxModel(modelPath);
    }

    private static IEnumerable<string> TokenizeEnglish(string text)
    {
        var matches = Regex.Matches(text, @"\w+|[^\w\s]");
        foreach (Match m in matches)
            yield return m.Value;
    }

    /// <summary>
    /// Synthesize speech from text and return audio buffer (no file write). Use for piping into tone conversion or custom output.
    /// </summary>
    /// <param name="text">Input text to synthesize.</param>
    /// <param name="speed">Playback speed (1.0 = normal, &gt;1 = faster, &lt;1 = slower).</param>
    /// <returns>(audio samples, sample rate).</returns>
    public (float[] Audio, int SampleRate) SynthesizeToBuffer(string text, float speed = 1.0f)
    {
        var tokens = TokenizeEnglish(text).ToList();
        var (phones, tones) = _lexicon.Convert(tokens);

        if (_model.AddBlank != 0)
        {
            var newPhones = new int[2 * phones.Count + 1];
            var newTones = new int[2 * tones.Count + 1];
            for (var i = 0; i < phones.Count; i++)
            {
                newPhones[2 * i + 1] = phones[i];
                newTones[2 * i + 1] = tones[i];
            }
            phones = newPhones.ToList();
            tones = newTones.ToList();
        }

        var phonesArr = phones.Select(p => (long)p).ToArray();
        var tonesArr = tones.Select(t => (long)t).ToArray();

        var audio = _model.Infer(phonesArr, tonesArr, speed);
        return (audio, _model.SampleRate);
    }

    /// <summary>
    /// Synthesize speech from text and save to WAV file.
    /// </summary>
    /// <param name="text">Input text to synthesize.</param>
    /// <param name="outputPath">Path for output WAV file.</param>
    /// <param name="speed">Playback speed (1.0 = normal, &gt;1 = faster, &lt;1 = slower).</param>
    /// <returns>Path to the written WAV file.</returns>
    public string Synthesize(string text, string outputPath, float speed = 1.0f)
    {
        var (audio, sampleRate) = SynthesizeToBuffer(text, speed);

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        using (var writer = new WaveFileWriter(outputPath, format))
        {
            writer.WriteSamples(audio, 0, audio.Length);
        }

        return outputPath;
    }
}
