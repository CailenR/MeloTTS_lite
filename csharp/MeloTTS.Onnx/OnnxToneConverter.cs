using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Microsoft.ML.OnnxRuntime;

namespace MeloTTS.Onnx;

/// <summary>
/// OpenVoice v2 voice conversion via ONNX. Converts audio from source to target tone color using speaker embeddings (SE).
/// </summary>
public class OnnxToneConverter : IDisposable
{
    private readonly InferenceSession _session;
    private readonly ToneConverterConfig _config;
    private bool _disposed;

    public ToneConverterConfig Config => _config;
    public int SamplingRate => _config.SamplingRate;
    public int GinChannels => _config.GinChannels;

    /// <summary>Elapsed time in milliseconds of the last Convert call.</summary>
    public double LastConvertMs { get; private set; }

    public OnnxToneConverter(string onnxPath, string? configPath = null)
    {
        var config = ToneConverterConfig.LoadForConverter(configPath, onnxPath)
            ?? throw new InvalidOperationException("Tone converter config not found. Pass config path or place converter_config.json next to the ONNX file.");

        _config = config;

        var options = new SessionOptions
        {
            InterOpNumThreads = 1,
            IntraOpNumThreads = 4
        };
        _session = new InferenceSession(onnxPath, options);
    }

    /// <summary>
    /// Convert audio from source to target tone color.
    /// </summary>
    /// <param name="audio">Mono float32 samples (any length/sample rate; resampled to converter SR if inputSampleRate is set).</param>
    /// <param name="srcSe">Source speaker embedding: flat float[] of length GinChannels (from SeLoader.LoadFromFile or preloaded).</param>
    /// <param name="tgtSe">Target speaker embedding: same shape as srcSe.</param>
    /// <param name="inputSampleRate">If set and different from converter sampling rate, audio is resampled.</param>
    /// <param name="tau">Conversion strength (default 0.3).</param>
    /// <returns>Converted mono float32 at SamplingRate.</returns>
    public float[] Convert(
        ReadOnlySpan<float> audio,
        ReadOnlySpan<float> srcSe,
        ReadOnlySpan<float> tgtSe,
        int? inputSampleRate = null,
        float tau = 0.3f)
    {
        float[] audioIn = audio.ToArray();
        if (inputSampleRate.HasValue && inputSampleRate.Value != _config.SamplingRate)
            audioIn = AudioDsp.ResampleLinear(audioIn, inputSampleRate.Value, _config.SamplingRate);

        var (spec, specFreq, nFrames) = AudioDsp.Spectrogram(
            audioIn,
            _config.FilterLength,
            _config.HopLength,
            _config.WinLength);

        // ONNX expects spec shape [1, specFreq, nFrames] (last dim contiguous). Our spec is [nFrames, specFreq].
        var specOnnx = new float[specFreq * nFrames];
        for (int f = 0; f < specFreq; f++)
            for (int t = 0; t < nFrames; t++)
                specOnnx[f * nFrames + t] = spec[t * specFreq + f];

        var specShape = new long[] { 1, specFreq, nFrames };
        var specLengths = new long[] { nFrames };
        var srcSeArr = srcSe.ToArray();
        var tgtSeArr = tgtSe.ToArray();
        if (srcSeArr.Length != _config.GinChannels || tgtSeArr.Length != _config.GinChannels)
            throw new ArgumentException($"SE must have length {_config.GinChannels}. Got src={srcSeArr.Length}, tgt={tgtSeArr.Length}.");

        // sid_src / sid_tgt: shape [1, gin_channels, 1]
        var sidShape = new long[] { 1, _config.GinChannels, 1 };
        using var specTensor = OrtValue.CreateTensorValueFromMemory(specOnnx, specShape);
        using var specLengthsTensor = OrtValue.CreateTensorValueFromMemory(specLengths, new long[] { 1 });
        using var sidSrcTensor = OrtValue.CreateTensorValueFromMemory(srcSeArr, sidShape);
        using var sidTgtTensor = OrtValue.CreateTensorValueFromMemory(tgtSeArr, sidShape);
        using var tauTensor = OrtValue.CreateTensorValueFromMemory(new[] { tau }, new long[] { 1 });

        var inputs = new Dictionary<string, OrtValue>
        {
            ["spec"] = specTensor,
            ["spec_lengths"] = specLengthsTensor,
            ["sid_src"] = sidSrcTensor,
            ["sid_tgt"] = sidTgtTensor,
            ["tau"] = tauTensor
        };

        using var runOptions = new RunOptions();
        var sw = Stopwatch.StartNew();
        using var outputs = _session.Run(runOptions, inputs, _session.OutputNames);
        sw.Stop();
        LastConvertMs = sw.Elapsed.TotalMilliseconds;
        Trace.WriteLine($"[MeloTTS.Onnx] ToneConverter Run: {LastConvertMs:F2} ms");

        if (outputs.Count == 0)
            throw new InvalidOperationException("No outputs from tone converter ONNX.");

        var output = outputs[0];
        var span = output.GetTensorDataAsSpan<float>();
        var audioOut = new float[span.Length];
        span.CopyTo(audioOut);
        return audioOut;
    }

    /// <summary>
    /// Convenience: convert using SE file paths.
    /// </summary>
    public float[] Convert(
        ReadOnlySpan<float> audio,
        string srcSePath,
        string tgtSePath,
        int? inputSampleRate = null,
        float tau = 0.3f)
    {
        var srcSe = SeLoader.LoadFromFile(srcSePath);
        var tgtSe = SeLoader.LoadFromFile(tgtSePath);
        return Convert(audio, srcSe, tgtSe, inputSampleRate, tau);
    }

    /// <summary>
    /// Apply tone conversion to audio. Loads SEs from paths and uses the given converter (or creates one from paths).
    /// </summary>
    /// <returns>(converted audio, output sample rate).</returns>
    public static (float[] Audio, int SampleRate) ApplyToneConversion(
        ReadOnlySpan<float> audio,
        int sampleRate,
        string srcSePath,
        string tgtSePath,
        OnnxToneConverter? converter = null,
        string? converterOnnxPath = null,
        string? converterConfigPath = null,
        float tau = 0.3f)
    {
        if (converter == null)
        {
            if (string.IsNullOrEmpty(converterOnnxPath) || !File.Exists(converterOnnxPath))
                throw new ArgumentException("Pass a converter instance or valid converterOnnxPath (and optional converterConfigPath).", nameof(converterOnnxPath));
            using var c = new OnnxToneConverter(converterOnnxPath, converterConfigPath);
            var converted = c.Convert(audio, srcSePath, tgtSePath, sampleRate, tau);
            return (converted, c.SamplingRate);
        }
        var outAudio = converter.Convert(audio, srcSePath, tgtSePath, sampleRate, tau);
        return (outAudio, converter.SamplingRate);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
