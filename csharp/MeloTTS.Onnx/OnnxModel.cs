using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;

namespace MeloTTS.Onnx;

/// <summary>
/// ONNX inference for MeloTTS. Uses ONNX Runtime only, no PyTorch.
/// </summary>
public class OnnxModel : IDisposable
{
    private readonly InferenceSession _session;
    private bool _disposed;

    /// <summary>Elapsed time in milliseconds of the last Run call (for profiling).</summary>
    public double LastRunMs { get; private set; }

    public int AddBlank { get; }
    public int SampleRate { get; }
    public int SpeakerId { get; }

    public OnnxModel(string modelPath)
    {
        var options = new SessionOptions
        {
            InterOpNumThreads = 1,
            IntraOpNumThreads = 4
        };

        _session = new InferenceSession(modelPath, options);

        var meta = _session.ModelMetadata.CustomMetadataMap;
        AddBlank = int.Parse(meta["add_blank"]);
        SampleRate = int.Parse(meta["sample_rate"]);
        SpeakerId = int.Parse(meta["speaker_id"]);
    }

    public float[] Infer(long[] phones, long[] tones, float speed = 1.0f, float noiseScale = 0.6f, float noiseScaleW = 0.8f)
    {
        var xShape = new long[] { 1, phones.Length };
        var tonesShape = new long[] { 1, tones.Length };

        using var x = OrtValue.CreateTensorValueFromMemory(phones, xShape);
        using var tonesTensor = OrtValue.CreateTensorValueFromMemory(tones, tonesShape);
        using var sid = OrtValue.CreateTensorValueFromMemory(new long[] { SpeakerId }, new long[] { 1 });
        using var noiseScaleTensor = OrtValue.CreateTensorValueFromMemory(new float[] { noiseScale }, new long[] { 1 });
        using var lengthScale = OrtValue.CreateTensorValueFromMemory(new float[] { speed }, new long[] { 1 });
        using var noiseScaleWTensor = OrtValue.CreateTensorValueFromMemory(new float[] { noiseScaleW }, new long[] { 1 });
        using var xLengths = OrtValue.CreateTensorValueFromMemory(new long[] { phones.Length }, new long[] { 1 });

        var inputs = new Dictionary<string, OrtValue>
        {
            ["x"] = x,
            ["x_lengths"] = xLengths,
            ["tones"] = tonesTensor,
            ["sid"] = sid,
            ["noise_scale"] = noiseScaleTensor,
            ["noise_scale_w"] = noiseScaleWTensor,
            ["length_scale"] = lengthScale
        };

        using var runOptions = new RunOptions();
        var sw = Stopwatch.StartNew();
        using var outputs = _session.Run(runOptions, inputs, _session.OutputNames);
        sw.Stop();
        LastRunMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"[MeloTTS.Onnx] Run: {LastRunMs:F2} ms");
        Trace.WriteLine($"[MeloTTS.Onnx] Run: {LastRunMs:F2} ms");

        if (outputs.Count == 0)
            throw new InvalidOperationException("No outputs from ONNX model.");

        var output = outputs[0];
        var span = output.GetTensorDataAsSpan<float>();
        var audio = new float[span.Length];
        span.CopyTo(audio);
        return audio;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
