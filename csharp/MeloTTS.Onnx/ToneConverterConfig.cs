using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace MeloTTS.Onnx;

/// <summary>
/// Config for OpenVoice v2 tone converter (converter_config.json).
/// </summary>
public class ToneConverterConfig
{
    public int SamplingRate { get; set; }
    public int FilterLength { get; set; }
    public int HopLength { get; set; }
    public int WinLength { get; set; }
    public int GinChannels { get; set; }

    public const int DefaultGinChannels = 256;

    /// <summary>
    /// Load config from JSON file. Expects keys: sampling_rate, filter_length, hop_length, win_length, gin_channels (optional).
    /// </summary>
    public static ToneConverterConfig Load(string configPath)
    {
        var json = File.ReadAllText(configPath);
        var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        return new ToneConverterConfig
        {
            SamplingRate = root.GetProperty("sampling_rate").GetInt32(),
            FilterLength = root.GetProperty("filter_length").GetInt32(),
            HopLength = root.GetProperty("hop_length").GetInt32(),
            WinLength = root.GetProperty("win_length").GetInt32(),
            GinChannels = root.TryGetProperty("gin_channels", out var g) ? g.GetInt32() : DefaultGinChannels
        };
    }

    /// <summary>
    /// Try to load config from path, or from {onnxPath}_config.json or {onnxPath without extension}_config.json.
    /// </summary>
    public static ToneConverterConfig? LoadForConverter(string? configPath, string onnxPath)
    {
        if (!string.IsNullOrEmpty(configPath) && File.Exists(configPath))
            return Load(configPath);
        var dir = Path.GetDirectoryName(onnxPath) ?? ".";
        var baseName = Path.GetFileNameWithoutExtension(onnxPath);
        var guess = Path.Combine(dir, baseName + "_config.json");
        if (File.Exists(guess))
            return Load(guess);
        return null;
    }
}
