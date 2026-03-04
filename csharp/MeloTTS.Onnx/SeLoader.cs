using System;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace MeloTTS.Onnx;

/// <summary>
/// Load tone-color speaker embedding (SE) from portable format.
/// Returns array shape (1, gin_channels, 1) as flat float[] of length gin_channels.
/// Supports: .json with "se" or "embedding", or .bin with .bin.json metadata.
/// </summary>
public static class SeLoader
{
    /// <summary>
    /// Load SE from file. Returns float array of length gin_channels (shape 1, gin_channels, 1 in Python).
    /// </summary>
    public static float[] LoadFromFile(string path)
    {
        path = Path.GetFullPath(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("SE file not found.", path);

        if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && !path.EndsWith(".bin.json", StringComparison.OrdinalIgnoreCase))
            return LoadFromJson(path);

        var binPath = path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)
            ? path
            : Path.ChangeExtension(path, ".bin");
        var metaPath = binPath + ".json";
        if (!File.Exists(metaPath))
            throw new FileNotFoundException("Binary SE requires metadata file.", metaPath);

        var meta = File.ReadAllText(metaPath);
        var doc = JsonDocument.Parse(meta);
        var ginChannels = doc.RootElement.GetProperty("gin_channels").GetInt32();

        var raw = File.ReadAllBytes(binPath);
        int floatCount = raw.Length / sizeof(float);
        if (floatCount < ginChannels)
            throw new InvalidDataException($"Binary SE has {floatCount} floats, need {ginChannels}.");
        var se = new float[ginChannels];
        Buffer.BlockCopy(raw, 0, se, 0, ginChannels * sizeof(float));
        return se;
    }

    private static float[] LoadFromJson(string path)
    {
        var json = File.ReadAllText(path);
        var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.TryGetProperty("se", out var seProp))
        {
            var list = seProp.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
            return list;
        }
        if (root.TryGetProperty("embedding", out var embProp))
        {
            var list = embProp.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
            return list;
        }
        throw new InvalidDataException($"No 'se' or 'embedding' key in {path}");
    }

    /// <summary>
    /// Reshape flat SE to ONNX shape (1, gin_channels, 1). Returns same data as 3D tensor dimensions [1, C, 1].
    /// </summary>
    public static float[] ToOnnxShape(float[] seFlat)
    {
        return seFlat; // ONNX expects [1, C, 1]; we pass flat C and create tensor with shape [1, C, 1]
    }
}
