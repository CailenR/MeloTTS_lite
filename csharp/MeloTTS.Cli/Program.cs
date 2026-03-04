using MeloTTS.Onnx;
using NAudio.Wave;

var argsList = args.ToList();
string? GetArg(string name)
{
    var i = argsList.IndexOf(name);
    if (i >= 0 && i + 1 < argsList.Count)
        return argsList[i + 1];
    return null;
}

var text = GetArg("--text");
var output = GetArg("--output") ?? "output.wav";
var speedStr = GetArg("--speed");
var model = GetArg("--model") ?? "model_en.onnx";
var lexicon = GetArg("--lexicon") ?? "lexicon.txt";
var tokens = GetArg("--tokens") ?? "tokens.txt";

// Tone conversion after TTS; defaults to converter.onnx, converter_config.json, base_se.json, target_se.json in current folder
var converterOnnx = GetArg("--converter") ?? "converter.onnx";
var configPath = GetArg("--config") ?? "converter_config.json";
var srcSe = GetArg("--src-se") ?? "base_se.json";
var tgtSe = GetArg("--tgt-se") ?? "target_se.json";
var tauStr = GetArg("--tau");
var tau = 0.3f;
if (!string.IsNullOrEmpty(tauStr) && float.TryParse(tauStr, out var tauParsed))
    tau = tauParsed;

bool useToneConverter = true; // always use converter with above defaults; override by passing different paths

var speed = 1.0f;
if (!string.IsNullOrEmpty(speedStr) && float.TryParse(speedStr, out var s))
    speed = s;

try
{
    var MeloEngine = new MeloTTSEngine(srcSePath: srcSe, tgtSePath: tgtSe, lexicon: lexicon, tokens: tokens);

    while (true)
    {
        if (string.IsNullOrEmpty(text))
        {
            Console.Write("Enter text to synthesize (empty to exit): ");
            text = Console.ReadLine()?.Trim();
        }

        if (string.IsNullOrEmpty(text))
        {
            Console.WriteLine("Done.");
            return 0;
        }
        var (audio, sampleRate) = MeloEngine.SynthesizeToBuffer(text, null, speed);
        var dir = Path.GetDirectoryName(output);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);
        var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        using (var writer = new WaveFileWriter(output, format))
            writer.WriteSamples(audio, 0, audio.Length);

        Console.WriteLine($"Wrote: {output} ({sampleRate} Hz)");
        text = null;
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 1;
}
