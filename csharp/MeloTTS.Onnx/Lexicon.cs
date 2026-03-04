using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MeloTTS.Onnx;

/// <summary>
/// Grapheme-to-phoneme lookup using lexicon and token files.
/// </summary>
public class Lexicon
{
    private readonly Dictionary<string, (int[] Phones, int[] Tones)> _lexicon = new();

    public Lexicon(string lexiconPath, string tokensPath)
    {
        var tokens = new Dictionary<string, int>();
        foreach (var line in File.ReadAllLines(tokensPath, System.Text.Encoding.UTF8))
        {
            var parts = line.Split(' ', 2, System.StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2)
                tokens[parts[0]] = int.Parse(parts[1]);
        }

        // Map "v" to "V" token ID (same as post_replace_ph in MeloTTS, only for English models)
        if (tokens.TryGetValue("V", out var vId) && vId == 14 && tokens.ContainsKey("v"))
            tokens["v"] = tokens["V"];

        foreach (var line in File.ReadAllLines(lexiconPath, System.Text.Encoding.UTF8))
        {
            var splits = line.Split(' ', System.StringSplitOptions.RemoveEmptyEntries);
            if (splits.Length < 2) continue;

            var wordOrPhrase = splits[0];
            var phoneToneList = splits.Skip(1).ToArray();
            if ((phoneToneList.Length & 1) != 0) continue;

            var half = phoneToneList.Length / 2;
            var phones = phoneToneList.Take(half).Select(p => tokens[p]).ToArray();
            var tones = phoneToneList.Skip(half).Select(t => int.Parse(t)).ToArray();
            _lexicon[wordOrPhrase] = (phones, tones);
        }

        var punctuation = new[] { "!", "?", "…", ",", ".", "'", "-" };
        foreach (var p in punctuation)
        {
            if (tokens.TryGetValue(p, out var i))
                _lexicon[p] = (new[] { i }, new[] { 0 });
        }
        if (tokens.TryGetValue("_", out var underscoreId))
            _lexicon[" "] = (new[] { underscoreId }, new[] { 0 });
    }

    private (List<int> Phones, List<int> Tones) ConvertSingle(string text, bool lowercase)
    {
        var phones = new List<int>();
        var tones = new List<int>();

        text = text switch
        {
            "，" => ",",
            "。" => ".",
            "！" => "!",
            "？" => "?",
            _ => text
        };

        var lookup = (lowercase && text.Length > 1) ? text.ToLowerInvariant() : text;
        if (!_lexicon.TryGetValue(lookup, out var entry))
        {
            if (text.Length > 1)
            {
                foreach (var c in text)
                {
                    var (p, t) = ConvertSingle(c.ToString(), false);
                    if (p.Count > 0)
                    {
                        phones.AddRange(p);
                        tones.AddRange(t);
                    }
                }
            }
            return (phones, tones);
        }

        phones.AddRange(entry.Phones);
        tones.AddRange(entry.Tones);
        return (phones, tones);
    }

    public (List<int> Phones, List<int> Tones) Convert(IEnumerable<string> textList, bool lowercase = true)
    {
        var phones = new List<int>();
        var tones = new List<int>();

        foreach (var text in textList)
        {
            var lookup = (lowercase && text.Length > 1) ? text.ToLowerInvariant() : text;
            var (p, t) = ConvertSingle(lookup, false);
            phones.AddRange(p);
            tones.AddRange(t);
        }

        return (phones, tones);
    }
}
