using System;

namespace MeloTTS.Onnx;

/// <summary>
/// DSP helpers for tone converter: real FFT magnitude, spectrogram (reflect pad + Hann), linear resample.
/// Matches OpenVoice v2 / tone_converter_onnx.py behavior.
/// </summary>
public static class AudioDsp
{
    /// <summary>
    /// Linear interpolation resample. Prefer a proper resampler in production for large rate changes.
    /// </summary>
    public static float[] ResampleLinear(float[] audio, int origSr, int targetSr)
    {
        if (origSr == targetSr)
            return (float[])audio.Clone();
        double duration = (double)audio.Length / origSr;
        int nOut = (int)Math.Round(duration * targetSr);
        var result = new float[nOut];
        for (int i = 0; i < nOut; i++)
        {
            double t = (double)i / targetSr * origSr;
            int i0 = (int)Math.Floor(t);
            if (i0 >= audio.Length - 1)
            {
                result[i] = audio[audio.Length - 1];
                continue;
            }
            if (i0 < 0) i0 = 0;
            float a = (float)(t - i0);
            result[i] = audio[i0] * (1f - a) + audio[i0 + 1] * a;
        }
        return result;
    }

    /// <summary>
    /// Magnitude spectrogram matching OpenVoice spectrogram_torch (reflect pad, Hann, onesided).
    /// y: mono float32 samples; returns [specFreq, nFrames] as single array (specFreq * nFrames).
    /// specFreq = nFft / 2 + 1.
    /// </summary>
    public static (float[] spec, int specFreq, int nFrames) Spectrogram(
        ReadOnlySpan<float> y,
        int nFft,
        int hopLength,
        int winLength)
    {
        int pad = (nFft - hopLength) / 2;
        int paddedLen = y.Length + 2 * pad;
        var yPad = new float[paddedLen];
        for (int i = 0; i < pad; i++)
            yPad[i] = y[Math.Min(pad - 1 - i, y.Length - 1)];
        for (int i = 0; i < y.Length; i++)
            yPad[pad + i] = y[i];
        for (int i = 0; i < pad; i++)
            yPad[pad + y.Length + i] = y[Math.Max(y.Length - 1 - i, 0)];

        var window = HannWindow(winLength);
        int nFrames = 1 + (paddedLen - winLength) / hopLength;
        int specFreq = nFft / 2 + 1;
        var outSpec = new float[specFreq * nFrames];

        var frame = new float[nFft];
        for (int i = 0; i < nFrames; i++)
        {
            int start = i * hopLength;
            for (int j = 0; j < winLength; j++)
                frame[j] = yPad[start + j] * window[j];
            for (int j = winLength; j < nFft; j++)
                frame[j] = 0;

            RealFftMagnitude(frame, nFft, outSpec, i * specFreq);
        }

        return (outSpec, specFreq, nFrames);
    }

    private static float[] HannWindow(int length)
    {
        var w = new float[length];
        for (int i = 0; i < length; i++)
            w[i] = (float)(0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1))));
        return w;
    }

    /// <summary>
    /// Compute magnitude of real FFT (first nFft/2+1 bins). Input real signal in frame[] (length nFft); output written to magnitude[] at offset.
    /// </summary>
    private static void RealFftMagnitude(float[] frame, int nFft, float[] magnitude, int offset)
    {
        // Complex FFT in-place: frame will hold [re0, im0, re1, im1, ...]. We use frame as interleaved re/im.
        int n = nFft;
        var re = new float[n];
        var im = new float[n];
        for (int i = 0; i < n; i++)
        {
            re[i] = frame[i];
            im[i] = 0;
        }
        Fft(re, im, n);
        int half = n / 2 + 1;
        const float eps = 1e-6f;
        for (int i = 0; i < half; i++)
        {
            float r = re[i], m = im[i];
            magnitude[offset + i] = (float)Math.Sqrt(r * r + m * m + eps);
        }
    }

    /// <summary>
    /// In-place complex FFT (radix-2). re and im arrays of length n (power of 2).
    /// </summary>
    private static void Fft(float[] re, float[] im, int n)
    {
        int bits = 0;
        for (int t = n; t > 1; t >>= 1) bits++;
        if (1 << bits != n)
            throw new ArgumentException("n must be power of 2", nameof(n));

        // Bit-reverse permutation
        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, bits);
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        for (int len = 2; len <= n; len *= 2)
        {
            float angle = (float)(-2 * Math.PI / len);
            float wLenRe = (float)Math.Cos(angle);
            float wLenIm = (float)Math.Sin(angle);
            for (int start = 0; start < n; start += len)
            {
                float wRe = 1, wIm = 0;
                for (int k = 0; k < len / 2; k++)
                {
                    int i = start + k;
                    int j = start + k + len / 2;
                    float tRe = wRe * re[j] - wIm * im[j];
                    float tIm = wRe * im[j] + wIm * re[j];
                    re[j] = re[i] - tRe;
                    im[j] = im[i] - tIm;
                    re[i] += tRe;
                    im[i] += tIm;
                    float nextWRe = wRe * wLenRe - wIm * wLenIm;
                    float nextWIm = wRe * wLenIm + wIm * wLenRe;
                    wRe = nextWRe;
                    wIm = nextWIm;
                }
            }
        }
    }

    private static int BitReverse(int x, int bits)
    {
        int r = 0;
        for (int i = 0; i < bits; i++)
        {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        return r;
    }
}
