using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TediousTasks;

/// <summary>
/// Hand-crafted pixel-feature classification engine (Engine B).
///
/// Nine features, weights calibrated from 162 false-cartoon and 85 false-real images:
///
///   Feature            Direction    Weight    Notes
///   ─────────────────────────────────────────────────────────────────────
///   ChannelNoise       real↑        0.30      dominant signal
///   InkOutline         anime↑       0.20
///   Saturation         anime↑       0.15
///   Palette            real↑        0.13      inverted: real photos use MORE colours
///   FlatRegion         real↑        0.10      inverted: real photos are FLATTER
///   EdgeBimodal        anime↑       0.05
///   ColorTemp          real↑        0.04      inverted: camera WB makes real MORE uniform
///   SkinDiscrete       real↑        0.02      inverted: real photos have more skin area
///   FlatNoise          real↑        0.01      near-zero separating power; kept for completeness
/// </summary>
internal static class HeuristicEngine
{
    // ── Scoring ───────────────────────────────────────────────────────────────

    public static bool IsCartoon(string filePath, out string reason)
    {
        using var bmp     = ImageUtils.LoadAndResize(filePath, maxDimension: 512);
        var px             = new PixelBuffer(bmp);
        ImageFeatures f    = ComputeFeatures(px);
        double composite   = ScoreFeatures(f);
        reason             = FormatReason(composite, f);
        return composite  >= ImageClassifier.HeuristicCartoonThreshold;
    }

    public static ImageFeatures ComputeFeatures(PixelBuffer px) => new()
    {
        Palette      = PaletteScore(px),
        Saturation   = SaturationScore(px),
        FlatRegion   = FlatRegionScore(px),
        EdgeBimodal  = EdgeSharpnessScore(px),
        InkOutline   = InkOutlineScore(px),
        SkinDiscrete = SkinDiscretenessScore(px),
        FlatNoise    = FlatRegionNoiseScore(px),
        ColorTemp    = ColourTemperatureScore(px),
        ChannelNoise = ChannelNoiseScore(px),
    };

    public static double ScoreFeatures(ImageFeatures f) =>
        // Inverted features (↑ = real) contribute (1 − value) to the cartoon score.
          0.30 * (1.0 - f.ChannelNoise)
        + 0.20 * f.InkOutline
        + 0.15 * f.Saturation
        + 0.13 * (1.0 - f.Palette)
        + 0.10 * (1.0 - f.FlatRegion)
        + 0.05 * f.EdgeBimodal
        + 0.04 * (1.0 - f.ColorTemp)
        + 0.02 * (1.0 - f.SkinDiscrete)
        + 0.01 * (1.0 - f.FlatNoise);

    public static string FormatReason(double composite, ImageFeatures f) =>
        $"score={composite:F3} " +
        $"[chnoise={f.ChannelNoise:F2}↓ outline={f.InkOutline:F2} sat={f.Saturation:F2} " +
        $"palette={f.Palette:F2}↓ flat={f.FlatRegion:F2}↓ edge={f.EdgeBimodal:F2} " +
        $"colortemp={f.ColorTemp:F2}↓ skin={f.SkinDiscrete:F2}↓ noise={f.FlatNoise:F2}↓]";

    // ── Feature 1 – Colour palette diversity ─────────────────────────────────
    // Real photos have more distinct colours than cartoon fills.
    private static double PaletteScore(PixelBuffer px)
    {
        const int Bits = 5, Shift = 8 - Bits;
        var buckets = new HashSet<int>();
        for (int y = 0; y < px.Height; y++)
        for (int x = 0; x < px.Width;  x++)
            buckets.Add(
                ((px.R(x, y) >> Shift) << (2 * Bits)) |
                ((px.G(x, y) >> Shift) <<      Bits)  |
                 (px.B(x, y) >> Shift));
        return 1.0 - Math.Min(1.0, (double)buckets.Count / (1 << (3 * Bits)) / 0.12);
    }

    // ── Feature 2 – HSV saturation ────────────────────────────────────────────
    // Anime colours are more vivid than real-world photographs.
    private static double SaturationScore(PixelBuffer px)
    {
        double total = 0;
        for (int y = 0; y < px.Height; y++)
        for (int x = 0; x < px.Width;  x++)
            total += px.Sat(x, y);
        return Math.Min(1.0, total / (px.Width * px.Height) / 0.65);
    }

    // ── Feature 3 – Flat region ratio ────────────────────────────────────────
    // Counter-intuitively real photos score higher here because camera blur
    // and overexposed skin produce very flat luminance regions.
    private static double FlatRegionScore(PixelBuffer px)
    {
        const int Threshold = 12;
        int flat = 0, total = 0;
        for (int y = 1; y < px.Height - 1; y++)
        for (int x = 1; x < px.Width  - 1; x++)
        {
            byte cr = px.R(x, y), cg = px.G(x, y), cb = px.B(x, y);
            int maxDelta = 0;
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0) continue;
                int d = Math.Abs(cr - px.R(x + dx, y + dy)); if (d > maxDelta) maxDelta = d;
                    d = Math.Abs(cg - px.G(x + dx, y + dy)); if (d > maxDelta) maxDelta = d;
                    d = Math.Abs(cb - px.B(x + dx, y + dy)); if (d > maxDelta) maxDelta = d;
            }
            if (maxDelta <= Threshold) flat++;
            total++;
        }
        return Math.Max(0.0, Math.Min(1.0, ((double)flat / total - 0.30) / 0.45));
    }

    // ── Feature 4 – Edge bimodality (Sobel) ──────────────────────────────────
    // Anime edges are either very sharp (outlines) or completely flat (fills).
    private static double EdgeSharpnessScore(PixelBuffer px)
    {
        int w = px.Width, h = px.Height;
        var grey = new float[h * w];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            grey[y * w + x] = px.Lum(x, y);

        int highEdge = 0, midEdge = 0;
        for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++)
        {
            float gx = grey[(y-1)*w+(x+1)] + 2*grey[y*w+(x+1)] + grey[(y+1)*w+(x+1)]
                     - grey[(y-1)*w+(x-1)] - 2*grey[y*w+(x-1)] - grey[(y+1)*w+(x-1)];
            float gy = grey[(y+1)*w+(x-1)] + 2*grey[(y+1)*w+x] + grey[(y+1)*w+(x+1)]
                     - grey[(y-1)*w+(x-1)] - 2*grey[(y-1)*w+x] - grey[(y-1)*w+(x+1)];
            float mag = MathF.Sqrt(gx * gx + gy * gy);
            if      (mag > 80) highEdge++;
            else if (mag > 20) midEdge++;
        }
        return Math.Min(1.0,
            (double)highEdge / Math.Max(0.01, highEdge + midEdge) * 1.4);
    }

    // ── Feature 5 – Ink outline detection ────────────────────────────────────
    // Anime always has dark ink outlines directly adjacent to bright fill areas.
    private static double InkOutlineScore(PixelBuffer px)
    {
        const float DarkThresh = 80f, BrightThresh = 160f;
        int outlinePixels = 0, total = 0;
        for (int y = 1; y < px.Height - 1; y++)
        for (int x = 1; x < px.Width  - 1; x++)
        {
            if (px.Lum(x, y) < DarkThresh)
            {
                bool hasBright = false;
                for (int dy = -1; dy <= 1 && !hasBright; dy++)
                for (int dx = -1; dx <= 1 && !hasBright; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    if (px.Lum(x + dx, y + dy) > BrightThresh) hasBright = true;
                }
                if (hasBright) outlinePixels++;
            }
            total++;
        }
        return Math.Min(1.0, (double)outlinePixels / total / 0.04);
    }

    // ── Feature 6 – Skin tone discreteness ───────────────────────────────────
    // Inverted: real photos contain more detected skin area overall.
    private static double SkinDiscretenessScore(PixelBuffer px)
    {
        const int BucketBits = 4, Shift = 8 - BucketBits;
        var skinBuckets = new Dictionary<int, int>();
        int skinPixels  = 0;

        for (int y = 0; y < px.Height; y++)
        for (int x = 0; x < px.Width;  x++)
        {
            byte r = px.R(x, y), g = px.G(x, y), b = px.B(x, y);
            if (!IsSkinTone(r, g, b)) continue;
            skinPixels++;
            int key = ((r >> Shift) << (2 * BucketBits)) |
                      ((g >> Shift) <<      BucketBits)  |
                       (b >> Shift);
            skinBuckets[key] = skinBuckets.GetValueOrDefault(key) + 1;
        }

        if (skinPixels < 200) return 0.0;

        int top4 = skinBuckets.Values
                               .OrderByDescending(v => v)
                               .Take(4)
                               .Sum();
        return Math.Max(0.0, Math.Min(1.0, ((double)top4 / skinPixels - 0.45) / 0.40));
    }

    private static bool IsSkinTone(byte r, byte g, byte b)
    {
        if (r <= 100 || r <= g || r <= b || g <= 50 || b <= 30 || (r - b) <= 20)
            return false;
        float max = MathF.Max(r, MathF.Max(g, b)) / 255f;
        float min = MathF.Min(r, MathF.Min(g, b)) / 255f;
        float sat = max < 1e-6f ? 0f : (max - min) / max;
        float bri = (max + min) / 2f;
        return sat > 0.08f && bri > 0.25f;
    }

    // ── Feature 7 – Flat-region micro-noise ──────────────────────────────────
    // Real camera sensors leave measurable noise even in flat-looking areas.
    // Near-zero separating power in practice; weight is 0.01.
    private static double FlatRegionNoiseScore(PixelBuffer px)
    {
        const int   PatchRadius = 3, SampleStep = 4;
        const float MaxRange    = 20f;
        var noiseValues = new List<double>();

        for (int y = PatchRadius; y < px.Height - PatchRadius; y += SampleStep)
        for (int x = PatchRadius; x < px.Width  - PatchRadius; x += SampleStep)
        {
            float lumMin = float.MaxValue, lumMax = float.MinValue, lumSum = 0;
            int   count  = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++)
            for (int dx = -PatchRadius; dx <= PatchRadius; dx++)
            {
                float l = px.Lum(x + dx, y + dy);
                if (l < lumMin) lumMin = l;
                if (l > lumMax) lumMax = l;
                lumSum += l;
                count++;
            }

            if (lumMax - lumMin > MaxRange) continue;

            float  mean     = lumSum / count;
            double variance = 0;
            for (int dy = -PatchRadius; dy <= PatchRadius; dy++)
            for (int dx = -PatchRadius; dx <= PatchRadius; dx++)
            {
                float d = px.Lum(x + dx, y + dy) - mean;
                variance += d * d;
            }
            noiseValues.Add(Math.Sqrt(variance / count));
        }

        return noiseValues.Count < 10
            ? 0.5
            : Math.Min(1.0, noiseValues.Average() / 5.0);
    }

    // ── Feature 8 – Colour temperature uniformity ────────────────────────────
    // Inverted: camera auto-white-balance makes real photos more uniform in R/B ratio.
    private static double ColourTemperatureScore(PixelBuffer px)
    {
        const int SampleStep = 4;
        var ratios = new List<double>();

        for (int y = 0; y < px.Height; y += SampleStep)
        for (int x = 0; x < px.Width;  x += SampleStep)
            ratios.Add((double)px.R(x, y) / (px.B(x, y) + 1));

        double mean   = ratios.Average();
        double stdDev = Math.Sqrt(ratios.Average(r => (r - mean) * (r - mean)));
        return Math.Max(0.0, Math.Min(1.0, 1.0 - stdDev / 0.7));
    }

    // ── Feature 9 – RGB channel noise independence ───────────────────────────
    // Dominant real signal: camera sensor noise is channel-independent. In flat
    // regions R−G and R−B vary randomly per pixel. Cartoon fills stay locked.
    private static double ChannelNoiseScore(PixelBuffer px)
    {
        const int   SampleStep   = 3;
        const float MaxLumDelta  = 25f;
        var rgDiffs = new List<float>();
        var rbDiffs = new List<float>();

        for (int y = 0; y < px.Height;     y += SampleStep)
        for (int x = 0; x < px.Width - 1; x += SampleStep)
        {
            if (MathF.Abs(px.Lum(x, y) - px.Lum(x + 1, y)) > MaxLumDelta) continue;
            rgDiffs.Add((px.R(x+1,y) - px.G(x+1,y)) - (px.R(x,y) - px.G(x,y)));
            rbDiffs.Add((px.R(x+1,y) - px.B(x+1,y)) - (px.R(x,y) - px.B(x,y)));
        }

        if (rgDiffs.Count < 50) return 0.5;

        return Math.Min(1.0, (Variance(rgDiffs) + Variance(rbDiffs)) / 2.0 / 25.0);
    }

    private static double Variance(List<float> values)
    {
        double mean = 0;
        foreach (var v in values) mean += v;
        mean /= values.Count;
        double variance = 0;
        foreach (var v in values) variance += (v - mean) * (v - mean);
        return variance / values.Count;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PixelBuffer – LockBits-based pixel accessor (zero GetPixel calls)
//
// Layout in memory for Format32bppArgb: B G R A per pixel, stride may include
// row-end padding. All feature methods operate on this buffer directly.
// ─────────────────────────────────────────────────────────────────────────────
internal sealed class PixelBuffer
{
    public readonly byte[] Data;
    public readonly int    Width, Height, Stride;

    public PixelBuffer(Bitmap bmp)
    {
        Width  = bmp.Width;
        Height = bmp.Height;

        var rect = new Rectangle(0, 0, Width, Height);
        BitmapData bd = bmp.LockBits(rect, ImageLockMode.ReadOnly,
                                     PixelFormat.Format32bppArgb);
        try
        {
            Stride = Math.Abs(bd.Stride);
            Data   = new byte[Stride * Height];
            Marshal.Copy(bd.Scan0, Data, 0, Data.Length);
        }
        finally
        {
            bmp.UnlockBits(bd);   // guaranteed even if Marshal.Copy throws
        }
    }

    public byte  B(int x, int y) => Data[y * Stride + x * 4];
    public byte  G(int x, int y) => Data[y * Stride + x * 4 + 1];
    public byte  R(int x, int y) => Data[y * Stride + x * 4 + 2];

    public float Lum(int x, int y) =>
        0.299f * R(x, y) + 0.587f * G(x, y) + 0.114f * B(x, y);

    public float Sat(int x, int y)
    {
        float r = R(x, y) / 255f, g = G(x, y) / 255f, b = B(x, y) / 255f;
        float max = MathF.Max(r, MathF.Max(g, b));
        float min = MathF.Min(r, MathF.Min(g, b));
        return max < 1e-6f ? 0f : (max - min) / max;
    }
}
