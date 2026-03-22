using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TediousTasks;

/// <summary>
/// Hand-crafted pixel-feature classification engine (Engine B).
///
/// Features and weights calibrated across four rounds of false-positive analysis.
///
///   Feature          Direction  Weight  Consistency
///   ──────────────────────────────────────────────────────────────────────────
///   ChannelNoise     real↑      0.40    ★★★ dominant, consistent all rounds
///   GlobalPalette    real↑      0.25    ★★★ consistent rounds 1,3,4; low R2
///   FlatRegion       real↑      0.15    ★★  real↑ in R1,R4; anime↑ in R2,R3 — fragile
///   FlatNoise        real↑      0.08    ★★  consistent R2,R3,R4
///   EdgeBimodal      anime↑     0.07    ★★  consistent all rounds
///   InkOutline       anime↑     0.05    ★   consistent direction, weak magnitude
///
/// PERMANENTLY REMOVED (data-driven):
///   Saturation       — direction flipped R3. Dataset-dependent. Gone.
///   ColorTemp        — Cohen's d ~0 in 2/3 rounds. Gone.
///   SkinDiscrete     — direction flipped R1→R2. Fragile and ~0 separation in R4. Gone.
///   JpegBlockArtifact— conceptually sound but DESTROYED by the LoadAndResize step.
///                      Bicubic interpolation erases JPEG block boundaries.
///                      Scored 0.0 for 100% of images. Gone.
///   GradientBimodality— calibration failure: scored 1.0 for 100% of BOTH groups.
///                      Added +0.18 constant to every score regardless of content.
///                      Gone.
///   LocalPalette     — scored ~0.001 for 100% of images; normalisation was off
///                      by ~20x. GlobalPalette is more reliable. Gone.
///
/// Thresholds (passed by ImageClassifier):
///   Consensus mode (both engines): 0.64
///   Standalone mode (heuristic only): 0.66 — raised because the 49 hard false-reals
///     all have chnoise≈1.0 and cannot be caught by the heuristic regardless of
///     threshold; setting 0.66 achieves 0% false cartoons on the full dataset sample.
/// </summary>
internal static class HeuristicEngine
{
    // ── Public interface ──────────────────────────────────────────────────────

    public static bool IsCartoon(string filePath, double threshold, out string reason)
    {
        using var bmp    = ImageUtils.LoadAndResize(filePath, maxDimension: 512);
        var px            = new PixelBuffer(bmp);
        ImageFeatures f   = ComputeFeatures(px);
        double composite  = ScoreFeatures(f);
        reason            = FormatReason(composite, f);
        return composite >= threshold;
    }

    public static ImageFeatures ComputeFeatures(PixelBuffer px) => new()
    {
        ChannelNoise = ChannelNoiseScore(px),
        FlatNoise    = FlatRegionNoiseScore(px),
        InkOutline   = InkOutlineScore(px),
        EdgeBimodal  = EdgeSharpnessScore(px),
        FlatRegion   = FlatRegionScore(px),
        Palette      = GlobalPaletteScore(px),
    };

    /// <summary>
    /// Weighted composite cartoon score in [0, 1].
    /// All real↑ features are inverted so higher always means "more cartoon".
    /// </summary>
    public static double ScoreFeatures(ImageFeatures f) =>
          0.40 * (1.0 - f.ChannelNoise)   // real↑ → invert  dominant
        + 0.25 * (1.0 - f.Palette)        // real↑ → invert  strong
        + 0.15 * (1.0 - f.FlatRegion)     // real↑ → invert  fragile but net useful
        + 0.08 * (1.0 - f.FlatNoise)      // real↑ → invert
        + 0.07 * f.EdgeBimodal            // anime↑
        + 0.05 * f.InkOutline;            // anime↑

    public static string FormatReason(double composite, ImageFeatures f) =>
        $"score={composite:F3} " +
        $"[chnoise={f.ChannelNoise:F2}↓ palette={f.Palette:F2}↓ " +
        $"flat={f.FlatRegion:F2}↓ fnoise={f.FlatNoise:F2}↓ " +
        $"edge={f.EdgeBimodal:F2} outline={f.InkOutline:F2}]";

    // ─────────────────────────────────────────────────────────────────────────
    // Features
    // ─────────────────────────────────────────────────────────────────────────

    // ── ChannelNoise – RGB channel independence (real↑, weight 0.40) ─────────
    // Camera sensor noise is channel-independent: R−G and R−B differences between
    // adjacent pixels in flat regions vary randomly and independently.
    // Cartoon fills have channels locked together (same fill colour, or JPEG
    // block artefacts shift all channels together). Consistent dominant signal
    // across all four rounds of data.
    private static double ChannelNoiseScore(PixelBuffer px)
    {
        const int   SampleStep  = 3;
        const float MaxLumDelta = 25f;
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

    // ── GlobalPalette – total colour diversity (real↑, weight 0.25) ──────────
    // Real photos use many more distinct colours than cartoon flat fills.
    // Quantise to 5-bit per channel and count occupied buckets.
    // Strong separator in rounds 1, 3, and 4; inconsistent in round 2.
    private static double GlobalPaletteScore(PixelBuffer px)
    {
        const int Bits = 5, Shift = 8 - Bits;
        var buckets = new HashSet<int>();
        for (int y = 0; y < px.Height; y++)
        for (int x = 0; x < px.Width;  x++)
            buckets.Add(
                ((px.R(x, y) >> Shift) << (2 * Bits)) |
                ((px.G(x, y) >> Shift) <<      Bits)  |
                 (px.B(x, y) >> Shift));
        // Real photos: typically fill 15–50% of the 32768-bucket space.
        // Anime: typically 2–10%.
        return 1.0 - Math.Min(1.0, (double)buckets.Count / (1 << (3 * Bits)) / 0.12);
    }

    // ── FlatRegion – 3×3 neighbourhood uniformity (real↑, weight 0.15) ───────
    // Real photos score higher in rounds 1 and 4 (camera blur, overexposed skin,
    // plain backgrounds). Anime scored higher in rounds 2 and 3 (cel-shading fills).
    // Direction is fragile and dataset-dependent. Kept at moderate weight because
    // it's strongly separating in the current round 4 data (FC=0.879, FR=0.430).
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
                int d = Math.Abs(cr - px.R(x+dx, y+dy)); if (d > maxDelta) maxDelta = d;
                    d = Math.Abs(cg - px.G(x+dx, y+dy)); if (d > maxDelta) maxDelta = d;
                    d = Math.Abs(cb - px.B(x+dx, y+dy)); if (d > maxDelta) maxDelta = d;
            }
            if (maxDelta <= Threshold) flat++;
            total++;
        }
        return Math.Max(0.0, Math.Min(1.0, ((double)flat / total - 0.30) / 0.45));
    }

    // ── FlatNoise – micro-noise in flat regions (real↑, weight 0.08) ─────────
    // Real camera sensors leave measurable luminance noise in visually flat areas.
    // Cartoon/anime flat fills are mathematically smooth.
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

        return noiseValues.Count < 10 ? 0.5 : Math.Min(1.0, noiseValues.Average() / 5.0);
    }

    // ── EdgeBimodal – Sobel edge bimodality (anime↑, weight 0.07) ────────────
    // Anime edges are either very sharp (outlines) or completely flat (fills).
    // Real photos have many mid-level gradients producing a unimodal distribution.
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
        return Math.Min(1.0, (double)highEdge / Math.Max(0.01, highEdge + midEdge) * 1.4);
    }

    // ── InkOutline – dark lines adjacent to bright fill (anime↑, weight 0.05) ─
    // Anime has dark ink outlines directly bordering bright fill areas.
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

    // ── Helpers ───────────────────────────────────────────────────────────────

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
// PixelBuffer – LockBits-based pixel accessor (zero GetPixel calls).
// Layout: Format32bppArgb → B G R A per pixel; stride may include row padding.
// ─────────────────────────────────────────────────────────────────────────────
internal sealed class PixelBuffer
{
    public readonly byte[] Data;
    public readonly int    Width, Height, Stride;

    public PixelBuffer(Bitmap bmp)
    {
        Width  = bmp.Width;
        Height = bmp.Height;
        BitmapData bd = bmp.LockBits(new Rectangle(0, 0, Width, Height),
                                     ImageLockMode.ReadOnly,
                                     PixelFormat.Format32bppArgb);
        try
        {
            Stride = Math.Abs(bd.Stride);
            Data   = new byte[Stride * Height];
            Marshal.Copy(bd.Scan0, Data, 0, Data.Length);
        }
        finally
        {
            bmp.UnlockBits(bd);
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
