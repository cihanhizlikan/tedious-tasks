using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TediousTasks;

/// <summary>
/// Hand-crafted pixel-feature classification engine (Engine B).
///
/// Features and weights calibrated across three rounds of false-positive analysis
/// (round 1: 162+85 images; round 2: 1049+84 images; round 3: 5+1681 images).
///
///   Feature              Direction  Weight  Notes
///   ──────────────────────────────────────────────────────────────────────────
///   ChannelNoise         real↑      0.22    Dominant signal, consistent all rounds
///   FlatNoise            real↑      0.15    Strong rounds 2+3
///   JpegBlockArtifact    anime↑     0.20    NEW: rescues compressed anime (round 3 1681 FRs)
///   GradientBimodality   anime↑     0.18    NEW: cel-shading hard transitions
///   InkOutline           anime↑     0.10    Consistent all rounds
///   LocalPalette         real↑      0.08    NEW: patch-level diversity, replaces global palette
///   EdgeBimodal          anime↑     0.04    Consistent but weak
///   FlatRegion           anime↑     0.02    Fragile (flipped round 1), kept at minimal weight
///   SkinDiscrete         anime↑     0.01    Fragile (flipped round 1), kept at minimal weight
///
/// REMOVED (data-driven decision):
///   Saturation  — direction flipped in round 3 (real↑ instead of anime↑). Unreliable.
///   ColorTemp   — near-zero Cohen's d in 2/3 rounds. Dead weight.
///   Palette     — replaced by LocalPalette (patch-level is far more robust).
///
/// Thresholds (set by ImageClassifier):
///   Consensus mode (both engines): 0.64 — conservative, minimises false cartoons
///   Standalone mode (heuristic only): 0.38 — balanced, catches ~78% of anime
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
        ChannelNoise       = ChannelNoiseScore(px),
        FlatNoise          = FlatRegionNoiseScore(px),
        InkOutline         = InkOutlineScore(px),
        EdgeBimodal        = EdgeSharpnessScore(px),
        FlatRegion         = FlatRegionScore(px),
        SkinDiscrete       = SkinDiscretenessScore(px),
        Palette            = GlobalPaletteScore(px),        // retained for CSV continuity
        JpegBlockArtifact  = JpegBlockArtifactScore(px),
        GradientBimodality = GradientBimodalityScore(px),
        LocalPalette       = LocalPaletteScore(px),
    };

    /// <summary>
    /// Weighted composite cartoon score in [0, 1].
    /// Real↑ features are inverted so that higher score always means "more cartoon".
    /// </summary>
    public static double ScoreFeatures(ImageFeatures f) =>
          0.22 * (1.0 - f.ChannelNoise)      // real↑ → invert
        + 0.15 * (1.0 - f.FlatNoise)         // real↑ → invert
        + 0.20 * f.JpegBlockArtifact         // anime↑ NEW
        + 0.18 * f.GradientBimodality        // anime↑ NEW
        + 0.10 * f.InkOutline                // anime↑
        + 0.08 * (1.0 - f.LocalPalette)      // real↑ → invert NEW
        + 0.04 * f.EdgeBimodal               // anime↑
        + 0.02 * f.FlatRegion                // anime↑ (fragile)
        + 0.01 * f.SkinDiscrete;             // anime↑ (fragile)

    public static string FormatReason(double composite, ImageFeatures f) =>
        $"score={composite:F3} " +
        $"[chnoise={f.ChannelNoise:F2}↓ fnoise={f.FlatNoise:F2}↓ " +
        $"jpeg={f.JpegBlockArtifact:F2} grad={f.GradientBimodality:F2} " +
        $"outline={f.InkOutline:F2} lpalet={f.LocalPalette:F2}↓ " +
        $"edge={f.EdgeBimodal:F2} flat={f.FlatRegion:F2} skin={f.SkinDiscrete:F2}]";

    // ─────────────────────────────────────────────────────────────────────────
    // Retained features
    // ─────────────────────────────────────────────────────────────────────────

    // ── ChannelNoise – RGB channel independence (real↑) ──────────────────────
    // Camera sensor noise is channel-independent: in flat regions, R−G and R−B
    // vary randomly per pixel. Cartoon/anime fills have channels locked together.
    // JPEG compression introduces correlated noise (all channels shift together
    // inside each 8×8 block), so compressed anime still differs from real photos.
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

    // ── FlatNoise – micro-noise in flat regions (real↑) ──────────────────────
    // Real camera sensors leave measurable luminance noise even in visually flat
    // areas. Cartoon/anime flat fills are mathematically smooth.
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

    // ── InkOutline – dark lines adjacent to bright fill (anime↑) ────────────
    // Anime always has dark ink outlines directly bordering bright fill areas.
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

    // ── EdgeBimodal – Sobel edge bimodality (anime↑) ─────────────────────────
    // Anime edges are either very sharp (outlines) or completely flat (fills),
    // producing a bimodal distribution. Real photos have many mid-level gradients.
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

    // ── FlatRegion – 3×3 neighbourhood uniformity (anime↑, fragile) ─────────
    // Anime fill areas have near-zero local colour variance. Direction was
    // inconsistent in round 1; rounds 2+3 agree anime↑. Kept at low weight.
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

    // ── SkinDiscrete – skin-range pixel concentration (anime↑, fragile) ──────
    // Whether anime or real photos score higher depends on image content
    // (portrait vs landscape). Kept at minimal weight.
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

        int top4 = skinBuckets.Values.OrderByDescending(v => v).Take(4).Sum();
        return Math.Max(0.0, Math.Min(1.0, ((double)top4 / skinPixels - 0.45) / 0.40));
    }

    private static bool IsSkinTone(byte r, byte g, byte b)
    {
        if (r <= 100 || r <= g || r <= b || g <= 50 || b <= 30 || (r - b) <= 20)
            return false;
        float max = MathF.Max(r, MathF.Max(g, b)) / 255f;
        float min = MathF.Min(r, MathF.Min(g, b)) / 255f;
        return (max < 1e-6f ? 0f : (max - min) / max) > 0.08f && (max + min) / 2f > 0.25f;
    }

    // ── GlobalPalette – retained for CSV continuity only (real↑) ─────────────
    // Replaced in the scoring function by LocalPalette (more robust).
    // Still computed so historical CSVs remain comparable.
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
        return 1.0 - Math.Min(1.0, (double)buckets.Count / (1 << (3 * Bits)) / 0.12);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // New features
    // ─────────────────────────────────────────────────────────────────────────

    // ── JpegBlockArtifact – 8-pixel periodic boundary noise (anime↑) ─────────
    //
    // JPEG and GIF compression divide the image into 8×8 pixel blocks and
    // encode each independently. This creates subtle luminance discontinuities
    // AT the block boundaries (multiples of 8 pixels) that are larger than the
    // discontinuities WITHIN blocks.
    //
    // For REAL PHOTOS: the image has natural continuous texture everywhere. The
    // ratio of 8-pixel-step differences to 1-pixel-step differences is close to
    // 1.0 — the compression adds uniform noise on top of existing texture.
    //
    // For COMPRESSED ANIME: the flat fill areas within a block are nearly uniform,
    // but the BOUNDARY between two blocks containing different fill colours creates
    // a large jump at exactly the 8-pixel interval. The ratio spikes above 1.0.
    //
    // This directly distinguishes "compressed anime with high ChannelNoise"
    // (which mimics real photos) from actual real photos, fixing the 1681 hard FRs.
    //
    // Returns values in [0,1] where higher = more block-periodic = more anime.
    private static double JpegBlockArtifactScore(PixelBuffer px)
    {
        const int BlockSize  = 8;
        const int SampleStep = 2;

        double sumBlock    = 0;  // avg |lum[x] - lum[x+BlockSize]| at block boundaries
        double sumAdjacent = 0;  // avg |lum[x] - lum[x+1]| at the same positions
        int    count       = 0;

        // Sample horizontal block boundaries (columns that are multiples of 8)
        for (int y = 0; y < px.Height; y += SampleStep)
        for (int bx = BlockSize; bx < px.Width - BlockSize; bx += BlockSize)
        {
            // Compare luminance across the block boundary (BlockSize apart)
            float lumL  = px.Lum(bx - 1, y);   // last pixel of left block
            float lumR  = px.Lum(bx,     y);   // first pixel of right block
            float lumR2 = px.Lum(bx + 1, y);   // second pixel of right block

            sumBlock    += MathF.Abs(lumL - lumR);  // boundary jump
            sumAdjacent += MathF.Abs(lumR - lumR2); // within-block step
            count++;
        }

        // Sample vertical block boundaries (rows that are multiples of 8)
        for (int x = 0; x < px.Width; x += SampleStep)
        for (int by = BlockSize; by < px.Height - BlockSize; by += BlockSize)
        {
            float lumT  = px.Lum(x, by - 1);
            float lumB  = px.Lum(x, by);
            float lumB2 = px.Lum(x, by + 1);

            sumBlock    += MathF.Abs(lumT - lumB);
            sumAdjacent += MathF.Abs(lumB - lumB2);
            count++;
        }

        if (count < 20) return 0.5;

        double avgBlock    = sumBlock    / count;
        double avgAdjacent = sumAdjacent / count + 0.5; // +0.5 avoids div-by-zero

        // Ratio > 1 means block boundaries are NOISIER than interior → anime signal
        // Ratio ≈ 1 means uniform noise throughout → real photo signal
        // Clamp: ratio of 2.5+ = strongly periodic = score 1.0
        double ratio = avgBlock / avgAdjacent;
        return Math.Max(0.0, Math.Min(1.0, (ratio - 1.0) / 1.5));
    }

    // ── GradientBimodality – colour transition distribution shape (anime↑) ───
    //
    // This is the fundamental rendering difference between anime and photography:
    //
    // ANIME (cel-shading): pixels are either in a flat fill (near-zero change to
    // neighbour) or crossing an outline/shading boundary (large sudden change).
    // The distribution of inter-pixel colour change magnitudes is strongly BIMODAL:
    // a tall spike near zero, a gap in the middle, and a secondary peak at large values.
    //
    // REAL PHOTOS: lighting gradients, texture, depth-of-field blur all produce
    // smooth continuous variation. The distribution is UNIMODAL — a bell curve
    // centred around small-to-moderate changes, tailing off at both extremes.
    //
    // We measure bimodality via the "dip" between the two modes:
    //   1. Compute all horizontal inter-pixel luminance differences.
    //   2. Build a histogram with 32 buckets.
    //   3. Find the peak in the lower half (flat fill bucket).
    //   4. Find the minimum count between that peak and the right half (the dip).
    //   5. Score = 1 - (dip / peak) clamped to [0,1].
    //      Deep dip → high score (bimodal → anime).
    //      No dip (smooth distribution) → low score (real photo).
    //
    // Returns values in [0,1] where higher = more bimodal = more anime.
    private static double GradientBimodalityScore(PixelBuffer px)
    {
        const int  NumBuckets  = 32;
        const int  SampleStep  = 2;
        const float MaxDiff    = 128f;  // clamp diffs above this (large jumps all go to last bucket)

        var hist = new int[NumBuckets];
        int total = 0;

        for (int y = 0; y < px.Height; y += SampleStep)
        for (int x = 0; x < px.Width - 1; x += SampleStep)
        {
            float diff = MathF.Abs(px.Lum(x, y) - px.Lum(x + 1, y));
            int   bin  = (int)Math.Min(diff / MaxDiff * NumBuckets, NumBuckets - 1);
            hist[bin]++;
            total++;
        }

        if (total < 100) return 0.5;

        // Find the peak in the lower third of the histogram (the "flat fill" mode)
        int lowerThird = NumBuckets / 3;
        int peakBin    = 0;
        int peakCount  = 0;
        for (int i = 0; i < lowerThird; i++)
            if (hist[i] > peakCount) { peakCount = hist[i]; peakBin = i; }

        if (peakCount == 0) return 0.0;

        // Find the minimum count between the peak and the upper half (the "dip")
        int upperHalf = NumBuckets / 2;
        int dipCount  = peakCount;
        for (int i = peakBin + 1; i < upperHalf; i++)
            if (hist[i] < dipCount) dipCount = hist[i];

        // Bimodality index: deep dip relative to peak = strongly bimodal
        double bimodalIndex = 1.0 - (double)dipCount / peakCount;

        // Scale: index > 0.85 is strongly bimodal (score → 1.0)
        return Math.Max(0.0, Math.Min(1.0, bimodalIndex / 0.85));
    }

    // ── LocalPalette – patch-level colour diversity (real↑) ──────────────────
    //
    // The global palette feature counted distinct colours across the entire image,
    // which was inconsistent (varied wildly with scene complexity). Local palette
    // analysis tiles the image into patches and counts distinct colours PER PATCH.
    //
    // ANIME: even a "complex" scene has fills — each small patch is dominated by
    // 1–3 flat colours. The average distinct-colour count per patch stays low.
    //
    // REAL PHOTOS: even visually "flat" areas (clear sky, plain wall) contain
    // continuous texture, lighting gradients, and sensor noise that produce many
    // distinct colours per patch, even after quantisation.
    //
    // Returns values in [0,1] where higher = more colours per patch = more real.
    private static double LocalPaletteScore(PixelBuffer px)
    {
        const int PatchSize  = 24;      // pixels per patch side
        const int Bits       = 4;       // quantise to 4 bits per channel (16 levels)
        const int Shift      = 8 - Bits;

        int   patchCount  = 0;
        double totalRatio = 0;

        for (int py = 0; py + PatchSize <= px.Height; py += PatchSize)
        for (int px2 = 0; px2 + PatchSize <= px.Width; px2 += PatchSize)
        {
            var patchBuckets = new HashSet<int>();
            for (int dy = 0; dy < PatchSize; dy++)
            for (int dx = 0; dx < PatchSize; dx++)
            {
                int x = px2 + dx, y = py + dy;
                int key = ((px.R(x, y) >> Shift) << (2 * Bits)) |
                          ((px.G(x, y) >> Shift) <<      Bits)  |
                           (px.B(x, y) >> Shift);
                patchBuckets.Add(key);
            }

            // Max possible distinct colours in a patch with 4-bit quantisation
            int maxPossible = 1 << (3 * Bits);  // 4096
            totalRatio += (double)patchBuckets.Count / maxPossible;
            patchCount++;
        }

        if (patchCount == 0) return 0.5;

        double avgRatio = totalRatio / patchCount;

        // Calibration from analysis:
        // Anime patches: avg ratio typically 0.002–0.015 (very few colours per patch)
        // Real photo patches: avg ratio typically 0.025–0.120
        // Normalise: 0.01 → score 0, 0.06 → score 1
        return Math.Max(0.0, Math.Min(1.0, (avgRatio - 0.01) / 0.05));
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
