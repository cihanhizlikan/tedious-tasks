using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TediousTasks
{
    /// <summary>
    /// Classifies images as real photos or anime/cartoon using a dual-engine approach:
    ///
    ///   Engine A – ONNX (deepghs/anime_real_cls mobilenetv3_v1.4_dist, 98.78 % accuracy)
    ///   Engine B – Hand-crafted pixel-feature pipeline (9 features)
    ///
    /// A file is moved only when BOTH engines agree on the classification.
    /// Disagreements are moved to an "Unclassified" folder for manual review.
    ///
    /// Pipeline:
    ///   0. Deduplication  – size-bucket filter, then SHA-256 only on collisions
    ///   1. Dual inference – ONNX + hand-crafted features, consensus required
    ///
    /// Model setup (one-time):
    ///   Download mobilenetv3_v1.4_dist/model.onnx from:
    ///     https://huggingface.co/deepghs/anime_real_cls
    ///   Place at the path specified in ModelPath (default: model.onnx beside the exe).
    ///
    /// NuGet:
    ///   Microsoft.ML.OnnxRuntime  >= 1.17.0
    ///   System.Drawing.Common     >= 8.0.0  (Windows only)
    /// </summary>
    public static class ImageClassifier
    {
        // ── Configurable ──────────────────────────────────────────────────────
        public static string RealPhotoFolderName    { get; set; } = "RealPhotos";
        public static string CartoonFolderName      { get; set; } = "Cartoons";
        public static string UnclassifiedFolderName { get; set; } = "Unclassified";

        /// <summary>Full path to mobilenetv3_v1.4_dist/model.onnx.</summary>
        public static string ModelPath { get; set; } =
            Path.Combine(AppContext.BaseDirectory, "model.onnx");

        /// <summary>0 = one thread per logical processor.</summary>
        public static int MaxDegreeOfParallelism { get; set; } = 0;

        /// <summary>Anime probability >= this → ONNX says cartoon.</summary>
        public static float OnnxCartoonThreshold { get; set; } = 0.5f;

        /// <summary>Hand-crafted composite score >= this → heuristic says cartoon.</summary>
        public static double HeuristicCartoonThreshold { get; set; } = 0.50;

        // ── Supported extensions ──────────────────────────────────────────────
        private static readonly HashSet<string> SupportedExtensions =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"
                // .jfif files are renamed to .jpg before processing (see RenameJfifFiles)
                // .webp requires SkiaSharp – add here once that package is referenced
            };

        // ── ONNX / ImageNet constants ─────────────────────────────────────────
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };
        private const int InputSize = 384;

        // ── Shared ONNX session (thread-safe) ────────────────────────────────
        private static InferenceSession? _session;
        private static readonly object   _sessionLock = new();

        private static InferenceSession GetSession()
        {
            if (_session is not null) return _session;
            lock (_sessionLock)
            {
                if (_session is not null) return _session;
                if (!File.Exists(ModelPath))
                    throw new FileNotFoundException(
                        $"ONNX model not found at '{ModelPath}'.\n" +
                        "Download mobilenetv3_v1.4_dist/model.onnx from:\n" +
                        "  https://huggingface.co/deepghs/anime_real_cls\n" +
                        $"and place it at '{ModelPath}'.");
                var opts = new SessionOptions();
                opts.InterOpNumThreads = 1;
                opts.IntraOpNumThreads = 1;
                opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _session = new InferenceSession(ModelPath, opts);
                Console.WriteLine($"  Model loaded: {Path.GetFileName(ModelPath)}");
                return _session;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Public entry point – ClassifyImages
        // ─────────────────────────────────────────────────────────────────────
        public static void ClassifyImages(string workingDirectory)
        {
            if (string.IsNullOrWhiteSpace(RealPhotoFolderName) ||
                string.IsNullOrWhiteSpace(CartoonFolderName)   ||
                string.IsNullOrWhiteSpace(UnclassifiedFolderName))
                throw new InvalidOperationException("Folder names must not be empty.");

            var folderNames = new[] { RealPhotoFolderName, CartoonFolderName, UnclassifiedFolderName };
            if (folderNames.Distinct(StringComparer.OrdinalIgnoreCase).Count() != 3)
                throw new InvalidOperationException("RealPhotoFolderName, CartoonFolderName and UnclassifiedFolderName must all be different.");

            GetSession();

            string realDir         = Path.Combine(workingDirectory, RealPhotoFolderName);
            string cartoonDir      = Path.Combine(workingDirectory, CartoonFolderName);
            string unclassifiedDir = Path.Combine(workingDirectory, UnclassifiedFolderName);

            Directory.CreateDirectory(realDir);
            Directory.CreateDirectory(cartoonDir);
            Directory.CreateDirectory(unclassifiedDir);

            // ── Step 0a: Rename .jfif → .jpg ─────────────────────────────────
            // JFIF is structurally identical to JPEG. GDI+ decodes the bytes fine
            // but only recognises .jpg/.jpeg extensions, so rename in-place first.
            int renamedJfif = RenameJfifToJpg(workingDirectory);
            if (renamedJfif > 0)
                Console.WriteLine($"  Renamed {renamedJfif} .jfif file(s) to .jpg.");

            var imageFiles = Directory
                .EnumerateFiles(workingDirectory, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => SupportedExtensions.Contains(Path.GetExtension(f)))
                .ToList();

            if (imageFiles.Count == 0)
            {
                Console.WriteLine("  No image files found.");
                return;
            }

            Console.WriteLine($"  Found {imageFiles.Count} image file(s).");

            // ── Step 0b: Deduplicate ──────────────────────────────────────────
            int removedDupes = RemoveBatchDuplicates(imageFiles);
            if (removedDupes > 0)
                Console.WriteLine($"  Removed {removedDupes} duplicate(s). {imageFiles.Count} remaining.");

            Console.WriteLine($"  Classifying with {ActualParallelism()} thread(s)...\n");

            int cartoon = 0, real = 0, unclassified = 0, skipped = 0;
            int processed = 0;
            int total = imageFiles.Count;
            var consoleLock = new object();

            Parallel.ForEach(imageFiles,
                new ParallelOptions { MaxDegreeOfParallelism = ActualParallelism() },
                srcPath =>
                {
                    string label, reason, destDir;
                    bool moveOk = true;

                    try
                    {
                        bool onnxCartoon      = OnnxIsCartoon(srcPath, out string onnxReason);
                        bool heuristicCartoon = HeuristicIsCartoon(srcPath, out string heuristicReason);

                        bool agree = onnxCartoon == heuristicCartoon;

                        if (agree)
                        {
                            destDir = onnxCartoon ? cartoonDir : realDir;
                            label   = onnxCartoon ? "CARTOON" : "REAL   ";
                        }
                        else
                        {
                            destDir = unclassifiedDir;
                            label   = "UNSURE ";
                        }

                        reason = $"onnx=[{onnxReason}] heuristic=[{heuristicReason}]";

                        try { MoveWithConflictResolution(srcPath, destDir); }
                        catch (Exception ex)
                        {
                            reason = $"Move failed: {ex.Message}";
                            label  = "ERROR  ";
                            moveOk = false;
                        }
                    }
                    catch (Exception ex)
                    {
                        label   = "ERROR  ";
                        reason  = ex.Message;
                        destDir = string.Empty;
                        moveOk  = false;
                    }

                    int n = System.Threading.Interlocked.Increment(ref processed);

                    lock (consoleLock)
                    {
                        Console.WriteLine($"  [{label}] ({n}/{total}) {Path.GetFileName(srcPath)}");
                        Console.WriteLine($"           {reason}");

                        if      (!moveOk)                          skipped++;
                        else if (label.StartsWith("CARTOON"))      cartoon++;
                        else if (label.StartsWith("REAL"))         real++;
                        else if (label.StartsWith("UNSURE"))       unclassified++;
                        else                                       skipped++;
                    }
                });

            Console.WriteLine();
            Console.WriteLine($"  Classified: {cartoon} cartoon/anime, {real} real, " +
                              $"{unclassified} unclassified (engines disagreed), {skipped} errors.");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Engine A – ONNX inference
        // ─────────────────────────────────────────────────────────────────────
        private static bool OnnxIsCartoon(string filePath, out string reason)
        {
            float[] tensor = BuildInputTensor(filePath);

            var session    = GetSession();
            string inName  = session.InputMetadata.Keys.First();
            string outName = session.OutputMetadata.Keys.First();

            var inputTensor = new DenseTensor<float>(tensor,
                new[] { 1, 3, InputSize, InputSize });

            using var results = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inName, inputTensor)
            });

            var output     = results.First(r => r.Name == outName).AsTensor<float>();
            float logitA   = output.GetValue(0);
            float logitR   = output.GetValue(1);
            float animeP   = Softmax(logitA, logitR);

            reason = $"anime={animeP:F4}";
            return animeP >= OnnxCartoonThreshold;
        }

        private static float[] BuildInputTensor(string filePath)
        {
            using var original = new Bitmap(filePath);
            using var resized  = new Bitmap(InputSize, InputSize, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(original, 0, 0, InputSize, InputSize);
            }

            var rect   = new Rectangle(0, 0, InputSize, InputSize);
            BitmapData bd = resized.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            int stride = Math.Abs(bd.Stride);
            var raw    = new byte[stride * InputSize];
            Marshal.Copy(bd.Scan0, raw, 0, raw.Length);
            resized.UnlockBits(bd);

            var tensor    = new float[3 * InputSize * InputSize];
            int planeSize = InputSize * InputSize;

            for (int y = 0; y < InputSize; y++)
            for (int x = 0; x < InputSize; x++)
            {
                int b0  = y * stride + x * 4;
                float r = raw[b0 + 2] / 255f;
                float g = raw[b0 + 1] / 255f;
                float b = raw[b0 + 0] / 255f;
                int   p = y * InputSize + x;
                tensor[0 * planeSize + p] = (r - Mean[0]) / Std[0];
                tensor[1 * planeSize + p] = (g - Mean[1]) / Std[1];
                tensor[2 * planeSize + p] = (b - Mean[2]) / Std[2];
            }
            return tensor;
        }

        private static float Softmax(float a, float b)
        {
            float m  = MathF.Max(a, b);
            float ea = MathF.Exp(a - m);
            float eb = MathF.Exp(b - m);
            return ea / (ea + eb);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Engine B – Hand-crafted heuristic pipeline
        // ─────────────────────────────────────────────────────────────────────
        private static bool HeuristicIsCartoon(string filePath, out string reason)
        {
            using var bmp = LoadAndResize(filePath, maxDimension: 512);
            var px = new PixelBuffer(bmp);
            var features = ComputeFeatures(px);
            double composite = ScoreFeatures(features);
            reason = FormatReason(composite, features);
            return composite >= HeuristicCartoonThreshold;
        }

        /// <summary>
        /// Computes all 9 heuristic features for a pixel buffer.
        /// Exposed internally so FeatureReporter can reuse it without re-implementing.
        /// </summary>
        internal static ImageFeatures ComputeFeatures(PixelBuffer px)
        {
            return new ImageFeatures
            {
                Palette       = PaletteScore(px),
                Saturation    = SaturationScore(px),
                FlatRegion    = FlatRegionScore(px),
                EdgeBimodal   = EdgeSharpnessScore(px),
                InkOutline    = InkOutlineScore(px),
                SkinDiscrete  = SkinDiscretenessScore(px),
                FlatNoise     = FlatRegionNoiseScore(px),
                ColorTemp     = ColourTemperatureUniformityScore(px),
                ChannelNoise  = ChannelNoiseIndependenceScore(px),
            };
        }

        internal static double ScoreFeatures(ImageFeatures f) =>
            // Weights and signs derived from statistical analysis of 162 false-cartoon
            // and 85 false-real images.
            //
            // Key findings:
            //   - chnoise is the dominant separator (FC=0.13, FR=0.79 mean)
            //   - palette, flat, colortemp, skin all score HIGHER for real photos
            //     than for anime — they must be INVERTED from original assumption
            //   - flat_noise has zero separating power (both groups ~0.36) → weight=0.01
            //   - sat, edge, outline correctly score higher for anime
              0.30 * (1.0 - f.ChannelNoise)   // chnoise HIGH = real → invert
            + 0.15 * f.Saturation              // sat HIGH = anime ✓
            + 0.20 * f.InkOutline              // outline HIGH = anime ✓
            + 0.05 * f.EdgeBimodal             // edge HIGH = anime ✓
            + 0.13 * (1.0 - f.Palette)         // palette HIGH = real → invert
            + 0.10 * (1.0 - f.FlatRegion)      // flat HIGH = real → invert
            + 0.04 * (1.0 - f.ColorTemp)       // colortemp HIGH = real → invert
            + 0.02 * (1.0 - f.SkinDiscrete)    // skin HIGH = real → invert
            + 0.01 * (1.0 - f.FlatNoise);      // flat_noise: no separation, minimal weight

        internal static string FormatReason(double composite, ImageFeatures f) =>
            $"score={composite:F3} " +
            $"[chnoise={f.ChannelNoise:F2}↓ sat={f.Saturation:F2} outline={f.InkOutline:F2} " +
            $"edge={f.EdgeBimodal:F2} palette={f.Palette:F2}↓ flat={f.FlatRegion:F2}↓ " +
            $"colortemp={f.ColorTemp:F2}↓ skin={f.SkinDiscrete:F2}↓ noise={f.FlatNoise:F2}↓]";

        // ── Feature 1 – Colour palette size ──────────────────────────────────
        private static double PaletteScore(PixelBuffer px)
        {
            const int bits = 5, shift = 8 - bits;
            var buckets = new HashSet<int>();
            for (int y = 0; y < px.Height; y++)
            for (int x = 0; x < px.Width;  x++)
                buckets.Add(((px.R(x,y) >> shift) << (2*bits)) | ((px.G(x,y) >> shift) << bits) | (px.B(x,y) >> shift));
            return 1.0 - Math.Min(1.0, (double)buckets.Count / (1 << (3*bits)) / 0.12);
        }

        // ── Feature 2 – HSV Saturation ────────────────────────────────────────
        private static double SaturationScore(PixelBuffer px)
        {
            double total = 0;
            for (int y = 0; y < px.Height; y++)
            for (int x = 0; x < px.Width;  x++)
                total += px.Sat(x, y);
            return Math.Min(1.0, (total / (px.Width * px.Height)) / 0.65);
        }

        // ── Feature 3 – Flat region ratio ────────────────────────────────────
        private static double FlatRegionScore(PixelBuffer px)
        {
            const int threshold = 12;
            int flat = 0, total = 0;
            for (int y = 1; y < px.Height-1; y++)
            for (int x = 1; x < px.Width-1;  x++)
            {
                byte cr = px.R(x,y), cg = px.G(x,y), cb = px.B(x,y);
                int maxDelta = 0;
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx==0 && dy==0) continue;
                    int d = Math.Abs(cr - px.R(x+dx,y+dy)); if (d>maxDelta) maxDelta=d;
                        d = Math.Abs(cg - px.G(x+dx,y+dy)); if (d>maxDelta) maxDelta=d;
                        d = Math.Abs(cb - px.B(x+dx,y+dy)); if (d>maxDelta) maxDelta=d;
                }
                if (maxDelta <= threshold) flat++;
                total++;
            }
            return Math.Max(0.0, Math.Min(1.0, ((double)flat/total - 0.30) / 0.45));
        }

        // ── Feature 4 – Edge bimodality (Sobel) ──────────────────────────────
        private static double EdgeSharpnessScore(PixelBuffer px)
        {
            int w = px.Width, h = px.Height;
            var grey = new float[h * w];
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                grey[y*w+x] = px.Lum(x,y);

            int highEdge = 0, midEdge = 0, total = 0;
            for (int y = 1; y < h-1; y++)
            for (int x = 1; x < w-1; x++)
            {
                float gx = grey[(y-1)*w+(x+1)] + 2*grey[y*w+(x+1)] + grey[(y+1)*w+(x+1)]
                         - grey[(y-1)*w+(x-1)] - 2*grey[y*w+(x-1)] - grey[(y+1)*w+(x-1)];
                float gy = grey[(y+1)*w+(x-1)] + 2*grey[(y+1)*w+x] + grey[(y+1)*w+(x+1)]
                         - grey[(y-1)*w+(x-1)] - 2*grey[(y-1)*w+x] - grey[(y-1)*w+(x+1)];
                float mag = MathF.Sqrt(gx*gx + gy*gy);
                if (mag > 80)      highEdge++;
                else if (mag > 20) midEdge++;
                total++;
            }
            return Math.Min(1.0, ((double)highEdge / Math.Max(0.01, highEdge + midEdge)) * 1.4);
        }

        // ── Feature 5 – Ink outline detection ────────────────────────────────
        private static double InkOutlineScore(PixelBuffer px)
        {
            const float darkThresh = 80f, brightThresh = 160f;
            int outlinePixels = 0, total = 0;
            for (int y = 1; y < px.Height-1; y++)
            for (int x = 1; x < px.Width-1;  x++)
            {
                if (px.Lum(x,y) < darkThresh)
                {
                    bool hasBright = false;
                    for (int dy=-1; dy<=1 && !hasBright; dy++)
                    for (int dx=-1; dx<=1 && !hasBright; dx++)
                    {
                        if (dx==0 && dy==0) continue;
                        if (px.Lum(x+dx,y+dy) > brightThresh) hasBright = true;
                    }
                    if (hasBright) outlinePixels++;
                }
                total++;
            }
            return Math.Min(1.0, (double)outlinePixels / total / 0.04);
        }

        // ── Feature 6 – Skin tone discreteness ───────────────────────────────
        private static double SkinDiscretenessScore(PixelBuffer px)
        {
            const int bucketBits = 4, shift = 8 - bucketBits;
            var skinBuckets = new Dictionary<int, int>();
            int skinPixels = 0;
            for (int y = 0; y < px.Height; y++)
            for (int x = 0; x < px.Width;  x++)
            {
                byte r = px.R(x,y), g = px.G(x,y), b = px.B(x,y);
                if (!IsSkinTone(r,g,b)) continue;
                skinPixels++;
                int key = ((r>>shift) << (2*bucketBits)) | ((g>>shift) << bucketBits) | (b>>shift);
                skinBuckets.TryGetValue(key, out int cnt);
                skinBuckets[key] = cnt + 1;
            }
            if (skinPixels < 200) return 0.0;
            int top4 = skinBuckets.Values.OrderByDescending(v=>v).Take(4).Sum();
            return Math.Max(0.0, Math.Min(1.0, ((double)top4/skinPixels - 0.45) / 0.40));
        }

        private static bool IsSkinTone(byte r, byte g, byte b)
        {
            if (r<=100 || r<=g || r<=b || g<=50 || b<=30 || (r-b)<=20) return false;
            float max = MathF.Max(r, MathF.Max(g,b))/255f;
            float min = MathF.Min(r, MathF.Min(g,b))/255f;
            return (max<1e-6f ? 0f : (max-min)/max) > 0.08f && (max+min)/2f > 0.25f;
        }

        // ── Feature 7 – Flat-region micro-noise (REAL signal) ────────────────
        private static double FlatRegionNoiseScore(PixelBuffer px)
        {
            const int pr = 3, sampleStep = 4;
            const float maxRange = 20f;
            var noiseValues = new List<double>();
            for (int y = pr; y < px.Height-pr; y += sampleStep)
            for (int x = pr; x < px.Width-pr;  x += sampleStep)
            {
                float lumMin = float.MaxValue, lumMax = float.MinValue, lumSum = 0;
                int count = 0;
                for (int dy=-pr; dy<=pr; dy++)
                for (int dx=-pr; dx<=pr; dx++)
                { float l=px.Lum(x+dx,y+dy); if(l<lumMin)lumMin=l; if(l>lumMax)lumMax=l; lumSum+=l; count++; }
                if (lumMax-lumMin > maxRange) continue;
                float mean = lumSum/count;
                double variance = 0;
                for (int dy=-pr; dy<=pr; dy++)
                for (int dx=-pr; dx<=pr; dx++)
                { float d=px.Lum(x+dx,y+dy)-mean; variance+=d*d; }
                noiseValues.Add(Math.Sqrt(variance/count));
            }
            if (noiseValues.Count < 10) return 0.5;
            return Math.Min(1.0, noiseValues.Average() / 5.0);
        }

        // ── Feature 8 – Colour temperature uniformity ────────────────────────
        private static double ColourTemperatureUniformityScore(PixelBuffer px)
        {
            const int sampleStep = 4;
            var ratios = new List<double>();
            for (int y = 0; y < px.Height; y += sampleStep)
            for (int x = 0; x < px.Width;  x += sampleStep)
                ratios.Add((double)px.R(x,y) / (px.B(x,y)+1));
            double mean = ratios.Average();
            double std  = Math.Sqrt(ratios.Average(r => (r-mean)*(r-mean)));
            return Math.Max(0.0, Math.Min(1.0, 1.0 - std/0.7));
        }

        // ── Feature 9 – RGB channel noise independence (REAL signal) ─────────
        private static double ChannelNoiseIndependenceScore(PixelBuffer px)
        {
            const int sampleStep = 3;
            const float maxLumDelta = 25f;
            var rgDiffs = new List<float>();
            var rbDiffs = new List<float>();
            for (int y = 0; y < px.Height; y += sampleStep)
            for (int x = 0; x < px.Width-1; x += sampleStep)
            {
                if (MathF.Abs(px.Lum(x,y) - px.Lum(x+1,y)) > maxLumDelta) continue;
                rgDiffs.Add((px.R(x+1,y)-px.G(x+1,y)) - (px.R(x,y)-px.G(x,y)));
                rbDiffs.Add((px.R(x+1,y)-px.B(x+1,y)) - (px.R(x,y)-px.B(x,y)));
            }
            if (rgDiffs.Count < 50) return 0.5;
            return Math.Min(1.0, (Variance(rgDiffs) + Variance(rbDiffs)) / 2.0 / 25.0);
        }

        private static double Variance(List<float> v)
        {
            double mean = 0; foreach (var x in v) mean += x; mean /= v.Count;
            double var  = 0; foreach (var x in v) var  += (x-mean)*(x-mean);
            return var / v.Count;
        }

        // ── Pixel buffer (LockBits-based, zero GetPixel) ──────────────────────
        internal sealed class PixelBuffer
        {
            public readonly byte[] Data;
            public readonly int Width, Height, Stride;

            public PixelBuffer(Bitmap bmp)
            {
                Width = bmp.Width; Height = bmp.Height;
                BitmapData bd = bmp.LockBits(new Rectangle(0,0,Width,Height),
                    ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                Stride = Math.Abs(bd.Stride);
                Data   = new byte[Stride * Height];
                Marshal.Copy(bd.Scan0, Data, 0, Data.Length);
                bmp.UnlockBits(bd);
            }

            public byte B(int x, int y) => Data[y*Stride + x*4];
            public byte G(int x, int y) => Data[y*Stride + x*4+1];
            public byte R(int x, int y) => Data[y*Stride + x*4+2];
            public float Lum(int x, int y) => 0.299f*R(x,y) + 0.587f*G(x,y) + 0.114f*B(x,y);
            public float Sat(int x, int y)
            {
                float r=R(x,y)/255f, g=G(x,y)/255f, b=B(x,y)/255f;
                float max=MathF.Max(r,MathF.Max(g,b)), min=MathF.Min(r,MathF.Min(g,b));
                return max<1e-6f ? 0f : (max-min)/max;
            }
        }

        private static Bitmap LoadAndResize(string path, int maxDimension)
        {
            using var original = new Bitmap(path);
            int w = original.Width, h = original.Height;
            if (w <= maxDimension && h <= maxDimension) return new Bitmap(original);
            double scale = Math.Min((double)maxDimension/w, (double)maxDimension/h);
            int nw = (int)(w*scale), nh = (int)(h*scale);
            var resized = new Bitmap(nw, nh, PixelFormat.Format32bppArgb);
            using var g = Graphics.FromImage(resized);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.DrawImage(original, 0, 0, nw, nh);
            return resized;
        }

        // ─────────────────────────────────────────────────────────────────────
        // JFIF → JPG rename
        // ─────────────────────────────────────────────────────────────────────
        private static int RenameJfifToJpg(string workingDirectory)
        {
            int count = 0;
            foreach (string path in Directory.EnumerateFiles(
                workingDirectory, "*.jfif", SearchOption.TopDirectoryOnly))
            {
                string dest = BuildUniquePath(
                    workingDirectory,
                    Path.GetFileNameWithoutExtension(path),
                    ".jpg");
                try
                {
                    File.Move(path, dest);
                    count++;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"  [ERROR] Could not rename \"{Path.GetFileName(path)}\": {ex.Message}");
                }
            }
            return count;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Batch deduplication
        // ─────────────────────────────────────────────────────────────────────
        private static int RemoveBatchDuplicates(List<string> files)
        {
            var bySize = files.GroupBy(f => new FileInfo(f).Length)
                              .Where(g => g.Count() > 1).ToList();
            if (bySize.Count == 0) return 0;

            var hashMap = new ConcurrentDictionary<string, string>();
            Parallel.ForEach(bySize.SelectMany(g => g),
                new ParallelOptions { MaxDegreeOfParallelism = ActualParallelism() },
                path => { using var sha = SHA256.Create(); hashMap[path] = Convert.ToHexString(ComputeHash(sha, path)); });

            var toDelete = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var sg in bySize)
            foreach (var hg in sg.GroupBy(f => hashMap[f]).Where(g => g.Count() > 1))
            {
                var sorted = hg.OrderBy(f => Path.GetFileName(f), StringComparer.OrdinalIgnoreCase).ToList();
                for (int i = 1; i < sorted.Count; i++)
                {
                    try { Console.WriteLine($"  [DUPE   ] Deleting \"{Path.GetFileName(sorted[i])}\" (identical to \"{Path.GetFileName(sorted[0])}\")");
                          File.Delete(sorted[i]); toDelete.Add(sorted[i]); }
                    catch (Exception ex) { Console.Error.WriteLine($"  [ERROR  ] Could not delete \"{Path.GetFileName(sorted[i])}\": {ex.Message}"); }
                }
            }
            files.RemoveAll(f => toDelete.Contains(f));
            return toDelete.Count;
        }

        // ─────────────────────────────────────────────────────────────────────
        // File-move helpers
        // ─────────────────────────────────────────────────────────────────────
        private static void MoveWithConflictResolution(string srcPath, string destDir)
        {
            string fileName = Path.GetFileName(srcPath);
            string destPath = Path.Combine(destDir, fileName);
            if (!File.Exists(destPath)) { File.Move(srcPath, destPath); return; }

            if (new FileInfo(srcPath).Length == new FileInfo(destPath).Length && CrcMatch(srcPath, destPath))
            { Console.WriteLine($"           ↳ Duplicate of existing; deleting source."); File.Delete(srcPath); }
            else
            { string u = BuildUniquePath(destDir, Path.GetFileNameWithoutExtension(fileName), Path.GetExtension(fileName));
              Console.WriteLine($"           ↳ Name clash; renaming to {Path.GetFileName(u)}"); File.Move(srcPath, u); }
        }

        private static bool CrcMatch(string a, string b)
        { using var sha = SHA256.Create(); return ComputeHash(sha, a).SequenceEqual(ComputeHash(sha, b)); }

        private static byte[] ComputeHash(HashAlgorithm alg, string path)
        { using var fs = File.OpenRead(path); return alg.ComputeHash(fs); }

        private static string BuildUniquePath(string dir, string baseName, string ext)
        { int c=1; string p; do { p=Path.Combine(dir,$"{baseName}_{c++}{ext}"); } while(File.Exists(p)); return p; }

        private static int ActualParallelism() =>
            MaxDegreeOfParallelism <= 0 ? Environment.ProcessorCount : MaxDegreeOfParallelism;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Feature data bag – passed between classifier and reporter
    // ─────────────────────────────────────────────────────────────────────────
    internal sealed class ImageFeatures
    {
        public double Palette;
        public double Saturation;
        public double FlatRegion;
        public double EdgeBimodal;
        public double InkOutline;
        public double SkinDiscrete;
        public double FlatNoise;       // REAL signal (high = real)
        public double ColorTemp;
        public double ChannelNoise;    // REAL signal (high = real)

        public string ToCsvRow(string fileName) =>
            $"{fileName},{Palette:F4},{Saturation:F4},{FlatRegion:F4},{EdgeBimodal:F4}," +
            $"{InkOutline:F4},{SkinDiscrete:F4},{FlatNoise:F4},{ColorTemp:F4},{ChannelNoise:F4}";

        public static string CsvHeader =>
            "file,palette(real↑),saturation(anime↑),flat_region(real↑),edge_bimodal(anime↑)," +
            "ink_outline(anime↑),skin_discrete(real↑),flat_noise(useless),color_temp(real↑),channel_noise(real↑)";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FeatureReporter – Step 6
    // Reads images from FalsePositiveCartoon and FalsePositiveRealPhoto folders
    // and writes their heuristic feature vectors to two CSV files so the
    // feature weights can be tuned.
    // ─────────────────────────────────────────────────────────────────────────
    public static class FeatureReporter
    {
        public static string FalsePositiveCartoonFolder  { get; set; } = "FalsePositiveCartoon";
        public static string FalsePositiveRealFolder     { get; set; } = "FalsePositiveRealPhoto";
        public static string OutputCartoonCsv            { get; set; } = "features_false_cartoon.csv";
        public static string OutputRealCsv               { get; set; } = "features_false_real.csv";

        private static readonly HashSet<string> SupportedExtensions =
            new(StringComparer.OrdinalIgnoreCase)
            { ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif" };

        public static void WriteFeatureReports(string workingDirectory)
        {
            ProcessFolder(
                Path.Combine(workingDirectory, FalsePositiveCartoonFolder),
                Path.Combine(workingDirectory, OutputCartoonCsv),
                "false-positive CARTOON (should be REAL)");

            ProcessFolder(
                Path.Combine(workingDirectory, FalsePositiveRealFolder),
                Path.Combine(workingDirectory, OutputRealCsv),
                "false-positive REAL (should be CARTOON)");
        }

        private static void ProcessFolder(string folder, string csvPath, string description)
        {
            if (!Directory.Exists(folder))
            {
                Console.WriteLine($"  Skipping '{Path.GetFileName(folder)}' — folder not found.");
                return;
            }

            var files = Directory
                .EnumerateFiles(folder, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => SupportedExtensions.Contains(Path.GetExtension(f)))
                .OrderBy(f => f)
                .ToList();

            if (files.Count == 0)
            {
                Console.WriteLine($"  No images in '{Path.GetFileName(folder)}'.");
                return;
            }

            Console.WriteLine($"  Analysing {files.Count} image(s) in '{Path.GetFileName(folder)}'...");

            var rows = new ConcurrentBag<(string name, ImageFeatures features)>();
            int processed = 0;

            Parallel.ForEach(files,
                new ParallelOptions { MaxDegreeOfParallelism = ImageClassifier.MaxDegreeOfParallelism <= 0
                    ? Environment.ProcessorCount : ImageClassifier.MaxDegreeOfParallelism },
                path =>
                {
                    try
                    {
                        using var bmp = LoadAndResize(path, 512);
                        var px       = new ImageClassifier.PixelBuffer(bmp);
                        var features = ImageClassifier.ComputeFeatures(px);
                        rows.Add((Path.GetFileName(path), features));
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"  [ERROR] {Path.GetFileName(path)}: {ex.Message}");
                    }
                    System.Threading.Interlocked.Increment(ref processed);
                });

            // Write CSV — sorted by filename for reproducibility
            using var sw = new StreamWriter(csvPath, append: false, encoding: System.Text.Encoding.UTF8);
            sw.WriteLine($"# {description}");
            sw.WriteLine($"# Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sw.WriteLine($"# Composite score = weighted sum; flat_noise and channel_noise are REAL signals (higher = more real)");
            sw.WriteLine($"# Weights: palette=0.13 saturation=0.09 flat_region=0.09 edge=0.09 outline=0.18 skin=0.13 flat_noise(inv)=0.12 colortemp=0.05 channelnoise(inv)=0.12");
            sw.WriteLine(ImageFeatures.CsvHeader);

            foreach (var (name, features) in rows.OrderBy(r => r.name))
            {
                double composite = ImageClassifier.ScoreFeatures(features);
                sw.WriteLine($"{features.ToCsvRow(name)},composite={composite:F4}");
            }

            Console.WriteLine($"  Written: {csvPath}  ({rows.Count} row(s))");
        }

        private static Bitmap LoadAndResize(string path, int maxDimension)
        {
            using var original = new Bitmap(path);
            int w = original.Width, h = original.Height;
            if (w <= maxDimension && h <= maxDimension) return new Bitmap(original);
            double scale = Math.Min((double)maxDimension/w, (double)maxDimension/h);
            int nw = (int)(w*scale), nh = (int)(h*scale);
            var resized = new Bitmap(nw, nh, PixelFormat.Format32bppArgb);
            using var g = Graphics.FromImage(resized);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.DrawImage(original, 0, 0, nw, nh);
            return resized;
        }
    }
}
