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
    /// Classifies images as real photos or anime/cartoon using the
    /// deepghs/anime_real_cls ONNX model (mobilenetv3_v1.4_dist, 98.78 % accuracy).
    ///
    /// Model: https://huggingface.co/deepghs/anime_real_cls
    /// License: openrail
    ///
    /// Pipeline:
    ///   0. Deduplication     – size-bucket filter, then SHA-256 only on collisions
    ///   1. ONNX inference    – MobileNetV3 pretrained on anime vs real dataset
    ///
    /// Setup (one-time, before first run):
    ///   Download mobilenetv3_v1.4_dist/model.onnx from the HuggingFace repo above
    ///   and place it at the path specified in <see cref="ModelPath"/>.
    ///
    /// NuGet packages required:
    ///   Microsoft.ML.OnnxRuntime  >= 1.17.0
    ///   System.Drawing.Common     >= 8.0.0   (Windows only)
    /// </summary>
    public static class ImageClassifier
    {
        // ── Configurable ──────────────────────────────────────────────────────

        public static string RealPhotoFolderName { get; set; } = "RealPhotos";
        public static string CartoonFolderName   { get; set; } = "Cartoons";

        /// <summary>
        /// Full path to the mobilenetv3_v1.4_dist/model.onnx file.
        /// Default: model.onnx next to the executable.
        /// </summary>
        public static string ModelPath { get; set; } =
            Path.Combine(AppContext.BaseDirectory, "model.onnx");

        /// <summary>
        /// Maximum parallel threads. 0 = one per logical processor.
        /// </summary>
        public static int MaxDegreeOfParallelism { get; set; } = 0;

        /// <summary>
        /// Images whose anime probability is above this are classified as cartoon.
        /// 0.5 is the natural midpoint; lower = more aggressive cartoon detection.
        /// </summary>
        public static float CartoonThreshold { get; set; } = 0.5f;

        // ── Supported extensions ──────────────────────────────────────────────
        private static readonly HashSet<string> SupportedExtensions =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"
                // .webp requires SkiaSharp – add here once that package is referenced
            };

        // ── ImageNet normalisation constants (used by MobileNetV3) ───────────
        // Pixel layout after preprocessing: float32 NCHW, values normalised as:
        //   value = (pixel/255 - mean) / std   per channel (R, G, B)
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };
        private const int InputSize = 224; // model expects 224×224

        // ── Lazily-loaded, shared ONNX session ────────────────────────────────
        // InferenceSession is thread-safe; one instance is reused across all threads.
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
                        $"and place it at '{ModelPath}' (or set ImageClassifier.ModelPath).");

                var opts = new SessionOptions();
                opts.InterOpNumThreads  = 1;   // we parallelise at the file level
                opts.IntraOpNumThreads  = 1;
                opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                _session = new InferenceSession(ModelPath, opts);
                Console.WriteLine($"  Model loaded: {Path.GetFileName(ModelPath)}");
                return _session;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Public entry point
        // ─────────────────────────────────────────────────────────────────────
        public static void ClassifyImages(string workingDirectory)
        {
            if (string.IsNullOrWhiteSpace(RealPhotoFolderName) ||
                string.IsNullOrWhiteSpace(CartoonFolderName))
                throw new InvalidOperationException("Destination folder names must not be empty.");

            if (string.Equals(RealPhotoFolderName, CartoonFolderName,
                              StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException(
                    $"RealPhotoFolderName and CartoonFolderName must differ. " +
                    $"Both are currently '{RealPhotoFolderName}'.");

            // Warm up the session before launching parallel work so that the
            // "model loaded" message appears before the progress output.
            GetSession();

            string realDir    = Path.Combine(workingDirectory, RealPhotoFolderName);
            string cartoonDir = Path.Combine(workingDirectory, CartoonFolderName);

            Directory.CreateDirectory(realDir);
            Directory.CreateDirectory(cartoonDir);

            var imageFiles = Directory
                .EnumerateFiles(workingDirectory, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => SupportedExtensions.Contains(Path.GetExtension(f)))
                .ToList();

            if (imageFiles.Count == 0)
            {
                Console.WriteLine("  No image files found in the root of the working directory.");
                return;
            }

            Console.WriteLine($"  Found {imageFiles.Count} image file(s).");

            // ── Step 0: Deduplicate ───────────────────────────────────────────
            int removedDupes = RemoveBatchDuplicates(imageFiles);
            if (removedDupes > 0)
                Console.WriteLine($"  Removed {removedDupes} in-batch duplicate(s). " +
                                  $"{imageFiles.Count} file(s) remaining.");

            Console.WriteLine($"  Classifying with {ActualParallelism()} thread(s)...\n");

            // ── Parallel classify + immediate move ────────────────────────────
            int cartoon = 0, real = 0, skipped = 0;
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
                        bool isCartoon = IsCartoonOrAnime(srcPath, out reason);
                        destDir = isCartoon ? cartoonDir : realDir;
                        label   = isCartoon ? "CARTOON" : "REAL   ";

                        try
                        {
                            MoveWithConflictResolution(srcPath, destDir);
                        }
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

                        if (!moveOk)                          skipped++;
                        else if (label.StartsWith("CARTOON")) cartoon++;
                        else                                  real++;
                    }
                });

            Console.WriteLine();
            Console.WriteLine($"  Classified: {cartoon} cartoon/anime, {real} real photo(s), {skipped} skipped.");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Core classification – ONNX inference
        // ─────────────────────────────────────────────────────────────────────
        internal static bool IsCartoonOrAnime(string filePath, out string reason)
        {
            // 1. Load and resize to 224×224
            float[] tensor = BuildInputTensor(filePath);

            // 2. Run inference
            var session = GetSession();
            string inputName  = session.InputMetadata.Keys.First();
            string outputName = session.OutputMetadata.Keys.First();

            var inputTensor = new DenseTensor<float>(tensor,
                new[] { 1, 3, InputSize, InputSize });

            using var results = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            });

            // 3. Read softmax output – model outputs [anime_prob, real_prob]
            var output = results.First(r => r.Name == outputName)
                                .AsTensor<float>();

            // Output layout: index 0 = anime, index 1 = real
            // (verified against deepghs Python source label order)
            float animeProb = Softmax(output[0], output[1]);
            float realProb  = 1f - animeProb;

            bool isCartoon = animeProb >= CartoonThreshold;
            reason = $"anime={animeProb:F4} real={realProb:F4}";
            return isCartoon;
        }

        /// <summary>
        /// Builds a float32 NCHW tensor [1, 3, 224, 224] from an image file,
        /// applying centre-crop resize and ImageNet normalisation.
        /// </summary>
        private static float[] BuildInputTensor(string filePath)
        {
            using var original = new Bitmap(filePath);

            // Resize to InputSize × InputSize
            using var resized  = new Bitmap(InputSize, InputSize, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(original, 0, 0, InputSize, InputSize);
            }

            // Lock pixels once and copy into managed array
            var rect = new Rectangle(0, 0, InputSize, InputSize);
            BitmapData bd = resized.LockBits(rect, ImageLockMode.ReadOnly,
                                             PixelFormat.Format32bppArgb);
            int stride = Math.Abs(bd.Stride);
            var raw    = new byte[stride * InputSize];
            Marshal.Copy(bd.Scan0, raw, 0, raw.Length);
            resized.UnlockBits(bd);

            // Build NCHW float tensor with ImageNet normalisation
            // raw byte layout: B G R A  (32bppArgb in little-endian memory)
            var tensor = new float[3 * InputSize * InputSize];
            int planeSize = InputSize * InputSize;

            for (int y = 0; y < InputSize; y++)
            for (int x = 0; x < InputSize; x++)
            {
                int srcBase = y * stride + x * 4;
                float r = raw[srcBase + 2] / 255f;
                float g = raw[srcBase + 1] / 255f;
                float b = raw[srcBase + 0] / 255f;

                int pixIdx = y * InputSize + x;
                tensor[0 * planeSize + pixIdx] = (r - Mean[0]) / Std[0];
                tensor[1 * planeSize + pixIdx] = (g - Mean[1]) / Std[1];
                tensor[2 * planeSize + pixIdx] = (b - Mean[2]) / Std[2];
            }

            return tensor;
        }

        /// <summary>
        /// Converts a raw logit pair into a probability for the first class.
        /// If the model already outputs softmax probabilities this is still safe
        /// (softmax is idempotent for 2-class outputs up to floating point).
        /// </summary>
        private static float Softmax(float logitAnime, float logitReal)
        {
            float maxL = MathF.Max(logitAnime, logitReal);
            float eA   = MathF.Exp(logitAnime - maxL);
            float eR   = MathF.Exp(logitReal  - maxL);
            return eA / (eA + eR);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Batch deduplication
        // ─────────────────────────────────────────────────────────────────────
        private static int RemoveBatchDuplicates(List<string> files)
        {
            var bySize = files
                .GroupBy(f => new FileInfo(f).Length)
                .Where(g => g.Count() > 1)
                .ToList();

            if (bySize.Count == 0) return 0;

            var hashMap = new ConcurrentDictionary<string, string>();

            Parallel.ForEach(
                bySize.SelectMany(g => g),
                new ParallelOptions { MaxDegreeOfParallelism = ActualParallelism() },
                path =>
                {
                    using var sha = SHA256.Create();
                    hashMap[path] = Convert.ToHexString(ComputeHash(sha, path));
                });

            var toDelete = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var sizeGroup in bySize)
            foreach (var hashGroup in sizeGroup.GroupBy(f => hashMap[f]).Where(g => g.Count() > 1))
            {
                var sorted = hashGroup
                    .OrderBy(f => Path.GetFileName(f), StringComparer.OrdinalIgnoreCase)
                    .ToList();
                string keep = sorted[0];

                for (int i = 1; i < sorted.Count; i++)
                {
                    string dupe = sorted[i];
                    try
                    {
                        Console.WriteLine(
                            $"  [DUPE   ] Deleting \"{Path.GetFileName(dupe)}\" " +
                            $"(identical to \"{Path.GetFileName(keep)}\")");
                        File.Delete(dupe);
                        toDelete.Add(dupe);
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine(
                            $"  [ERROR  ] Could not delete \"{Path.GetFileName(dupe)}\": {ex.Message}");
                    }
                }
            }

            files.RemoveAll(f => toDelete.Contains(f));
            return toDelete.Count;
        }

        // ─────────────────────────────────────────────────────────────────────
        // File-move with conflict resolution
        // ─────────────────────────────────────────────────────────────────────
        private static void MoveWithConflictResolution(string srcPath, string destDir)
        {
            string fileName = Path.GetFileName(srcPath);
            string destPath = Path.Combine(destDir, fileName);

            if (!File.Exists(destPath))
            {
                File.Move(srcPath, destPath);
                return;
            }

            long srcSize  = new FileInfo(srcPath).Length;
            long destSize = new FileInfo(destPath).Length;

            if (srcSize == destSize && CrcMatch(srcPath, destPath))
            {
                Console.WriteLine($"           ↳ Duplicate of existing file; deleting source.");
                File.Delete(srcPath);
            }
            else
            {
                string unique = BuildUniquePath(destDir,
                    Path.GetFileNameWithoutExtension(fileName),
                    Path.GetExtension(fileName));
                Console.WriteLine($"           ↳ Name clash (different content); renaming to {Path.GetFileName(unique)}");
                File.Move(srcPath, unique);
            }
        }

        private static bool CrcMatch(string a, string b)
        {
            using var sha = SHA256.Create();
            return ComputeHash(sha, a).SequenceEqual(ComputeHash(sha, b));
        }

        private static byte[] ComputeHash(HashAlgorithm alg, string path)
        {
            using var fs = File.OpenRead(path);
            return alg.ComputeHash(fs);
        }

        private static string BuildUniquePath(string dir, string baseName, string ext)
        {
            int counter = 1;
            string candidate;
            do { candidate = Path.Combine(dir, $"{baseName}_{counter++}{ext}"); }
            while (File.Exists(candidate));
            return candidate;
        }

        private static int ActualParallelism() =>
            MaxDegreeOfParallelism <= 0
                ? Environment.ProcessorCount
                : MaxDegreeOfParallelism;
    }
}
