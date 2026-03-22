using System.Collections.Concurrent;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TediousTasks;

/// <summary>
/// Classifies images as real photos or anime/cartoon using a dual-engine approach.
///
///   Engine A – ONNX   : deepghs/anime_real_cls mobilenetv3_v1.4_dist (98.78 % accuracy)
///   Engine B – Heuristic: hand-crafted pixel-feature pipeline (9 features)
///
/// When both engines are active a file is moved only when they agree;
/// disagreements go to the Unclassified folder for manual review.
/// Either engine can be disabled independently; both disabled skips the step.
///
/// Model setup (one-time):
///   Download mobilenetv3_v1.4_dist/model.onnx from:
///     https://huggingface.co/deepghs/anime_real_cls
///   Place at ModelPath (default: model.onnx beside the executable).
///
/// NuGet required:
///   Microsoft.ML.OnnxRuntime  >= 1.17.0
///   System.Drawing.Common     >= 8.0.0  (Windows only)
/// </summary>
public static class ImageClassifier
{
    // ── Configuration ─────────────────────────────────────────────────────────
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

    /// <summary>
    /// Heuristic composite score threshold used when BOTH engines are active (consensus mode).
    /// Set conservatively high (0.64) so the heuristic only confirms clear cartoon cases,
    /// minimising false cartoons at the cost of more Unclassified images.
    /// </summary>
    public static double HeuristicConsensusThreshold { get; set; } = 0.64;

    /// <summary>
    /// Heuristic composite score threshold used when the heuristic runs ALONE
    /// (ONNX engine disabled). Lower than <see cref="HeuristicConsensusThreshold"/>
    /// because there is no second engine to catch the errors this looser threshold lets through.
    /// Calibrated to catch ~78% of anime while accepting ~5 false cartoons per 1700 images.
    /// </summary>
    public static double HeuristicStandaloneThreshold { get; set; } = 0.66;

    /// <summary>Enable the ONNX neural-network engine.</summary>
    public static bool UseOnnxEngine { get; set; } = true;

    /// <summary>Enable the hand-crafted heuristic engine.</summary>
    public static bool UseHeuristicEngine { get; set; } = true;

    // ── ONNX / ImageNet constants ─────────────────────────────────────────────
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };
    private const int InputSize = 384;

    // ── Shared ONNX session (lazy, thread-safe) ───────────────────────────────
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

            _session = new InferenceSession(ModelPath, new SessionOptions
            {
                InterOpNumThreads       = 1,   // we parallelise at the file level
                IntraOpNumThreads       = 1,
                GraphOptimizationLevel  = GraphOptimizationLevel.ORT_ENABLE_ALL,
            });

            Console.WriteLine($"  Model loaded: {Path.GetFileName(ModelPath)}");
            return _session;
        }
    }

    // Exposed so FeatureReporter can reuse without duplicating the formula.
    internal static int ActualParallelism() =>
        MaxDegreeOfParallelism <= 0 ? Environment.ProcessorCount : MaxDegreeOfParallelism;

    // ─────────────────────────────────────────────────────────────────────────
    // Public entry point – ConvertImagesToJxl
    // Converts jpeg/png/gif/bmp/tiff files to JPEG XL in-place.
    // Call this BEFORE ClassifyImages so the classifier sees only .jxl files.
    // ─────────────────────────────────────────────────────────────────────────
    public static void ConvertImagesToJxl(string workingDirectory)
    {
        var candidates = Directory
            .EnumerateFiles(workingDirectory, "*.*", SearchOption.TopDirectoryOnly)
            .Where(f => ImageUtils.ConvertToJxlExtensions.Contains(Path.GetExtension(f)))
            .ToList();

        if (candidates.Count == 0)
        {
            Console.WriteLine("  No convertible image files found.");
            return;
        }

        Console.WriteLine($"  Converting {candidates.Count} file(s) to JPEG XL...");

        int ok = 0, failed = 0, processed = 0, total = candidates.Count;
        var consoleLock = new object();

        Parallel.ForEach(candidates,
            new ParallelOptions { MaxDegreeOfParallelism = ActualParallelism() },
            srcPath =>
            {
                string? result = ImageUtils.ConvertToJxl(srcPath);
                int n = Interlocked.Increment(ref processed);
                lock (consoleLock)
                {
                    if (result is not null)
                    {
                        Console.WriteLine(
                            $"  [JXL] ({n}/{total}) {Path.GetFileName(srcPath)} → {Path.GetFileName(result)}");
                        ok++;
                    }
                    else
                    {
                        failed++;
                    }
                }
            });

        Console.WriteLine($"  Converted: {ok} succeeded, {failed} failed.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public entry point – ClassifyImages
    public static void ClassifyImages(string workingDirectory)
    {
        if (!UseOnnxEngine && !UseHeuristicEngine)
        {
            Console.WriteLine("  Both engines are disabled — skipping image classification.");
            return;
        }

        ValidateFolderNames();

        if (UseOnnxEngine)
            GetSession();   // fail fast on missing model before touching any files

        string realDir         = Path.Combine(workingDirectory, RealPhotoFolderName);
        string cartoonDir      = Path.Combine(workingDirectory, CartoonFolderName);
        string unclassifiedDir = Path.Combine(workingDirectory, UnclassifiedFolderName);
        Directory.CreateDirectory(realDir);
        Directory.CreateDirectory(cartoonDir);
        Directory.CreateDirectory(unclassifiedDir);

        Console.WriteLine($"  Active engines: {EngineLabel()}");

        // ── Step 0a: Rename .jfif → .jpg ─────────────────────────────────────
        // (JXL conversion is a separate step run before ClassifyImages)
        int renamedJfif = RenameJfifToJpg(workingDirectory);
        if (renamedJfif > 0)
            Console.WriteLine($"  Renamed {renamedJfif} .jfif file(s) to .jpg.");

        // ── Step 0b: Enumerate ────────────────────────────────────────────────
        var imageFiles = Directory
            .EnumerateFiles(workingDirectory, "*.*", SearchOption.TopDirectoryOnly)
            .Where(f => ImageUtils.SupportedExtensions.Contains(Path.GetExtension(f)))
            .ToList();

        if (imageFiles.Count == 0)
        {
            Console.WriteLine("  No image files found.");
            return;
        }
        Console.WriteLine($"  Found {imageFiles.Count} image file(s).");

        // ── Step 0c: Deduplicate ──────────────────────────────────────────────
        int removedDupes = DuplicateRemover.RemoveDuplicates(imageFiles, ActualParallelism());
        if (removedDupes > 0)
            Console.WriteLine($"  Removed {removedDupes} duplicate(s). {imageFiles.Count} remaining.");

        // ── Parallel classify + immediate move ────────────────────────────────
        Console.WriteLine($"  Classifying with {ActualParallelism()} thread(s)...\n");

        int cartoon = 0, real = 0, unclassified = 0, errors = 0;
        int processed = 0;
        int total     = imageFiles.Count;
        var consoleLock = new object();

        Parallel.ForEach(imageFiles,
            new ParallelOptions { MaxDegreeOfParallelism = ActualParallelism() },
            srcPath =>
            {
                var (label, reason) = Classify(srcPath,
                    cartoonDir, realDir, unclassifiedDir);

                int n = Interlocked.Increment(ref processed);
                lock (consoleLock)
                {
                    Console.WriteLine($"  [{label}] ({n}/{total}) {Path.GetFileName(srcPath)}");
                    Console.WriteLine($"           {reason}");

                    if      (label == "CARTOON") cartoon++;
                    else if (label == "REAL   ") real++;
                    else if (label == "UNSURE ") unclassified++;
                    else                         errors++;
                }
            });

        Console.WriteLine();
        Console.WriteLine(
            $"  Classified: {cartoon} cartoon/anime, {real} real, " +
            $"{unclassified} unclassified (engines disagreed), {errors} errors.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Classification logic — returns (label, reason) for one file
    // ─────────────────────────────────────────────────────────────────────────
    private static (string Label, string Reason) Classify(
        string srcPath,
        string cartoonDir, string realDir, string unclassifiedDir)
    {
        try
        {
            bool   isCartoon;
            string reason;

            if (UseOnnxEngine && UseHeuristicEngine)
            {
                // Consensus mode: use the conservative high threshold so the heuristic
                // only confirms ONNX when it's very confident — minimises false cartoons.
                bool onnxResult      = OnnxEngine.IsCartoon(srcPath, out string onnxReason);
                bool heuristicResult = HeuristicEngine.IsCartoon(srcPath,
                                           HeuristicConsensusThreshold, out string heuristicReason);
                reason = $"onnx=[{onnxReason}] heuristic=[{heuristicReason}]";

                if (onnxResult != heuristicResult)
                    return MoveAndLabel("UNSURE ", reason, srcPath, unclassifiedDir);

                isCartoon = onnxResult;
            }
            else if (UseOnnxEngine)
            {
                isCartoon = OnnxEngine.IsCartoon(srcPath, out string r);
                reason    = $"onnx=[{r}]";
            }
            else
            {
                // Standalone heuristic mode: use the lower threshold calibrated for
                // operating without ONNX backup. Catches more anime at the cost of
                // a small increase in false cartoons.
                isCartoon = HeuristicEngine.IsCartoon(srcPath,
                                HeuristicStandaloneThreshold, out string r);
                reason    = $"heuristic=[{r}]";
            }

            string destDir = isCartoon ? cartoonDir : realDir;
            string label   = isCartoon ? "CARTOON" : "REAL   ";
            return MoveAndLabel(label, reason, srcPath, destDir);
        }
        catch (Exception ex)
        {
            return ("ERROR  ", ex.Message);
        }
    }

    private static (string Label, string Reason) MoveAndLabel(
        string label, string reason, string srcPath, string destDir)
    {
        try
        {
            MoveWithConflictResolution(srcPath, destDir);
        }
        catch (Exception ex)
        {
            return ("ERROR  ", $"Move failed: {ex.Message}");
        }
        return (label, reason);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Engine A – ONNX inference (inner class keeps ONNX-specific code contained)
    // ─────────────────────────────────────────────────────────────────────────
    private static class OnnxEngine
    {
        public static bool IsCartoon(string filePath, out string reason)
        {
            float[] tensor = BuildInputTensor(filePath);

            var    session = GetSession();
            string inName  = session.InputMetadata.Keys.First();
            string outName = session.OutputMetadata.Keys.First();

            var inputTensor = new DenseTensor<float>(tensor,
                new[] { 1, 3, InputSize, InputSize });

            using var results = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inName, inputTensor)
            });

            var   output  = results.First(r => r.Name == outName).AsTensor<float>();
            float animeP  = Softmax(output.GetValue(0), output.GetValue(1));

            reason = $"anime={animeP:F4}";
            return animeP >= OnnxCartoonThreshold;
        }

        private static float[] BuildInputTensor(string filePath)
        {
            // GDI+ cannot decode .jxl — use Magick.NET via ImageUtils for those.
            using var original = Path.GetExtension(filePath)
                                     .Equals(".jxl", StringComparison.OrdinalIgnoreCase)
                                 ? ImageUtils.JxlToBitmap(filePath)
                                 : new Bitmap(filePath);
            using var resized  = new Bitmap(InputSize, InputSize, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(resized))
            {
                g.InterpolationMode =
                    System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(original, 0, 0, InputSize, InputSize);
            }

            var        rect   = new Rectangle(0, 0, InputSize, InputSize);
            BitmapData bd     = resized.LockBits(rect, ImageLockMode.ReadOnly,
                                                 PixelFormat.Format32bppArgb);
            int    stride;
            byte[] raw;
            try
            {
                stride = Math.Abs(bd.Stride);
                raw    = new byte[stride * InputSize];
                Marshal.Copy(bd.Scan0, raw, 0, raw.Length);
            }
            finally
            {
                resized.UnlockBits(bd);   // guaranteed even if Marshal.Copy throws
            }

            var tensor    = new float[3 * InputSize * InputSize];
            int planeSize = InputSize * InputSize;

            for (int y = 0; y < InputSize; y++)
            for (int x = 0; x < InputSize; x++)
            {
                int   b0 = y * stride + x * 4;
                float r  = raw[b0 + 2] / 255f;
                float g  = raw[b0 + 1] / 255f;
                float b  = raw[b0 + 0] / 255f;
                int   p  = y * InputSize + x;
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
    }

    // ─────────────────────────────────────────────────────────────────────────
    // File helpers
    // ─────────────────────────────────────────────────────────────────────────
    private static void MoveWithConflictResolution(string srcPath, string destDir)
    {
        string fileName = Path.GetFileName(srcPath);
        string destPath = Path.Combine(destDir, fileName);

        if (!File.Exists(destPath))
        {
            File.Move(srcPath, destPath);
            return;
        }

        // Conflict: same name already exists in the target directory.
        long srcSize  = new FileInfo(srcPath).Length;
        long destSize = new FileInfo(destPath).Length;

        if (srcSize == destSize && FilesMatch(srcPath, destPath))
        {
            Console.WriteLine($"           ↳ Duplicate of existing; deleting source.");
            File.Delete(srcPath);
        }
        else
        {
            string unique = BuildUniquePath(destDir,
                Path.GetFileNameWithoutExtension(fileName),
                Path.GetExtension(fileName));
            Console.WriteLine($"           ↳ Name clash; renaming to {Path.GetFileName(unique)}");
            File.Move(srcPath, unique);
        }
    }

    private static bool FilesMatch(string a, string b)
    {
        using var sha = SHA256.Create();
        return HashFile(sha, a).SequenceEqual(HashFile(sha, b));
    }

    private static byte[] HashFile(HashAlgorithm alg, string path)
    {
        using var fs = File.OpenRead(path);
        return alg.ComputeHash(fs);
    }

    private static string BuildUniquePath(string dir, string baseName, string ext)
    {
        int    counter   = 1;
        string candidate;
        do { candidate = Path.Combine(dir, $"{baseName}_{counter++}{ext}"); }
        while (File.Exists(candidate));
        return candidate;
    }

    private static int RenameJfifToJpg(string directory)
    {
        int count = 0;
        foreach (string path in Directory.EnumerateFiles(
            directory, "*.jfif", SearchOption.TopDirectoryOnly))
        {
            string dest = BuildUniquePath(directory,
                Path.GetFileNameWithoutExtension(path), ".jpg");
            try   { File.Move(path, dest); count++; }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"  [ERROR] Could not rename \"{Path.GetFileName(path)}\": {ex.Message}");
            }
        }
        return count;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Validation helpers
    // ─────────────────────────────────────────────────────────────────────────
    private static void ValidateFolderNames()
    {
        if (string.IsNullOrWhiteSpace(RealPhotoFolderName)    ||
            string.IsNullOrWhiteSpace(CartoonFolderName)      ||
            string.IsNullOrWhiteSpace(UnclassifiedFolderName))
            throw new InvalidOperationException("Folder names must not be empty.");

        if (new[] { RealPhotoFolderName, CartoonFolderName, UnclassifiedFolderName }
                .Distinct(StringComparer.OrdinalIgnoreCase).Count() != 3)
            throw new InvalidOperationException(
                "RealPhotoFolderName, CartoonFolderName and UnclassifiedFolderName must all be different.");
    }

    private static string EngineLabel() => (UseOnnxEngine, UseHeuristicEngine) switch
    {
        (true,  true)  => "ONNX + Heuristic (consensus)",
        (true,  false) => "ONNX only",
        (false, true)  => "Heuristic only",
        _              => "none"   // unreachable — guarded at entry
    };
}
