using System.Drawing;
using System.Drawing.Imaging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Webp;

namespace TediousTasks;

/// <summary>
/// Shared image utilities used by <see cref="ImageClassifier"/> and
/// <see cref="FeatureReporter"/>.
/// </summary>
internal static class ImageUtils
{
    /// <summary>
    /// Image file extensions accepted as input by the pipeline.
    /// .webp is fully supported for both reading and writing.
    /// .jfif files are renamed to .jpg before this list is consulted.
    /// </summary>
    public static readonly HashSet<string> SupportedExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"
        };

    /// <summary>
    /// Extensions that will be converted to lossless WebP by
    /// <see cref="ImageClassifier.ConvertImagesToWebP"/>.
    /// BMP and TIFF are intentionally included — they are large and benefit most.
    /// </summary>
    public static readonly HashSet<string> ConvertToWebPExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"
        };

    /// <summary>
    /// Converts a single image file to lossless WebP in-place:
    ///   1. Reads the source file with ImageSharp.
    ///   2. Encodes to a lossless WebP alongside the original.
    ///   3. Verifies the output file is non-empty.
    ///   4. Deletes the original only after a successful write.
    ///
    /// Lossless WebP preserves every pixel exactly — no data loss.
    /// Animated GIFs retain all frames.
    ///
    /// Returns the path of the new .webp file, or null if conversion failed.
    /// </summary>
    public static string? ConvertToWebP(string sourcePath)
    {
        string destPath = Path.ChangeExtension(sourcePath, ".webp");

        // If a .webp with the same stem already exists, build a unique name.
        if (File.Exists(destPath))
            destPath = BuildUniquePath(
                Path.GetDirectoryName(sourcePath)!,
                Path.GetFileNameWithoutExtension(sourcePath),
                ".webp");

        try
        {
            using var image = SixLabors.ImageSharp.Image.Load(sourcePath);
            image.Save(destPath, new WebpEncoder
            {
                FileFormat     = WebpFileFormatType.Lossless,
                // Quality is ignored for lossless but set explicitly for clarity.
                Quality        = 100,
                // Use the slowest (most efficient) compression method.
                // Method 6 = best compression, no quality loss.
                Method         = WebpEncodingMethod.BestQuality,
                TransparentColorMode = WebpTransparentColorMode.Preserve,
            });

            // Sanity check — make sure the output was actually written.
            if (!File.Exists(destPath) || new FileInfo(destPath).Length == 0)
            {
                Console.Error.WriteLine(
                    $"  [ERROR] WebP output empty or missing for \"{Path.GetFileName(sourcePath)}\"");
                if (File.Exists(destPath)) File.Delete(destPath);
                return null;
            }

            // Only delete the original after a confirmed successful write.
            File.Delete(sourcePath);
            return destPath;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"  [ERROR] WebP conversion failed for \"{Path.GetFileName(sourcePath)}\": {ex.Message}");
            // Clean up any partial output.
            if (File.Exists(destPath)) File.Delete(destPath);
            return null;
        }
    }

    /// <summary>
    /// Loads an image and scales it so its longest edge is at most
    /// <paramref name="maxDimension"/> pixels. The caller owns the result.
    /// Uses System.Drawing (GDI+) — suitable for the heuristic engine's
    /// pixel-level analysis.
    /// </summary>
    public static System.Drawing.Bitmap LoadAndResize(string path, int maxDimension)
    {
        using var original = new System.Drawing.Bitmap(path);
        int w = original.Width, h = original.Height;

        if (w <= maxDimension && h <= maxDimension)
            return new System.Drawing.Bitmap(original);

        double scale = Math.Min((double)maxDimension / w, (double)maxDimension / h);
        int nw = (int)(w * scale), nh = (int)(h * scale);

        var resized = new System.Drawing.Bitmap(nw, nh, PixelFormat.Format32bppArgb);
        using var g = System.Drawing.Graphics.FromImage(resized);
        g.InterpolationMode =
            System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
        g.DrawImage(original, 0, 0, nw, nh);
        return resized;
    }

    private static string BuildUniquePath(string dir, string baseName, string ext)
    {
        int    counter  = 1;
        string candidate;
        do { candidate = Path.Combine(dir, $"{baseName}_{counter++}{ext}"); }
        while (File.Exists(candidate));
        return candidate;
    }
}
