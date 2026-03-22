using System.Drawing;
using System.Drawing.Imaging;
using ImageMagick;

namespace TediousTasks;

/// <summary>
/// Shared image utilities used by <see cref="ImageClassifier"/> and
/// <see cref="FeatureReporter"/>.
/// </summary>
internal static class ImageUtils
{
    /// <summary>
    /// Image file extensions accepted as input by the pipeline.
    /// .jxl is the primary format after conversion.
    /// .webp is accepted because ~350 images were already converted to lossless WebP.
    /// .jfif files are renamed to .jpg before this list is consulted.
    /// </summary>
    public static readonly HashSet<string> SupportedExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".jxl"
        };

    /// <summary>
    /// Extensions that will be converted to JPEG XL by
    /// <see cref="ImageClassifier.ConvertImagesToJxl"/>.
    /// .jxl files are already in the target format and are skipped.
    /// </summary>
    public static readonly HashSet<string> ConvertToJxlExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"
        };

    /// <summary>
    /// Source extensions that are already lossy — converted with lossy JXL at
    /// high quality to avoid a second generation of lossless bloat.
    /// </summary>
    private static readonly HashSet<string> LossySourceExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg"
        };

    /// <summary>
    /// Converts a single image file to JPEG XL in-place, format-aware:
    ///
    ///   JPEG → lossy JXL quality=90
    ///     Already lossy; high-quality JXL recompress avoids lossless bloat
    ///     (lossless WebP/JXL of a JPEG is 3-5× larger than the original JPEG).
    ///     Quality 90 in JXL is visually near-identical to the JPEG source.
    ///
    ///   PNG / GIF / BMP / TIFF / WebP → lossless JXL
    ///     True lossless originals (or our own lossless WebP conversions);
    ///     every pixel is preserved exactly.
    ///
    /// The original is only deleted after a confirmed successful write.
    /// Returns the path of the new .jxl file, or null if conversion failed.
    /// </summary>
    public static string? ConvertToJxl(string sourcePath)
    {
        string destPath = Path.ChangeExtension(sourcePath, ".jxl");

        if (File.Exists(destPath))
            destPath = BuildUniquePath(
                Path.GetDirectoryName(sourcePath)!,
                Path.GetFileNameWithoutExtension(sourcePath),
                ".jxl");

        bool isLossy = LossySourceExtensions.Contains(Path.GetExtension(sourcePath));

        try
        {
            using var image = new MagickImage(sourcePath);

            if (isLossy)
            {
                // Lossy JXL at quality 90.
                // JXL quality scale: 100=lossless, lower=more compression.
                // 90 is visually near-identical to a typical JPEG source.
                image.Quality = 90;
                image.Settings.SetDefine("jxl:lossless", "false");
            }
            else
            {
                // Lossless JXL — pixel-perfect for PNG/BMP/TIFF/WebP originals.
                image.Settings.SetDefine("jxl:lossless", "true");
            }

            image.Write(destPath, MagickFormat.Jxl);

            if (!File.Exists(destPath) || new FileInfo(destPath).Length == 0)
            {
                Console.Error.WriteLine(
                    $"  [ERROR] JXL output empty or missing for \"{Path.GetFileName(sourcePath)}\"");
                if (File.Exists(destPath)) File.Delete(destPath);
                return null;
            }

            File.Delete(sourcePath);
            return destPath;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"  [ERROR] JXL conversion failed for \"{Path.GetFileName(sourcePath)}\": {ex.Message}");
            if (File.Exists(destPath)) File.Delete(destPath);
            return null;
        }
    }

    /// <summary>
    /// Loads an image and scales it so its longest edge is at most
    /// <paramref name="maxDimension"/> pixels. The caller owns the result.
    /// Uses System.Drawing (GDI+) for all formats except .jxl, which is
    /// decoded via Magick.NET (GDI+ does not support JPEG XL).
    /// </summary>
    public static System.Drawing.Bitmap LoadAndResize(string path, int maxDimension)
    {
        System.Drawing.Bitmap original;

        if (Path.GetExtension(path).Equals(".jxl", StringComparison.OrdinalIgnoreCase))
            original = JxlToBitmap(path);
        else
            original = new System.Drawing.Bitmap(path);

        using (original)
        {
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
    }

    /// <summary>
    /// Decodes a .jxl file to a System.Drawing.Bitmap via Magick.NET.
    /// Magick.NET reads the JXL pixels; we copy them into a 32bppArgb Bitmap
    /// that the rest of the pipeline (GDI+ / heuristic engine) can consume.
    /// </summary>
    public static System.Drawing.Bitmap JxlToBitmap(string path)
    {
        using var magick = new MagickImage(path);
        magick.ColorSpace = ColorSpace.sRGB;
        magick.Format     = MagickFormat.Bmp3;

        using var ms = new MemoryStream();
        magick.Write(ms);
        ms.Position = 0;
        // Return a copy so the MemoryStream can be disposed safely.
        return new System.Drawing.Bitmap(System.Drawing.Image.FromStream(ms));
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
