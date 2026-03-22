using System.Drawing;
using System.Drawing.Imaging;

namespace TediousTasks;

/// <summary>
/// Shared image utilities used by both <see cref="ImageClassifier"/>
/// and <see cref="FeatureReporter"/>.
/// </summary>
internal static class ImageUtils
{
    /// <summary>Image file extensions processed by the pipeline.</summary>
    public static readonly HashSet<string> SupportedExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"
            // .jfif → renamed to .jpg before processing (see ImageClassifier.RenameJfifToJpg)
            // .webp → requires SkiaSharp; add here once that package is referenced
        };

    /// <summary>
    /// Loads an image and scales it so its longest edge is at most
    /// <paramref name="maxDimension"/> pixels. If the image already fits,
    /// a copy is returned. The caller owns (and must dispose) the result.
    /// </summary>
    public static Bitmap LoadAndResize(string path, int maxDimension)
    {
        using var original = new Bitmap(path);
        int w = original.Width, h = original.Height;

        if (w <= maxDimension && h <= maxDimension)
            return new Bitmap(original);

        double scale = Math.Min((double)maxDimension / w, (double)maxDimension / h);
        int nw = (int)(w * scale), nh = (int)(h * scale);

        var resized = new Bitmap(nw, nh, PixelFormat.Format32bppArgb);
        using var g = Graphics.FromImage(resized);
        g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
        g.DrawImage(original, 0, 0, nw, nh);
        return resized;
    }
}
