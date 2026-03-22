namespace TediousTasks;

/// <summary>
/// Immutable bag of heuristic feature scores for a single image. All values in [0, 1].
///
/// Features retained after four rounds of false-positive analysis:
///   real↑  (higher = more real photo): ChannelNoise, Palette, FlatRegion, FlatNoise
///   anime↑ (higher = more anime):      EdgeBimodal, InkOutline
///
/// Features permanently removed:
///   Saturation        — direction flipped R3, dataset-dependent
///   ColorTemp         — near-zero Cohen's d in 2/3 rounds
///   SkinDiscrete      — direction flipped R1→R2, ~0 separation in R4
///   JpegBlockArtifact — destroyed by LoadAndResize (bicubic erases 8px block boundaries)
///   GradientBimodality— calibration failure, scored 1.0 for 100% of all images
///   LocalPalette      — normalisation off by ~20x, scored ~0 for 100% of all images
/// </summary>
internal sealed class ImageFeatures
{
    public required double ChannelNoise { get; init; }  // real↑  dominant, consistent
    public required double Palette      { get; init; }  // real↑  strong R1,R3,R4
    public required double FlatRegion   { get; init; }  // real↑  fragile direction
    public required double FlatNoise    { get; init; }  // real↑  consistent R2,R3,R4
    public required double EdgeBimodal  { get; init; }  // anime↑ consistent all rounds
    public required double InkOutline   { get; init; }  // anime↑ consistent, weak magnitude

    // ── CSV serialisation ──────────────────────────────────────────────────────

    public static string CsvHeader =>
        "file," +
        "channel_noise(real↑),palette(real↑),flat_region(real↑),flat_noise(real↑)," +
        "edge_bimodal(anime↑),ink_outline(anime↑)," +
        "composite";

    public string ToCsvRow(string fileName, double composite) =>
        $"{fileName}," +
        $"{ChannelNoise:F4},{Palette:F4},{FlatRegion:F4},{FlatNoise:F4}," +
        $"{EdgeBimodal:F4},{InkOutline:F4}," +
        $"{composite:F4}";
}
