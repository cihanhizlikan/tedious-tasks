namespace TediousTasks;

/// <summary>
/// Immutable bag of heuristic feature scores for a single image. All values in [0, 1].
///
/// Removed after statistical analysis across 3 rounds of false-positive data:
///   Saturation  — direction flipped in round 3 (unreliable, dataset-dependent)
///   ColorTemp   — near-zero Cohen's d in 2/3 rounds (effectively dead weight)
///
/// Added:
///   JpegBlockArtifact  — detects 8-pixel periodic noise from JPEG/GIF compression
///   GradientBimodality — measures bimodal vs unimodal colour transition distribution
///   LocalPalette       — patch-level colour count (more robust than global palette)
///
/// Direction key:
///   anime↑  : higher value = more likely anime/cartoon
///   real↑   : higher value = more likely real photo (inverted in ScoreFeatures)
/// </summary>
internal sealed class ImageFeatures
{
    // ── Retained features ──────────────────────────────────────────────────────
    public required double ChannelNoise      { get; init; }  // real↑  dominant signal
    public required double FlatNoise         { get; init; }  // real↑  strong in rounds 2+3
    public required double InkOutline        { get; init; }  // anime↑ consistent all rounds
    public required double EdgeBimodal       { get; init; }  // anime↑ consistent all rounds
    public required double FlatRegion        { get; init; }  // anime↑ (rounds 2+3; low weight)
    public required double SkinDiscrete      { get; init; }  // anime↑ (fragile; low weight)
    public required double Palette           { get; init; }  // real↑  replaced by LocalPalette below but kept for CSV continuity

    // ── New features ───────────────────────────────────────────────────────────
    public required double JpegBlockArtifact { get; init; }  // anime↑ JPEG/GIF block periodicity
    public required double GradientBimodality{ get; init; }  // anime↑ hard cel-shading transitions
    public required double LocalPalette      { get; init; }  // real↑  patch-level colour diversity

    // ── CSV serialisation ──────────────────────────────────────────────────────

    public static string CsvHeader =>
        "file," +
        "channel_noise(real↑),flat_noise(real↑),ink_outline(anime↑),edge_bimodal(anime↑)," +
        "flat_region(anime↑),skin_discrete(anime↑),palette(real↑)," +
        "jpeg_block(anime↑),gradient_bimodal(anime↑),local_palette(real↑)," +
        "composite";

    public string ToCsvRow(string fileName, double composite) =>
        $"{fileName}," +
        $"{ChannelNoise:F4},{FlatNoise:F4},{InkOutline:F4},{EdgeBimodal:F4}," +
        $"{FlatRegion:F4},{SkinDiscrete:F4},{Palette:F4}," +
        $"{JpegBlockArtifact:F4},{GradientBimodality:F4},{LocalPalette:F4}," +
        $"{composite:F4}";
}
