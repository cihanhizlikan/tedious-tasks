namespace TediousTasks;

/// <summary>
/// Immutable bag of the nine heuristic feature scores computed for a single image.
/// All values are in [0, 1].
///
/// Directional notes (derived from statistical analysis of labelled false-positives):
///   Higher = more ANIME : Saturation, EdgeBimodal, InkOutline
///   Higher = more REAL  : Palette, FlatRegion, SkinDiscrete, ColorTemp,
///                         FlatNoise, ChannelNoise
/// </summary>
internal sealed class ImageFeatures
{
    public required double Palette       { get; init; }   // real↑
    public required double Saturation    { get; init; }   // anime↑
    public required double FlatRegion    { get; init; }   // real↑
    public required double EdgeBimodal   { get; init; }   // anime↑
    public required double InkOutline    { get; init; }   // anime↑
    public required double SkinDiscrete  { get; init; }   // real↑
    public required double FlatNoise     { get; init; }   // real↑  (near-zero separating power)
    public required double ColorTemp     { get; init; }   // real↑
    public required double ChannelNoise  { get; init; }   // real↑  (dominant signal)

    // ── CSV serialisation ─────────────────────────────────────────────────────

    public static string CsvHeader =>
        "file," +
        "palette(real↑),saturation(anime↑),flat_region(real↑),edge_bimodal(anime↑)," +
        "ink_outline(anime↑),skin_discrete(real↑),flat_noise(weak),color_temp(real↑)," +
        "channel_noise(real↑),composite";

    public string ToCsvRow(string fileName, double composite) =>
        $"{fileName}," +
        $"{Palette:F4},{Saturation:F4},{FlatRegion:F4},{EdgeBimodal:F4}," +
        $"{InkOutline:F4},{SkinDiscrete:F4},{FlatNoise:F4},{ColorTemp:F4}," +
        $"{ChannelNoise:F4},{composite:F4}";
}
