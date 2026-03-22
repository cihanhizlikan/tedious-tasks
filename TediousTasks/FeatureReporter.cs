using System.Collections.Concurrent;
using System.Drawing;

namespace TediousTasks;

/// <summary>
/// Step 6 — writes heuristic feature vectors for manually labelled false-positive
/// images to CSV files, enabling data-driven tuning of the heuristic weights.
///
/// Workflow:
///   1. After a classification run, move misclassified images into:
///        FalsePositiveCartoon/   – real photos wrongly sent to Cartoons
///        FalsePositiveRealPhoto/ – anime wrongly sent to RealPhotos
///   2. Run the program; this step writes two CSV files.
///   3. Feed the CSVs back to retune ScoreFeatures() weights.
/// </summary>
public static class FeatureReporter
{
    public static string FalsePositiveCartoonFolder { get; set; } = "FalsePositiveCartoon";
    public static string FalsePositiveRealFolder    { get; set; } = "FalsePositiveRealPhoto";
    public static string OutputCartoonCsv           { get; set; } = "features_false_cartoon.csv";
    public static string OutputRealCsv              { get; set; } = "features_false_real.csv";

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

    // ─────────────────────────────────────────────────────────────────────────

    private static void ProcessFolder(string folder, string csvPath, string description)
    {
        if (!Directory.Exists(folder))
        {
            Console.WriteLine($"  Skipping '{Path.GetFileName(folder)}' — folder not found.");
            return;
        }

        var files = Directory
            .EnumerateFiles(folder, "*.*", SearchOption.TopDirectoryOnly)
            .Where(f => ImageUtils.SupportedExtensions.Contains(Path.GetExtension(f)))
            .OrderBy(f => f)
            .ToList();

        if (files.Count == 0)
        {
            Console.WriteLine($"  No images in '{Path.GetFileName(folder)}'.");
            return;
        }

        Console.WriteLine($"  Analysing {files.Count} image(s) in '{Path.GetFileName(folder)}'...");

        var rows = new ConcurrentBag<(string Name, ImageFeatures Features)>();

        Parallel.ForEach(files,
            new ParallelOptions { MaxDegreeOfParallelism = ImageClassifier.ActualParallelism() },
            path =>
            {
                try
                {
                    using var bmp = ImageUtils.LoadAndResize(path, maxDimension: 512);
                    var px        = new PixelBuffer(bmp);
                    var features  = HeuristicEngine.ComputeFeatures(px);
                    rows.Add((Path.GetFileName(path), features));
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  [ERROR] {Path.GetFileName(path)}: {ex.Message}");
                }
            });

        Writecsv(csvPath, description, rows);
        Console.WriteLine($"  Written: {csvPath}  ({rows.Count} row(s))");
    }

    private static void Writecsv(
        string csvPath,
        string description,
        ConcurrentBag<(string Name, ImageFeatures Features)> rows)
    {
        using var sw = new StreamWriter(csvPath, append: false, encoding: System.Text.Encoding.UTF8);
        sw.WriteLine($"# {description}");
        sw.WriteLine($"# Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        sw.WriteLine($"# channel_noise is the dominant real/anime separator (real photos score high).");
        sw.WriteLine(ImageFeatures.CsvHeader);

        foreach (var (name, features) in rows.OrderBy(r => r.Name))
        {
            double composite = HeuristicEngine.ScoreFeatures(features);
            sw.WriteLine(features.ToCsvRow(name, composite));
        }
    }
}
