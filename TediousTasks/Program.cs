using TediousTasks;

// ── Configuration ──────────────────────────────────────────────────────────────
// Each media type has its own working directory.
// Leave a path as empty string to default to the executable's directory.

string storiesDirectory = @"G:\Stories";
string picturesDirectory = @"E:\Pictures";
// string videosDirectory = @"";  // reserved for future video classification step

if (string.IsNullOrWhiteSpace(storiesDirectory))
    storiesDirectory = AppContext.BaseDirectory;

if (string.IsNullOrWhiteSpace(picturesDirectory))
    picturesDirectory = AppContext.BaseDirectory;

// Validate directories
foreach (var (label, dir) in new[] { ("Stories", storiesDirectory), ("Pictures", picturesDirectory) })
{
    if (!Directory.Exists(dir))
    {
        Console.Error.WriteLine($"[ERROR] {label} directory does not exist: {dir}");
        return 1;
    }
}

Console.WriteLine($"Stories  directory: {storiesDirectory}");
Console.WriteLine($"Pictures directory: {picturesDirectory}\n");

// ── Step 1: Rename extension-less text files to .txt ──────────────────────────
//Console.WriteLine("=== Step 1: Rename extension-less files ===");
//TextFileChecker.RenameExtensionlessTextFiles(storiesDirectory);

// ── Step 2: Convert .txt files to UTF-8 + LF ──────────────────────────────────
//Console.WriteLine("\n=== Step 2: Convert to UTF-8 + LF ===");
//EncodingConverter.ConvertAllToUtf8Lf(storiesDirectory);

// ── Step 3: Organise root-level .txt files into subdirectories ─────────────────
//Console.WriteLine("\n=== Step 3: Organise files into subdirectories ===");
//FileOrganizer.OrganizeFiles(storiesDirectory);

// ── Step 4: Remove leading tab characters from paragraphs ─────────────────────
//Console.WriteLine("\n=== Step 4: Remove leading tabs from paragraphs ===");
//TabRemover.RemoveLeadingTabs(storiesDirectory);

// ── Step 5: Classify images – dual engine (ONNX + heuristic, consensus only) ──
Console.WriteLine("\n=== Step 5: Classify images ===");

ImageClassifier.RealPhotoFolderName    = "RealPhotos";
ImageClassifier.CartoonFolderName      = "Cartoons";
ImageClassifier.UnclassifiedFolderName = "Unclassified";

// Path to model.onnx (default: beside the executable, copied via .csproj)
// ImageClassifier.ModelPath = @"C:\custom\path\model.onnx";

ImageClassifier.ClassifyImages(picturesDirectory);

// ── Step 6: Write feature reports for manual false-positive correction ─────────
// Move misclassified images into the two folders below, then run this step.
// Two CSV files will be written with raw feature scores for every image,
// which can be fed back here to tune the heuristic weights.
Console.WriteLine("\n=== Step 6: Write feature reports for false positives ===");

FeatureReporter.FalsePositiveCartoonFolder = "FalsePositiveCartoon";   // real photos wrongly sent to Cartoons
FeatureReporter.FalsePositiveRealFolder    = "FalsePositiveRealPhoto"; // anime wrongly sent to RealPhotos
FeatureReporter.OutputCartoonCsv           = "features_false_cartoon.csv";
FeatureReporter.OutputRealCsv              = "features_false_real.csv";

FeatureReporter.WriteFeatureReports(picturesDirectory);

Console.WriteLine("\nAll done.");
return 0;
