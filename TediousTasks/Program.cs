using TediousTasks;

// ── Configuration ──────────────────────────────────────────────────────────────
// Each media type has its own working directory.
// Leave empty to default to the executable's directory.

string storiesDirectory  = @"G:\Stories";
string picturesDirectory = @"E:\Pictures";
// string videosDirectory = @"";   // reserved for a future video classification step

if (string.IsNullOrWhiteSpace(storiesDirectory))
    storiesDirectory = AppContext.BaseDirectory;

if (string.IsNullOrWhiteSpace(picturesDirectory))
    picturesDirectory = AppContext.BaseDirectory;

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
Console.WriteLine("=== Step 1: Rename extension-less files ===");
TextFileChecker.RenameExtensionlessTextFiles(storiesDirectory);

// ── Step 2: Convert .txt files to UTF-8 + CRLF ────────────────────────────────
Console.WriteLine("\n=== Step 2: Convert to UTF-8 + CRLF ===");
// Change to LineEnding.Lf if you need Unix-style line endings.
EncodingConverter.TargetLineEnding = EncodingConverter.LineEnding.CrLf;
EncodingConverter.ConvertAllToUtf8(storiesDirectory);

// ── Step 3: Organise root-level .txt files into subdirectories ─────────────────
Console.WriteLine("\n=== Step 3: Organise files into subdirectories ===");
FileOrganizer.OrganizeFiles(storiesDirectory);

// ── Step 4: Convert images to lossless WebP ───────────────────────────────────
// Converts jpeg/png/gif/bmp/tiff → lossless WebP in-place.
// Lossless = zero data loss; every pixel is preserved exactly.
// Run this before classification so the classifier only handles .webp files.
// Existing .webp files are left untouched and accepted as input already.
Console.WriteLine("\n=== Step 4: Convert images to lossless WebP ===");
ImageClassifier.ConvertImagesToWebP(picturesDirectory);

// ── Step 5: Classify images ───────────────────────────────────────────────────
Console.WriteLine("\n=== Step 5: Classify images ===");

ImageClassifier.RealPhotoFolderName    = "RealPhotos";
ImageClassifier.CartoonFolderName      = "Cartoons";
ImageClassifier.UnclassifiedFolderName = "Unclassified";

// Engine selection:
//   Both true  → consensus mode; disagreements go to Unclassified/
//   One true   → that engine's verdict is used directly
//   Both false → step is skipped entirely
ImageClassifier.UseOnnxEngine      = true;
ImageClassifier.UseHeuristicEngine = true;

// Uncomment to override the default model path (beside the executable):
// ImageClassifier.ModelPath = @"C:\custom\path\model.onnx";

ImageClassifier.ClassifyImages(picturesDirectory);

// ── Step 6: Write feature reports for false-positive tuning ───────────────────
// After a run, move misclassified images into the two folders below, then
// re-run this step. The resulting CSVs show each image's raw feature scores
// and can be used to retune the heuristic weights in HeuristicEngine.cs.
Console.WriteLine("\n=== Step 6: Write feature reports for false positives ===");

FeatureReporter.FalsePositiveCartoonFolder = "FalsePositiveCartoon";    // real photos in Cartoons
FeatureReporter.FalsePositiveRealFolder    = "FalsePositiveRealPhoto";  // anime in RealPhotos
FeatureReporter.OutputCartoonCsv           = "features_false_cartoon.csv";
FeatureReporter.OutputRealCsv              = "features_false_real.csv";

FeatureReporter.WriteFeatureReports(picturesDirectory);

Console.WriteLine("\nAll done.");
return 0;
