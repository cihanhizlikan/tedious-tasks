using TediousTasks;

// ── Configuration ──────────────────────────────────────────────────────────────
string workingDirectory = @"E:\Pictures";

if (string.IsNullOrWhiteSpace(workingDirectory))
    workingDirectory = AppContext.BaseDirectory;

if (!Directory.Exists(workingDirectory))
{
    Console.Error.WriteLine($"[ERROR] Working directory does not exist: {workingDirectory}");
    return 1;
}

Console.WriteLine($"Working directory: {workingDirectory}\n");

// ── Step 1: Rename extension-less text files to .txt ──────────────────────────
//Console.WriteLine("=== Step 1: Rename extension-less files ===");
//TextFileChecker.RenameExtensionlessTextFiles(workingDirectory);

// ── Step 2: Convert .txt files to UTF-8 + LF ──────────────────────────────────
//Console.WriteLine("\n=== Step 2: Convert to UTF-8 + LF ===");
//EncodingConverter.ConvertAllToUtf8Lf(workingDirectory);

// ── Step 3: Organise root-level .txt files into subdirectories ─────────────────
//Console.WriteLine("\n=== Step 3: Organise files into subdirectories ===");
//FileOrganizer.OrganizeFiles(workingDirectory);

// ── Step 4: Remove leading tab characters from paragraphs ─────────────────────
//Console.WriteLine("\n=== Step 4: Remove leading tabs from paragraphs ===");
//TabRemover.RemoveLeadingTabs(workingDirectory);

// ── Step 5: Classify images – dual engine (ONNX + heuristic, consensus only) ──
Console.WriteLine("\n=== Step 5: Classify images ===");

// Destination folder names — all three must be different
ImageClassifier.RealPhotoFolderName    = "RealPhotos";
ImageClassifier.CartoonFolderName      = "Cartoons";
ImageClassifier.UnclassifiedFolderName = "Unclassified";  // engines disagreed

// Path to model.onnx (default: beside the executable, copied from project root via .csproj)
// ImageClassifier.ModelPath = @"C:\custom\path\model.onnx";

ImageClassifier.ClassifyImages(workingDirectory);

// ── Step 6: Write feature reports for manual false-positive correction ─────────
// Place misclassified images into the two folders below, then run this step.
// Two CSV files will be written with the raw feature scores for every image,
// which can be fed back to tune the heuristic weights.
Console.WriteLine("\n=== Step 6: Write feature reports for false positives ===");

FeatureReporter.FalsePositiveCartoonFolder = "FalsePositiveCartoon";   // real photos wrongly sent to Cartoons
FeatureReporter.FalsePositiveRealFolder    = "FalsePositiveRealPhoto"; // anime wrongly sent to RealPhotos
FeatureReporter.OutputCartoonCsv           = "features_false_cartoon.csv";
FeatureReporter.OutputRealCsv              = "features_false_real.csv";

FeatureReporter.WriteFeatureReports(workingDirectory);

Console.WriteLine("\nAll done.");
return 0;
