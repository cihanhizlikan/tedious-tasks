using TediousTasks;

// ── Configuration ──────────────────────────────────────────────────────────────
// Set this to whatever root directory you want to process.
// Defaults to the directory that contains the executable when left as an empty string.
string workingDirectory = @"E:\";

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

// ── Step 5: Classify images into Real Photos / Cartoon-Anime folders ──────────
Console.WriteLine("\n=== Step 5: Classify images ===");

// ↓↓ Configure destination folder names here — they must not be identical ↓↓
ImageClassifier.RealPhotoFolderName = "RealPhotos";
ImageClassifier.CartoonFolderName   = "Cartoons";

ImageClassifier.ClassifyImages(workingDirectory);

Console.WriteLine("\nAll done.");
return 0;
