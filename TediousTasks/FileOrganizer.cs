using System.Text.RegularExpressions;

namespace TediousTasks;

/// <summary>
/// Moves .txt files in the root of the working directory into sub-directories
/// derived from their names.
///
/// Rule: a file named  X-Y.txt  (where Y is the trailing numeric segment)
/// is moved to  workingDirectory/X/X-Y.txt.
///
/// "Trailing numeric segment" means the last hyphen-separated token that is
/// a pure integer, e.g.:
///   apple-1-1.txt      → subdirectory  apple-1
///   austins-new-life-1.txt → subdirectory  austins-new-life
///
/// If the target file already exists:
///   • Identical content  → delete the source (treat as already moved).
///   • Different content  → leave the source and emit a warning.
/// </summary>
internal static class FileOrganizer
{
    // Matches a filename that ends with  -<digits>  (before the extension).
    // Group 1 = everything before the last  -<digits>  segment  (the sub-dir name).
    // Group 2 = the last  -<digits>  segment (kept as part of the filename).
    private static readonly Regex TrailingNumber =
        new(@"^(.+)-(\d+)$", RegexOptions.Compiled);

    public static void OrganizeFiles(string rootDirectory)
    {
        // Only look at files directly inside rootDirectory, not subdirectories.
        var files = Directory
            .GetFiles(rootDirectory, "*.txt", SearchOption.TopDirectoryOnly)
            .ToList();

        if (files.Count == 0)
        {
            Console.WriteLine("  No .txt files in root directory.");
            return;
        }

        int moved = 0, skipped = 0, warnings = 0, noMatch = 0;

        foreach (var filePath in files)
        {
            string baseName = Path.GetFileNameWithoutExtension(filePath); // e.g. "apple-1-1"
            var match = TrailingNumber.Match(baseName);

            if (!match.Success)
            {
                // Filename doesn't follow the X-Y pattern; leave it alone.
                noMatch++;
                continue;
            }

            string subDirName = match.Groups[1].Value; // e.g. "apple-1"
            string subDirPath = Path.Combine(rootDirectory, subDirName);
            string destPath   = Path.Combine(subDirPath, Path.GetFileName(filePath));

            // Create the subdirectory if it doesn't exist yet.
            Directory.CreateDirectory(subDirPath);

            if (File.Exists(destPath))
            {
                if (FilesAreIdentical(filePath, destPath))
                {
                    // Already there with identical content — remove duplicate.
                    File.Delete(filePath);
                    Console.WriteLine($"  [DELETED DUPLICATE] {Path.GetFileName(filePath)} (identical copy exists in {subDirName}/)");
                    skipped++;
                }
                else
                {
                    Console.WriteLine($"  [WARNING] Cannot move {Path.GetFileName(filePath)}: a different file already exists at {Path.GetRelativePath(rootDirectory, destPath)}");
                    warnings++;
                }

                continue;
            }

            File.Move(filePath, destPath);
            Console.WriteLine($"  [MOVED] {Path.GetFileName(filePath)} → {subDirName}/");
            moved++;
        }

        Console.WriteLine($"  Result — moved: {moved}, duplicate removed: {skipped}, warnings: {warnings}, no pattern match: {noMatch}");
    }

    /// <summary>
    /// Byte-for-byte comparison to decide whether two files are identical.
    /// Uses buffered reading so large files don't load entirely into RAM.
    /// </summary>
    private static bool FilesAreIdentical(string pathA, string pathB)
    {
        var infoA = new FileInfo(pathA);
        var infoB = new FileInfo(pathB);

        if (infoA.Length != infoB.Length)
            return false;

        const int bufferSize = 65536;
        byte[] bufA = new byte[bufferSize];
        byte[] bufB = new byte[bufferSize];

        using var fsA = new FileStream(pathA, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var fsB = new FileStream(pathB, FileMode.Open, FileAccess.Read, FileShare.Read);

        int readA, readB;
        while ((readA = fsA.Read(bufA, 0, bufferSize)) > 0)
        {
            readB = fsB.Read(bufB, 0, bufferSize);
            if (readA != readB) return false;
            if (!bufA.AsSpan(0, readA).SequenceEqual(bufB.AsSpan(0, readB))) return false;
        }

        return true;
    }
}
