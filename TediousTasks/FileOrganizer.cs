using System.Text.RegularExpressions;

namespace TediousTasks;

/// <summary>
/// Moves .txt files in the root of the working directory into sub-directories
/// derived from their names.
///
/// Rule: a file named X-Y.txt (where Y is the trailing numeric segment) is moved
/// to workingDirectory/X/X-Y.txt.
///
/// "Trailing numeric segment" means the last hyphen-separated token that is a
/// pure integer, e.g.:
///   apple-1-1.txt          → subdirectory  apple-1
///   austins-new-life-1.txt → subdirectory  austins-new-life
///
/// Conflict resolution:
///   Identical content  → delete the source (already moved).
///   Different content  → leave source and emit a warning.
/// </summary>
internal static class FileOrganizer
{
    // Group 1 = everything before the last -<digits> segment (the sub-dir name).
    private static readonly Regex TrailingNumber =
        new(@"^(.+)-(\d+)$", RegexOptions.Compiled);

    public static void OrganizeFiles(string rootDirectory)
    {
        var files = Directory
            .GetFiles(rootDirectory, "*.txt", SearchOption.TopDirectoryOnly)
            .ToList();

        if (files.Count == 0)
        {
            Console.WriteLine("  No .txt files in root directory.");
            return;
        }

        int moved = 0, skipped = 0, warnings = 0, noMatch = 0;

        foreach (string filePath in files)
        {
            string baseName = Path.GetFileNameWithoutExtension(filePath);
            var    match    = TrailingNumber.Match(baseName);

            if (!match.Success) { noMatch++; continue; }

            string subDirName = match.Groups[1].Value;
            string subDirPath = Path.Combine(rootDirectory, subDirName);
            string destPath   = Path.Combine(subDirPath, Path.GetFileName(filePath));

            Directory.CreateDirectory(subDirPath);

            if (File.Exists(destPath))
            {
                if (FilesIdentical(filePath, destPath))
                {
                    File.Delete(filePath);
                    Console.WriteLine(
                        $"  [DELETED DUPLICATE] {Path.GetFileName(filePath)} " +
                        $"(identical copy exists in {subDirName}/)");
                    skipped++;
                }
                else
                {
                    Console.Error.WriteLine(
                        $"  [WARNING] Cannot move {Path.GetFileName(filePath)}: " +
                        $"a different file already exists at {Path.GetRelativePath(rootDirectory, destPath)}");
                    warnings++;
                }
                continue;
            }

            File.Move(filePath, destPath);
            Console.WriteLine($"  [MOVED] {Path.GetFileName(filePath)} → {subDirName}/");
            moved++;
        }

        Console.WriteLine(
            $"  Result — moved: {moved}, duplicate removed: {skipped}, " +
            $"warnings: {warnings}, no pattern match: {noMatch}");
    }

    /// <summary>
    /// Size-check first (O(1)), then buffered byte-for-byte comparison.
    /// Uses the same two-stage approach as <see cref="DuplicateRemover"/> but
    /// without hashing, since we are only ever comparing two specific files.
    /// </summary>
    private static bool FilesIdentical(string pathA, string pathB)
    {
        if (new FileInfo(pathA).Length != new FileInfo(pathB).Length)
            return false;

        const int BufferSize = 65536;
        byte[] bufA = new byte[BufferSize];
        byte[] bufB = new byte[BufferSize];

        using var fsA = new FileStream(pathA, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var fsB = new FileStream(pathB, FileMode.Open, FileAccess.Read, FileShare.Read);

        int read;
        while ((read = fsA.Read(bufA, 0, BufferSize)) > 0)
        {
            if (fsB.Read(bufB, 0, BufferSize) != read) return false;
            if (!bufA.AsSpan(0, read).SequenceEqual(bufB.AsSpan(0, read))) return false;
        }
        return true;
    }
}
