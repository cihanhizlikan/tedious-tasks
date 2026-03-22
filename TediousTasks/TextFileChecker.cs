namespace TediousTasks;

/// <summary>
/// Finds files with no extension in the working directory tree and renames
/// them to .txt if their content is text; warns and leaves binary files alone.
/// </summary>
internal static class TextFileChecker
{
    // Bytes we treat as proof of binary content when found in the sample window.
    // NUL (0x00) virtually never appears in plain text files.
    private const int SampleSize = 8192;

    public static void RenameExtensionlessTextFiles(string rootDirectory)
    {
        var candidates = Directory
            .GetFiles(rootDirectory, "*", SearchOption.AllDirectories)
            .Where(f => string.IsNullOrEmpty(Path.GetExtension(f)))
            .ToList();

        if (candidates.Count == 0)
        {
            Console.WriteLine("  No extension-less files found.");
            return;
        }

        Console.WriteLine($"  Found {candidates.Count} extension-less file(s).");

        foreach (var filePath in candidates)
        {
            if (IsTextFile(filePath))
            {
                string newPath = filePath + ".txt";

                // If a .txt counterpart already exists, skip to avoid overwriting.
                if (File.Exists(newPath))
                {
                    Console.WriteLine($"  [SKIP] Target already exists, leaving as-is: {Rel(rootDirectory, filePath)}");
                    continue;
                }

                File.Move(filePath, newPath);
                Console.WriteLine($"  [RENAMED] {Rel(rootDirectory, filePath)} → {Path.GetFileName(newPath)}");
            }
            else
            {
                Console.WriteLine($"  [WARNING] Binary file detected, not renamed: {Rel(rootDirectory, filePath)}");
            }
        }
    }

    /// <summary>
    /// Heuristic: reads up to <see cref="SampleSize"/> bytes; if a NUL byte is
    /// found the file is considered binary.  This covers the vast majority of
    /// real-world cases without pulling in any third-party library.
    /// </summary>
    private static bool IsTextFile(string filePath)
    {
        try
        {
            using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            byte[] buffer = new byte[SampleSize];
            int bytesRead = fs.Read(buffer, 0, buffer.Length);

            for (int i = 0; i < bytesRead; i++)
            {
                if (buffer[i] == 0x00)
                    return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [ERROR] Could not read {filePath}: {ex.Message}");
            return false;
        }
    }

    private static string Rel(string root, string full) =>
        Path.GetRelativePath(root, full);
}
