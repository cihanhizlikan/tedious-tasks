namespace TediousTasks;

/// <summary>
/// Removes leading tab characters from every line in every .txt file found
/// under the working directory (recursive).  Files are overwritten in place.
/// Files that contain no leading tabs are left untouched.
/// </summary>
internal static class TabRemover
{
    public static void RemoveLeadingTabs(string rootDirectory)
    {
        var files = Directory.GetFiles(rootDirectory, "*.txt", SearchOption.AllDirectories);
        Console.WriteLine($"  Found {files.Length} .txt file(s).");

        int modified = 0, skipped = 0, failed = 0;

        foreach (var filePath in files)
        {
            try
            {
                string original = File.ReadAllText(filePath, System.Text.Encoding.UTF8);

                // Strip one or more leading tab characters from every line.
                string updated = StripLeadingTabs(original);

                if (updated == original)
                {
                    skipped++;
                    continue;
                }

                File.WriteAllText(filePath, updated, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
                Console.WriteLine($"  [STRIPPED TABS] {Path.GetRelativePath(rootDirectory, filePath)}");
                modified++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [ERROR] {Path.GetRelativePath(rootDirectory, filePath)}: {ex.Message}");
                failed++;
            }
        }

        Console.WriteLine($"  Result — modified: {modified}, no tabs found: {skipped}, failed: {failed}");
    }

    /// <summary>
    /// Splits the content on newlines, removes all leading tab characters from
    /// each line, then reassembles with the same line endings that were present.
    /// </summary>
    private static string StripLeadingTabs(string content)
    {
        // We process character-by-character to preserve whatever line ending
        // style the file already uses (LF, CRLF, or CR) without normalising it
        // again here — that was already handled in Step 2.
        var result = new System.Text.StringBuilder(content.Length);
        bool atLineStart = true;

        foreach (char c in content)
        {
            if (atLineStart && c == '\t')
            {
                // Consume (skip) the leading tab.
                continue;
            }

            atLineStart = c is '\n' or '\r';
            result.Append(c);
        }

        return result.ToString();
    }
}
