using System.Text;

namespace TediousTasks;

/// <summary>
/// Converts every .txt file in the tree to UTF-8 (no BOM) with LF line endings.
/// Files that are already UTF-8-LF are left untouched.
/// </summary>
internal static class EncodingConverter
{
    private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

    public static void ConvertAllToUtf8Lf(string rootDirectory)
    {
        var files = Directory.GetFiles(rootDirectory, "*.txt", SearchOption.AllDirectories);
        Console.WriteLine($"  Found {files.Length} .txt file(s).");

        int converted = 0, skipped = 0, failed = 0;

        foreach (var filePath in files)
        {
            try
            {
                Encoding detected = DetectEncoding(filePath);
                string content = File.ReadAllText(filePath, detected);

                bool needsEncodingFix = detected.CodePage != Encoding.UTF8.CodePage;
                bool needsLineEndingFix = content.Contains('\r');

                if (!needsEncodingFix && !needsLineEndingFix)
                {
                    skipped++;
                    continue;
                }

                // Normalise line endings to LF only
                content = content.Replace("\r\n", "\n").Replace('\r', '\n');

                File.WriteAllText(filePath, content, Utf8NoBom);

                string reason = (needsEncodingFix, needsLineEndingFix) switch
                {
                    (true,  true)  => $"{detected.EncodingName} + CRLF → UTF-8 + LF",
                    (true,  false) => $"{detected.EncodingName} → UTF-8",
                    (false, true)  => "CRLF → LF",
                    _              => string.Empty
                };

                Console.WriteLine($"  [CONVERTED] {Rel(rootDirectory, filePath)}  ({reason})");
                converted++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [ERROR] {Rel(rootDirectory, filePath)}: {ex.Message}");
                failed++;
            }
        }

        Console.WriteLine($"  Result — converted: {converted}, already OK: {skipped}, failed: {failed}");
    }

    // ── Encoding detection ────────────────────────────────────────────────────

    internal static Encoding DetectEncoding(string filePath)
    {
        // Read first 4 bytes to check for a BOM.
        byte[] bom = new byte[4];
        int bomRead;
        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            bomRead = fs.Read(bom, 0, 4);

        if (bomRead >= 3 && bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF)
            return Encoding.UTF8;

        if (bomRead >= 4 && bom[0] == 0xFF && bom[1] == 0xFE && bom[2] == 0x00 && bom[3] == 0x00)
            return Encoding.UTF32; // UTF-32 LE

        if (bomRead >= 2 && bom[0] == 0xFF && bom[1] == 0xFE)
            return Encoding.Unicode; // UTF-16 LE

        if (bomRead >= 2 && bom[0] == 0xFE && bom[1] == 0xFF)
            return Encoding.BigEndianUnicode; // UTF-16 BE

        // No BOM — try strict UTF-8 validation.
        try
        {
            byte[] bytes = File.ReadAllBytes(filePath);
            var strictUtf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);
            strictUtf8.GetString(bytes); // throws if invalid
            return Encoding.UTF8;
        }
        catch { /* fall through */ }

        // Check for Windows-1252 high bytes (0x80–0x9F range)
        string defaultContent = File.ReadAllText(filePath, Encoding.Default);
        if (defaultContent.Any(c => c is >= '\x80' and <= '\x9F'))
            return Encoding.GetEncoding(1252);

        return Encoding.Default;
    }

    private static string Rel(string root, string full) =>
        Path.GetRelativePath(root, full);
}
