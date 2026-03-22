using System.Text;

namespace TediousTasks;

/// <summary>
/// Converts every .txt file in the tree to UTF-8 (no BOM) with configurable
/// line endings, and removes duplicate .txt files before processing.
///
/// Default line ending: <see cref="LineEnding.CrLf"/> (Windows standard).
/// </summary>
internal static class EncodingConverter
{
    public enum LineEnding { CrLf, Lf }

    /// <summary>
    /// Line ending written to every converted file.
    /// Defaults to <see cref="LineEnding.CrLf"/> (Windows convention).
    /// </summary>
    public static LineEnding TargetLineEnding { get; set; } = LineEnding.CrLf;

    private static readonly UTF8Encoding Utf8NoBom =
        new(encoderShouldEmitUTF8Identifier: false);

    public static void ConvertAllToUtf8(string rootDirectory)
    {
        var files = Directory
            .GetFiles(rootDirectory, "*.txt", SearchOption.AllDirectories)
            .ToList();

        Console.WriteLine($"  Found {files.Count} .txt file(s).");

        int removed = DuplicateRemover.RemoveDuplicates(files);
        if (removed > 0)
            Console.WriteLine($"  Removed {removed} duplicate .txt file(s). {files.Count} remaining.");

        string targetNewline = TargetLineEnding == LineEnding.CrLf ? "\r\n" : "\n";
        string endingLabel   = TargetLineEnding == LineEnding.CrLf ? "CRLF"  : "LF";

        int converted = 0, skipped = 0, failed = 0;

        foreach (string filePath in files)
        {
            try
            {
                Encoding detected       = DetectEncoding(filePath);
                string   content        = File.ReadAllText(filePath, detected);
                bool     needsEncoding  = detected.CodePage != Encoding.UTF8.CodePage;
                bool     needsLineEnds  = NeedsLineEndingFix(content, TargetLineEnding);

                if (!needsEncoding && !needsLineEnds) { skipped++; continue; }

                // Normalise: collapse all CR/LF variants to bare LF, then expand.
                content = content.Replace("\r\n", "\n").Replace('\r', '\n');
                if (TargetLineEnding == LineEnding.CrLf)
                    content = content.Replace("\n", "\r\n");

                File.WriteAllText(filePath, content, Utf8NoBom);

                string reason = (needsEncoding, needsLineEnds) switch
                {
                    (true,  true)  => $"{detected.EncodingName} + endings → UTF-8 + {endingLabel}",
                    (true,  false) => $"{detected.EncodingName} → UTF-8",
                    (false, true)  => $"endings → {endingLabel}",
                    _              => string.Empty
                };
                Console.WriteLine($"  [CONVERTED] {Rel(rootDirectory, filePath)}  ({reason})");
                converted++;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"  [ERROR] {Rel(rootDirectory, filePath)}: {ex.Message}");
                failed++;
            }
        }

        Console.WriteLine($"  Result — converted: {converted}, already OK: {skipped}, failed: {failed}");
    }

    private static bool NeedsLineEndingFix(string content, LineEnding target) => target switch
    {
        LineEnding.CrLf =>
            // Needs fix if there are bare LFs not preceded by CR, or bare CRs without LF.
            (content.Contains('\n') && !content.Contains("\r\n")) ||
            (content.Contains('\r') && !content.Contains("\r\n")),
        LineEnding.Lf => content.Contains('\r'),
        _             => false
    };

    internal static Encoding DetectEncoding(string filePath)
    {
        byte[] bom    = new byte[4];
        int    bomRead;
        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            bomRead = fs.Read(bom, 0, 4);

        if (bomRead >= 3 && bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF)
            return Encoding.UTF8;
        if (bomRead >= 4 && bom[0] == 0xFF && bom[1] == 0xFE && bom[2] == 0x00 && bom[3] == 0x00)
            return Encoding.UTF32;
        if (bomRead >= 2 && bom[0] == 0xFF && bom[1] == 0xFE)
            return Encoding.Unicode;
        if (bomRead >= 2 && bom[0] == 0xFE && bom[1] == 0xFF)
            return Encoding.BigEndianUnicode;

        try
        {
            byte[] bytes = File.ReadAllBytes(filePath);
            new UTF8Encoding(false, throwOnInvalidBytes: true).GetString(bytes);
            return Encoding.UTF8;
        }
        catch { /* fall through to Windows-1252 check */ }

        if (File.ReadAllText(filePath, Encoding.Default).Any(c => c is >= '\x80' and <= '\x9F'))
            return Encoding.GetEncoding(1252);

        return Encoding.Default;
    }

    private static string Rel(string root, string full) => Path.GetRelativePath(root, full);
}
