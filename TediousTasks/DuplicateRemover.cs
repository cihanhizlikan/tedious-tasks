using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace TediousTasks;

/// <summary>
/// Generic file deduplication utility shared by all pipeline steps.
///
/// Algorithm:
///   1. Group files by exact byte size — O(n) dictionary pass.
///      Files with a unique size cannot be duplicates; they are never hashed.
///   2. Within each size-collision group, compute SHA-256 in parallel.
///   3. Within each hash group, keep the alphabetically first filename;
///      delete all others and remove them from the working list.
///
/// Thread safety: the public method is not thread-safe with respect to the
/// <paramref name="files"/> list — call it from a single thread before
/// launching any parallel work over that list.
/// </summary>
public static class DuplicateRemover
{
    /// <summary>
    /// Removes duplicate files from <paramref name="files"/> in-place.
    /// Duplicate files are deleted from disk and removed from the list.
    /// The first file alphabetically within each duplicate group is kept.
    /// </summary>
    /// <param name="files">
    ///   Mutable list of absolute file paths. Modified in-place: duplicates
    ///   are removed from the list after being deleted from disk.
    /// </param>
    /// <param name="parallelism">
    ///   Maximum hashing threads. 0 = one per logical processor.
    /// </param>
    /// <returns>Number of duplicate files deleted.</returns>
    public static int RemoveDuplicates(List<string> files, int parallelism = 0)
    {
        // ── Stage 1: size buckets (free filter) ───────────────────────────────
        var bySize = files
            .GroupBy(f => new FileInfo(f).Length)
            .Where(g => g.Count() > 1)
            .ToList();

        if (bySize.Count == 0) return 0;

        // ── Stage 2: hash candidates in parallel ──────────────────────────────
        int actualParallelism = parallelism <= 0 ? Environment.ProcessorCount : parallelism;
        var hashMap = new ConcurrentDictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        Parallel.ForEach(
            bySize.SelectMany(g => g),
            new ParallelOptions { MaxDegreeOfParallelism = actualParallelism },
            path =>
            {
                using var sha  = SHA256.Create();
                using var fs   = File.OpenRead(path);
                hashMap[path]  = Convert.ToHexString(sha.ComputeHash(fs));
            });

        // ── Stage 3: keep first alphabetically, delete the rest ───────────────
        var toDelete = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var sizeGroup in bySize)
        foreach (var hashGroup in sizeGroup
                     .GroupBy(f => hashMap[f])
                     .Where(g => g.Count() > 1))
        {
            var sorted = hashGroup
                .OrderBy(f => Path.GetFileName(f), StringComparer.OrdinalIgnoreCase)
                .ToList();

            string keep = sorted[0];

            for (int i = 1; i < sorted.Count; i++)
            {
                string dupe = sorted[i];
                try
                {
                    Console.WriteLine(
                        $"  [DUPE] Deleting \"{Path.GetFileName(dupe)}\" " +
                        $"(identical to \"{Path.GetFileName(keep)}\")");
                    File.Delete(dupe);
                    toDelete.Add(dupe);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"  [ERROR] Could not delete \"{Path.GetFileName(dupe)}\": {ex.Message}");
                }
            }
        }

        files.RemoveAll(f => toDelete.Contains(f));
        return toDelete.Count;
    }
}
