// Text-content position helpers for TipTap / ProseMirror comment anchoring.
//
// Comments are anchored by (start_index, end_index) in the raw file and by
// anchor_content (the verbatim selected text).  These helpers bridge the gap
// between raw file offsets and ProseMirror integer positions.
//
// Strategy (both directions):
//   1. Build the PM text content with "\n" between blocks as a proxy for the
//      raw file content.
//   2. Locate anchor_content together with the text around it on the same line,
//      so a short selection resolves to the copy the user picked even when the
//      same text repeats nearby.  The scaled offset only breaks ties.
//   3. Map between text-content offset and PM position via binary search on
//      doc.textBetween(0, mid, "\n").length — O(log n · n) for typical docs.

import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { Comment } from "@/hooks/useComments";

const SEP = "\n";

/** Longest run of surrounding text tried on each side of the anchor. */
const MAX_CONTEXT = 256;
/** Shorter runs tried when the longer ones do not occur verbatim. */
const CONTEXT_STEPS = [128, 64, 32, 16, 8, 4, 2, 1, 0];

/**
 * Returns the occurrence of `needle` whose start is closest to `hint`.
 *
 * A fixed forward search window is not enough for short selections: another
 * copy of the same text can sit well within that window, and `indexOf`
 * returns the first one rather than the one the user selected.
 */
function nearestOccurrence(haystack: string, needle: string, hint: number): number {
  if (!needle) return -1;
  let best = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  let from = 0;
  while (from <= haystack.length) {
    const found = haystack.indexOf(needle, from);
    if (found === -1) break;
    const distance = Math.abs(found - hint);
    if (distance < bestDistance) {
      best = found;
      bestDistance = distance;
    } else if (found > hint) {
      break;
    }
    from = found + 1;
  }
  return best;
}

/** The text sharing a line with `[from, to)`, split into prefix and suffix. */
function lineContext(text: string, from: number, to: number): { prefix: string; suffix: string } {
  const lineStart = from > 0 ? text.lastIndexOf(SEP, from - 1) + 1 : 0;
  const nextSep = text.indexOf(SEP, to);
  const lineEnd = nextSep === -1 ? text.length : nextSep;
  return { prefix: text.slice(lineStart, from), suffix: text.slice(to, lineEnd) };
}

function contextLengths(available: number): number[] {
  const longest = Math.min(available, MAX_CONTEXT);
  return [longest, ...CONTEXT_STEPS.filter((n) => n < longest)];
}

/**
 * Locates `needle` in `haystack`, telling repeated copies apart by the text
 * that surrounded it where it came from. Markdown syntax exists only on the raw
 * side, so context pairs are tried from the longest total down to the bare
 * needle; `hint` breaks ties. Returns the start of `needle`, or -1.
 */
function locateWithContext(
  haystack: string,
  needle: string,
  prefix: string,
  suffix: string,
  hint: number,
): number {
  if (!needle || haystack.indexOf(needle) === -1) return -1;

  const pairs: [number, number][] = [];
  for (const p of contextLengths(prefix.length)) {
    for (const s of contextLengths(suffix.length)) pairs.push([p, s]);
  }
  pairs.sort((a, b) => b[0] + b[1] - (a[0] + a[1]) || Math.min(b[0], b[1]) - Math.min(a[0], a[1]));

  for (const [p, s] of pairs) {
    const context = prefix.slice(prefix.length - p) + needle + suffix.slice(0, s);
    const found = nearestOccurrence(haystack, context, hint - p);
    if (found !== -1) return found + p;
  }
  return -1;
}

/**
 * Returns the smallest PM position p where
 * doc.textBetween(0, p, SEP).length >= offset.
 *
 * Binary search over PM positions — O(log(doc.size) * doc.size).
 * Adequate for typical markdown documents (< 200 KB).
 */
function textOffsetToPmPos(doc: ProseMirrorNode, offset: number): number {
  const maxSize = doc.content.size;
  if (offset <= 0) return 0;
  const total = doc.textBetween(0, maxSize, SEP).length;
  if (offset >= total) return maxSize;
  let lo = 0;
  let hi = maxSize;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (doc.textBetween(0, mid, SEP).length < offset) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Finds the PM [from, to) range for a saved comment.
 *
 * Uses anchor_content as the text to locate. The raw text around start_index
 * identifies which copy of a repeated anchor the comment belongs to; the
 * start_index scaled by the textContent/rawContent ratio breaks ties.
 *
 * Returns null when anchor_content is absent or not found in the document.
 */
export function findPmRangeForComment(
  doc: ProseMirrorNode,
  comment: Comment,
  rawContent: string,
): { from: number; to: number } | null {
  const { anchor_content, start_index } = comment;
  if (!anchor_content?.trim()) return null;

  const textContent = doc.textBetween(0, doc.content.size, SEP);
  if (!textContent) return null;

  const hint =
    rawContent.length > 0 ? Math.round((start_index * textContent.length) / rawContent.length) : 0;

  // The surrounding raw text is only meaningful while the stored offset still
  // points at the anchor (the file may have changed since the comment was made).
  const rawEnd = start_index + anchor_content.length;
  const { prefix, suffix } =
    rawContent.slice(start_index, rawEnd) === anchor_content
      ? lineContext(rawContent, start_index, rawEnd)
      : { prefix: "", suffix: "" };

  const textFrom = locateWithContext(textContent, anchor_content, prefix, suffix, hint);
  if (textFrom === -1) return null;

  const from = textOffsetToPmPos(doc, textFrom);
  const to = textOffsetToPmPos(doc, textFrom + anchor_content.length);
  if (from >= to) return null;

  return { from, to };
}

/**
 * Computes raw-file comment anchor data for a PM selection range.
 *
 * Extracts the selected text as anchor_content, then searches for it in
 * rawContent together with the text around the selection; the scaled
 * text-content offset breaks ties between identical copies.
 *
 * When the text cannot be found verbatim in the raw file (e.g. multi-line
 * selections, table cells, or code blocks whose markdown syntax the parser
 * strips), falls back to proportionally scaled indices so the button is never
 * blocked.  The anchor_content from the PM doc is still used for re-locating
 * the highlight later via findPmRangeForComment.
 *
 * Returns null only when the selection contains no text.
 */
export function computeSelectionData(
  from: number,
  to: number,
  doc: ProseMirrorNode,
  rawContent: string,
): { start_index: number; end_index: number; anchor_content: string } | null {
  const anchor_content = doc.textBetween(from, to, SEP);
  if (!anchor_content.trim()) return null;

  const textContent = doc.textBetween(0, doc.content.size, SEP);
  const textFrom = doc.textBetween(0, from, SEP).length;
  const textTo = textFrom + anchor_content.length;

  const hint =
    textContent.length > 0 ? Math.round((textFrom * rawContent.length) / textContent.length) : 0;

  const { prefix, suffix } =
    textContent.slice(textFrom, textTo) === anchor_content
      ? lineContext(textContent, textFrom, textTo)
      : { prefix: "", suffix: "" };

  const idx = locateWithContext(rawContent, anchor_content, prefix, suffix, hint);

  // Fall back to proportional indices when the anchor text isn't found
  // verbatim (multi-line, table, code block selections).
  if (idx === -1) {
    return {
      start_index: hint,
      end_index: hint + anchor_content.length,
      anchor_content,
    };
  }

  return {
    start_index: idx,
    end_index: idx + anchor_content.length,
    anchor_content,
  };
}
