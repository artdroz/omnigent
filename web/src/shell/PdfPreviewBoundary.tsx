import { Component, type ErrorInfo, type ReactNode } from "react";
import { isChunkLoadError } from "@/lib/chunkLoadRecovery";

// Keep rejected PDF imports and rendering errors inside the file preview.
// Stale-chunk failures are rethrown so ChunkLoadErrorBoundary can refresh the page.
export class PdfPreviewBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  override state = { failed: false };

  static getDerivedStateFromError(error: unknown) {
    if (isChunkLoadError(error)) throw error;
    return { failed: true };
  }

  override componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("PDF preview failed", error, info.componentStack);
  }

  override render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div
        role="alert"
        className="flex flex-col items-center justify-center p-8 text-center text-ui"
      >
        <p className="text-destructive">Unable to render PDF.</p>
        <p className="mt-2 text-muted-foreground">Download the file to open it in another app.</p>
      </div>
    );
  }
}
