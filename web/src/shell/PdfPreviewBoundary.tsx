import { Component, type ErrorInfo, type ReactNode } from "react";
import { isChunkLoadError } from "@/lib/chunkLoadRecovery";

// Keep rejected PDF imports and rendering errors inside the file preview.
// Stale-chunk failures are rethrown from render so ChunkLoadErrorBoundary can refresh the page.
export class PdfPreviewBoundary extends Component<
  { children: ReactNode },
  { failed: boolean; error: unknown }
> {
  override state: { failed: boolean; error: unknown } = { failed: false, error: null };

  static getDerivedStateFromError(error: unknown) {
    return { failed: true, error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo) {
    if (isChunkLoadError(error)) return;
    console.error("PDF preview failed", error, info.componentStack);
  }

  override render() {
    if (!this.state.failed) return this.props.children;
    if (isChunkLoadError(this.state.error)) throw this.state.error;
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
