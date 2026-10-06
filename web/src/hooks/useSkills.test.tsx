import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { applySessionSkillCatalog } from "@/lib/skillCatalogs";
import { fetchSkills, skillsQueryKey, useSkills } from "./useSkills";

const { fetchMock, subscribe, watchSkills, listeners } = vi.hoisted(() => ({
  fetchMock: vi.fn(),
  subscribe: vi.fn(),
  watchSkills: vi.fn(),
  listeners: new Set<(frame: unknown) => void>(),
}));
vi.mock("@/lib/identity", () => ({ authenticatedFetch: fetchMock }));
vi.mock("@/lib/sessionUpdatesSocket", () => ({
  subscribeHostSkills: (id: string, target: object, listener: (frame: unknown) => void) => {
    const unsubscribe = subscribe(listener);
    const unwatch = watchSkills(id, target);
    return () => {
      unsubscribe();
      unwatch();
    };
  },
}));
let client: QueryClient;
const target = { hostId: "host", harness: "claude-native", path: "/repo" };
const ready = {
  status: "ready" as const,
  revision: 1,
  skills: [{ name: "review", description: "Review" }],
};
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
beforeEach(() => {
  vi.clearAllMocks();
  listeners.clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  subscribe.mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  watchSkills.mockImplementation(() => vi.fn());
});
function push(value: object) {
  act(() => {
    for (const listener of listeners) listener(value);
  });
}
describe("pushed skills", () => {
  it("waits for host data without polling or a separate session subscription", async () => {
    const { result } = renderHook(() => useSkills({ target: { sessionId: "a" } }), { wrapper });
    expect(result.current.skillsStatus).toBe("loading");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(watchSkills).not.toHaveBeenCalled();
    act(() => applySessionSkillCatalog(client, "a", ready));
    await waitFor(() => expect(result.current.skills).toEqual(ready.skills));
  });
  it("uses data that arrived before the composer mounted", () => {
    applySessionSkillCatalog(client, "a", ready);
    const { result } = renderHook(() => useSkills({ target: { sessionId: "a" } }), { wrapper });
    expect(result.current.skills).toEqual(ready.skills);
  });
  it("isolates sessions and ignores older revisions", async () => {
    const a = renderHook(() => useSkills({ target: { sessionId: "a" } }), { wrapper });
    const b = renderHook(() => useSkills({ target: { sessionId: "b" } }), { wrapper });
    act(() => {
      applySessionSkillCatalog(client, "a", { ...ready, revision: 3 });
      applySessionSkillCatalog(client, "a", { ...ready, revision: 2, skills: [] });
      applySessionSkillCatalog(client, "b", { ...ready, skills: [] });
    });
    await waitFor(() => expect(a.result.current.skills).toEqual(ready.skills));
    await waitFor(() => expect(b.result.current.skillsStatus).toBe("ready"));
    expect(b.result.current.skills).toEqual([]);
  });
  it("retains the catalog on error but clears it on lost access", async () => {
    const { result } = renderHook(() => useSkills({ target: { sessionId: "a" } }), { wrapper });
    act(() => {
      applySessionSkillCatalog(client, "a", ready);
      applySessionSkillCatalog(client, "a", { status: "error", revision: 2 });
    });
    await waitFor(() => expect(result.current.skillsStatus).toBe("error"));
    expect(result.current.skills).toEqual(ready.skills);
    act(() => applySessionSkillCatalog(client, "a", { status: "unavailable", revision: 3 }));
    await waitFor(() => expect(result.current.skills).toEqual([]));
  });
  it("rejects a cached catalog from another workspace", () => {
    applySessionSkillCatalog(client, "a", {
      ...ready,
      host_id: "host",
      workspace: "/old",
      agent_id: "agent",
    });
    const { result } = renderHook(
      () =>
        useSkills({
          target: {
            sessionId: "a",
            scope: {
              hostId: "host",
              workspace: "/new",
              agentId: "agent",
              harness: "claude-native",
              subAgentName: null,
            },
          },
        }),
      { wrapper },
    );
    expect(result.current.skillsStatus).toBe("loading");
  });
  it("registers and releases pre-session targets and ignores stale frames", async () => {
    const unwatch = vi.fn();
    watchSkills.mockReturnValue(unwatch);
    const { result, rerender, unmount } = renderHook(
      ({ path }) => useSkills({ target: { ...target, path } }),
      { wrapper, initialProps: { path: "/repo" } },
    );
    const identity = JSON.stringify(skillsQueryKey(target));
    expect(watchSkills).toHaveBeenCalledWith(identity, {
      host_id: "host",
      harness: "claude-native",
      path: "/repo",
    });
    push({ type: "skills", target_id: identity, ...ready });
    await waitFor(() => expect(result.current.skills).toEqual(ready.skills));
    rerender({ path: "/other" });
    expect(unwatch).toHaveBeenCalledOnce();
    push({ type: "skills", target_id: identity, ...ready });
    expect(result.current.skills).toEqual([]);
    unmount();
    expect(unwatch).toHaveBeenCalledTimes(2);
  });
  it("waits for a complete target", () => {
    const { result } = renderHook(() => useSkills({ target: null, starting: true }), { wrapper });
    expect(result.current.skillsStatus).toBe("loading");
    expect(watchSkills).not.toHaveBeenCalled();
  });
  it("keeps explicit inventory reads available", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ skills: ready.skills })));
    expect(await fetchSkills(target, new AbortController().signal)).toEqual(ready.skills);
  });
});
