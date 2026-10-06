import type { QueryClient } from "@tanstack/react-query";
import type { SkillSummary } from "@/lib/types";

export interface SkillCatalog {
  status: "ready" | "error" | "unavailable";
  revision: number;
  skills?: SkillSummary[];
  host_id?: string;
  workspace?: string;
  agent_id?: string | null;
  sub_agent_name?: string | null;
}

export function parseSkillCatalog(data: Record<string, unknown>): SkillCatalog | null {
  if (!["ready", "error", "unavailable"].includes(String(data.status))) return null;
  if (
    typeof data.revision !== "number" ||
    !Number.isSafeInteger(data.revision) ||
    data.revision < 0
  )
    return null;
  if (
    data.status === "ready" &&
    (!Array.isArray(data.skills) ||
      !data.skills.every(
        (skill) => skill && typeof skill.name === "string" && typeof skill.description === "string",
      ))
  )
    return null;
  return data as unknown as SkillCatalog;
}

export function catalogMatchesScope(
  catalog: SkillCatalog,
  scope?: {
    hostId?: string | null;
    workspace?: string | null;
    agentId?: string | null;
    subAgentName?: string | null;
  },
): boolean {
  return (
    !scope ||
    (!catalog.skills && catalog.status !== "ready") ||
    (catalog.host_id === scope.hostId &&
      catalog.workspace === scope.workspace &&
      catalog.agent_id === scope.agentId &&
      (catalog.sub_agent_name ?? null) === (scope.subAgentName ?? null))
  );
}

export function mergeSkillCatalog(
  previous: SkillCatalog | undefined,
  catalog: SkillCatalog,
): SkillCatalog {
  if (previous && previous.revision > catalog.revision) return previous;
  return catalog.status === "error"
    ? { ...previous, ...catalog, skills: previous?.skills }
    : catalog;
}

export function applySessionSkillCatalog(
  client: QueryClient,
  sessionId: string,
  catalog: SkillCatalog,
): void {
  client.setQueryData<SkillCatalog>(["session-skill-catalog", sessionId], (old) =>
    mergeSkillCatalog(old, catalog),
  );
  client.setQueriesData<SkillCatalog>(
    {
      queryKey: ["skills", sessionId],
      predicate: (query) =>
        catalogMatchesScope(
          catalog,
          (query.queryKey[2] as { scope?: Parameters<typeof catalogMatchesScope>[1] })?.scope,
        ),
    },
    (old) => mergeSkillCatalog(old, catalog),
  );
}
