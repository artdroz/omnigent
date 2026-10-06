import { useEffect } from "react";
import { refreshHostSkills, subscribeHostSkills } from "@/lib/sessionUpdatesSocket";
import {
  catalogMatchesScope,
  mergeSkillCatalog,
  parseSkillCatalog,
  type SkillCatalog,
} from "@/lib/skillCatalogs";
import { skipToken, useQuery, useQueryClient } from "@tanstack/react-query";
import { authenticatedFetch } from "@/lib/identity";
import { ApiError } from "@/lib/sessionsApi";
import type { Session, SkillSummary, SkillsStatus } from "@/lib/types";

export type SkillsTarget =
  | {
      sessionId: string;
      hostId?: never;
      harness?: never;
      path?: never;
      agentId?: never;
      /** Cache dependencies for the scope derived by the server. */
      scope?: Pick<Session, "hostId" | "harness" | "workspace" | "agentId" | "subAgentName">;
    }
  | {
      sessionId?: never;
      hostId: string;
      harness: string;
      path: string;
      agentId?: string;
      scope?: never;
    };

/** Explicit discovery for inventory and compatibility callers. */
export async function fetchSkills(
  target: SkillsTarget,
  signal: AbortSignal,
): Promise<SkillSummary[]> {
  const params =
    target.sessionId !== undefined
      ? new URLSearchParams({ session_id: target.sessionId })
      : new URLSearchParams({ host_id: target.hostId, harness: target.harness, path: target.path });
  if (target.agentId !== undefined) params.set("agent_id", target.agentId);
  const response = await authenticatedFetch(`/v1/skills?${params}`, { signal });
  if (!response.ok) {
    throw new ApiError(`${response.status} ${response.statusText}`, response.status, null);
  }
  const body = (await response.json()) as { skills?: SkillSummary[] };
  if (!Array.isArray(body.skills)) throw new Error("Invalid host skills response");
  return body.skills;
}

/** Cache key for a composer's streamed catalog. */
export function skillsQueryKey(target: SkillsTarget | null) {
  return ["skills", target?.sessionId, target] as const;
}

interface SkillsOptions {
  /** Null until the composer has a complete discovery target. */
  target: SkillsTarget | null;
  enabled?: boolean;
  starting?: boolean;
}

/** Discover composer skills, with optional session authorization and agent scope. */
export function useSkills({ target, enabled = true, starting = false }: SkillsOptions) {
  const client = useQueryClient();
  const available = enabled && target !== null;
  const key = skillsQueryKey(target);
  const identity = JSON.stringify(key);
  const query = useQuery<SkillCatalog>({
    queryKey: key,
    queryFn: skipToken,
    enabled: false,
    staleTime: Infinity,
    initialData: () => {
      if (!target?.sessionId) return undefined;
      const cached = client.getQueryData<SkillCatalog>(["session-skill-catalog", target.sessionId]);
      return cached && catalogMatchesScope(cached, target.scope) ? cached : undefined;
    },
  });
  useEffect(() => {
    const stableKey = JSON.parse(identity) as ReturnType<typeof skillsQueryKey>;
    const stableTarget = stableKey[2];
    if (!available || !stableTarget || stableTarget.sessionId !== undefined) return;
    return subscribeHostSkills(
      identity,
      {
        host_id: stableTarget.hostId,
        harness: stableTarget.harness,
        path: stableTarget.path,
        ...(stableTarget.agentId !== undefined ? { agent_id: stableTarget.agentId } : {}),
      },
      (frame) => {
        if (frame.type !== "skills" || frame.target_id !== identity) return;
        const catalog = parseSkillCatalog(frame as unknown as Record<string, unknown>);
        if (catalog)
          client.setQueryData<SkillCatalog>(stableKey, (old) => mergeSkillCatalog(old, catalog));
      },
    );
    // The serialized key includes every target field.
  }, [client, available, identity]);

  const skillsStatus: SkillsStatus = !available
    ? starting
      ? "loading"
      : "unavailable"
    : (query.data?.status ?? "loading");
  const refetch = async () => {
    if (!target || !available) return;
    try {
      if (target.sessionId !== undefined) {
        await authenticatedFetch(
          `/v1/sessions/${encodeURIComponent(target.sessionId)}/skills/refresh`,
          { method: "POST" },
        );
      } else {
        refreshHostSkills(target.hostId, identity);
      }
    } catch {
      client.setQueryData<SkillCatalog>(key, (old) =>
        mergeSkillCatalog(old, { status: "error", revision: old?.revision ?? 0 }),
      );
    }
  };
  return { skills: available ? (query.data?.skills ?? []) : [], skillsStatus, refetch };
}
