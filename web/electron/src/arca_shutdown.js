"use strict";

const DEFAULT_OPTIONS = Object.freeze({
  finalLeadMs: 60 * 60e3,
  launchMaxLeadMs: 8 * 60 * 60e3,
  maxTimerMs: 30 * 60e3,
  retryMs: 15 * 60e3,
  businessHours: Object.freeze({ start: 6, end: 18 }),
});

/**
 * Account for the weekday business-hours lock when the nominal deadline has
 * already passed.
 *
 * @param {number} shutdownAt
 * @param {number} now
 * @param {{isWeekday: (ms: number) => boolean, localHour: (ms: number) => number,
 *   localTimeAt: (ms: number, hour: number) => number,
 *   businessHours: {start: number, end: number}}} deps
 * @returns {number}
 */
function effectiveShutdownAt(
  shutdownAt,
  now,
  { isWeekday, localHour, localTimeAt, businessHours },
) {
  if (
    shutdownAt < now &&
    isWeekday(now) &&
    localHour(now) >= businessHours.start &&
    localHour(now) < businessHours.end
  ) {
    return localTimeAt(now, businessHours.end);
  }
  return shutdownAt;
}

/**
 * @param {{effectiveAt: number, now: number, launch: boolean,
 *   prompted: Set<string>, finalLeadMs: number, launchMaxLeadMs: number}} input
 * @returns {{promptNow: "launch" | "final" | null, finalAt: number | null}}
 */
function planPrompt({ effectiveAt, now, launch, prompted, finalLeadMs, launchMaxLeadMs }) {
  const lead = effectiveAt - now;
  const finalPending = !prompted.has(`final:${effectiveAt}`);
  let promptNow = null;
  if (finalPending && lead <= finalLeadMs) promptNow = "final";
  else if (launch && lead <= launchMaxLeadMs && !prompted.has(`launch:${effectiveAt}`)) {
    promptNow = "launch";
  }
  return { promptNow, finalAt: finalPending ? effectiveAt - finalLeadMs : null };
}

/**
 * @param {{
 *   readStatus: () => Promise<{ok: true, state: string | null, shutdownAt: number | null,
 *     rawShutdownTime?: string | null} | {ok: false, errorKind: string, error: string}>,
 *   extend: (mode: "default" | "overnight" | "workweek") => Promise<
 *     {ok: true, message: string} | {ok: false, errorKind: string, error: string}>,
 *   prompt: (request: {kind: "launch" | "final", shutdownAt: number,
 *     now: number, modes: string[]}) => Promise<{mode: string | null}>,
 *   onExtendResult?: (result: object) => void,
 *   isEnabled: () => boolean,
 *   now?: () => number,
 *   setTimer?: (fn: () => void, ms: number) => unknown,
 *   clearTimer?: (timer: unknown) => void,
 *   isFriday?: (ms: number) => boolean,
 *   isWeekday?: (ms: number) => boolean,
 *   localTimeAt?: (ms: number, hour: number) => number,
 *   localHour?: (ms: number) => number,
 *   log?: (message: string) => void,
 *   options?: Partial<typeof DEFAULT_OPTIONS>,
 * }} deps
 */
function createArcaShutdownWatch({
  readStatus,
  extend,
  prompt,
  onExtendResult = () => {},
  isEnabled,
  now = () => Date.now(),
  setTimer = (fn, ms) => {
    const timer = setTimeout(fn, ms);
    timer.unref?.();
    return timer;
  },
  clearTimer = clearTimeout,
  isFriday = (ms) => new Date(ms).getDay() === 5,
  isWeekday = (ms) => {
    const day = new Date(ms).getDay();
    return day >= 1 && day <= 5;
  },
  localTimeAt = (ms, hour) => {
    const date = new Date(ms);
    date.setHours(hour, 0, 0, 0);
    return date.getTime();
  },
  localHour = (ms) => new Date(ms).getHours() + new Date(ms).getMinutes() / 60,
  log = () => {},
  options = {},
}) {
  const settings = {
    ...DEFAULT_OPTIONS,
    ...options,
    businessHours: { ...DEFAULT_OPTIONS.businessHours, ...options.businessHours },
  };
  const prompted = new Set();
  let started = false;
  let disposed = false;
  let phase = "idle";
  let shutdownAt = null;
  let effectiveAt = null;
  let nextCheckAt = null;
  let timer = null;
  let busy = false;
  let pendingFresh = false;
  let launchPending = true;

  function safeLog(message) {
    try {
      log(message);
    } catch {
      // Logging cannot interrupt the watch.
    }
  }

  function enabled() {
    try {
      return isEnabled();
    } catch (error) {
      safeLog(`arca shutdown: eligibility failed: ${error}`);
      return false;
    }
  }

  function clearScheduled() {
    if (timer !== null) {
      try {
        clearTimer(timer);
      } catch (error) {
        safeLog(`arca shutdown: clearing timer failed: ${error}`);
      }
    }
    timer = null;
    nextCheckAt = null;
  }

  function armAt(target) {
    clearScheduled();
    if (disposed) return;
    nextCheckAt = target;
    try {
      const delay = Math.min(Math.max(target - now(), 0), settings.maxTimerMs);
      timer = setTimer(() => {
        timer = null;
        nextCheckAt = null;
        if (disposed) return;
        try {
          if (!enabled()) {
            phase = "idle";
          } else if (now() < target) {
            armAt(target);
          } else {
            requestFresh(launchPending);
          }
        } catch (error) {
          safeLog(`arca shutdown: timer failed: ${error}`);
          requestFresh(launchPending);
        }
      }, delay);
      if (!busy) phase = "scheduled";
    } catch (error) {
      nextCheckAt = null;
      safeLog(`arca shutdown: scheduling failed: ${error}`);
    }
  }

  function armRetry() {
    try {
      armAt(now() + settings.retryMs);
    } catch (error) {
      safeLog(`arca shutdown: retry scheduling failed: ${error}`);
    }
  }

  async function readAndCache() {
    if (!enabled()) {
      clearScheduled();
      shutdownAt = null;
      effectiveAt = null;
      phase = "idle";
      return false;
    }
    phase = "reading";
    let status;
    try {
      status = await readStatus();
    } catch (error) {
      status = { ok: false, errorKind: "unknown", error: String(error) };
    }
    if (disposed) return false;
    if (!status?.ok) {
      shutdownAt = null;
      effectiveAt = null;
      safeLog(`arca shutdown: status failed (${status?.errorKind ?? "unknown"}): ${status?.error}`);
      armRetry();
      return false;
    }
    launchPending = false;
    if (status.state !== "running") {
      shutdownAt = null;
      effectiveAt = null;
      clearScheduled();
      safeLog(`arca shutdown: instance state is ${status.state}`);
      return false;
    }
    if (status.shutdownAt === null) {
      shutdownAt = null;
      effectiveAt = null;
      clearScheduled();
      safeLog(`arca shutdown: no shutdown time (${status.rawShutdownTime ?? "null"})`);
      return false;
    }
    shutdownAt = status.shutdownAt;
    effectiveAt = effectiveShutdownAt(shutdownAt, now(), {
      isWeekday,
      localHour,
      localTimeAt,
      businessHours: settings.businessHours,
    });
    return true;
  }

  function notifyExtend(result) {
    if (disposed) return;
    try {
      onExtendResult(result);
    } catch (error) {
      safeLog(`arca shutdown: result callback failed: ${error}`);
    }
  }

  async function planCached(launch) {
    if (disposed || pendingFresh || shutdownAt === null) return;
    if (!enabled()) {
      clearScheduled();
      phase = "idle";
      return;
    }
    effectiveAt = effectiveShutdownAt(shutdownAt, now(), {
      isWeekday,
      localHour,
      localTimeAt,
      businessHours: settings.businessHours,
    });
    const plan = planPrompt({
      effectiveAt,
      now: now(),
      launch,
      prompted,
      finalLeadMs: settings.finalLeadMs,
      launchMaxLeadMs: settings.launchMaxLeadMs,
    });
    if (plan.finalAt !== null && plan.finalAt > now()) armAt(plan.finalAt);
    else clearScheduled();
    if (!plan.promptNow) return;
    if (!enabled()) {
      clearScheduled();
      phase = "idle";
      return;
    }

    const kind = plan.promptNow;
    const promptedAt = effectiveAt;
    const modes = isFriday(now()) ? ["overnight"] : ["overnight", "workweek"];
    prompted.add(`${kind}:${promptedAt}`);
    phase = "prompting";
    let answer;
    try {
      answer = await prompt({ kind, shutdownAt: promptedAt, now: now(), modes });
    } catch (error) {
      safeLog(`arca shutdown: prompt failed: ${error}`);
    }
    if (disposed) return;
    if (!modes.includes(answer?.mode)) {
      if (!pendingFresh) await planCached(false);
      return;
    }

    const mode = answer.mode;
    phase = "extending";
    let result;
    try {
      result = await extend(mode);
    } catch (error) {
      result = { ok: false, errorKind: "unknown", error: String(error) };
    }
    if (disposed) return;
    if (!result?.ok) {
      notifyExtend({
        ok: false,
        mode,
        errorKind: result?.errorKind ?? "unknown",
        error: result?.error ?? "Extending Arca failed.",
      });
      if (!pendingFresh) await planCached(false);
      return;
    }
    clearScheduled();
    const active = await readAndCache();
    if (disposed) return;
    if (active && !pendingFresh) await planCached(false);
    notifyExtend({ ok: true, mode, shutdownAt: active ? effectiveAt : null });
  }

  async function evaluate(launch) {
    if (disposed || busy) return;
    busy = true;
    clearScheduled();
    try {
      const active = await readAndCache();
      if (active && !pendingFresh) await planCached(launch);
    } catch (error) {
      if (!disposed) {
        safeLog(`arca shutdown: evaluation failed: ${error}`);
        armRetry();
      }
    } finally {
      busy = false;
      if (!disposed && pendingFresh) {
        pendingFresh = false;
        clearScheduled();
        void evaluate(false);
      } else {
        phase = timer !== null ? "scheduled" : "idle";
      }
    }
  }

  function requestFresh(launch = false) {
    if (disposed) return;
    if (busy) pendingFresh = true;
    else void evaluate(launch);
  }

  function start() {
    if (disposed || busy || timer !== null) return;
    started = true;
    void evaluate(launchPending);
  }

  function onResume() {
    if (!started || disposed) return;
    clearScheduled();
    requestFresh();
  }

  function dispose() {
    disposed = true;
    clearScheduled();
    phase = "idle";
  }

  function getState() {
    return {
      started,
      disposed,
      phase,
      shutdownAt,
      effectiveAt,
      nextCheckAt,
      prompted: [...prompted],
    };
  }

  return { start, onResume, dispose, getState };
}

module.exports = { DEFAULT_OPTIONS, effectiveShutdownAt, planPrompt, createArcaShutdownWatch };
