"use strict";

/**
 * Schedule launch and final warnings for an Arca instance. Its box may shut
 * down after 60 idle minutes once the shutdown time has passed, outside
 * weekday business hours. Electron-free: all I/O and timing are injected.
 */

const DEFAULT_OPTIONS = Object.freeze({
  finalLeadMs: 60 * 60e3,
  launchMaxLeadMs: 8 * 60 * 60e3,
  maxTimerMs: 30 * 60e3,
  firstRetryMs: 2 * 60e3,
  retryMs: 15 * 60e3,
  refreshMs: 3 * 60 * 60e3,
  businessHours: Object.freeze({ start: 6, end: 18 }),
});
const TRANSIENT_STATUS_ERROR_KINDS = new Set(["timeout", "unreachable", "unknown"]);

/**
 * Find the current warning window's effective deadline.
 *
 * @param {number} shutdownAt
 * @param {number} now
 * @param {{isWeekday: (ms: number) => boolean, localHour: (ms: number) => number,
 *   localTimeAt: (ms: number, hour: number) => number,
 *   localDayOffset: (ms: number, days: number) => number,
 *   businessHours: {start: number, end: number}}} deps
 * @returns {number}
 */
function effectiveShutdownAt(
  shutdownAt,
  now,
  { isWeekday, localHour, localTimeAt, localDayOffset, businessHours },
) {
  const locked = (ms) =>
    isWeekday(ms) && localHour(ms) >= businessHours.start && localHour(ms) < businessHours.end;
  if (shutdownAt > now) {
    return locked(shutdownAt) ? localTimeAt(shutdownAt, businessHours.end) : shutdownAt;
  }
  if (locked(now)) return localTimeAt(now, businessHours.end);
  // Each weekday evening is a new warning window for an instance still running.
  for (let daysAgo = 0; daysAgo <= 7; daysAgo++) {
    const day = localDayOffset(now, -daysAgo);
    const end = localTimeAt(day, businessHours.end);
    if (isWeekday(day) && end <= now) return Math.max(shutdownAt, end);
  }
  return shutdownAt;
}

/**
 * Choose a prompt and its next final check for one effective deadline.
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
 * Watch an Arca instance and serialize warnings, extensions, and status reads.
 * @param {{
 *   readStatus: () => Promise<{ok: true, state: string | null, shutdownAt: number | null,
 *     rawShutdownTime?: string | null} | {ok: false, errorKind: string, error: string}>,
 *   extend: (mode: "default" | "overnight" | "workweek") => Promise<
 *     {ok: true, message: string} | {ok: false, errorKind: string, error: string}>,
 *   prompt: (request: {kind: "launch" | "final", shutdownAt: number,
 *     now: number, modes: string[]}) => Promise<{mode: string | null}>,
 *   onExtendResult?: (result: object) => void | Promise<void>,
 *   isEnabled: () => boolean,
 *   now?: () => number,
 *   setTimer?: (fn: () => void, ms: number) => unknown,
 *   clearTimer?: (timer: unknown) => void,
 *   offersWorkweek?: (ms: number) => boolean,
 *   isWeekday?: (ms: number) => boolean,
 *   localTimeAt?: (ms: number, hour: number) => number,
 *   localDayOffset?: (ms: number, days: number) => number,
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
  offersWorkweek = (ms) => ![4, 5].includes(new Date(ms).getDay()),
  isWeekday = (ms) => {
    const day = new Date(ms).getDay();
    return day >= 1 && day <= 5;
  },
  localTimeAt = (ms, hour) => {
    const date = new Date(ms);
    date.setHours(hour, 0, 0, 0);
    return date.getTime();
  },
  localDayOffset = (ms, days) => {
    const date = new Date(ms);
    date.setDate(date.getDate() + days);
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
  let failedReads = 0;

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
      armAt(now() + (failedReads <= 1 ? settings.firstRetryMs : settings.retryMs));
    } catch (error) {
      safeLog(`arca shutdown: retry scheduling failed: ${error}`);
    }
  }

  function armRefresh() {
    try {
      armAt(now() + settings.refreshMs);
    } catch (error) {
      safeLog(`arca shutdown: refresh scheduling failed: ${error}`);
    }
  }

  async function readAndCache({ confirmExtend = false } = {}) {
    if (!confirmExtend && !enabled()) {
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
      if (TRANSIENT_STATUS_ERROR_KINDS.has(status?.errorKind)) {
        failedReads++;
        if (!confirmExtend || enabled()) armRetry();
      } else {
        failedReads = 0;
        if (!confirmExtend || enabled()) armRefresh();
      }
      return false;
    }
    failedReads = 0;
    if (status.state !== "running") {
      if (!pendingFresh) launchPending = false;
      shutdownAt = null;
      effectiveAt = null;
      clearScheduled();
      safeLog(`arca shutdown: instance state is ${status.state}`);
      return false;
    }
    if (!Number.isFinite(status.shutdownAt)) {
      if (!pendingFresh) launchPending = false;
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
      localDayOffset,
      businessHours: settings.businessHours,
    });
    return true;
  }

  async function notifyExtend(result) {
    if (disposed) return;
    try {
      await onExtendResult(result);
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
      localDayOffset,
      businessHours: settings.businessHours,
    });
    for (const key of prompted) {
      if (Number(key.slice(key.indexOf(":") + 1)) < effectiveAt) prompted.delete(key);
    }
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
    const modes = offersWorkweek(now()) ? ["overnight", "workweek"] : ["overnight"];
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
      await notifyExtend({
        ok: false,
        mode,
        errorKind: result?.errorKind ?? "unknown",
        error: result?.error ?? "Extending Arca failed.",
      });
      if (disposed) return;
      if (!pendingFresh) await planCached(false);
      return;
    }
    clearScheduled();
    const active = await readAndCache({ confirmExtend: true });
    if (disposed) return;
    phase = "extending";
    await notifyExtend({ ok: true, mode, shutdownAt: active ? effectiveAt : null });
    if (disposed) return;
    if (active && !pendingFresh && enabled()) await planCached(false);
  }

  async function evaluate(launch) {
    if (disposed || busy) return;
    busy = true;
    clearScheduled();
    try {
      const active = await readAndCache();
      if (active && !pendingFresh) {
        launchPending = false;
        await planCached(launch);
      }
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
        void evaluate(launchPending);
      } else {
        if (!disposed && timer === null && enabled()) armRefresh();
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
    requestFresh(launchPending);
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
