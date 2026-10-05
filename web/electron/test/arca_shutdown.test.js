"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const {
  DEFAULT_OPTIONS,
  createArcaShutdownWatch,
  effectiveShutdownAt,
  planPrompt,
} = require("../src/arca_shutdown");

const HOUR = 60 * 60e3;
const MINUTE = 60e3;
const DAY = 24 * HOUR;

function weekdayOf(ms) {
  return ((Math.floor(ms / DAY) % 7) + 7) % 7;
}

function calendarWeekday(ms) {
  return weekdayOf(ms) < 5;
}

function flush() {
  return Array.from({ length: 12 }).reduce((pending) => pending.then(() => {}), Promise.resolve());
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function fakeClock(initial = 0) {
  let time = initial;
  let nextId = 0;
  const timers = new Map();
  const delays = [];
  return {
    now: () => time,
    setTimer(fn, ms) {
      const id = ++nextId;
      delays.push(ms);
      timers.set(id, { fn, at: time + ms });
      return id;
    },
    clearTimer(id) {
      timers.delete(id);
    },
    jump(ms) {
      time += ms;
    },
    async advance(ms) {
      time += ms;
      const due = [...timers].filter(([, timer]) => timer.at <= time);
      for (const [id, timer] of due) {
        if (!timers.delete(id)) continue;
        timer.fn();
      }
      await flush();
    },
    async advanceTo(target) {
      const due = [...timers].filter(([, timer]) => timer.at <= target);
      if (due.length === 0) {
        time = target;
        await flush();
        return;
      }
      const [id, timer] = due.reduce((earliest, entry) =>
        entry[1].at < earliest[1].at ? entry : earliest,
      );
      time = timer.at;
      timers.delete(id);
      timer.fn();
      await flush();
      await this.advanceTo(target);
    },
    delays,
    timers,
  };
}

function running(shutdownAt) {
  return { ok: true, state: "running", shutdownAt };
}

function makeWatch({
  time = 0,
  statuses = [running(2 * HOUR)],
  readStatus,
  prompt,
  extend,
  enabled = () => true,
  offersWorkweek = () => true,
  isWeekday = () => true,
  options = {},
  onExtendResult = () => {},
  log = () => {},
} = {}) {
  const clock = fakeClock(time);
  const prompts = [];
  let reads = 0;
  let lastStatus = statuses.at(-1) ?? running(2 * HOUR);
  const watch = createArcaShutdownWatch({
    readStatus: async () => {
      reads++;
      if (readStatus) return readStatus();
      if (statuses.length > 0) lastStatus = statuses.shift();
      return lastStatus;
    },
    prompt: async (request) => {
      prompts.push(request);
      return prompt ? prompt(request) : { mode: null };
    },
    extend: extend ?? (async () => ({ ok: true, message: "extended" })),
    onExtendResult,
    isEnabled: enabled,
    now: clock.now,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
    offersWorkweek,
    isWeekday,
    localHour: (ms) => (((ms % DAY) + DAY) % DAY) / HOUR,
    localTimeAt: (ms, hour) => Math.floor(ms / DAY) * DAY + hour * HOUR,
    localDayOffset: (ms, days) => ms + days * DAY,
    log,
    options,
  });
  return {
    watch,
    clock,
    prompts,
    get reads() {
      return reads;
    },
  };
}

describe("Arca shutdown planning helpers", () => {
  it("groups past deadlines by eligibility window and moves future locked times to 18:00", () => {
    const deps = {
      isWeekday: calendarWeekday,
      localHour: (ms) => (((ms % DAY) + DAY) % DAY) / HOUR,
      localTimeAt: (ms, hour) => Math.floor(ms / DAY) * DAY + hour * HOUR,
      localDayOffset: (ms, days) => ms + days * DAY,
      businessHours: { start: 6, end: 18 },
    };
    assert.equal(effectiveShutdownAt(11 * HOUR, 12 * HOUR, deps), 18 * HOUR);
    assert.equal(effectiveShutdownAt(13 * HOUR, 12 * HOUR, deps), 18 * HOUR);
    assert.equal(effectiveShutdownAt(20 * HOUR, 12 * HOUR, deps), 20 * HOUR);
    assert.equal(effectiveShutdownAt(11 * HOUR, 20 * HOUR, deps), 18 * HOUR);
    assert.equal(effectiveShutdownAt(19 * HOUR, 20 * HOUR, deps), 19 * HOUR);
    const fridayEnd = 4 * DAY + 18 * HOUR;
    assert.equal(effectiveShutdownAt(fridayEnd, 5 * DAY + 10 * HOUR, deps), fridayEnd);
    assert.equal(effectiveShutdownAt(fridayEnd, 6 * DAY + 10 * HOUR, deps), fridayEnd);
    const sundayEnd = 6 * DAY + 18 * HOUR;
    assert.equal(effectiveShutdownAt(sundayEnd, 7 * DAY + 3 * HOUR, deps), sundayEnd);
  });

  it("prioritizes final and records a final check only while unprompted", () => {
    const input = {
      effectiveAt: 2 * HOUR,
      now: 0,
      launch: true,
      prompted: new Set(),
      finalLeadMs: HOUR,
      launchMaxLeadMs: 8 * HOUR,
    };
    assert.deepEqual(planPrompt(input), { promptNow: "launch", finalAt: HOUR });
    assert.deepEqual(planPrompt({ ...input, now: 90 * MINUTE }), {
      promptNow: "final",
      finalAt: HOUR,
    });
    assert.deepEqual(planPrompt({ ...input, prompted: new Set([`final:${2 * HOUR}`]) }), {
      promptNow: "launch",
      finalAt: null,
    });
    assert.equal(Object.isFrozen(DEFAULT_OPTIONS), true);
  });
});

describe("Arca shutdown watch", () => {
  it("prompts at launch within eight hours, then at the final hour after a decline", async () => {
    const h = makeWatch({ statuses: [running(2 * HOUR), running(2 * HOUR)] });
    h.watch.start();
    await flush();
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch"],
    );
    assert.equal(h.watch.getState().nextCheckAt, HOUR);
    await h.clock.advance(HOUR);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch", "final"],
    );
    assert.equal(h.reads, 2);
  });

  it("skips a distant launch prompt and reads again at the final deadline", async () => {
    const h = makeWatch({ statuses: [running(10 * HOUR), running(10 * HOUR)] });
    h.watch.start();
    await flush();
    assert.equal(h.prompts.length, 0);
    assert.equal(h.watch.getState().nextCheckAt, 17 * HOUR);
    await h.clock.advance(17 * HOUR);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["final"],
    );
    assert.equal(h.reads, 2);
  });

  it("shows only final when already inside the final hour", async () => {
    const h = makeWatch({ statuses: [running(30 * MINUTE)] });
    h.watch.start();
    await flush();
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["final"],
    );
    assert.deepEqual(h.watch.getState().prompted, [`final:${30 * MINUTE}`]);
  });

  it("uses a past deadline immediately outside business hours", async () => {
    const h = makeWatch({ time: 20 * HOUR, statuses: [running(19 * HOUR)] });
    h.watch.start();
    await flush();
    assert.equal(h.prompts[0].kind, "final");
    assert.equal(h.prompts[0].shutdownAt, 19 * HOUR);
  });

  it("defers a past business-hours deadline to today's 18:00", async () => {
    const h = makeWatch({ time: 12 * HOUR, statuses: [running(11 * HOUR), running(11 * HOUR)] });
    h.watch.start();
    await flush();
    assert.equal(h.prompts[0].shutdownAt, 18 * HOUR);
    assert.equal(h.watch.getState().nextCheckAt, 17 * HOUR);
    await h.clock.advance(5 * HOUR);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch", "final"],
    );
  });

  it("keeps Monday's final key after hours and opens a new Tuesday window", async () => {
    const h = makeWatch({
      time: 17 * HOUR,
      statuses: [running(11 * HOUR)],
      isWeekday: calendarWeekday,
    });
    h.watch.start();
    await flush();
    assert.deepEqual(
      h.prompts.map((request) => request.shutdownAt),
      [18 * HOUR],
    );
    h.clock.jump(2 * HOUR);
    h.watch.onResume();
    await flush();
    assert.equal(h.prompts.length, 1);
    h.clock.jump(22 * HOUR);
    h.watch.onResume();
    await flush();
    assert.deepEqual(
      h.prompts.map((request) => request.shutdownAt),
      [18 * HOUR, DAY + 18 * HOUR],
    );
  });

  it("holds a future weekday shutdown time until 18:00", async () => {
    const h = makeWatch({
      time: 9 * HOUR,
      statuses: [running(11 * HOUR)],
      isWeekday: calendarWeekday,
    });
    h.watch.start();
    await flush();
    assert.equal(h.watch.getState().effectiveAt, 18 * HOUR);
    assert.equal(h.watch.getState().nextCheckAt, 17 * HOUR);
    await h.clock.advanceTo(10 * HOUR);
    assert.equal(h.prompts.length, 0);
    await h.clock.advanceTo(17 * HOUR);
    assert.deepEqual(
      h.prompts.map((request) => request.kind),
      ["final"],
    );
  });

  it("uses one Friday window throughout the weekend", async () => {
    const fridayEnd = 4 * DAY + 18 * HOUR;
    const h = makeWatch({
      time: 5 * DAY + 10 * HOUR,
      statuses: [running(fridayEnd)],
      isWeekday: calendarWeekday,
    });
    h.watch.start();
    await flush();
    assert.deepEqual(
      h.prompts.map((request) => request.shutdownAt),
      [fridayEnd],
    );
    h.clock.jump(DAY);
    h.watch.onResume();
    await flush();
    assert.equal(h.prompts.length, 1);
  });

  it("re-reads across days without resume and warns again Tuesday evening", async () => {
    const h = makeWatch({
      time: 20 * HOUR,
      statuses: [running(11 * HOUR)],
      isWeekday: calendarWeekday,
    });
    h.watch.start();
    await flush();
    assert.deepEqual(
      h.prompts.map((request) => request.shutdownAt),
      [18 * HOUR],
    );
    await h.clock.advanceTo(DAY + 17 * HOUR);
    assert.ok(h.reads > 2);
    assert.deepEqual(
      h.prompts.map((request) => request.shutdownAt),
      [18 * HOUR, DAY + 18 * HOUR],
    );
  });

  it("silently re-plans when another client has extended the deadline", async () => {
    const h = makeWatch({ statuses: [running(2 * HOUR), running(5 * HOUR)] });
    h.watch.start();
    await flush();
    await h.clock.advance(HOUR);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch"],
    );
    assert.equal(h.watch.getState().nextCheckAt, 4 * HOUR);
  });

  it("re-reads after extension and schedules final for the new effective time", async () => {
    const results = [];
    const h = makeWatch({
      statuses: [running(2 * HOUR), running(5 * HOUR), running(5 * HOUR)],
      prompt: (request) => ({ mode: request.kind === "launch" ? "overnight" : null }),
      onExtendResult: (result) => results.push(result),
    });
    h.watch.start();
    await flush();
    assert.deepEqual(results, [{ ok: true, mode: "overnight", shutdownAt: 5 * HOUR }]);
    assert.equal(h.watch.getState().nextCheckAt, 4 * HOUR);
    await h.clock.advance(4 * HOUR);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch", "final"],
    );
    assert.equal(h.prompts[1].shutdownAt, 5 * HOUR);
  });

  it("confirms the new deadline after opt-out without scheduling more reads", async () => {
    let enabled = true;
    let extendCalls = 0;
    const results = [];
    const h = makeWatch({
      statuses: [running(2 * HOUR), running(11 * HOUR)],
      enabled: () => enabled,
      prompt: () => {
        enabled = false;
        return { mode: "overnight" };
      },
      extend: async () => {
        extendCalls++;
        return { ok: true, message: "extended" };
      },
      onExtendResult: (result) => results.push(result),
    });
    h.watch.start();
    await flush();
    assert.equal(extendCalls, 1);
    assert.equal(h.reads, 2);
    assert.deepEqual(results, [{ ok: true, mode: "overnight", shutdownAt: 18 * HOUR }]);
    assert.equal(h.watch.getState().phase, "idle");
    assert.equal(h.watch.getState().nextCheckAt, null);
    await h.clock.advanceTo(DAY);
    h.watch.onResume();
    await flush();
    assert.equal(h.reads, 2);
    assert.equal(h.clock.timers.size, 0);
  });

  it("reports extension failure without repeating the same prompt key", async () => {
    const results = [];
    const h = makeWatch({
      statuses: [running(2 * HOUR), running(2 * HOUR)],
      prompt: () => ({ mode: "overnight" }),
      extend: async () => ({ ok: false, errorKind: "network", error: "unreachable" }),
      onExtendResult: (result) => results.push(result),
    });
    h.watch.start();
    await flush();
    h.watch.onResume();
    await flush();
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch"],
    );
    assert.deepEqual(results, [
      { ok: false, mode: "overnight", errorKind: "network", error: "unreachable" },
    ]);
  });

  it("waits for an extension failure dialog before the final prompt", async () => {
    const dialog = deferred();
    const h = makeWatch({
      statuses: [running(2 * HOUR), running(2 * HOUR)],
      prompt: (request) => ({ mode: request.kind === "launch" ? "overnight" : null }),
      extend: async () => ({ ok: false, errorKind: "network", error: "offline" }),
      onExtendResult: () => dialog.promise,
    });
    h.watch.start();
    await flush();
    assert.equal(h.watch.getState().phase, "extending");
    await h.clock.advance(HOUR);
    assert.deepEqual(
      h.prompts.map((request) => request.kind),
      ["launch"],
    );
    dialog.resolve();
    await flush();
    assert.deepEqual(
      h.prompts.map((request) => request.kind),
      ["launch", "final"],
    );
  });

  it("waits for an extension success dialog before re-planning", async () => {
    const dialog = deferred();
    const h = makeWatch({
      statuses: [running(2 * HOUR), running(30 * MINUTE)],
      prompt: (request) => ({ mode: request.kind === "launch" ? "overnight" : null }),
      onExtendResult: () => dialog.promise,
    });
    h.watch.start();
    await flush();
    assert.deepEqual(
      h.prompts.map((request) => request.kind),
      ["launch"],
    );
    dialog.resolve();
    await flush();
    assert.deepEqual(
      h.prompts.map((request) => request.kind),
      ["launch", "final"],
    );
  });

  it("treats a throwing prompt as a decline and keeps the final timer", async () => {
    const h = makeWatch({
      prompt: () => {
        throw new Error("window closed");
      },
    });
    h.watch.start();
    await flush();
    assert.equal(h.watch.getState().nextCheckAt, HOUR);
    await h.clock.advance(HOUR);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch", "final"],
    );
  });

  it("prompts once per key across start and resume", async () => {
    const h = makeWatch({ statuses: [running(2 * HOUR), running(2 * HOUR), running(2 * HOUR)] });
    h.watch.start();
    h.watch.start();
    await flush();
    h.watch.onResume();
    await flush();
    h.watch.start();
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch"],
    );
  });

  it("does not read when disabled or prompt after eligibility flips", async () => {
    const disabled = makeWatch({ enabled: () => false });
    disabled.watch.start();
    await flush();
    assert.equal(disabled.reads, 0);
    assert.equal(disabled.prompts.length, 0);
    let checks = 0;
    const flipped = makeWatch({ enabled: () => ++checks < 3 });
    flipped.watch.start();
    await flush();
    assert.equal(flipped.reads, 1);
    assert.equal(flipped.prompts.length, 0);
    assert.equal(flipped.watch.getState().phase, "idle");
  });

  it("retries a failed status read and preserves launch eligibility", async () => {
    const h = makeWatch({
      statuses: [{ ok: false, errorKind: "network", error: "offline" }, running(2 * HOUR)],
    });
    h.watch.start();
    await flush();
    assert.equal(h.watch.getState().nextCheckAt, 2 * MINUTE);
    await h.clock.advance(2 * MINUTE);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch"],
    );
  });

  it("preserves launch eligibility when resume replaces a failed read's retry", async () => {
    const h = makeWatch({
      statuses: [{ ok: false, errorKind: "network", error: "offline" }, running(2 * HOUR)],
    });
    h.watch.start();
    await flush();
    assert.equal(h.watch.getState().nextCheckAt, 2 * MINUTE);
    h.watch.onResume();
    await flush();
    assert.equal(h.reads, 2);
    assert.deepEqual(
      h.prompts.map((request) => request.kind),
      ["launch"],
    );
  });

  it("preserves one launch prompt when resume supersedes the first status read", async () => {
    const firstRead = deferred();
    let calls = 0;
    const h = makeWatch({
      readStatus: () => (++calls === 1 ? firstRead.promise : running(2 * HOUR)),
    });
    h.watch.start();
    h.watch.onResume();
    firstRead.resolve(running(2 * HOUR));
    await flush();
    assert.equal(h.reads, 2);
    assert.deepEqual(
      h.prompts.map((request) => request.kind),
      ["launch"],
    );
  });

  it("uses a two-minute first retry, fifteen-minute later retries, and resets on success", async () => {
    const failure = { ok: false, errorKind: "network", error: "offline" };
    const h = makeWatch({
      statuses: [failure, failure, { ok: true, state: "stopped", shutdownAt: null }, failure],
    });
    h.watch.start();
    await flush();
    assert.equal(h.watch.getState().nextCheckAt, 2 * MINUTE);
    await h.clock.advance(2 * MINUTE);
    assert.equal(h.watch.getState().nextCheckAt, 17 * MINUTE);
    await h.clock.advance(15 * MINUTE);
    assert.equal(h.watch.getState().nextCheckAt, 17 * MINUTE + 3 * HOUR);
    h.watch.onResume();
    await flush();
    assert.equal(h.watch.getState().nextCheckAt, 19 * MINUTE);
  });

  it("refreshes stopped instances and missing shutdown times every three hours", async () => {
    await Promise.all(
      [
        { ok: true, state: "stopped", shutdownAt: 2 * HOUR },
        { ok: true, state: "running", shutdownAt: null, rawShutdownTime: "unknown" },
        { ok: true, state: "running", rawShutdownTime: "unparseable" },
      ].map(async (status) => {
        const messages = [];
        const h = makeWatch({ statuses: [status], log: (message) => messages.push(message) });
        h.watch.start();
        await flush();
        assert.equal(h.watch.getState().phase, "scheduled");
        assert.equal(h.watch.getState().nextCheckAt, 3 * HOUR);
        assert.equal(h.prompts.length, 0);
        assert.equal(messages.length, 1);
        await h.clock.advanceTo(6 * HOUR);
        assert.equal(h.reads, 3);
        assert.equal(h.prompts.length, 0);
      }),
    );
  });

  it("omits workweek on Thursday and Friday", async () => {
    await Promise.all(
      [3, 4].map(async (day) => {
        const time = day * DAY;
        const h = makeWatch({
          time,
          statuses: [running(time + 2 * HOUR)],
          offersWorkweek: (ms) => ![3, 4].includes(weekdayOf(ms)),
        });
        h.watch.start();
        await flush();
        assert.deepEqual(h.prompts[0].modes, ["overnight"]);
      }),
    );
  });

  it("caps timer sleeps without reading status before the target", async () => {
    const h = makeWatch({ time: 18 * HOUR, statuses: [running(24 * HOUR)] });
    h.watch.start();
    await flush();
    await Array.from({ length: 9 }).reduce(
      (pending) => pending.then(() => h.clock.advance(30 * MINUTE)),
      Promise.resolve(),
    );
    assert.equal(h.reads, 1);
    assert.equal(h.watch.getState().nextCheckAt, 23 * HOUR);
    assert.ok(h.clock.delays.every((delay) => delay <= 30 * MINUTE));
    await h.clock.advance(30 * MINUTE);
    assert.equal(h.reads, 2);
  });

  it("re-reads on resume after a clock jump into the final hour", async () => {
    const h = makeWatch({ statuses: [running(2 * HOUR), running(2 * HOUR)] });
    h.watch.start();
    await flush();
    h.clock.jump(90 * MINUTE);
    h.watch.onResume();
    await flush();
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch", "final"],
    );
    assert.equal(h.reads, 2);
  });

  it("ignores disposal during a status read", async () => {
    const pending = deferred();
    const h = makeWatch({ readStatus: () => pending.promise });
    h.watch.start();
    h.watch.dispose();
    pending.resolve(running(2 * HOUR));
    await flush();
    assert.equal(h.prompts.length, 0);
    assert.equal(h.clock.timers.size, 0);
    assert.equal(h.watch.getState().disposed, true);
  });

  it("ignores disposal during a prompt", async () => {
    const pending = deferred();
    let extendCalls = 0;
    const h = makeWatch({
      prompt: () => pending.promise,
      extend: async () => {
        extendCalls++;
        return { ok: true, message: "extended" };
      },
    });
    h.watch.start();
    await flush();
    h.watch.dispose();
    pending.resolve({ mode: "overnight" });
    await flush();
    assert.equal(extendCalls, 0);
    assert.equal(h.clock.timers.size, 0);
  });

  it("ignores disposal during an extension", async () => {
    const pending = deferred();
    const results = [];
    const h = makeWatch({
      prompt: () => ({ mode: "overnight" }),
      extend: () => pending.promise,
      onExtendResult: (result) => results.push(result),
    });
    h.watch.start();
    await flush();
    h.watch.dispose();
    pending.resolve({ ok: true, message: "extended" });
    await flush();
    assert.equal(h.reads, 1);
    assert.deepEqual(results, []);
  });

  it("keeps one prompt open when its final timer fires", async () => {
    const pending = deferred();
    const h = makeWatch({
      statuses: [running(2 * HOUR), running(2 * HOUR)],
      prompt: (request) => (request.kind === "launch" ? pending.promise : { mode: null }),
    });
    h.watch.start();
    await flush();
    await h.clock.advance(HOUR);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch"],
    );
    pending.resolve({ mode: null });
    await flush();
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch", "final"],
    );
  });
});
