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
  isFriday = () => false,
  options = {},
  onExtendResult = () => {},
  log = () => {},
} = {}) {
  const clock = fakeClock(time);
  const prompts = [];
  let reads = 0;
  const watch = createArcaShutdownWatch({
    readStatus: async () => {
      reads++;
      return readStatus ? readStatus() : (statuses.shift() ?? statuses.at(-1) ?? running(2 * HOUR));
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
    isFriday,
    isWeekday: () => true,
    localHour: (ms) => (ms % (24 * HOUR)) / HOUR,
    localTimeAt: (ms, hour) => Math.floor(ms / (24 * HOUR)) * 24 * HOUR + hour * HOUR,
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
  it("applies the weekday business-hours lock only to past deadlines", () => {
    const deps = {
      isWeekday: () => true,
      localHour: () => 12,
      localTimeAt: (_, hour) => hour * HOUR,
      businessHours: { start: 6, end: 18 },
    };
    assert.equal(effectiveShutdownAt(11 * HOUR, 12 * HOUR, deps), 18 * HOUR);
    assert.equal(effectiveShutdownAt(13 * HOUR, 12 * HOUR, deps), 13 * HOUR);
    assert.equal(
      effectiveShutdownAt(11 * HOUR, 12 * HOUR, { ...deps, isWeekday: () => false }),
      11 * HOUR,
    );
    assert.equal(
      effectiveShutdownAt(11 * HOUR, 12 * HOUR, { ...deps, localHour: () => 18 }),
      11 * HOUR,
    );
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
    assert.equal(h.watch.getState().nextCheckAt, 9 * HOUR);
    await h.clock.advance(9 * HOUR);
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
    assert.equal(h.watch.getState().nextCheckAt, 15 * MINUTE);
    await h.clock.advance(15 * MINUTE);
    assert.deepEqual(
      h.prompts.map((p) => p.kind),
      ["launch"],
    );
  });

  it("stays idle for a stopped instance or missing shutdown time", async () => {
    await Promise.all(
      [
        { ok: true, state: "stopped", shutdownAt: 2 * HOUR },
        { ok: true, state: "running", shutdownAt: null, rawShutdownTime: "unknown" },
      ].map(async (status) => {
        const messages = [];
        const h = makeWatch({ statuses: [status], log: (message) => messages.push(message) });
        h.watch.start();
        await flush();
        assert.equal(h.watch.getState().phase, "idle");
        assert.equal(h.clock.timers.size, 0);
        assert.equal(h.prompts.length, 0);
        assert.equal(messages.length, 1);
      }),
    );
  });

  it("omits workweek on Friday", async () => {
    const h = makeWatch({ isFriday: () => true });
    h.watch.start();
    await flush();
    assert.deepEqual(h.prompts[0].modes, ["overnight"]);
  });

  it("caps timer sleeps without reading status before the target", async () => {
    const h = makeWatch({ statuses: [running(6 * HOUR)] });
    h.watch.start();
    await flush();
    await Array.from({ length: 9 }).reduce(
      (pending) => pending.then(() => h.clock.advance(30 * MINUTE)),
      Promise.resolve(),
    );
    assert.equal(h.reads, 1);
    assert.equal(h.watch.getState().nextCheckAt, 5 * HOUR);
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
