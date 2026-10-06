"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const { spawn: realSpawn } = require("node:child_process");
const { EventEmitter } = require("node:events");
const {
  ARCA_EXTEND_MODES,
  EXTEND_TIMEOUT_MS,
  STATUS_TIMEOUT_MS,
  buildConnectArgs,
  buildLoginArgs,
  connectArcaHost,
  describeArcaCliFailure,
  describeConnectFailure,
  parseArcaStatus,
  readArcaStatus,
  resolveArcaPath,
  runArcaExtend,
  startArcaConnect,
  startArcaLogin,
} = require("../src/arca");

/** A fake connect child: an EventEmitter with stdout/stderr stream stubs. */
function fakeConnectChild() {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  for (const stream of [child.stdout, child.stderr]) {
    stream.destroyed = false;
    stream.destroy = () => {
      stream.destroyed = true;
    };
  }
  child.killed = false;
  child.kill = () => {
    child.killed = true;
  };
  return child;
}

function fakeCliSpawn(onSpawn) {
  const calls = [];
  const spawn = (file, args, options) => {
    const child = fakeConnectChild();
    calls.push({ file, args, options, child });
    onSpawn?.(child);
    return child;
  };
  return { spawn, calls, resolveArcaPath: () => "/fake/arca" };
}

describe("arca status output", () => {
  it("parses a pretty-printed status and explicit UTC offsets", () => {
    const status = {
      name: "dev",
      status: "running",
      region: "us-west-2",
      volume: null,
      instance: { launch_time: "2026-10-04T01:00:00Z", shutdown_time: "2026-10-05T01:00:00Z" },
    };
    assert.deepEqual(parseArcaStatus(JSON.stringify(status, null, 2)), {
      state: "running",
      shutdownAt: Date.UTC(2026, 9, 5, 1),
      rawShutdownTime: "2026-10-05T01:00:00Z",
    });
    status.instance.shutdown_time = "2026-10-05T01:00:00+00:00";
    assert.equal(parseArcaStatus(JSON.stringify(status)).shutdownAt, Date.UTC(2026, 9, 5, 1));
    status.instance.shutdown_time = "2026-10-05T01:00:00+0000";
    assert.deepEqual(parseArcaStatus(JSON.stringify(status)), {
      state: "running",
      shutdownAt: Date.UTC(2026, 9, 5, 1),
      rawShutdownTime: "2026-10-05T01:00:00+0000",
    });
  });

  it("handles null instance, null time, and the minimal top-level shape", () => {
    assert.deepEqual(parseArcaStatus('{"status":"terminated","instance":null}'), {
      state: "terminated",
      shutdownAt: null,
      rawShutdownTime: null,
    });
    assert.deepEqual(parseArcaStatus('{"status":"running","instance":{"shutdown_time":null}}'), {
      state: "running",
      shutdownAt: null,
      rawShutdownTime: null,
    });
    assert.equal(
      parseArcaStatus('{"shutdown_time":"2026-10-05T01:00:00Z"}').shutdownAt,
      Date.UTC(2026, 9, 5, 1),
    );
  });

  it("ignores notices and keeps invalid or zone-less raw times", () => {
    const stdout =
      'Upgrade available\n{"status":"running","instance":{"shutdown_time":"2026-10-05T01:00:00"}}\nDone';
    assert.deepEqual(parseArcaStatus(stdout), {
      state: "running",
      shutdownAt: null,
      rawShutdownTime: "2026-10-05T01:00:00",
    });
    assert.equal(parseArcaStatus('{"instance":{"shutdown_time":"badZ"}}').shutdownAt, null);
    assert.equal(parseArcaStatus("garbage"), null);
    assert.equal(parseArcaStatus("{broken}"), null);
  });

  it("finds status JSON around notices containing braces", () => {
    const json = '{"status":"running","instance":{"shutdown_time":"2026-10-05T01:00:00Z"}}';
    for (const stdout of [`Notice: run {arca upgrade}\n${json}`, `${json}\nTip: see {docs}`]) {
      assert.equal(parseArcaStatus(stdout).shutdownAt, Date.UTC(2026, 9, 5, 1));
    }
  });

  it("skips a status-shaped notice before the real payload", () => {
    const stdout =
      '{"status":"update available"}\n{"status":"running","instance":{"shutdown_time":"2026-10-05T01:00:00Z"}}';
    assert.deepEqual(parseArcaStatus(stdout), {
      state: "running",
      shutdownAt: Date.UTC(2026, 9, 5, 1),
      rawShutdownTime: "2026-10-05T01:00:00Z",
    });
  });

  it("prefers an instance payload over a minimal status candidate", () => {
    const stdout =
      '{"status":"pending","shutdown_time":null}\n{"status":"running","instance":{"shutdown_time":"2026-10-05T01:00:00Z"}}';
    assert.equal(parseArcaStatus(stdout).shutdownAt, Date.UTC(2026, 9, 5, 1));
  });

  it("finds a status payload after more than twenty brace notices", () => {
    const notices = Array.from({ length: 25 }, (_, index) => `Notice ${index}: {arca upgrade}`);
    const stdout = `${notices.join("\n")}\n{"status":"running","instance":null}`;
    assert.equal(parseArcaStatus(stdout).state, "running");
  });

  it("handles brace-heavy malformed output promptly", () => {
    const malformed = "{not-json}".repeat(1_250);
    const started = performance.now();
    assert.equal(parseArcaStatus(malformed), null);
    assert.ok(performance.now() - started < 100);
  });
});

describe("arca status command", () => {
  it("runs headlessly with exact argv and returns parsed status", async () => {
    const deps = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit(
          "data",
          'notice\n{"status":"running","instance":{"shutdown_time":"2026-10-05T01:00:00Z"}}\n',
        );
        child.emit("close", 0);
      }),
    );
    assert.deepEqual(await readArcaStatus(deps), {
      ok: true,
      state: "running",
      shutdownAt: Date.UTC(2026, 9, 5, 1),
      rawShutdownTime: "2026-10-05T01:00:00Z",
    });
    assert.deepEqual(deps.calls[0].args, ["status", "--json"]);
    assert.equal(deps.calls[0].options.stdio[0], "ignore");
    assert.equal(STATUS_TIMEOUT_MS, 60_000);
  });

  it("waits for close when status output arrives after exit", async () => {
    const deps = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit("data", '{"status":"running","instance":{"shutdown_time":"2026-');
        child.emit("exit", 0);
        child.stdout.emit("data", '10-05T01:00:00Z"}}');
        child.emit("close", 0);
      }),
    );
    assert.deepEqual(await readArcaStatus(deps), {
      ok: true,
      state: "running",
      shutdownAt: Date.UTC(2026, 9, 5, 1),
      rawShutdownTime: "2026-10-05T01:00:00Z",
    });
  });

  it("uses captured output after exit when close never arrives", async () => {
    const deps = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit("data", '{"status":"running","instance":null}');
        child.emit("exit", 0);
      }),
    );
    assert.deepEqual(await readArcaStatus({ ...deps, timeoutMs: 1_000 }), {
      ok: true,
      state: "running",
      shutdownAt: null,
      rawShutdownTime: null,
    });
    assert.equal(deps.calls[0].child.killed, false);
    assert.equal(deps.calls[0].child.stdout.destroyed, true);
    assert.equal(deps.calls[0].child.stderr.destroyed, true);
  });

  it(
    "closes inherited pipes after a real child exits",
    { skip: process.platform === "win32" },
    async () => {
      let child;
      const started = performance.now();
      const result = await readArcaStatus({
        resolveArcaPath: () => "/bin/sh",
        spawn: (_file, _args, options) => {
          child = realSpawn(
            "/bin/sh",
            ["-c", 'printf \'%s\\n\' \'{"status":"running","instance":null}\'; sleep 2 &'],
            options,
          );
          return child;
        },
        timeoutMs: 900,
      });
      assert.equal(result.ok, true);
      assert.ok(performance.now() - started < 900);
      assert.equal(child.stdout.destroyed, true);
      assert.equal(child.stderr.destroyed, true);
    },
  );

  it("uses the exit code at the overall timeout when close never arrives", async () => {
    const deps = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit("data", '{"status":"running","instance":null}');
        child.emit("exit", 0);
      }),
    );
    assert.equal((await readArcaStatus({ ...deps, timeoutMs: 20 })).ok, true);
    assert.equal(deps.calls[0].child.killed, false);
    assert.equal(deps.calls[0].child.stdout.destroyed, true);
    assert.equal(deps.calls[0].child.stderr.destroyed, true);
  });

  it("caps captured status stdout", async () => {
    const deps = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit("data", "noise".repeat(60_000));
        child.stdout.emit("data", '{"status":"running","instance":null}');
        child.emit("close", 0);
      }),
    );
    assert.match((await readArcaStatus(deps)).error, /unexpected arca status output/i);
  });

  it("decodes UTF-8 split across stdout chunks", async () => {
    const deps = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        const output = Buffer.from('{"status":"runníng","instance":null}');
        const split = output.indexOf(0xc3) + 1;
        child.stdout.emit("data", output.subarray(0, split));
        child.stdout.emit("data", output.subarray(split));
        child.emit("close", 0);
      }),
    );
    assert.equal((await readArcaStatus(deps)).state, "runníng");
  });

  it("maps non-zero exit, timeout, spawn errors, and malformed success", async () => {
    const fail = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stderr.emit("data", "No running arca instance found for 'dev'.");
        child.emit("close", 1);
      }),
    );
    assert.equal((await readArcaStatus(fail)).errorKind, "no-instance");
    const timeout = fakeCliSpawn();
    assert.equal((await readArcaStatus({ ...timeout, timeoutMs: 20 })).errorKind, "timeout");
    assert.equal(timeout.calls[0].child.killed, true);
    assert.equal(timeout.calls[0].child.stdout.destroyed, true);
    assert.equal(timeout.calls[0].child.stderr.destroyed, true);
    const spawnError = fakeCliSpawn((child) =>
      queueMicrotask(() => child.emit("error", new Error("EACCES"))),
    );
    assert.deepEqual(await readArcaStatus(spawnError), {
      ok: false,
      errorKind: "unknown",
      error: "Couldn't run arca: EACCES",
    });
    const garbage = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit("data", "garbage");
        child.emit("close", 0);
      }),
    );
    assert.match((await readArcaStatus(garbage)).error, /unexpected arca status output/i);
  });

  it("does not spawn when arca is absent", async () => {
    const result = await readArcaStatus({
      resolveArcaPath: () => null,
      spawn: () => {
        throw new Error("must not spawn");
      },
    });
    assert.equal(result.errorKind, "not-installed");
  });
});

describe("arca extend command", () => {
  it("validates modes and returns the last output line", async () => {
    assert.deepEqual(ARCA_EXTEND_MODES, ["default", "overnight", "workweek"]);
    assert.equal(Object.isFrozen(ARCA_EXTEND_MODES), true);
    assert.equal(EXTEND_TIMEOUT_MS, 120_000);
    const deps = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit(
          "data",
          "Starting\nExtend succeeded. Instance will not be shutdown until October 06\n",
        );
        child.emit("close", 0);
      }),
    );
    assert.deepEqual(await runArcaExtend("overnight", deps), {
      ok: true,
      message: "Extend succeeded. Instance will not be shutdown until October 06",
    });
    assert.deepEqual(deps.calls[0].args, ["extend", "overnight"]);
    assert.equal(deps.calls[0].options.stdio[0], "ignore");
    const invalidResults = await Promise.all(
      ["reset", "overnight; rm -rf ~"].map((mode) => runArcaExtend(mode, deps)),
    );
    for (const result of invalidResults) {
      assert.deepEqual(result, {
        ok: false,
        errorKind: "unknown",
        error: "Unsupported arca extend mode.",
      });
    }
    assert.equal(deps.calls.length, 1);
  });

  it("maps every documented failure kind", async () => {
    const failures = [
      ["No running arca instance found for 'dev'.", "no-instance"],
      [
        "Cannot set shutdown_after: reaches its one-week runtime limit at tomorrow",
        "runtime-limit",
      ],
      ["Your dbcert login expired", "arca-auth"],
      ["Error connecting to arca", "unreachable"],
      ["noise\nopaque failure", "unknown"],
    ];
    const results = await Promise.all(
      failures.map(([output]) => {
        const deps = fakeCliSpawn((child) =>
          queueMicrotask(() => {
            child.stderr.emit("data", output);
            child.emit("close", 1);
          }),
        );
        return runArcaExtend("workweek", deps);
      }),
    );
    results.forEach((result, index) => {
      const kind = failures[index][1];
      assert.equal(result.errorKind, kind);
      if (kind === "unknown") assert.match(result.error, /opaque failure/);
    });
    const timeout = fakeCliSpawn();
    assert.equal(
      (await runArcaExtend("default", { ...timeout, timeoutMs: 20 })).errorKind,
      "timeout",
    );
    assert.equal(timeout.calls[0].child.killed, true);
    assert.equal(
      (await runArcaExtend("default", { resolveArcaPath: () => null })).errorKind,
      "not-installed",
    );
  });

  it("caps extend output and decodes UTF-8 split across stderr chunks", async () => {
    const capped = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        child.stdout.emit("data", "n".repeat(300_000));
        child.stdout.emit("data", "SUCCESS");
        child.emit("close", 0);
      }),
    );
    const result = await runArcaExtend("overnight", capped);
    assert.equal(Buffer.byteLength(result.message), 256 * 1024);
    assert.equal(result.message.includes("SUCCESS"), false);

    const splitError = fakeCliSpawn((child) =>
      queueMicrotask(() => {
        const output = Buffer.from("défaillance");
        child.stderr.emit("data", output.subarray(0, 2));
        child.stderr.emit("data", output.subarray(2));
        child.emit("close", 1);
      }),
    );
    assert.match((await runArcaExtend("overnight", splitError)).error, /défaillance/);
  });

  it("uses action-specific no-instance and runtime-limit messages", () => {
    for (const output of ["No running arca instance", "one-week runtime limit"]) {
      const run = { code: 1, stdout: "", stderr: output };
      const status = describeArcaCliFailure(run, "status");
      const extend = describeArcaCliFailure(run, "extend");
      assert.equal(status.errorKind, extend.errorKind);
      assert.notEqual(status.error, extend.error);
      assert.match(status.error, /start|restart/i);
    }
  });

  it("uses timeout before text classification", () => {
    assert.equal(
      describeArcaCliFailure({ code: null, stdout: "dbcert", stderr: "", timedOut: true }, "status")
        .errorKind,
      "timeout",
    );
    assert.match(
      describeArcaCliFailure({ code: null, stdout: "", stderr: "", timedOut: true }, "status")
        .error,
      /checking arca's status timed out/i,
    );
  });
});

describe("arca binary resolution", () => {
  it("prefers PATH, then falls back to well-known locations", () => {
    assert.equal(
      resolveArcaPath({
        whichArca: () => "/from/path/arca",
        isExecutableFile: (p) => p === "/from/path/arca",
        candidatePaths: () => ["/usr/local/bin/arca"],
      }),
      "/from/path/arca",
    );
    assert.equal(
      resolveArcaPath({
        whichArca: () => null,
        isExecutableFile: (p) => p === "/usr/local/bin/arca",
        candidatePaths: () => ["/opt/homebrew/bin/arca", "/usr/local/bin/arca"],
      }),
      "/usr/local/bin/arca",
    );
    assert.equal(
      resolveArcaPath({
        whichArca: () => null,
        isExecutableFile: () => false,
        candidatePaths: () => ["/usr/local/bin/arca"],
      }),
      null,
    );
  });
});

describe("arca connect command", () => {
  it("binds remote login and noninteractive startup to the same SPOG workspace", () => {
    const server = "https://account.databricks.com/omnigent?o=123&test=value";
    const args = buildLoginArgs(server);
    assert.deepEqual(args, [
      "ssh",
      "-o",
      "ClearAllForwardings=yes",
      "isaac",
      "omni",
      "login",
      `'${server}'`,
    ]);
    assert.equal(buildConnectArgs(server)[7], args[6]);
  });
  it("passes the remote isaac omni host command through ssh", () => {
    assert.deepEqual(buildConnectArgs("https://workspace.example.com/ml/omnigents"), [
      "ssh",
      "-o",
      "ClearAllForwardings=yes",
      "isaac",
      "omni",
      "host",
      "--server",
      "'https://workspace.example.com/ml/omnigents'",
      "--background",
      "--non-interactive",
    ]);
  });

  it("quotes the URL for the remote shell so a query's `?` isn't globbed", () => {
    const args = buildConnectArgs("https://ws.cloud.databricks.com/omnigent?o=123");
    assert.equal(
      args[args.indexOf("--server") + 1],
      "'https://ws.cloud.databricks.com/omnigent?o=123'",
    );
  });

  it("rejects non-http(s) server URLs", () => {
    assert.throws(() => buildConnectArgs("file:///etc/passwd"));
    assert.throws(() => buildConnectArgs("not a url"));
  });

  it("rejects URLs smuggling shell metacharacters through path or query", () => {
    // ssh re-parses the remote command in a shell, so URL-legal but
    // shell-hostile characters must be refused, not passed through.
    assert.throws(() => buildConnectArgs("https://ws.cloud.databricks.com/omnigent;id"));
    assert.throws(() => buildConnectArgs("https://ws.cloud.databricks.com/a$(id)"));
    assert.throws(() => buildConnectArgs("https://ws.cloud.databricks.com/a'b"));
    // The ordinary workspace-mount shape stays accepted.
    assert.doesNotThrow(() => buildConnectArgs("https://ws.cloud.databricks.com/omnigent?o=123"));
  });
});

describe("arca connect failures", () => {
  it("classifies managed OAuth failures without treating them as network errors", () => {
    for (const stderr of [
      "Error: OMNIGENT_AUTH_REQUIRED: sign in",
      "Authentication failed (HTTP 401): rejected",
    ]) {
      assert.equal(describeConnectFailure({ code: 1, stdout: "", stderr }).errorKind, "omni-auth");
    }
  });
  it("maps a timeout, sign-in, missing-CLI, and unreachable instance", () => {
    assert.match(
      describeConnectFailure({ code: null, stdout: "", stderr: "", timedOut: true }).error,
      /timed out/i,
    );

    const auth = describeConnectFailure({
      code: 1,
      stdout: "",
      stderr: "Not signed in to https://srv (\u2026). Run `omnigent login https://srv` and retry.",
    });
    assert.equal(auth.authError, true);
    assert.match(auth.error, /isaac omni login/);

    assert.match(
      describeConnectFailure({ code: 127, stdout: "", stderr: "bash: isaac: command not found" })
        .error,
      /isn't available on the Arca instance/,
    );
    assert.match(
      describeConnectFailure({ code: 1, stdout: "", stderr: "isaac: omni: command not found" })
        .error,
      /isn't available on the Arca instance/,
    );

    assert.match(
      describeConnectFailure({
        code: 1,
        stdout: "",
        stderr: "Error connecting to arca. The instance may be stopped or unreachable.",
      }).error,
      /arca stop && arca start/,
    );
  });

  it("tags each failure with the kind of fix it needs", () => {
    const kind = (run) =>
      describeConnectFailure({ code: 1, stdout: "", stderr: "", ...run }).errorKind;
    assert.equal(kind({ code: null, timedOut: true }), "timeout");
    assert.equal(kind({ stderr: "Not signed in to https://srv." }), "omni-auth");
    assert.equal(
      kind({ code: 127, stderr: "bash: isaac: command not found" }),
      "missing-remote-cli",
    );
    assert.equal(kind({ stderr: "Error connecting to arca." }), "unreachable");
    assert.equal(kind({ stderr: "Your certificate has expired. Run `arca login`." }), "arca-auth");
    assert.equal(kind({ stderr: "user@host: Permission denied (publickey)." }), "arca-auth");
    assert.equal(kind({ stderr: "something else" }), "unknown");
  });

  it("falls back to the last output line for unrecognized failures", () => {
    const result = describeConnectFailure({
      code: 1,
      stdout: "",
      stderr: "noise line\nssh: connect to host 1.2.3.4 port 22: Connection refused",
    });
    assert.match(result.error, /Connection refused/);
    assert.doesNotMatch(result.error, /noise line/);
  });
});

describe("startArcaConnect / connectArcaHost", () => {
  it("settles canceled login without waiting for a remote process exit", async () => {
    const child = fakeConnectChild();
    const run = startArcaLogin("https://account.databricks.com/omnigent?o=123", {
      resolveArcaPath: () => "/bin/arca",
      spawn: () => child,
    });
    run.cancel();
    assert.equal((await run.promise).canceled, true);
    assert.equal(child.killed, true);
    child.emit("exit", 0);
    assert.equal((await run.promise).ok, false);
  });
  it("does not forward remote login output or error tickets to the renderer", async () => {
    for (const [code, errorKind] of [
      [0, undefined],
      [1, "omni-auth"],
      [127, "missing-remote-cli"],
      [255, "unreachable"],
    ]) {
      const child = fakeConnectChild();
      const chunks = [];
      const run = startArcaLogin("https://account.databricks.com/omnigent?o=123", {
        resolveArcaPath: () => "/bin/arca",
        spawn: (_file, args, opts) => {
          assert.equal(args[5], "login");
          assert.equal(opts.stdio[0], "ignore");
          return child;
        },
        onOutput: (text) => chunks.push(text),
      });
      child.stdout.emit("data", "https://example.com/auth/login?ticket=SECRET");
      child.stderr.emit("data", "SECRET");
      child.emit("exit", code);
      // oxlint-disable-next-line no-await-in-loop -- Exercise both process outcomes.
      const result = await run.promise;
      assert.equal(result.ok, code === 0);
      assert.equal(result.errorKind, errorKind);
      assert.equal(result.authError === true, code === 1);
      assert.deepEqual(chunks, []);
      assert.doesNotMatch(JSON.stringify(result), /SECRET|ticket=/);
    }
  });

  it("settles a timed-out login even if the child never emits exit", async () => {
    const child = fakeConnectChild();
    const run = startArcaLogin("https://account.databricks.com/omnigent?o=123", {
      resolveArcaPath: () => "/bin/arca",
      spawn: () => child,
      timeoutMs: 1,
    });
    const keepAlive = setTimeout(() => {}, 1000);
    try {
      const result = await run.promise;
      assert.equal(result.errorKind, "timeout");
      assert.match(result.error, /sign-in timed out.*Arca Companion/);
      assert.equal(child.killed, true);
      child.emit("exit", 0);
      assert.equal((await run.promise).ok, false);
    } finally {
      clearTimeout(keepAlive);
    }
  });
  it("streams live output, exposes the command, and resolves ok on exit 0", async () => {
    const chunks = [];
    let child;
    const run = startArcaConnect("https://srv.example.com", {
      resolveArcaPath: () => "/usr/local/bin/arca",
      spawn: (file, args) => {
        assert.equal(file, "/usr/local/bin/arca");
        assert.equal(args[0], "ssh");
        child = fakeConnectChild();
        return child;
      },
      onOutput: (text) => chunks.push(text),
    });
    assert.equal(
      run.command,
      "arca ssh -o ClearAllForwardings=yes isaac omni host --server 'https://srv.example.com/' --background --non-interactive",
    );
    child.stdout.emit("data", "Attempting to start your Arca instance\n");
    child.stderr.emit("data", "synced dbcert\n");
    child.emit("exit", 0);
    assert.deepEqual(await run.promise, { ok: true, alreadyRunning: false });
    assert.deepEqual(chunks, ["Attempting to start your Arca instance\n", "synced dbcert\n"]);
  });

  it("reports a reused daemon so callers don't wait for a new host", async () => {
    let child;
    const run = startArcaConnect("https://srv.example.com", {
      resolveArcaPath: () => "/usr/local/bin/arca",
      spawn: () => {
        child = fakeConnectChild();
        return child;
      },
    });
    child.stdout.emit("data", "Host daemon already running (pid 4242).\n");
    child.emit("exit", 0);
    assert.deepEqual(await run.promise, { ok: true, alreadyRunning: true });
  });

  it("maps a failing exit through the captured output and never rejects", async () => {
    let child;
    const failRun = startArcaConnect("https://srv.example.com", {
      resolveArcaPath: () => "/usr/local/bin/arca",
      spawn: () => {
        child = fakeConnectChild();
        return child;
      },
    });
    child.stderr.emit("data", "bash: isaac: command not found");
    child.emit("exit", 1);
    const result = await failRun.promise;
    assert.equal(result.ok, false);
    assert.match(result.error, /Arca instance/);
  });

  it("cancel kills the child and resolves as canceled", async () => {
    let child;
    const run = startArcaConnect("https://srv.example.com", {
      resolveArcaPath: () => "/usr/local/bin/arca",
      spawn: () => {
        child = fakeConnectChild();
        return child;
      },
    });
    run.cancel();
    assert.equal(child.killed, true);
    child.emit("exit", null); // the kill lands
    const result = await run.promise;
    assert.equal(result.ok, false);
    assert.equal(result.canceled, true);
  });

  it("fails cleanly when arca is not installed", async () => {
    const result = await connectArcaHost("https://srv.example.com", {
      resolveArcaPath: () => null,
      spawn: () => {
        throw new Error("must not spawn");
      },
    });
    assert.deepEqual(result, {
      ok: false,
      error: "The arca CLI was not found on this machine.",
    });
  });
});
