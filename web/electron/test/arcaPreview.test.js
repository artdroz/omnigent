"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const {
  createArcaPreviewManager,
  loopbackPreview,
  parseStatusJson,
  sameServer,
} = require("../src/arcaPreview");

function child() {
  const value = new EventEmitter();
  value.stdout = new EventEmitter();
  value.stderr = new EventEmitter();
  value.killed = false;
  value.kill = () => {
    value.killed = true;
  };
  return value;
}

function successfulSpawner({ hostId = "host_arca", serverUrl = "https://srv.example.com/" } = {}) {
  const calls = [];
  const children = [];
  const spawn = (file, args) => {
    const proc = child();
    calls.push({ file, args });
    children.push(proc);
    if (args.includes("status")) {
      queueMicrotask(() => {
        proc.stdout.emit(
          "data",
          JSON.stringify({
            daemons: [
              { host_id: hostId, server_url: serverUrl, process: "online", host_status: "online" },
            ],
          }),
        );
        proc.emit("exit", 0);
      });
    } else if (args.includes("-O")) {
      queueMicrotask(() => proc.emit("exit", 0));
    }
    return proc;
  };
  return { spawn, calls, children };
}

describe("Arca localhost preview URL", () => {
  it("accepts explicit loopback previews and preserves their exact origin", () => {
    assert.deepEqual(loopbackPreview("http://localhost:5173/app"), {
      origin: "http://localhost:5173",
      host: "localhost",
      port: 5173,
    });
    assert.deepEqual(loopbackPreview("http://127.0.0.1:7331/"), {
      origin: "http://127.0.0.1:7331",
      host: "127.0.0.1",
      port: 7331,
    });
    assert.equal(loopbackPreview("https://example.com"), null);
    assert.equal(loopbackPreview("http://10.0.0.2:5173"), null);
  });

  it("parses status JSON after Arca startup notices", () => {
    assert.deepEqual(parseStatusJson('Starting Arca…\n{"daemons":[]}'), { daemons: [] });
  });

  it("matches workspace UI and API mounts without relaxing host or explicit selectors", () => {
    assert.equal(
      sameServer(
        "https://acme.cloud.databricks.com/omnigent?o=123",
        "https://acme.cloud.databricks.com/api/2.0/omnigent",
      ),
      true,
    );
    assert.equal(
      sameServer(
        "https://acme.cloud.databricks.com/omnigent?o=123",
        "https://acme.cloud.databricks.com/api/2.0/omnigent?o=456",
      ),
      false,
    );
    assert.equal(
      sameServer(
        "https://acme.cloud.databricks.com/omnigent",
        "https://other.cloud.databricks.com/api/2.0/omnigent",
      ),
      false,
    );
  });
});

describe("Arca preview manager", () => {
  it("verifies the exact server host then starts requested independent forwards", async () => {
    const fake = successfulSpawner();
    const manager = createArcaPreviewManager({
      resolveArcaPathFn: () => "/usr/local/bin/arca",
      spawnFn: fake.spawn,
      socketReady: () => true,
    });
    const first = await manager.prepare({
      conversationId: "a",
      url: "http://localhost:5173/app",
      hostId: "host_arca",
      serverUrl: "https://srv.example.com",
    });
    const second = await manager.prepare({
      conversationId: "b",
      url: "http://localhost:7331/",
      hostId: "host_arca",
      serverUrl: "https://srv.example.com",
    });

    assert.equal(first.origin, "http://localhost:5173");
    assert.equal(second.origin, "http://localhost:7331");
    const forwards = fake.calls.filter((call) => call.args.includes("-L"));
    assert.deepEqual(
      forwards.map((call) => call.args[call.args.indexOf("-L") + 1]),
      ["localhost:5173:localhost:5173", "localhost:7331:localhost:7331"],
    );
    assert.ok(forwards.every((call) => call.args.includes("/dev/null")));
    const masters = fake.calls.filter((call) => call.args.includes("-M"));
    assert.ok(masters.every((call) => call.args.includes("ClearAllForwardings=yes")));
  });

  it("rejects a different, offline, or unknown requesting host", async () => {
    const fake = successfulSpawner({ hostId: "other_host" });
    const manager = createArcaPreviewManager({
      resolveArcaPathFn: () => "/arca",
      spawnFn: fake.spawn,
      socketReady: () => true,
    });
    await assert.rejects(
      manager.prepare({
        conversationId: "a",
        url: "http://localhost:5173",
        hostId: "host_arca",
        serverUrl: "https://srv.example.com",
      }),
      /not running on this server's Arca host/,
    );
    await assert.rejects(
      manager.prepare({
        conversationId: "b",
        url: "http://localhost:5173",
        hostId: null,
        serverUrl: "https://srv.example.com",
      }),
      /host is unknown/,
    );
  });

  it("fails an occupied desktop port without treating that listener as readiness", async () => {
    const spawn = (_file, args) => {
      const proc = child();
      queueMicrotask(() => {
        if (args.includes("status")) {
          proc.stdout.emit(
            "data",
            JSON.stringify({
              daemons: [
                {
                  host_id: "host_arca",
                  server_url: "https://srv.example.com/",
                  process: "online",
                  host_status: "online",
                },
              ],
            }),
          );
          proc.emit("exit", 0);
        } else if (args.includes("-O")) {
          proc.stderr.emit("data", "bind [127.0.0.1]:5173: Address already in use\n");
          proc.emit("exit", 255);
        }
      });
      return proc;
    };
    const manager = createArcaPreviewManager({
      resolveArcaPathFn: () => "/arca",
      spawnFn: spawn,
      socketReady: () => true,
    });
    await assert.rejects(
      manager.prepare({
        conversationId: "a",
        url: "http://localhost:5173",
        hostId: "host_arca",
        serverUrl: "https://srv.example.com",
      }),
      /Address already in use/,
    );
  });

  it("cancels pending and active work, and reports an unexpected forward exit", async () => {
    const pending = child();
    const exits = [];
    let calls = 0;
    const manager = createArcaPreviewManager({
      resolveArcaPathFn: () => "/arca",
      onExit: (id) => exits.push(id),
      spawnFn: () => {
        calls += 1;
        if (calls === 1) return pending;
        throw new Error("unexpected spawn");
      },
    });
    const attempt = manager.prepare({
      conversationId: "a",
      url: "http://localhost:5173",
      hostId: "host_arca",
      serverUrl: "https://srv.example.com",
    });
    await manager.prepare({ conversationId: "a", url: "https://example.com" });
    assert.equal(pending.killed, true);
    pending.emit("exit", null);
    await assert.rejects(attempt, /cancelled|superseded/);

    const fake = successfulSpawner();
    const active = createArcaPreviewManager({
      resolveArcaPathFn: () => "/arca",
      spawnFn: fake.spawn,
      socketReady: () => true,
      onExit: (id) => exits.push(id),
    });
    const owned = await active.prepare({
      conversationId: "live",
      url: "http://localhost:5173",
      hostId: "host_arca",
      serverUrl: "https://srv.example.com",
    });
    fake.children.at(-2).emit("exit", 1);
    assert.deepEqual(exits, ["live"]);
    owned.release();
  });

  it("settles a preparation deadline and terminates the owned status process", async () => {
    const proc = child();
    const manager = createArcaPreviewManager({
      resolveArcaPathFn: () => "/arca",
      spawnFn: () => proc,
      timeoutMs: 5,
    });
    await assert.rejects(
      manager.prepare({
        conversationId: "slow",
        url: "http://localhost:5173",
        hostId: "host_arca",
        serverUrl: "https://srv.example.com",
      }),
      /timed out preparing/,
    );
    assert.equal(proc.killed, true);
  });
});
