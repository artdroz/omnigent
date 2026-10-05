"use strict";

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { normalizeSafeServerUrl, resolveArcaPath } = require("./arca");
const { WORKSPACE_UI_PATH } = require("./url");

const PREPARE_TIMEOUT_MS = 25_000;
const OUTPUT_LIMIT = 8_192;
const WORKSPACE_API_PATHS = new Set(["/api/2.0/omnigent", "/api/2.0/omnigents"]);

function loopbackPreview(url) {
  try {
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol)) return null;
    if (!["localhost", "127.0.0.1"].includes(parsed.hostname)) return null;
    const port = Number(parsed.port || (parsed.protocol === "https:" ? 443 : 80));
    if (!Number.isInteger(port) || port < 1 || port > 65535) return null;
    return { origin: parsed.origin, host: parsed.hostname, port };
  } catch {
    return null;
  }
}

function serverIdentity(value) {
  try {
    const url = new URL(value);
    const pathname = url.pathname.replace(/\/+$/, "") || "/";
    const workspace = pathname === WORKSPACE_UI_PATH || WORKSPACE_API_PATHS.has(pathname);
    return {
      base: `${url.protocol}//${url.host}${workspace ? WORKSPACE_UI_PATH : pathname}`,
      workspace: url.searchParams.get("o"),
    };
  } catch {
    return null;
  }
}

function sameServer(left, right) {
  const a = serverIdentity(left);
  const b = serverIdentity(right);
  if (!a || !b || a.base !== b.base) return false;
  // The scoped status request is routed with the requested selector, but its
  // daemon record normally omits `?o=` and does not attest that selector.
  // Reject an explicit mismatch whenever the record does carry one.
  return !a.workspace || (!!b.workspace && a.workspace === b.workspace);
}

function parseStatusJson(stdout) {
  for (let index = stdout.indexOf("{"); index >= 0; index = stdout.indexOf("{", index + 1)) {
    try {
      return JSON.parse(stdout.slice(index));
    } catch {
      // Arca may print startup notices before the JSON payload.
    }
  }
  return null;
}

function terminate(child) {
  if (!child) return;
  const signal = (name) => {
    try {
      if (child.pid && child.spawnargs) process.kill(-child.pid, name);
      else child.kill(name);
    } catch {
      /* already exited */
    }
  };
  signal("SIGTERM");
  const timer = setTimeout(() => signal("SIGKILL"), 1_000);
  timer.unref?.();
}

function run(file, args, { spawnFn, deadline, onChild }) {
  return new Promise((resolve, reject) => {
    let child;
    let stdout = "";
    let stderr = "";
    let settled = false;
    let timer;
    const finish = (error, code = null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve({ code, stdout, stderr });
    };
    try {
      child = spawnFn(file, args, { stdio: ["ignore", "pipe", "pipe"], detached: true });
      onChild(child, () => finish(new Error("preview was cancelled")));
    } catch (error) {
      reject(error);
      return;
    }
    timer = setTimeout(
      () => {
        terminate(child);
        finish(new Error("timed out preparing the Arca localhost preview"));
      },
      Math.max(0, deadline - Date.now()),
    );
    timer.unref?.();
    child.stdout?.on("data", (chunk) => (stdout = (stdout + String(chunk)).slice(-OUTPUT_LIMIT)));
    child.stderr?.on("data", (chunk) => (stderr = (stderr + String(chunk)).slice(-OUTPUT_LIMIT)));
    child.on("error", (error) => finish(error));
    child.on("exit", (code) => finish(null, code));
  });
}

async function verifyArcaHost({ arcaPath, serverUrl, hostId, spawnFn, deadline, onChild }) {
  const safeServerUrl = normalizeSafeServerUrl(serverUrl);
  const result = await run(
    arcaPath,
    [
      "ssh",
      "-o",
      "ClearAllForwardings=yes",
      "isaac",
      "omni",
      "host",
      "status",
      "--server",
      safeServerUrl,
      "--json",
    ],
    { spawnFn, deadline, onChild },
  );
  if (result.code !== 0) throw new Error(result.stderr.trim() || "could not read Arca host status");
  const daemon = parseStatusJson(result.stdout)?.daemons?.find(
    (item) =>
      item?.host_id === hostId &&
      sameServer(item.server_url, safeServerUrl) &&
      item.process === "online" &&
      item.host_status === "online",
  );
  if (!daemon) throw new Error("the requesting session is not running on this server's Arca host");
}

function waitForSocket(socketPath, deadline, isSocketReady) {
  return new Promise((resolve, reject) => {
    const poll = () => {
      if (isSocketReady(socketPath)) return resolve();
      if (Date.now() >= deadline) return reject(new Error("timed out starting Arca SSH"));
      const timer = setTimeout(poll, 20);
      timer.unref?.();
    };
    poll();
  });
}

function createArcaPreviewManager({
  resolveArcaPathFn = resolveArcaPath,
  spawnFn = spawn,
  timeoutMs = PREPARE_TIMEOUT_MS,
  socketReady = fs.existsSync,
  unlinkSocket = (socketPath) => fs.rmSync(socketPath, { force: true }),
  removeSocketDir = (socketDir) => fs.rmSync(socketDir, { recursive: true, force: true }),
  shutdownTimeoutMs = 500,
  socketPathFn = () => {
    const socketDir = fs.mkdtempSync(path.join("/tmp", "oa-"));
    return { socketPath: path.join(socketDir, "s"), socketDir };
  },
  onExit = () => {},
} = {}) {
  const owned = new Map();
  let sequence = 0;

  function cleanupSocket(state) {
    try {
      if (state.socketPath) unlinkSocket(state.socketPath);
    } catch {
      /* already removed */
    }
    try {
      if (state.socketDir) removeSocketDir(state.socketDir);
    } catch {
      /* already removed */
    }
  }

  function release(conversationId, token) {
    const current = owned.get(conversationId);
    if (!current || (token && current.token !== token)) return null;
    owned.delete(conversationId);
    current.cancel?.();
    let control = null;
    if (current.master && !current.masterExited && current.socketPath) {
      try {
        // Keep the socket reachable until the exact owned mux master has been
        // asked to exit; the timer below bounds wrapper incompatibility.
        control = spawnFn(
          current.arcaPath,
          ["ssh", "-F", "/dev/null", "-S", current.socketPath, "-O", "exit"],
          { stdio: ["ignore", "ignore", "ignore"], detached: true },
        );
      } catch {
        /* fall through to process-group termination */
      }
    }
    if (!control) {
      terminate(current.child);
      if (current.master !== current.child) terminate(current.master);
      cleanupSocket(current);
      return null;
    }
    return new Promise((resolve) => {
      let settled = false;
      const finish = (timedOut = false) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (timedOut) terminate(control);
        terminate(current.child);
        if (current.master !== current.child) terminate(current.master);
        cleanupSocket(current);
        resolve();
      };
      const timer = setTimeout(() => finish(true), shutdownTimeoutMs);
      control.once("error", () => finish());
      control.once("exit", () => finish());
    });
  }

  async function prepare({ conversationId, url, hostId, serverUrl, deadline: requestedDeadline }) {
    const preview = loopbackPreview(url);
    const priorRelease = release(conversationId);
    if (priorRelease) await priorRelease;
    if (!preview) return null;
    if (typeof hostId !== "string" || !hostId)
      throw new Error("the requesting session's host is unknown");
    const arcaPath = resolveArcaPathFn();
    if (!arcaPath) throw new Error("the arca CLI was not found on this machine");
    const token = ++sequence;
    const deadline = Math.min(requestedDeadline ?? Infinity, Date.now() + timeoutMs);
    if (deadline <= Date.now()) throw new Error("timed out preparing the Arca localhost preview");
    const state = {
      token,
      arcaPath,
      child: null,
      master: null,
      masterExited: false,
      cancel: null,
      socketPath: null,
      socketDir: null,
    };
    owned.set(conversationId, state);
    const onChild = (child, cancel) => {
      if (owned.get(conversationId)?.token !== token) {
        terminate(child);
        cancel();
        return;
      }
      state.child = child;
      state.cancel = cancel;
    };
    try {
      await verifyArcaHost({ arcaPath, serverUrl, hostId, spawnFn, deadline, onChild });
      if (owned.get(conversationId)?.token !== token) throw new Error("preview was superseded");
      const socket = socketPathFn();
      const socketPath = typeof socket === "string" ? socket : socket.socketPath;
      state.socketPath = socketPath;
      state.socketDir = typeof socket === "string" ? null : socket.socketDir;
      // ControlPersist=no keeps config from daemonizing away from our process
      // group. A sudden Electron SIGKILL can still orphan this stopgap.
      const master = spawnFn(
        arcaPath,
        [
          "ssh",
          "-M",
          "-S",
          socketPath,
          "-o",
          "ClearAllForwardings=yes",
          "-o",
          "ControlPersist=no",
          "-N",
        ],
        { stdio: ["ignore", "ignore", "pipe"], detached: true },
      );
      master.stderr?.resume?.();
      state.child = master;
      state.master = master;
      let rejectMaster;
      const masterExit = new Promise((_, reject) => {
        rejectMaster = reject;
        master.on("error", (error) => {
          state.masterExited = true;
          reject(error);
        });
        master.on("exit", (code) => {
          state.masterExited = true;
          reject(new Error(`Arca preview exited (${code ?? "unknown"})`));
        });
      });
      state.cancel = () => rejectMaster(new Error("preview was cancelled"));
      await Promise.race([waitForSocket(socketPath, deadline, socketReady), masterExit]);
      const bindHosts = preview.host === "localhost" ? ["127.0.0.1", "[::1]"] : [preview.host];
      for (const bindHost of bindHosts) {
        const spec = `${bindHost}:${preview.port}:${preview.host}:${preview.port}`;
        // Each exact family is acknowledged independently; either failure
        // tears down the master and therefore rolls back the other forward.
        // eslint-disable-next-line no-await-in-loop
        const acknowledged = await Promise.race([
          run(
            arcaPath,
            [
              "ssh",
              "-F",
              "/dev/null",
              "-S",
              socketPath,
              "-O",
              "forward",
              "-o",
              "ExitOnForwardFailure=yes",
              "-L",
              spec,
            ],
            {
              spawnFn,
              deadline,
              onChild: (child, cancel) => {
                state.cancel = () => {
                  terminate(child);
                  cancel();
                };
              },
            },
          ),
          masterExit,
        ]);
        if (acknowledged.code !== 0) {
          throw new Error(acknowledged.stderr.trim() || "Arca rejected the localhost forward");
        }
        if (owned.get(conversationId)?.token !== token) {
          throw new Error("preview was superseded");
        }
      }
      state.child = master;
      state.cancel = null;
      master.on("exit", () => {
        if (owned.get(conversationId)?.token !== token) return;
        owned.delete(conversationId);
        unlinkSocket(socketPath);
        if (state.socketDir) removeSocketDir(state.socketDir);
        onExit(conversationId);
      });
      return { origin: preview.origin, release: () => release(conversationId, token) };
    } catch (error) {
      await release(conversationId, token);
      throw error;
    }
  }

  return { prepare, release: (conversationId) => release(conversationId) };
}

module.exports = {
  createArcaPreviewManager,
  loopbackPreview,
  parseStatusJson,
  sameServer,
};
