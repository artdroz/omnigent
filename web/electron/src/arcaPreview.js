"use strict";

const { spawn } = require("node:child_process");
const { normalizeSafeServerUrl, resolveArcaPath } = require("./arca");

const STATUS_TIMEOUT_MS = 30_000;
const FORWARD_TIMEOUT_MS = 30_000;

function loopbackPreview(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
  if (!new Set(["localhost", "127.0.0.1"]).has(parsed.hostname)) return null;
  const port = Number(parsed.port || (parsed.protocol === "https:" ? 443 : 80));
  if (!Number.isInteger(port) || port < 1 || port > 65535) return null;
  return { origin: parsed.origin, host: parsed.hostname, port };
}

function normalizeServerUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.toString().replace(/\/+$/, "");
  } catch {
    return null;
  }
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

function capture(
  file,
  args,
  { spawnFn = spawn, timeoutMs = STATUS_TIMEOUT_MS, onChild = () => {} } = {},
) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawnFn(file, args, { stdio: ["ignore", "pipe", "pipe"] });
      onChild(child);
    } catch (error) {
      resolve({ code: null, stdout: "", stderr: error.message });
      return;
    }
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => child.kill(), timeoutMs);
    timer.unref?.();
    child.stdout?.on("data", (chunk) => (stdout += String(chunk)));
    child.stderr?.on("data", (chunk) => (stderr += String(chunk)));
    let settled = false;
    const finish = (code, error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ code, stdout, stderr: error ? `${stderr}\n${error.message}` : stderr });
    };
    child.on("error", (error) => finish(null, error));
    child.on("exit", (code) => finish(code));
  });
}

async function verifyArcaHost({ arcaPath, serverUrl, hostId, spawnFn, timeoutMs, onChild }) {
  const safeServerUrl = normalizeSafeServerUrl(serverUrl);
  const result = await capture(
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
    { spawnFn, timeoutMs, onChild },
  );
  if (result.code !== 0) {
    throw new Error(result.stderr.trim() || "could not read Arca host status");
  }
  const payload = parseStatusJson(result.stdout);
  const expectedServer = normalizeServerUrl(safeServerUrl);
  const daemon = payload?.daemons?.find(
    (item) =>
      item?.host_id === hostId &&
      normalizeServerUrl(item.server_url) === expectedServer &&
      item.process === "online" &&
      item.host_status === "online",
  );
  if (!daemon) throw new Error("the requesting session is not running on this server's Arca host");
}

function startForward({ arcaPath, preview, spawnFn = spawn, timeoutMs = FORWARD_TIMEOUT_MS }) {
  const spec = `${preview.host}:${preview.port}:${preview.host}:${preview.port}`;
  let child;
  try {
    child = spawnFn(
      arcaPath,
      [
        "ssh",
        "-v",
        "-o",
        "ClearAllForwardings=no",
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
        "-L",
        spec,
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
  } catch (error) {
    return { child: null, promise: Promise.reject(error) };
  }
  const promise = new Promise((resolve, reject) => {
    let stderr = "";
    let settled = false;
    const timer = setTimeout(() => {
      child.kill();
      finish(new Error("timed out starting the Arca localhost preview"));
    }, timeoutMs);
    timer.unref?.();
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve();
    };
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
      if (/Local forwarding listening on .* port \d+/i.test(stderr)) finish();
    });
    child.on("error", finish);
    child.on("exit", (code) => {
      if (!settled)
        finish(new Error(stderr.trim() || `Arca preview exited (${code ?? "unknown"})`));
    });
  });
  return { child, promise };
}

function createArcaPreviewManager({
  resolveArcaPathFn = resolveArcaPath,
  spawnFn = spawn,
  statusTimeoutMs = STATUS_TIMEOUT_MS,
  forwardTimeoutMs = FORWARD_TIMEOUT_MS,
  onExit = () => {},
} = {}) {
  const owned = new Map();
  let sequence = 0;

  function release(conversationId, token) {
    const current = owned.get(conversationId);
    if (!current || (token && current.token !== token)) return;
    owned.delete(conversationId);
    current.child?.kill();
  }

  async function prepare({ conversationId, url, hostId, serverUrl }) {
    const preview = loopbackPreview(url);
    release(conversationId);
    if (!preview) return null;
    if (typeof hostId !== "string" || !hostId) {
      throw new Error("the requesting session's host is unknown");
    }
    const arcaPath = resolveArcaPathFn();
    if (!arcaPath) throw new Error("the arca CLI was not found on this machine");
    const token = ++sequence;
    owned.set(conversationId, { token, child: null });
    try {
      await verifyArcaHost({
        arcaPath,
        serverUrl,
        hostId,
        spawnFn,
        timeoutMs: statusTimeoutMs,
        onChild: (child) => {
          if (owned.get(conversationId)?.token === token) {
            owned.set(conversationId, { token, child });
          } else {
            child.kill();
          }
        },
      });
      if (owned.get(conversationId)?.token !== token) throw new Error("preview was superseded");
      const forward = startForward({
        arcaPath,
        preview,
        spawnFn,
        timeoutMs: forwardTimeoutMs,
      });
      owned.set(conversationId, { token, child: forward.child });
      let active = false;
      forward.child?.on("exit", () => {
        if (owned.get(conversationId)?.token !== token) return;
        owned.delete(conversationId);
        if (active) onExit(conversationId);
      });
      await forward.promise;
      if (owned.get(conversationId)?.token !== token) throw new Error("preview was superseded");
      active = true;
      return {
        origin: preview.origin,
        release: () => release(conversationId, token),
      };
    } catch (error) {
      release(conversationId, token);
      throw error;
    }
  }

  return { prepare, release: (conversationId) => release(conversationId) };
}

module.exports = {
  createArcaPreviewManager,
  loopbackPreview,
  parseStatusJson,
  verifyArcaHost,
};
