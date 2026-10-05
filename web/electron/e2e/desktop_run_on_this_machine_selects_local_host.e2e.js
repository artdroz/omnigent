// Desktop-shell journeys for the new-session host picker's local-machine path:
// "Use this machine" (reported as "Run on this machine") must connect and
// select this machine, and a host daemon the user already started in a
// terminal must be picked up — including for a returning user whose saved host
// pick still carries the legacy `host_` prefix.
//
// Run from web/electron after building the SPA:
//   OMNIGENT_PW_NO_SANDBOX=1 OMNIGENT_PYTHON=../../.venv/bin/python \
//     xvfb-run -a node --test --test-concurrency=1 \
//     e2e/desktop_run_on_this_machine_selects_local_host.e2e.js

"use strict";

const { describe, it, before, after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");

const {
  REPO_ROOT,
  desktopDepsAvailable,
  spawnServer,
  launchDesktop,
  saveRecording,
} = require("./desktopHarness");

const deps = desktopDepsAvailable();
const RECORD_ROOT =
  process.env.OMNIGENT_DESKTOP_RECORD_DIR ||
  path.join(__dirname, "recordings", "run-on-this-machine");

const PYTHON = process.env.OMNIGENT_PYTHON || "python3";
const CONNECTED_MARKER = "✓ Connected";
const SELECT_TIMEOUT_MS = 45_000;
const LAST_HOST_CHOICE_KEY = "omnigent:last-host-choice";
const CHIP = '[data-testid="new-chat-landing-host-chip"]';
const HOST_MENU = '[data-testid="new-chat-landing-host-menu"]';
const RUN_ON_THIS_MACHINE = '[data-testid="new-chat-landing-run-on-this-machine"]';
const CONNECT_ERROR = '[data-testid="new-chat-landing-connect-error"]';
// "This machine" on Linux/Windows, "This Mac" on macOS (localMachineLabel).
const THIS_MACHINE = /This (machine|Mac)\b/;

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function writeCliShim(dir) {
  const shim = path.join(dir, "omnigent");
  const pythonPath = [
    REPO_ROOT,
    path.join(REPO_ROOT, "sdks", "python-client"),
    path.join(REPO_ROOT, "sdks", "ui"),
  ].join(path.delimiter);
  fs.writeFileSync(
    shim,
    "#!/usr/bin/env bash\n" +
      `export PYTHONPATH="${pythonPath}\${PYTHONPATH:+:$PYTHONPATH}"\n` +
      `exec "${PYTHON}" -c "from omnigent.cli import main; main()" "$@"\n`,
    { mode: 0o755 },
  );
  return shim;
}

function prepareProfile(label, serverUrl, cliShim) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `omni-local-host-home-${label}-`));
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), `omni-local-host-data-${label}-`));
  const settings = {
    server_url: serverUrl,
    omnigent_path: cliShim,
    // Native enrollment is preapproved because Playwright cannot click its dialog.
    allowed_hosting_origins: [new URL(serverUrl).origin],
  };
  // Dev builds derive userData from appData (see launchDesktop); seed both.
  for (const profile of [userDataDir, path.join(userDataDir, "Omnigent Dev")]) {
    fs.mkdirSync(profile, { recursive: true });
    fs.writeFileSync(path.join(profile, "settings.json"), JSON.stringify(settings, null, 2));
  }
  return { home, userDataDir };
}

function isolateEnv(home) {
  // Electron and the daemon must read the same host identity under HOME, and
  // ambient runner/host identities would override the daemon's file-based one.
  const cleanEnv = Object.fromEntries(
    Object.entries(process.env).filter(
      ([key]) => !key.startsWith("OMNIGENT_HOST_") && !key.startsWith("OMNIGENT_RUNNER_"),
    ),
  );
  delete cleanEnv.OMNIGENT_CONFIG_HOME;
  delete cleanEnv.OMNIGENT_DATA_DIR;
  cleanEnv.HOME = home;
  const noProxy = "127.0.0.1,localhost";
  for (const key of ["NO_PROXY", "no_proxy"]) {
    cleanEnv[key] = cleanEnv[key] ? `${cleanEnv[key]},${noProxy}` : noProxy;
  }
  process.env = cleanEnv;
}

async function waitForSpaWindow(electronApp, firstWindow, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  // The update overlay can open first; select the served SPA window.
  /* oxlint-disable no-await-in-loop */
  while (Date.now() < deadline) {
    const spa = [firstWindow, ...electronApp.windows()].find((w) => w.url().startsWith("http"));
    if (spa) return spa;
    await sleep(500);
  }
  /* oxlint-enable no-await-in-loop */
  throw new Error("no SPA window (http…) appeared within the deadline");
}

function startHostDaemon(cliShim, serverUrl, logPath) {
  const child = spawn(cliShim, ["host", "--server", serverUrl, "--non-interactive"], {
    env: { ...process.env },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let log = "";
  const out = logPath ? fs.openSync(logPath, "w") : null;
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve({ child, connected: false, log }), 60_000);
    const onData = (buf) => {
      const text = buf.toString();
      log += text;
      if (out !== null) fs.writeSync(out, text);
      if (log.includes(CONNECTED_MARKER)) {
        clearTimeout(timer);
        resolve({ child, connected: true, log });
      }
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("exit", () => {
      clearTimeout(timer);
      resolve({ child, connected: false, log });
    });
  });
}

async function fetchHosts(serverUrl) {
  const res = await fetch(`${serverUrl}/v1/hosts`);
  const body = await res.json();
  return body.hosts ?? [];
}

async function waitForOnlineHost(serverUrl, timeoutMs = 20_000) {
  const deadline = Date.now() + timeoutMs;
  let hosts = [];
  /* oxlint-disable no-await-in-loop */
  while (Date.now() < deadline) {
    hosts = await fetchHosts(serverUrl);
    const online = hosts.find((h) => h.status === "online");
    if (online) return online;
    await sleep(500);
  }
  /* oxlint-enable no-await-in-loop */
  throw new Error(`no online host after connect: ${JSON.stringify(hosts)}`);
}

async function chipLabel(window) {
  // The chip renders only an icon; its label is the accessible name.
  return (await window.locator(CHIP).getAttribute("aria-label")) ?? "";
}

async function waitForChipOutcome(window) {
  const errorBox = window.locator(CONNECT_ERROR);
  const deadline = Date.now() + SELECT_TIMEOUT_MS;
  let label = "";
  /* oxlint-disable no-await-in-loop */
  while (Date.now() < deadline) {
    if ((await errorBox.count()) > 0) {
      return { label, error: (await errorBox.textContent()) ?? "(empty error)" };
    }
    label = await chipLabel(window);
    if (THIS_MACHINE.test(label)) return { label, error: null };
    await sleep(500);
  }
  /* oxlint-enable no-await-in-loop */
  return { label, error: null };
}

async function settleAndSnapshot(window, recordDir, name, extra) {
  // Open the host menu so the clip shows which row is selected, hold it, then
  // keep a still + facts.
  let menuText = null;
  try {
    await window.locator(CHIP).click();
    const menu = window.locator(HOST_MENU);
    await menu.waitFor({ state: "visible", timeout: 5_000 });
    menuText = await menu.innerText();
  } catch {
    // A chip whose menu will not open is still filmed and snapshotted.
  }
  await sleep(3000);
  await window.screenshot({ path: path.join(recordDir, `${name}.png`) });
  fs.writeFileSync(
    path.join(recordDir, `${name}.json`),
    JSON.stringify({ ...extra, menuText }, null, 2),
  );
}

describe(
  "desktop shell — 'Use this machine' selects the local host",
  { skip: deps.ok ? false : `missing deps: ${deps.missing.join(", ")}` },
  () => {
    let tmpDir;
    let cliShim;

    before(() => {
      tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "omni-local-host-e2e-"));
      cliShim = writeCliShim(tmpDir);
    });

    after(() => {
      if (tmpDir) fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    it("connects and selects this machine when 'Use this machine' is clicked", async () => {
      // Each journey gets its own host registry.
      const server = await spawnServer(fs.mkdtempSync(path.join(tmpDir, "srv-connect-")));
      const { home, userDataDir } = prepareProfile("connect", server.serverUrl, cliShim);
      isolateEnv(home);
      const recordDir = path.join(RECORD_ROOT, "run-on-this-machine");
      fs.mkdirSync(recordDir, { recursive: true });
      const launched = await launchDesktop({ recordDir, userDataDir });
      const { electronApp, stopDisplayCapture } = launched;
      let saved;
      try {
        const window = await waitForSpaWindow(electronApp, launched.window);
        const chip = window.locator(CHIP);
        await chip.waitFor({ state: "visible", timeout: 30_000 });
        await window
          .locator(`${CHIP}[aria-label*="No host"]`)
          .waitFor({ state: "visible", timeout: 30_000 });

        await chip.click();
        const runItem = window.locator(RUN_ON_THIS_MACHINE);
        await runItem.waitFor({ state: "visible", timeout: 15_000 });
        await sleep(1000);
        await runItem.click();

        const outcome = await waitForChipOutcome(window);
        const hosts = await fetchHosts(server.serverUrl);
        await settleAndSnapshot(window, recordDir, "outcome", { ...outcome, hosts });
        assert.equal(
          outcome.error,
          null,
          `"Use this machine" surfaced a connect error: ${outcome.error}`,
        );
        assert.match(
          outcome.label,
          THIS_MACHINE,
          `host chip never selected this machine — it reads ${JSON.stringify(outcome.label)}`,
        );
        const online = hosts.filter((h) => h.status === "online");
        assert.equal(
          online.length,
          1,
          `expected exactly one online host after the connect, got: ${JSON.stringify(hosts)}`,
        );
      } finally {
        // Stop filming before the window goes away so the clip ends on the
        // observed state rather than on teardown.
        await stopDisplayCapture();
        await electronApp.close();
        saved = saveRecording(recordDir, "run-on-this-machine");
        await server.close();
        fs.rmSync(userDataDir, { recursive: true, force: true });
      }
      assert.ok(saved && saved.length > 0, "no desktop recording was produced");
    });

    it("auto-selects a local host daemon the user already started in a terminal", async () => {
      const server = await spawnServer(fs.mkdtempSync(path.join(tmpDir, "srv-manual-")));
      const { home, userDataDir } = prepareProfile("manual", server.serverUrl, cliShim);
      isolateEnv(home);
      const recordDir = path.join(RECORD_ROOT, "manual-host");
      fs.mkdirSync(recordDir, { recursive: true });

      const { child, connected, log } = await startHostDaemon(
        cliShim,
        server.serverUrl,
        path.join(recordDir, "daemon.log"),
      );

      let electronApp;
      let stopDisplayCapture = async () => {};
      let saved;
      try {
        assert.ok(connected, `omnigent host did not connect:\n${log.slice(-2000)}`);
        await waitForOnlineHost(server.serverUrl);

        const launched = await launchDesktop({ recordDir, userDataDir });
        electronApp = launched.electronApp;
        stopDisplayCapture = launched.stopDisplayCapture;
        const window = await waitForSpaWindow(electronApp, launched.window);
        await window.locator(CHIP).waitFor({ state: "visible", timeout: 30_000 });

        const outcome = await waitForChipOutcome(window);
        const hosts = await fetchHosts(server.serverUrl);
        await settleAndSnapshot(window, recordDir, "outcome", { ...outcome, hosts });
        assert.match(
          outcome.label,
          THIS_MACHINE,
          `host chip never picked the running local host — it reads ${JSON.stringify(outcome.label)}`,
        );
      } finally {
        await stopDisplayCapture();
        if (electronApp) await electronApp.close();
        saved = saveRecording(recordDir, "manual-host");
        child.kill("SIGTERM");
        await server.close();
        fs.rmSync(userDataDir, { recursive: true, force: true });
      }
      assert.ok(saved && saved.length > 0, "no desktop recording was produced");
    });

    it("recovers when the persisted last-host choice carries the legacy host_ prefix", async () => {
      const server = await spawnServer(fs.mkdtempSync(path.join(tmpDir, "srv-legacy-")));
      const { home, userDataDir } = prepareProfile("legacy", server.serverUrl, cliShim);
      isolateEnv(home);
      const recordDir = path.join(RECORD_ROOT, "legacy-host-choice");
      fs.mkdirSync(recordDir, { recursive: true });

      const { child, connected, log } = await startHostDaemon(
        cliShim,
        server.serverUrl,
        path.join(recordDir, "daemon.log"),
      );

      let electronApp;
      let stopDisplayCapture = async () => {};
      let saved;
      try {
        assert.ok(connected, `omnigent host did not connect:\n${log.slice(-2000)}`);
        const { host_id: hostId } = await waitForOnlineHost(server.serverUrl);

        const launched = await launchDesktop({ recordDir, userDataDir });
        electronApp = launched.electronApp;
        stopDisplayCapture = launched.stopDisplayCapture;
        const window = await waitForSpaWindow(electronApp, launched.window);
        const chip = window.locator(CHIP);
        await chip.waitFor({ state: "visible", timeout: 30_000 });
        const beforeSeed = await waitForChipOutcome(window);

        // Recreate the pick a desktop build predating the id-format change saved.
        await window.evaluate(
          ([key, id]) => localStorage.setItem(key, `host_${id}`),
          [LAST_HOST_CHOICE_KEY, hostId],
        );
        await window.reload();
        await chip.waitFor({ state: "visible", timeout: 30_000 });

        const outcome = await waitForChipOutcome(window);
        const hosts = await fetchHosts(server.serverUrl);
        const storedChoice = await window.evaluate(
          (key) => localStorage.getItem(key),
          [LAST_HOST_CHOICE_KEY],
        );
        await settleAndSnapshot(window, recordDir, "outcome", {
          beforeSeed,
          ...outcome,
          storedChoice,
          hosts,
        });
        assert.match(
          outcome.label,
          THIS_MACHINE,
          `host chip never recovered from the legacy-prefixed stored choice — it reads ` +
            `${JSON.stringify(outcome.label)}`,
        );
      } finally {
        await stopDisplayCapture();
        if (electronApp) await electronApp.close();
        saved = saveRecording(recordDir, "legacy-host-choice");
        child.kill("SIGTERM");
        await server.close();
        fs.rmSync(userDataDir, { recursive: true, force: true });
      }
      assert.ok(saved && saved.length > 0, "no desktop recording was produced");
    });
  },
);
