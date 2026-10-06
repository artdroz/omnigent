// Loaded as an external asset so the bridge can start under an inherited CSP
// that blocks inline scripts. Managed deployments allow this app-authored asset
// and dynamic compilation while the iframe remains an opaque sandbox.
(function () {
  var script = document.currentScript;
  var source = script && script.getAttribute("data-omni-bridge");
  if (source) new Function(source)();
})();
