/* Token counter — persists across page navigations in the same tab via sessionStorage */
(function () {
  var KEY = 'bee_tokens';

  function load() {
    try {
      return JSON.parse(sessionStorage.getItem(KEY)) || { input: 0, output: 0 };
    } catch (_) {
      return { input: 0, output: 0 };
    }
  }

  function save(data) {
    try { sessionStorage.setItem(KEY, JSON.stringify(data)); } catch (_) {}
  }

  function fmt(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1).replace(/\.0$/, '') + 'm';
    if (n >= 1000)    return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
    return String(n);
  }

  function render(data) {
    var total = data.input + data.output;
    var badge = document.getElementById('token-total');
    var breakdown = document.getElementById('token-breakdown');
    if (!badge) return;
    badge.textContent = fmt(total) + ' tokens';
    if (breakdown) {
      breakdown.textContent = total > 0
        ? 'in: ' + fmt(data.input) + ' · out: ' + fmt(data.output)
        : '';
    }
  }

  function add(delta) {
    var data = load();
    data.input += (delta.input || 0);
    data.output += (delta.output || 0);
    save(data);
    render(data);
  }

  function reset() {
    save({ input: 0, output: 0 });
    render({ input: 0, output: 0 });
  }

  // Expose reset globally so the inline onclick button can call it
  window.beeTokensReset = reset;

  // Render on page load (defer until DOM is ready — script loads in <head>)
  document.addEventListener('DOMContentLoaded', function () { render(load()); });

  // Patch EventSource to intercept 'tokens' events from all SSE streams.
  // htmx-sse only registers listeners for events named in sse-swap/hx-trigger
  // attributes, so htmx:sseMessage never fires for 'tokens' — we listen directly.
  var _ES = window.EventSource;
  function PatchedES(url, init) {
    var es = new _ES(url, init);
    es.addEventListener('tokens', function (e) {
      try { add(JSON.parse(e.data)); } catch (_) {}
    });
    return es;
  }
  PatchedES.prototype = _ES.prototype;
  PatchedES.CONNECTING = _ES.CONNECTING;
  PatchedES.OPEN = _ES.OPEN;
  PatchedES.CLOSED = _ES.CLOSED;
  window.EventSource = PatchedES;
})();
