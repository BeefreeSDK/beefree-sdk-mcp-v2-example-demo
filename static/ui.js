/* Shared UI behaviour for every page. */
(function () {
  /* ── Show server errors that htmx would otherwise drop ──
     htmx 2 does not swap 4xx/5xx responses, so a "400 Invalid JSON" left the
     user with a stopped spinner and no message. Render the server's message
     as an inline error in the request's target instead. */
  function messageFrom(xhr) {
    var text = (xhr && xhr.responseText) || '';
    try {
      var data = JSON.parse(text);
      if (data && typeof data.detail === 'string') return data.detail;
      if (data && typeof data.error === 'string') return data.error;
    } catch (_) {}
    if (text) {
      var doc = new DOMParser().parseFromString(text, 'text/html');
      var plain = (doc.body.textContent || '').replace(/\s+/g, ' ').trim();
      if (plain) return plain.slice(0, 400);
    }
    return 'The server could not complete this request (status ' + (xhr ? xhr.status : '?') + ').';
  }

  function showError(target, message) {
    if (!target) return;
    var box = document.createElement('div');
    box.className = 'inline-error';
    box.setAttribute('role', 'alert');
    box.textContent = message;
    target.replaceChildren(box);
  }

  document.addEventListener('htmx:responseError', function (e) {
    showError(e.detail.target, messageFrom(e.detail.xhr));
  });
  document.addEventListener('htmx:sendError', function (e) {
    showError(e.detail.target, 'Could not reach the server. Check your connection and try again.');
  });

  /* ── Collapse buttons announce whether their card is open ── */
  function syncCollapse(card) {
    var btn = Array.prototype.find.call(card.querySelectorAll('.collapse-btn'), function (b) {
      return b.closest('.card') === card;
    });
    if (!btn) return;
    var open = !card.classList.contains('is-collapsed');
    btn.setAttribute('aria-expanded', String(open));
    btn.setAttribute('aria-label', open ? 'Collapse section' : 'Expand section');
  }

  var observer = new MutationObserver(function (records) {
    records.forEach(function (r) { syncCollapse(r.target); });
  });
  function watchCards(root) {
    root.querySelectorAll('.card').forEach(function (card) {
      if (card.dataset.collapseWatched) return;
      card.dataset.collapseWatched = '1';
      syncCollapse(card);
      observer.observe(card, { attributes: true, attributeFilter: ['class'] });
    });
  }
  /* Cards swapped in later by htmx (e.g. the sequence plan) are watched too */
  document.addEventListener('htmx:afterSwap', function (e) { watchCards(e.detail.target); });

  document.addEventListener('DOMContentLoaded', function () {
    watchCards(document);

  });
})();

/* Load the ready-made sample template into a template textarea and preview it. */
window.beeLoadSample = function (textareaId, statusId) {
  var ta = document.getElementById(textareaId);
  var status = statusId && document.getElementById(statusId);
  if (status) { status.textContent = 'Loading sample...'; status.className = 'json-status'; }
  fetch('/sample-template').then(function (r) {
    if (!r.ok) throw new Error();
    return r.json();
  }).then(function (template) {
    ta.value = JSON.stringify(template, null, 2);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    if (status) { status.textContent = 'Sample template loaded'; status.className = 'json-status is-valid'; }
  }).catch(function () {
    if (status) { status.textContent = 'Could not load the sample template'; status.className = 'json-status is-error'; }
  });
};
