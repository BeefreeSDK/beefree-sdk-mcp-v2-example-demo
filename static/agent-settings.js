/* Agent settings on /builder: brand rules, reusable rows and File Manager access.

   The dialog (templates/partials/agent_settings.html) offers the headless pages'
   examples, or your own JSON, and checks the choice with the same endpoints
   (/brand-rules, /rows-library, /files-access). The builder then passes
   options() to bee.startMcpSession() on every chat turn, so each turn's session
   carries them, and sends flags() with the message so the agent knows how to
   use them. The choice is remembered in this browser only. */
(function () {
  var dialog = document.getElementById('as-dialog');
  if (!dialog) return;
  function $(id) { return document.getElementById(id); }

  var DATA = JSON.parse($('as-data').textContent);
  var STORE_KEY = 'beefree-demo-agent-settings';
  var UID_KEY = 'beefree-demo-editor-uid';      // the editor's user, see /integration-auth
  var SECTIONS = ['brand', 'rows', 'files'];
  var DEFAULTS = {
    brand: { on: false, choice: DATA.brandPresets[0].id, json: '' },
    rows: { on: false, choice: 'sample', json: '' },
    files: { on: false, choice: DATA.filePresets[0].id, json: '' }
  };

  function clone(o) { return JSON.parse(JSON.stringify(o)); }
  function byId(list, id) { return list.filter(function (x) { return x.id === id; })[0]; }

  // ── Saved state (per-browser convenience; the page works without it) ──
  function load() {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch (_) {}
    var state = clone(DEFAULTS);
    if (saved && typeof saved === 'object') {
      SECTIONS.forEach(function (k) {
        var s = saved[k];
        if (s && typeof s === 'object') {
          state[k] = { on: s.on === true, choice: String(s.choice || state[k].choice), json: String(s.json || '') };
        }
      });
    }
    // A preset that no longer exists falls back to off
    if (state.brand.choice !== 'json' && !byId(DATA.brandPresets, state.brand.choice)) state.brand = clone(DEFAULTS.brand);
    if (state.files.choice !== 'json' && !byId(DATA.filePresets, state.files.choice)) state.files = clone(DEFAULTS.files);
    return state;
  }
  function save(state) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (_) {}
  }

  var applied = load();
  var draft = clone(applied);
  var listeners = [];

  // ── What each setting sends ──
  function brandRules(s) {
    if (s.choice === 'json') return JSON.parse(s.json);
    return byId(DATA.brandPresets, s.choice).rules;
  }
  function rowLibrary(s) {
    return s.choice === 'json' ? JSON.parse(s.json) : DATA.sampleLibrary;
  }
  function fileAccess(s) {
    if (s.choice === 'json') {
      var parsed = JSON.parse(s.json);
      if (parsed && typeof parsed === 'object') delete parsed.uid;   // the editor adds its own user
      return parsed;
    }
    return clone(byId(DATA.filePresets, s.choice).config);
  }

  // The options for bee.startMcpSession(); undefined when nothing is on.
  function options() {
    var o = {};
    try {
      if (applied.brand.on) o.brandRules = brandRules(applied.brand);
      if (applied.rows.on) o.reusableRows = rowLibrary(applied.rows);
      if (applied.files.on) o.fileManager = fileAccess(applied.files);
    } catch (_) { return undefined; }
    return Object.keys(o).length ? o : undefined;
  }

  // What the chat tells the server is on, so the agent's prompt explains it.
  function flags() {
    var f = { brand: applied.brand.on, rows: applied.rows.on };
    if (applied.files.on) { try { f.files = fileAccess(applied.files); } catch (_) {} }
    return f;
  }

  function labels() {
    var out = [];
    if (applied.brand.on) {
      var preset = byId(DATA.brandPresets, applied.brand.choice);
      out.push({ section: 'brand', text: 'Brand: ' + (preset ? preset.name : 'your rules') });
    }
    if (applied.rows.on) {
      var n = 0;
      try { n = (rowLibrary(applied.rows).entries || []).length; } catch (_) {}
      out.push({ section: 'rows', text: (applied.rows.choice === 'sample' ? 'Sample rows' : 'Your rows') + ' (' + n + ')' });
    }
    if (applied.files.on) {
      var fp = byId(DATA.filePresets, applied.files.choice);
      out.push({ section: 'files', text: 'Files: ' + (fp ? fp.tag : 'custom access') });
    }
    return out;
  }

  // ── Dialog rendering ──
  var currentTab = 'brand';
  function selectTab(section, focus) {
    currentTab = section;
    dialog.querySelectorAll('.as-tabs [role=tab]').forEach(function (t) {
      var on = t.dataset.section === section;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    SECTIONS.forEach(function (k) { $('as-panel-' + k).hidden = k !== section; });
    if (section === 'files') loadFiles();
  }

  function render() {
    SECTIONS.forEach(function (k) {
      var s = draft[k];
      $('as-' + k + '-on').checked = s.on;
      dialog.querySelector('[data-fields-for=' + k + ']').classList.toggle('is-off', !s.on);
      var state = dialog.querySelector('[data-state-for=' + k + ']');
      state.textContent = s.on ? 'On' : 'Off';
      state.classList.toggle('is-on', s.on);
      dialog.querySelectorAll('.as-choice[data-group=' + k + ']').forEach(function (b) {
        var checked = b.dataset.value === s.choice;
        b.setAttribute('aria-checked', String(checked));
        b.tabIndex = checked ? 0 : -1;
      });
      dialog.querySelector('[data-json-for=' + k + ']').hidden = s.choice !== 'json';
      $('as-' + k + '-json').value = s.json;
    });
    dialog.querySelector('[data-sample-for=rows]').hidden = draft.rows.choice !== 'sample';
  }

  // When switching to "Your own JSON", start from what was picked before.
  function seedJson(section) {
    if (draft[section].json.trim()) return;
    var prev = clone(draft[section]);
    try {
      var value = section === 'brand' ? brandRules(prev) : section === 'rows' ? rowLibrary(prev) : fileAccess(prev);
      draft[section].json = JSON.stringify(value, null, 2);
    } catch (_) {}
  }

  SECTIONS.forEach(function (k) {
    $('as-' + k + '-on').addEventListener('change', function (e) {
      draft[k].on = e.target.checked;
      hideErrors();
      render();
    });
    $('as-' + k + '-json').addEventListener('input', function (e) { draft[k].json = e.target.value; });
  });

  dialog.querySelectorAll('.as-choice').forEach(function (b) {
    b.addEventListener('click', function () {
      var k = b.dataset.group;
      if (b.dataset.value === 'json') seedJson(k);
      draft[k].choice = b.dataset.value;
      hideErrors();
      render();
    });
  });

  // Arrow keys move between radio cards and between tabs
  dialog.addEventListener('keydown', function (e) {
    var keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    if (!(e.key in keys)) return;
    var card = e.target.closest('.as-choice');
    var tab = e.target.closest('.as-tabs [role=tab]');
    if (card) {
      var cards = Array.prototype.slice.call(dialog.querySelectorAll('.as-choice[data-group=' + card.dataset.group + ']'));
      var next = cards[(cards.indexOf(card) + keys[e.key] + cards.length) % cards.length];
      e.preventDefault();
      next.click();
      next.focus();
    } else if (tab && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      var i = SECTIONS.indexOf(tab.dataset.section);
      e.preventDefault();
      selectTab(SECTIONS[(i + keys[e.key] + SECTIONS.length) % SECTIONS.length], true);
    }
  });
  dialog.querySelectorAll('.as-tabs [role=tab]').forEach(function (t) {
    t.addEventListener('click', function () { selectTab(t.dataset.section); });
  });

  // ── Errors ──
  function hideErrors() { var box = $('as-errors'); box.hidden = true; box.replaceChildren(); }
  function showErrors(groups) {
    var box = $('as-errors');
    box.replaceChildren();
    groups.forEach(function (g) {
      var p = document.createElement('p');
      p.textContent = g.title;
      box.appendChild(p);
      var ul = document.createElement('ul');
      g.errors.slice(0, 8).forEach(function (m) {
        var li = document.createElement('li');
        li.textContent = m;
        ul.appendChild(li);
      });
      box.appendChild(ul);
    });
    box.hidden = false;
  }

  // ── The visitor's File Manager (same one the editor's image picker opens) ──
  function uid() { try { return localStorage.getItem(UID_KEY); } catch (_) { return null; } }
  function rememberUid(value) { try { if (value) localStorage.setItem(UID_KEY, value); } catch (_) {} }
  function fmStatus(text) { $('as-fm-status').textContent = text || ''; }

  function renderFiles(data) {
    var list = $('as-fm-folders');
    list.replaceChildren();
    (data.folders || []).forEach(function (folder) {
      var li = document.createElement('li');
      var name = document.createElement('p');
      name.className = 'as-fm-folder-name';
      name.textContent = folder.path + ' ';
      var count = document.createElement('span');
      count.textContent = folder.files.length ? '(' + folder.files.length + ')' : '(empty)';
      name.appendChild(count);
      li.appendChild(name);
      if (folder.files.length) {
        var ul = document.createElement('ul');
        ul.className = 'as-fm-files';
        folder.files.forEach(function (f) {
          var item = document.createElement('li');
          item.className = 'as-fm-file';
          var img = document.createElement('img');
          img.src = f.thumbnail || f.url;
          img.alt = '';
          img.loading = 'lazy';
          item.appendChild(img);
          item.appendChild(document.createTextNode(f.name));
          ul.appendChild(item);
        });
        li.appendChild(ul);
      }
      list.appendChild(li);
    });
  }

  var filesLoaded = false;
  async function loadFiles(force) {
    if (filesLoaded && !force) return;
    filesLoaded = true;
    fmStatus('Reading your File Manager…');
    try {
      var id = uid();
      var resp = await fetch('/files-state' + (id ? '?uid=' + encodeURIComponent(id) : ''));
      var data = await resp.json().catch(function () { return {}; });
      if (data.uid) rememberUid(data.uid);
      if (!resp.ok) { fmStatus((data.errors || ['Your File Manager could not be read.'])[0]); filesLoaded = false; return; }
      renderFiles(data);
      var sample = (data.folders || [])[0];
      fmStatus(sample && sample.files.length ? '' : 'No sample files yet: add them so the agent has brand images to reuse.');
    } catch (_) {
      filesLoaded = false;
      fmStatus('The server could not be reached. Check your connection and try again.');
    }
  }
  $('as-fm-refresh').addEventListener('click', function () { loadFiles(true); });
  $('as-fm-samples').addEventListener('click', async function () {
    var btn = this;
    var id = uid();
    if (!id) { fmStatus('Wait for the editor to load, then try again.'); return; }
    btn.disabled = true;
    $('as-fm-spinner').hidden = false;
    fmStatus('Adding the sample files…');
    try {
      var resp = await fetch('/files-samples', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ uid: id })
      });
      var data = await resp.json().catch(function () { return {}; });
      if (data.folders) renderFiles(data);
      fmStatus(data.errors ? data.errors[0] : 'Added. The editor’s image picker shows them too.');
    } catch (_) {
      fmStatus('The server could not be reached. Check your connection and try again.');
    } finally {
      btn.disabled = false;
      $('as-fm-spinner').hidden = true;
    }
  });

  // ── Save: check every setting that is on, with Beefree ──
  async function check(url, body) {
    var resp = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    var data = await resp.json().catch(function () { return {}; });
    if (!resp.ok) throw (data.errors && data.errors.length ? data.errors : ['The server could not check this (status ' + resp.status + ').']);
  }

  var TITLES = { brand: 'Brand rules', rows: 'Reusable rows', files: 'File Manager' };

  async function saveDraft() {
    hideErrors();
    var groups = [];
    var checks = [];
    SECTIONS.forEach(function (k) {
      if (!draft[k].on) return;
      var value;
      try {
        value = k === 'brand' ? brandRules(draft[k]) : k === 'rows' ? rowLibrary(draft[k]) : fileAccess(draft[k]);
      } catch (e) {
        groups.push({ section: k, title: TITLES[k] + ': the JSON is not valid', errors: [e.message] });
        return;
      }
      if (k === 'brand') checks.push({ k: k, p: check('/brand-rules', { rules: value }) });
      // The sample library is the one /rows already checks; only pasted ones need a round trip.
      if (k === 'rows' && draft.rows.choice === 'json') checks.push({ k: k, p: check('/rows-library', { library: value }) });
      if (k === 'files') {
        var id = uid();
        if (!id) groups.push({ section: k, title: TITLES[k], errors: ['Wait for the editor to load, then save again.'] });
        else checks.push({ k: k, p: check('/files-access', { access: value, uid: id }) });
      }
    });

    var btn = $('as-save');
    btn.disabled = true;
    $('as-save-spinner').hidden = false;
    var results = await Promise.allSettled(checks.map(function (c) { return c.p; }));
    btn.disabled = false;
    $('as-save-spinner').hidden = true;
    results.forEach(function (r, i) {
      if (r.status === 'rejected') {
        var reason = r.reason;
        var errors = Array.isArray(reason) ? reason : ['The server could not be reached. Check your connection and try again.'];
        groups.push({ section: checks[i].k, title: TITLES[checks[i].k] + ': Beefree refused it', errors: errors });
      }
    });
    if (groups.length) {
      showErrors(groups);
      selectTab(groups[0].section);
      return;
    }
    applied = clone(draft);
    save(applied);
    dialog.close('saved');
    listeners.forEach(function (fn) { fn(labels()); });
  }

  $('as-save').addEventListener('click', saveDraft);
  $('as-cancel').addEventListener('click', function () { dialog.close('cancel'); });
  $('as-close').addEventListener('click', function () { dialog.close('cancel'); });

  function open(section) {
    draft = clone(applied);
    hideErrors();
    render();
    selectTab(section || currentTab);
    dialog.showModal();
  }

  // ── Strip under the chat header and the settings button ──
  function renderStrip() {
    var strip = $('as-strip');
    if (!strip) return;
    strip.replaceChildren();
    var items = labels();
    var dot = document.querySelector('.as-open-btn .as-dot');
    if (dot) dot.hidden = !items.length;
    if (!items.length) {
      var empty = document.createElement('span');
      empty.className = 'as-strip-empty';
      empty.textContent = 'No brand rules, saved rows or files yet.';
      strip.appendChild(empty);
    }
    items.forEach(function (item) {
      var name = TITLES[item.section].toLowerCase();
      var chip = document.createElement('span');
      chip.className = 'as-chip';
      var label = document.createElement('button');
      label.type = 'button';
      label.className = 'as-chip-label';
      label.textContent = item.text;
      label.title = 'Change ' + name;
      label.addEventListener('click', function () { open(item.section); });
      var remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'as-chip-remove';
      remove.title = 'Stop using ' + name;
      remove.setAttribute('aria-label', 'Stop using ' + name);
      remove.innerHTML = '<svg width="10" height="10" viewBox="0 0 12 12" fill="none" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
      remove.addEventListener('click', function () { turnOff(item.section); });
      chip.appendChild(label);
      chip.appendChild(remove);
      strip.appendChild(chip);
    });
    var edit = document.createElement('button');
    edit.type = 'button';
    edit.className = 'btn-link as-strip-edit';
    edit.textContent = items.length ? 'Change' : 'Add them';
    edit.addEventListener('click', function () { open(); });
    strip.appendChild(edit);
  }
  // The chip's x: switch one setting off without opening the dialog. Its choice
  // is kept, so switching it back on in the dialog brings it back as it was.
  function turnOff(section) {
    applied[section].on = false;
    save(applied);
    listeners.forEach(function (fn) { fn(labels()); });
    var next = document.querySelector('#as-strip .as-chip-remove') || $('as-open');
    if (next) next.focus({ preventScroll: true });
  }

  listeners.push(renderStrip);
  renderStrip();

  window.beeAgentSettings = {
    open: open,
    options: options,
    flags: flags,
    labels: labels,
    onChange: function (fn) { listeners.push(fn); }
  };
})();
