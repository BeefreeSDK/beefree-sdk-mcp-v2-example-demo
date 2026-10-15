(function () {
  var storageKey = 'bee_model';
  var selectedModel = null;
  var ready = false;
  var NativeEventSource = window.EventSource;

  function ModelEventSource(url, options) {
    var target = new URL(url, window.location.href);
    if (target.origin === window.location.origin && selectedModel && !target.searchParams.has('model')) {
      target.searchParams.set('model', selectedModel);
    }
    return new NativeEventSource(target.href, options);
  }
  ModelEventSource.prototype = NativeEventSource.prototype;
  ModelEventSource.CONNECTING = NativeEventSource.CONNECTING;
  ModelEventSource.OPEN = NativeEventSource.OPEN;
  ModelEventSource.CLOSED = NativeEventSource.CLOSED;
  window.EventSource = ModelEventSource;

  document.addEventListener('DOMContentLoaded', function () {
    var host = document.querySelector('.site-header .token-badge');
    if (!host) return;
    var picker = document.createElement('div');
    picker.className = 'model-picker';
    var trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'model-picker-trigger';
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', 'model-selection-panel');
    trigger.title = 'Choose AI provider and model';
    trigger.textContent = 'Loading models';
    var controls = document.createElement('div');
    controls.id = 'model-selection-panel';
    controls.className = 'model-selection';
    controls.hidden = true;
    controls.innerHTML = '<label>Provider<select aria-label="AI provider" disabled></select></label>' +
      '<label>Model<select aria-label="AI model" disabled></select></label>' +
      '<span role="status" aria-live="polite">Loading models...</span>';
    picker.append(trigger, controls);
    var usage = document.createElement('div');
    usage.className = 'header-usage';
    host.before(usage);
    usage.append(picker, host);
    var providerSelect = controls.querySelector('[aria-label="AI provider"]');
    var modelSelect = controls.querySelector('[aria-label="AI model"]');
    var status = controls.querySelector('[role="status"]');

    function closePicker() {
      controls.hidden = true;
      trigger.setAttribute('aria-expanded', 'false');
    }

    function updateTrigger() {
      var name = selectedModel ? selectedModel.split(':').slice(1).join(':') : 'Choose model';
      trigger.textContent = name.replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2');
      trigger.title = selectedModel || 'Choose AI provider and model';
    }

    trigger.addEventListener('click', function () {
      controls.hidden = !controls.hidden;
      trigger.setAttribute('aria-expanded', String(!controls.hidden));
      if (!controls.hidden) providerSelect.focus();
    });
    document.addEventListener('click', function (event) {
      if (!picker.contains(event.target)) closePicker();
    });
    picker.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') { closePicker(); trigger.focus(); }
    });

    document.addEventListener('submit', function (event) {
      if (!ready || !selectedModel) {
        event.preventDefault();
        event.stopImmediatePropagation();
        controls.hidden = false;
        trigger.setAttribute('aria-expanded', 'true');
        status.textContent = ready ? 'No provider key configured.' : 'Model selection unavailable.';
      }
    }, true);

    fetch('/ai-models', { cache: 'no-store' }).then(function (response) {
      if (!response.ok) throw new Error('Could not load models. Reload to retry.');
      return response.json();
    }).then(function (catalog) {
      var providers = catalog.providers;
      var saved = null;
      try { saved = sessionStorage.getItem(storageKey); } catch (_) {}
      selectedModel = providers.some(function (provider) { return provider.models.includes(saved); })
        ? saved : catalog.default_model;
      providers.forEach(function (provider) {
        providerSelect.add(new Option(provider.label, provider.id));
      });
      var initial = providers.find(function (provider) { return provider.models.includes(selectedModel); });
      if (initial) providerSelect.value = initial.id;

      function populateModels() {
        var provider = providers.find(function (item) { return item.id === providerSelect.value; });
        modelSelect.replaceChildren();
        if (!provider) return;
        provider.models.forEach(function (model) {
          modelSelect.add(new Option(model.split(':').slice(1).join(':'), model));
        });
        if (provider.models.includes(selectedModel)) modelSelect.value = selectedModel;
      }

      function saveSelection() {
        selectedModel = modelSelect.value || null;
        try {
          if (selectedModel) sessionStorage.setItem(storageKey, selectedModel);
          else sessionStorage.removeItem(storageKey);
        } catch (_) {}
        status.textContent = '';
        updateTrigger();
      }

      providerSelect.addEventListener('change', function () { populateModels(); saveSelection(); });
      modelSelect.addEventListener('change', saveSelection);
      populateModels();
      ready = true;
      updateTrigger();
      status.textContent = '';
      providerSelect.disabled = modelSelect.disabled = providers.length === 0;
      if (!providers.length) {
        status.textContent = 'No provider key configured.';
      } else if (!selectedModel) {
        modelSelect.prepend(new Option('Select a model', '', true, true));
        status.textContent = 'Choose a model to continue.';
      }
    }).catch(function () {
      trigger.textContent = 'Models unavailable';
      status.textContent = 'Could not load models. Reload to retry.';
    });
  });
})();