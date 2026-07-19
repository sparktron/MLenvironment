/* ================================================================
   RL Experiment Framework – GUI Frontend
   ================================================================ */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  let schema = {};                // env-type -> field schema
  let selectedEnv = null;         // 'walker_bullet' | 'organism_arena_parallel'
  let currentConfig = {};         // assembled config object
  let activeRunId = null;         // run_id being monitored
  let activeRunStatus = null;     // status for selected dashboard run
  let pollTimer = null;           // setInterval id for dashboard polling
  let rewardHistory = [];         // [{x: timestep, y: reward}, ...]
  let frameData = [];             // all captured frames (client-side buffer)
  let nextFrameSince = 0;         // high-water mark: fetch only frames >= this index
  let isPlaying = false;          // playback state
  let currentFrameIndex = 0;      // current frame being displayed
  let playbackSpeed = 1.0;        // 0.5x, 1x, 2x, 4x
  let frameUpdateTimer = null;    // timer for frame playback
  let lastFrameDisplayTime = 0;   // for frame timing
  let framePollTimer = null;      // timer for polling frames
  let frameDisplayGen = 0;        // generation counter; onload ignores stale loads

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function $(sel) { return document.querySelector(sel); }
  function $$(sel) { return document.querySelectorAll(sel); }

  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    try {
      const r = await fetch(path, opts);
      return r.json();
    } catch (err) {
      console.error("API error:", method, path, err);
      return { error: String(err) };
    }
  }

  function toast(msg, type) {
    type = type || "info";
    const el = document.createElement("div");
    el.className = "toast " + type;
    el.textContent = msg;
    $("#toast-container").appendChild(el);
    setTimeout(function () { el.remove(); }, 4000);
  }

  function fmt(v) {
    if (v === null || v === undefined || v === "--") return "--";
    var n = Number(v);
    if (isNaN(n)) return String(v);
    if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(3);
    if (Math.abs(n) >= 1000) return n.toFixed(0);
    return n.toFixed(4);
  }

  // ------------------------------------------------------------------
  // Tab navigation
  // ------------------------------------------------------------------
  $$(".nav-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      $$(".nav-btn").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      $$(".tab-content").forEach(function (t) { t.classList.remove("active"); });
      $("#tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "dashboard") refreshRuns();
      if (btn.dataset.tab === "outputs") refreshOutputs();
      if (btn.dataset.tab === "analysis") refreshAnalysis();
    });
  });

  // ------------------------------------------------------------------
  // Wizard step navigation
  // ------------------------------------------------------------------
  function goToStep(n) {
    $$(".wizard-panel").forEach(function (p) { p.classList.remove("active"); });
    $("#wiz-step-" + n).classList.add("active");
    $$(".wizard-steps .step").forEach(function (s) {
      var sn = parseInt(s.dataset.step);
      s.classList.remove("active", "done");
      if (sn < n) s.classList.add("done");
      if (sn === n) s.classList.add("active");
    });
  }

  // Step 1 -> 2
  $("#wiz-next-1").addEventListener("click", function () {
    if (!selectedEnv) return;
    goToStep(2);
    renderEnvParams();
  });
  // Step 2 -> 3
  $("#wiz-next-2").addEventListener("click", function () {
    goToStep(3);
    renderTrainingParams();
  });
  // Step 3 -> 4
  $("#wiz-next-3").addEventListener("click", function () {
    assembleConfig();
    $("#config-preview").textContent = JSON.stringify(currentConfig, null, 2);
    goToStep(4);
  });
  // Back buttons
  $("#wiz-prev-2").addEventListener("click", function () { goToStep(1); });
  $("#wiz-prev-3").addEventListener("click", function () { goToStep(2); });
  $("#wiz-prev-4").addEventListener("click", function () { goToStep(3); });

  // ------------------------------------------------------------------
  // Step 1: Environment selection
  // ------------------------------------------------------------------
  $$(".env-card").forEach(function (card) {
    card.addEventListener("click", function () {
      var nextEnv = card.dataset.env;
      var resetVisibleDefaults = false;
      if (selectedEnv && selectedEnv !== nextEnv) {
        currentConfig = {};
        resetVisibleDefaults = true;
      } else if (currentConfig.environment && currentConfig.environment.type !== nextEnv) {
        currentConfig = {};
        resetVisibleDefaults = true;
      }
      $$(".env-card").forEach(function (c) { c.classList.remove("selected"); });
      card.classList.add("selected");
      selectedEnv = nextEnv;
      if (resetVisibleDefaults) {
        $("#cfg-experiment-name").value = selectedEnv === "walker_bullet" ? "my_walker" : "my_arena";
        $("#cfg-seed").value = 42;
      }
      $("#wiz-next-1").disabled = false;
    });
  });

  // Load templates
  async function loadTemplates() {
    var configs = await api("GET", "/api/configs");
    var container = $("#template-list");
    container.innerHTML = "";
    configs.forEach(function (c) {
      var btn = document.createElement("button");
      btn.className = "template-btn";
      var nameText = document.createTextNode(c.name);
      var envTypeSpan = document.createElement("span");
      envTypeSpan.className = "env-type";
      envTypeSpan.textContent = c.env_type;
      btn.appendChild(nameText);
      btn.appendChild(envTypeSpan);
      btn.addEventListener("click", async function () {
        var full = await api("GET", "/api/configs/" + c.name);
        selectedEnv = full.environment.type;
        currentConfig = full;
        $$(".env-card").forEach(function (card) {
          card.classList.toggle("selected", card.dataset.env === selectedEnv);
        });
        $("#wiz-next-1").disabled = false;
        // Pre-fill and jump to step 2
        goToStep(2);
        renderEnvParams(full);
        toast("Loaded template: " + c.name, "success");
      });
      container.appendChild(btn);
    });
  }

  // ------------------------------------------------------------------
  // Step 2: Render environment parameter form
  // ------------------------------------------------------------------
  function renderEnvParams(prefill) {
    var container = $("#env-params-container");
    container.innerHTML = "";
    var envSchema = schema[selectedEnv];
    if (!envSchema) return;

    var envFields = envSchema.environment;
    var prefillEnv = prefill ? prefill.environment : null;

    if (prefill) {
      $("#cfg-experiment-name").value = prefill.experiment_name || "";
      $("#cfg-seed").value = prefill.seed != null ? prefill.seed : 42;
    } else {
      if (!$("#cfg-experiment-name").value) {
        $("#cfg-experiment-name").value = selectedEnv === "walker_bullet" ? "my_walker" : "my_arena";
      }
    }

    Object.keys(envFields).forEach(function (section) {
      var sectionData = envFields[section];
      if (sectionData.type) return; // leaf field like 'type' or 'seed'

      var title = document.createElement("div");
      title.className = "form-group-title";
      title.textContent = section.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
      container.appendChild(title);

      var group = document.createElement("div");
      group.className = "form-group";
      populateGroup(
        "env." + section,
        sectionData,
        prefillEnv ? prefillEnv[section] : null,
        group
      );
      container.appendChild(group);
    });
  }

  // Recursively populate a form group. Schema entries with a `type` field are
  // leaves and get an <input>; entries without `type` are sub-groups and get
  // a subtitle + recursive render. This preserves nested groups like
  // `sim.control` when the wizard later assembles the config.
  function populateGroup(pathPrefix, groupSpec, prefillData, container) {
    Object.keys(groupSpec).forEach(function (key) {
      var spec = groupSpec[key];
      var prefillVal = prefillData ? prefillData[key] : null;

      if (spec && spec.type) {
        // leaf
        var val = (prefillVal != null) ? prefillVal : spec.value;
        container.appendChild(createInput(pathPrefix + "." + key, spec, val));
      } else if (spec && typeof spec === "object") {
        // nested group — render sub-heading and recurse
        var sub = document.createElement("div");
        sub.className = "form-group-subtitle";
        sub.textContent = key.replace(/_/g, " ");
        container.appendChild(sub);
        populateGroup(pathPrefix + "." + key, spec, prefillVal, container);
      }
    });
  }

  // ------------------------------------------------------------------
  // Step 3: Render training parameter form
  // ------------------------------------------------------------------
  function renderTrainingParams(prefill) {
    var container = $("#training-params-container");
    container.innerHTML = "";
    var envSchema = schema[selectedEnv];
    if (!envSchema) return;

    var trainFields = envSchema.training;
    var selectedConfig =
      currentConfig.environment && currentConfig.environment.type === selectedEnv
        ? currentConfig
        : {};
    var prefillTrain = (prefill || selectedConfig).training || null;

    var title = document.createElement("div");
    title.className = "form-group-title";
    title.textContent = "PPO Hyperparameters";
    container.appendChild(title);

    var group = document.createElement("div");
    group.className = "form-group";

    Object.keys(trainFields).forEach(function (key) {
      var spec = trainFields[key];
      var prefillVal = prefillTrain ? prefillTrain[key] : null;
      var val = prefillVal != null ? prefillVal : spec.value;
      var inputEl = createInput("training." + key, spec, val);
      group.appendChild(inputEl);
    });

    container.appendChild(group);

    // Evaluation section
    var evalTitle = document.createElement("div");
    evalTitle.className = "form-group-title";
    evalTitle.textContent = "Evaluation";
    container.appendChild(evalTitle);

    var evalGroup = document.createElement("div");
    evalGroup.className = "form-group";
    var evalFields = envSchema.evaluation;
    var prefillEval = (prefill || selectedConfig).evaluation || null;
    Object.keys(evalFields).forEach(function (key) {
      var spec = evalFields[key];
      var prefillVal = prefillEval ? prefillEval[key] : null;
      var val = prefillVal != null ? prefillVal : spec.value;
      var inputEl = createInput("evaluation." + key, spec, val);
      evalGroup.appendChild(inputEl);
    });
    container.appendChild(evalGroup);

    // Arena-only extra groups (self_play / reward_annealing / curriculum).
    // These are top-level config keys (siblings of environment/training/
    // evaluation), not nested under environment — populateGroup's "top."
    // prefix (handled in assembleConfig) writes them there directly.
    if (envSchema.extra) {
      var prefillExtra = prefill || selectedConfig;
      Object.keys(envSchema.extra).forEach(function (groupKey) {
        var groupTitle = document.createElement("div");
        groupTitle.className = "form-group-title";
        groupTitle.textContent = groupKey
          .replace(/_/g, " ")
          .replace(/\b\w/g, function (c) { return c.toUpperCase(); });
        container.appendChild(groupTitle);

        var groupContainer = document.createElement("div");
        groupContainer.className = "form-group";
        populateGroup(
          "top." + groupKey,
          envSchema.extra[groupKey],
          prefillExtra ? prefillExtra[groupKey] : null,
          groupContainer
        );
        container.appendChild(groupContainer);
      });
    }
  }

  // ------------------------------------------------------------------
  // Generic input builder
  // ------------------------------------------------------------------
  function createInput(path, spec, value) {
    var label = document.createElement("label");
    var nameText = path.split(".").pop().replace(/_/g, " ");
    label.textContent = nameText;
    if (spec.desc) label.title = spec.desc;

    var input;
    if (spec.type === "choice") {
      input = document.createElement("select");
      spec.choices.forEach(function (ch) {
        var opt = document.createElement("option");
        opt.value = ch;
        opt.textContent = ch;
        if (ch === value) opt.selected = true;
        input.appendChild(opt);
      });
    } else if (spec.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!value;
    } else if (spec.type === "range") {
      // Two inputs for [lo, hi]
      var wrap = document.createElement("div");
      wrap.style.display = "flex";
      wrap.style.gap = "0.3rem";
      var lo = document.createElement("input");
      lo.type = "number";
      lo.step = "0.01";
      lo.value = Array.isArray(value) ? value[0] : spec.value[0];
      lo.dataset.path = path + ".0";
      lo.className = "cfg-input";
      var hi = document.createElement("input");
      hi.type = "number";
      hi.step = "0.01";
      hi.value = Array.isArray(value) ? value[1] : spec.value[1];
      hi.dataset.path = path + ".1";
      hi.className = "cfg-input";
      wrap.appendChild(lo);
      wrap.appendChild(hi);
      label.appendChild(wrap);
      return label;
    } else if (spec.type === "list_float") {
      input = document.createElement("input");
      input.type = "text";
      input.value = Array.isArray(value) ? value.join(", ") : String(value);
    } else if (spec.type === "text") {
      input = document.createElement("input");
      input.type = "text";
      input.value = value != null ? value : "";
    } else if (spec.type === "json") {
      // A dict-valued leaf (e.g. curriculum per-level thresholds/overrides)
      // rendered as raw JSON text — the wizard's flat-field form has no
      // generic editor for arbitrarily-shaped nested config.
      input = document.createElement("textarea");
      input.rows = 4;
      input.value = JSON.stringify(value != null ? value : spec.value, null, 2);
      input.dataset.jsonField = "1";
    } else if (spec.type === "int") {
      input = document.createElement("input");
      input.type = "number";
      input.step = "1";
      input.value = value;
      if (spec.min != null) input.min = spec.min;
      if (spec.max != null) input.max = spec.max;
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = value;
      if (spec.min != null) input.min = spec.min;
      if (spec.max != null) input.max = spec.max;
    }

    input.dataset.path = path;
    input.className = "cfg-input";
    label.appendChild(input);
    return label;
  }

  // ------------------------------------------------------------------
  // Assemble config from form inputs
  // ------------------------------------------------------------------
  function assembleConfig() {
    var _seedRaw = parseInt($("#cfg-seed").value);
    var _seed = isNaN(_seedRaw) ? 42 : _seedRaw;
    // Rebuild only the sections the wizard form actually renders
    // (experiment_name/seed/environment/training/evaluation/output, plus
    // self_play/reward_annealing/curriculum for the arena's "extra" schema
    // group when present), and mutate the existing currentConfig object in
    // place rather than replacing it wholesale. A loaded template can still
    // carry sections the wizard has no fields for at all (sweep, multi_seed,
    // reproducibility, ...); a full replacement here would silently drop
    // them, so a template's launch could fail validation or — via "Save
    // config as YAML template", checked by default — overwrite the source
    // YAML with a stripped copy. Selecting a different environment type
    // still clears currentConfig entirely (see the env-card click handler
    // above), which is the right point to drop template state that may not
    // apply to the new environment.
    if (!currentConfig || typeof currentConfig !== "object") currentConfig = {};
    currentConfig.experiment_name = $("#cfg-experiment-name").value || "experiment";
    currentConfig.seed = _seed;
    currentConfig.environment = { type: selectedEnv, seed: _seed };
    currentConfig.training = {};
    currentConfig.evaluation = {};
    if (!currentConfig.output || typeof currentConfig.output !== "object") {
      currentConfig.output = { base_dir: "outputs" };
    }

    $$(".cfg-input").forEach(function (input) {
      var path = input.dataset.path;
      if (!path) return;

      var val;
      if (input.dataset.jsonField) {
        try {
          val = JSON.parse(input.value);
        } catch (e) {
          toast("Invalid JSON in " + path + " — keeping previous value", "error");
          return;
        }
      } else if (input.type === "checkbox") {
        val = input.checked;
      } else if (input.type === "number") {
        val = input.step === "1" ? parseInt(input.value) : parseFloat(input.value);
      } else if (path.startsWith("env.") && input.type === "text") {
        // list_float
        val = input.value.split(",").map(function (s) { return parseFloat(s.trim()); });
      } else {
        val = input.value;
      }

      // Parse the path and set value
      var parts = path.split(".");

      if (parts[0] === "env") {
        // env.section.key or env.section.key.index
        var ref = currentConfig.environment;
        for (var i = 1; i < parts.length - 1; i++) {
          if (!ref[parts[i]]) ref[parts[i]] = {};
          ref = ref[parts[i]];
        }
        var lastKey = parts[parts.length - 1];
        // Handle range indices (0, 1)
        if (lastKey === "0" || lastKey === "1") {
          var rangeKey = parts[parts.length - 2];
          if (!ref[rangeKey]) ref[rangeKey] = [0, 0];
          // Navigate up one level
          var parentRef = currentConfig.environment;
          for (var j = 1; j < parts.length - 2; j++) {
            parentRef = parentRef[parts[j]];
          }
          if (!Array.isArray(parentRef[rangeKey])) parentRef[rangeKey] = [0, 0];
          parentRef[rangeKey][parseInt(lastKey)] = parseFloat(input.value);
        } else {
          ref[lastKey] = val;
        }
      } else if (parts[0] === "training") {
        currentConfig.training[parts[1]] = val;
      } else if (parts[0] === "evaluation") {
        currentConfig.evaluation[parts[1]] = val;
      } else if (parts[0] === "top") {
        // top.<group>.<key> — a top-level config section (self_play,
        // reward_annealing, curriculum), a sibling of environment/training/
        // evaluation rather than nested under any of them.
        var groupKey = parts[1];
        if (!currentConfig[groupKey] || typeof currentConfig[groupKey] !== "object") {
          currentConfig[groupKey] = {};
        }
        currentConfig[groupKey][parts[2]] = val;
      }
    });
  }

  // ------------------------------------------------------------------
  // Launch training
  // ------------------------------------------------------------------
  $("#launch-btn").addEventListener("click", async function () {
    assembleConfig();
    // Optionally save as template
    if ($("#save-config-check").checked) {
      await api("POST", "/api/configs", currentConfig);
    }
    var result = await api("POST", "/api/train/start", currentConfig);
    if (result.error) {
      toast(result.error, "error");
      return;
    }
    activeRunId = result.run_id;
    setDashboardRunState({ run_id: activeRunId, status: "running" });
    toast("Training started: " + activeRunId, "success");
    // Switch to dashboard
    $$(".nav-btn").forEach(function (b) { b.classList.remove("active"); });
    $(".nav-btn[data-tab='dashboard']").classList.add("active");
    $$(".tab-content").forEach(function (t) { t.classList.remove("active"); });
    $("#tab-dashboard").classList.add("active");
    rewardHistory = [];
    frameData = [];
    nextFrameSince = 0;
    currentFrameIndex = 0;
    frameDisplayGen = 0;
    stopFramePlayback();
    buildEnvTuningControls();
    setTuningEnabled(true);
    startPolling();
  });

  // ------------------------------------------------------------------
  // Dashboard polling
  // ------------------------------------------------------------------
  function startPolling() {
    stopPolling();
    refreshRuns();
    pollTimer = setInterval(pollStatus, 2000);
    framePollTimer = setInterval(pollFrames, 1000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    if (framePollTimer) { clearInterval(framePollTimer); framePollTimer = null; }
  }

  function setTuningEnabled(enabled) {
    $$(".tuning-row input, .tuning-row button").forEach(function (el) {
      el.disabled = !enabled;
    });
  }

  function resetMetrics() {
    [
      "#m-timesteps",
      "#m-reward",
      "#m-ep-len",
      "#m-loss",
      "#m-pg-loss",
      "#m-val-loss",
      "#m-entropy",
      "#m-lr",
    ].forEach(function (sel) {
      $(sel).textContent = "--";
    });
  }

  function resetFrameViewer() {
    frameData = [];
    nextFrameSince = 0;
    currentFrameIndex = 0;
    frameDisplayGen++;
    $("#frame-scrubber").value = 0;
    $("#frame-scrubber").max = 0;
    stopFramePlayback();
    updateFrameDisplay();
  }

  function setDashboardRunState(run) {
    activeRunId = run ? run.run_id : null;
    activeRunStatus = run ? run.status : null;

    var badge = $("#run-status-badge");
    badge.textContent = activeRunStatus || "--";
    badge.className = activeRunStatus ? "badge " + activeRunStatus : "badge";

    var isRunning = activeRunStatus === "running";
    $("#stop-run-btn").disabled = !isRunning;
    setTuningEnabled(isRunning);

    if (!run) {
      rewardHistory = [];
      resetMetrics();
      drawRewardChart();
      resetFrameViewer();
    }
  }

  async function refreshRuns() {
    var runs = await api("GET", "/api/train/runs");
    var sel = $("#active-run-select");
    sel.innerHTML = "";
    if (runs.length === 0) {
      sel.innerHTML = '<option value="">No runs</option>';
      setDashboardRunState(null);
      return;
    }
    runs.forEach(function (r) {
      var opt = document.createElement("option");
      opt.value = r.run_id;
      opt.dataset.status = r.status;
      opt.textContent = r.run_id + " (" + r.experiment + ") - " + r.status;
      sel.appendChild(opt);
    });
    var selectedRun = runs.find(function (r) { return r.run_id === activeRunId; }) || runs[runs.length - 1];
    sel.value = selectedRun.run_id;
    setDashboardRunState(selectedRun);
    // Do NOT call startPolling() here — startPolling already calls refreshRuns()
    // once at startup; calling it back creates an unbounded async loop.
  }

  $("#stop-run-btn").addEventListener("click", async function () {
    if (!activeRunId) { toast("No active run selected", "error"); return; }
    var result = await api("POST", "/api/train/stop/" + activeRunId);
    if (result.error) {
      toast("Stop failed: " + result.error, "error");
    } else {
      toast("Stop requested for " + activeRunId, "info");
    }
  });

  $("#active-run-select").addEventListener("change", function () {
    activeRunId = this.value;
    rewardHistory = [];
    frameData = [];
    nextFrameSince = 0;
    currentFrameIndex = 0;
    frameDisplayGen = 0;
    stopFramePlayback();
    if (activeRunId) {
      var opt = this.options[this.selectedIndex];
      setDashboardRunState({ run_id: activeRunId, status: opt ? opt.dataset.status : null });
      startPolling();
    } else {
      setDashboardRunState(null);
    }
  });

  async function pollStatus() {
    if (!activeRunId) return;
    var data = await api("GET", "/api/train/status/" + activeRunId);
    if (data.error) return;

    // Status badge
    var badge = $("#run-status-badge");
    badge.textContent = data.status;
    badge.className = "badge " + data.status;
    activeRunStatus = data.status;
    var isRunning = data.status === "running";
    $("#stop-run-btn").disabled = !isRunning;
    setTuningEnabled(isRunning);

    // Metrics
    var m = data.metrics || {};
    $("#m-timesteps").textContent = m.timesteps != null ? m.timesteps.toLocaleString() : "--";
    $("#m-reward").textContent = fmt(m["rollout/ep_rew_mean"]);
    $("#m-ep-len").textContent = fmt(m["rollout/ep_len_mean"]);
    $("#m-loss").textContent = fmt(m["train/loss"]);
    $("#m-pg-loss").textContent = fmt(m["train/policy_gradient_loss"]);
    $("#m-val-loss").textContent = fmt(m["train/value_loss"]);
    $("#m-entropy").textContent = fmt(m["train/entropy_loss"]);
    $("#m-lr").textContent = fmt(m["train/learning_rate"]);

    // Reward chart
    if (m.timesteps && m["rollout/ep_rew_mean"] != null) {
      var last = rewardHistory.length > 0 ? rewardHistory[rewardHistory.length - 1] : null;
      if (!last || last.x !== m.timesteps) {
        rewardHistory.push({ x: m.timesteps, y: m["rollout/ep_rew_mean"] });
        drawRewardChart();
      }
    }

    // Stop polling when done; flush final frames first so nothing is lost.
    if (data.status === "completed" || data.status === "failed") {
      await pollFrames();
      stopPolling();
      if (data.status === "completed") toast("Training completed! Model: " + data.model_path, "success");
      if (data.status === "failed") toast("Training failed: " + (data.error || "").slice(0, 100), "error");
    }
  }

  // ------------------------------------------------------------------
  // Reward chart (canvas-based, no dependencies)
  // ------------------------------------------------------------------
  function drawRewardChart() {
    var canvas = $("#reward-chart");
    var ctx = canvas.getContext("2d");
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 200 * dpr;
    ctx.scale(dpr, dpr);
    var W = rect.width;
    var H = 200;
    var pad = { top: 10, right: 15, bottom: 30, left: 55 };

    ctx.clearRect(0, 0, W, H);

    if (rewardHistory.length < 2) {
      ctx.fillStyle = "#8b8fa3";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for data...", W / 2, H / 2);
      return;
    }

    var xs = rewardHistory.map(function (p) { return p.x; });
    var ys = rewardHistory.map(function (p) { return p.y; });
    var xMin = Math.min.apply(null, xs), xMax = Math.max.apply(null, xs);
    var yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    var xRange = xMax - xMin || 1;
    var yRange = yMax - yMin;

    function tx(v) { return pad.left + (v - xMin) / xRange * (W - pad.left - pad.right); }
    function ty(v) { return pad.top + (1 - (v - yMin) / yRange) * (H - pad.top - pad.bottom); }

    // Grid lines
    ctx.strokeStyle = "#333746";
    ctx.lineWidth = 0.5;
    for (var i = 0; i <= 4; i++) {
      var yv = yMin + (yRange * i) / 4;
      var yy = ty(yv);
      ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(W - pad.right, yy); ctx.stroke();
      ctx.fillStyle = "#8b8fa3"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
      ctx.fillText(yv.toFixed(1), pad.left - 5, yy + 3);
    }

    // X-axis labels
    ctx.fillStyle = "#8b8fa3"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(xMin.toLocaleString(), tx(xMin), H - 5);
    ctx.fillText(xMax.toLocaleString(), tx(xMax), H - 5);
    ctx.fillText("timesteps", W / 2, H - 5);

    // Line
    ctx.strokeStyle = "#6c8cff";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    rewardHistory.forEach(function (p, idx) {
      if (idx === 0) ctx.moveTo(tx(p.x), ty(p.y));
      else ctx.lineTo(tx(p.x), ty(p.y));
    });
    ctx.stroke();

    // Fill under line
    ctx.lineTo(tx(rewardHistory[rewardHistory.length - 1].x), ty(yMin));
    ctx.lineTo(tx(rewardHistory[0].x), ty(yMin));
    ctx.closePath();
    ctx.fillStyle = "rgba(108,140,255,0.08)";
    ctx.fill();

    // Dots on last point
    var last = rewardHistory[rewardHistory.length - 1];
    ctx.beginPath();
    ctx.arc(tx(last.x), ty(last.y), 4, 0, Math.PI * 2);
    ctx.fillStyle = "#6c8cff";
    ctx.fill();
  }

  // ------------------------------------------------------------------
  // Live Tuning
  // ------------------------------------------------------------------
  $$(".tuning-row .btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var key = btn.dataset.tune;
      var inputId = btn.dataset.input;
      var val = parseFloat($("#" + inputId).value);
      if (isNaN(val)) { toast("Invalid value", "error"); return; }
      applyTuning(key, val);
    });
  });

  function buildEnvTuningControls() {
    var container = $("#env-tuning-params");
    container.innerHTML = "";
    if (!selectedEnv || !schema[selectedEnv]) return;

    var envSchema = schema[selectedEnv].environment;
    var tunableSections = ["reward", "termination", "domain_randomization", "battle_rules", "morphology"];

    tunableSections.forEach(function (section) {
      if (!envSchema[section]) return;
      var sectionTitle = document.createElement("div");
      sectionTitle.className = "form-group-title";
      sectionTitle.style.marginTop = "0.75rem";
      sectionTitle.textContent = section.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
      container.appendChild(sectionTitle);

      var grid = document.createElement("div");
      grid.className = "tuning-grid";

      Object.keys(envSchema[section]).forEach(function (key) {
        var spec = envSchema[section][key];
        if (spec.type === "range" || spec.type === "list_float") return; // skip complex types for live tuning
        var row = document.createElement("div");
        row.className = "tuning-row";
        var label = document.createElement("label");
        label.textContent = key.replace(/_/g, " ");
        label.title = spec.desc || "";
        var input = document.createElement("input");
        input.type = "number";
        input.step = spec.type === "int" ? "1" : "any";
        input.value = spec.value;
        input.id = "tune-" + section + "-" + key;
        var btn = document.createElement("button");
        btn.className = "btn small";
        btn.textContent = "Apply";
        btn.addEventListener("click", function () {
          var v = parseFloat(input.value);
          if (isNaN(v)) { toast("Invalid value", "error"); return; }
          applyTuning(section + "." + key, v);
        });
        row.appendChild(label);
        row.appendChild(input);
        row.appendChild(btn);
        grid.appendChild(row);
      });

      container.appendChild(grid);
    });
    setTuningEnabled(activeRunStatus === "running");
  }

  async function applyTuning(key, value) {
    if (!activeRunId) { toast("No active run", "error"); return; }
    if (activeRunStatus !== "running") { toast("Run is not active", "error"); return; }
    var params = {};
    params[key] = value;
    var result = await api("POST", "/api/train/tune/" + activeRunId, params);
    if (result.error) {
      toast("Tuning failed: " + result.error, "error");
      return;
    }
    toast("Applied: " + key + " = " + value, "success");
    var log = $("#tuning-log-entries");
    var entry = document.createElement("div");
    entry.className = "log-entry";
    entry.textContent = new Date().toLocaleTimeString() + "  " + key + " = " + value;
    log.prepend(entry);
  }

  // ------------------------------------------------------------------
  // Outputs tab
  // ------------------------------------------------------------------
  function formatAge(seconds) {
    if (seconds < 60) return Math.round(seconds) + "s";
    if (seconds < 3600) return Math.round(seconds / 60) + "m";
    if (seconds < 86400) return Math.round(seconds / 3600) + "h";
    return Math.round(seconds / 86400) + "d";
  }

  function appendLeaguePanel(item, o) {
    if (!o.league_size || o.league_size <= 0) return;
    var panel = document.createElement("div");
    panel.className = "league-panel";

    var badge = document.createElement("button");
    badge.className = "league-badge";
    badge.textContent =
      "⚔ League: " + o.league_size +
      (o.league_size === 1 ? " snapshot" : " snapshots") + " ▾";

    var detail = document.createElement("div");
    detail.className = "league-detail";
    detail.style.display = "none";

    var loaded = false;
    badge.addEventListener("click", async function () {
      var open = detail.style.display === "none";
      detail.style.display = open ? "block" : "none";
      if (!open || loaded) return;
      loaded = true;
      var data = await api("GET", "/api/league?path=" + encodeURIComponent(o.path));
      detail.innerHTML = "";
      if (!data.snapshots || data.snapshots.length === 0) {
        detail.innerHTML = '<p class="hint">No league snapshots.</p>';
        return;
      }
      // Newest first.
      data.snapshots.slice().reverse().forEach(function (s) {
        var row = document.createElement("div");
        row.className = "league-row";
        var size = (s.size_bytes / 1024).toFixed(0) + " KB";
        var warn = s.has_vecnorm ? "" : "  ⚠ no vecnorm";
        row.textContent =
          "t=" + s.timesteps + "  ·  " + size +
          "  ·  " + formatAge(s.age_seconds) + " ago" + warn;
        detail.appendChild(row);
      });
    });

    panel.appendChild(badge);
    panel.appendChild(detail);
    item.appendChild(panel);
  }

  async function refreshOutputs() {
    var outputs = await api("GET", "/api/outputs");
    var container = $("#outputs-list");
    if (outputs.length === 0) {
      container.innerHTML = '<p class="hint">No experiment outputs found. Run a training first.</p>';
      return;
    }
    container.innerHTML = "";
    outputs.forEach(function (o) {
      var item = document.createElement("div");
      item.className = "output-item";

      var h3 = document.createElement("h3");
      h3.textContent = o.run_id
        ? o.experiment + " / " + o.run_id + " / " + o.seed
        : o.experiment + " / " + o.seed;

      var metaDiv = document.createElement("div");
      metaDiv.className = "meta";
      metaDiv.textContent = o.path;

      var cpListDiv = document.createElement("div");
      cpListDiv.className = "checkpoint-list";
      o.checkpoints.forEach(function (c) {
        var span = document.createElement("span");
        span.className = c === "final_model.zip" ? "checkpoint-tag final" : "checkpoint-tag";
        span.textContent = c;
        cpListDiv.appendChild(span);
      });

      item.appendChild(h3);
      item.appendChild(metaDiv);
      item.appendChild(cpListDiv);
      appendLeaguePanel(item, o);
      container.appendChild(item);
    });
  }

  // ------------------------------------------------------------------
  // Analysis tab
  // ------------------------------------------------------------------
  function describeAnalysisJob(job) {
    if (job.status === "running") return "";
    if (job.error) return job.error;
    if (!job.result) return "";
    if (job.kind === "replay") return "Replay: " + job.result.saved_replay;
    var standings = (job.result.standings || []).map(function (s) {
      return s.rank + ". " + s.competitor + " (" + s.elo + ")";
    });
    return standings.join("  ·  ");
  }

  async function pollAnalysisJob(jobId, resultEl) {
    var job = await api("GET", "/api/analysis/jobs/" + jobId);
    if (job.status === "running") {
      resultEl.textContent = "Analysis job running...";
      setTimeout(function () { pollAnalysisJob(jobId, resultEl); }, 1000);
      return;
    }
    resultEl.textContent = job.error
      ? "Analysis failed: " + job.error
      : describeAnalysisJob(job) || job.status;
    renderRecentAnalysisJobs();
  }

  async function renderRecentAnalysisJobs() {
    var jobs = await api("GET", "/api/analysis/jobs");
    var container = $("#analysis-jobs");
    container.innerHTML = "";
    if (!Array.isArray(jobs) || jobs.length === 0) return;
    var heading = document.createElement("h3");
    heading.textContent = "Recent analysis jobs";
    container.appendChild(heading);
    jobs.forEach(function (job) {
      var item = document.createElement("div");
      item.className = "meta";
      var pieces = [job.kind, job.status];
      if (job.params && job.params.path) pieces.push(job.params.path);
      var detail = describeAnalysisJob(job);
      if (detail) pieces.push(detail);
      item.textContent = pieces.join(" · ");
      container.appendChild(item);
    });
  }

  async function refreshAnalysis() {
    renderRecentAnalysisJobs();
    var runs = await api("GET", "/api/analysis/runs");
    var container = $("#analysis-list");
    if (!Array.isArray(runs) || runs.length === 0) {
      $("#analysis-comparison").innerHTML = "";
      container.innerHTML = '<p class="hint">No registry runs yet. New training runs appear here automatically.</p>';
      return;
    }
    var comparison = $("#analysis-comparison");
    var table = document.createElement("table");
    table.className = "comparison-table";
    table.innerHTML = "<thead><tr><th>Experiment</th><th>Run</th><th>Status</th><th>Algorithm</th><th>Latest reward</th><th>Latest length</th></tr></thead>";
    var body = document.createElement("tbody");
    runs.forEach(function (run) {
      var row = document.createElement("tr");
      var metrics = run.metrics || {};
      [run.experiment_name, run.run_id, run.status, run.algorithm,
        fmt(metrics["rollout/ep_rew_mean"]), fmt(metrics["rollout/ep_len_mean"])
      ].forEach(function (value) { var cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell); });
      body.appendChild(row);
    });
    table.appendChild(body); comparison.innerHTML = ""; comparison.appendChild(table);
    container.innerHTML = "";
    runs.forEach(function (run) {
      var item = document.createElement("div");
      item.className = "output-item";
      var title = document.createElement("h3");
      title.textContent = run.experiment_name + " / " + run.run_id;
      var meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = run.status + " · " + run.algorithm + " · seed " + run.seed;
      var artifact = document.createElement("div");
      artifact.className = "checkpoint-list";
      (run.artifacts || []).forEach(function (a) {
        var tag = document.createElement("span");
        tag.className = "checkpoint-tag" + (a.path.indexOf("best_model") >= 0 ? " final" : "");
        tag.textContent = a.kind + ": " + a.path.split("/").pop();
        artifact.appendChild(tag);
      });
      var controls = document.createElement("div");
      controls.className = "analysis-controls";
      var result = document.createElement("div");
      result.className = "meta";
      var relPath = run.run_dir.replace(/\\/g, "/").split("outputs/").pop();
      var replay = document.createElement("button");
      replay.className = "btn secondary";
      replay.textContent = "Replay best";
      replay.addEventListener("click", async function () {
        var response = await api("POST", "/api/analysis/replay", { path: relPath });
        if (response.error) { result.textContent = response.error; return; }
        pollAnalysisJob(response.job_id, result);
      });
      controls.appendChild(replay);
      if (run.environment_type === "organism_arena_parallel") {
        var league = document.createElement("button");
        league.className = "btn secondary";
        league.textContent = "Rate league";
        league.addEventListener("click", async function () {
          var response = await api("POST", "/api/analysis/league-ratings", { path: relPath, episodes: 10 });
          if (response.error) { result.textContent = response.error; return; }
          pollAnalysisJob(response.job_id, result);
        });
        controls.appendChild(league);
      }
      item.appendChild(title); item.appendChild(meta); item.appendChild(artifact);
      item.appendChild(controls); item.appendChild(result); container.appendChild(item);
    });
  }

  // ------------------------------------------------------------------
  // Frame Playback
  // ------------------------------------------------------------------
  async function pollFrames() {
    if (!activeRunId) return;
    var result = await api("GET", "/api/train/frames/" + activeRunId + "?since=" + nextFrameSince);
    if (result.error || !result.frames || result.frames.length === 0) return;

    var wasEmpty = frameData.length === 0;

    // Append only new frames; update high-water mark.
    result.frames.forEach(function (f) { frameData.push(f); });
    nextFrameSince = frameData[frameData.length - 1].frame_index + 1;
    $("#frame-scrubber").max = frameData.length - 1;

    // Show the very first frame that arrives so the canvas isn't blank,
    // but only when the user hasn't manually positioned the scrubber yet.
    // Never disturb the scrubber position once playback or seeking has begun.
    if (wasEmpty) {
      updateFrameDisplay();
    }
  }

  function updateFrameDisplay() {
    if (frameData.length === 0) {
      $("#frame-indicator").textContent = "0 / 0";
      var canvas = $("#frame-canvas");
      var ctx = canvas.getContext("2d");
      ctx.fillStyle = "#333";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }

    currentFrameIndex = Math.max(0, Math.min(currentFrameIndex, frameData.length - 1));
    var frame = frameData[currentFrameIndex];
    if (!frame || !frame.image_base64) return;

    var canvas = $("#frame-canvas");
    var ctx = canvas.getContext("2d");
    var gen = ++frameDisplayGen;
    var img = new Image();
    img.onload = function () {
      // Discard if a newer paint request has already been issued.
      if (gen !== frameDisplayGen) return;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.onerror = function () {
      console.error("Failed to load frame image");
    };
    img.src = "data:image/jpeg;base64," + frame.image_base64;

    $("#frame-scrubber").value = currentFrameIndex;
    $("#frame-indicator").textContent = (currentFrameIndex + 1) + " / " + frameData.length;
  }

  function startFramePlayback() {
    stopFramePlayback();
    isPlaying = true;
    $("#play-pause-btn").textContent = "⏸ Pause";
    lastFrameDisplayTime = Date.now();

    frameUpdateTimer = setInterval(function () {
      if (!isPlaying || frameData.length === 0) return;

      var now = Date.now();
      var elapsed = now - lastFrameDisplayTime;
      var frameIntervalMs = 1000 / 30; // Assume 30 FPS for display
      var scaledInterval = frameIntervalMs / playbackSpeed;

      if (elapsed >= scaledInterval) {
        currentFrameIndex++;
        if (currentFrameIndex >= frameData.length) {
          currentFrameIndex = frameData.length - 1;
          updateFrameDisplay();
          stopFramePlayback();
        } else {
          updateFrameDisplay();
        }
        lastFrameDisplayTime = now;
      }
    }, 16); // ~60 FPS update rate
  }

  function stopFramePlayback() {
    isPlaying = false;
    $("#play-pause-btn").textContent = "▶ Play";
    if (frameUpdateTimer) { clearInterval(frameUpdateTimer); frameUpdateTimer = null; }
  }

  $("#play-pause-btn").addEventListener("click", function () {
    if (frameData.length === 0) {
      toast("No frames yet. Training will capture them.", "info");
      return;
    }
    if (isPlaying) {
      stopFramePlayback();
    } else {
      if (currentFrameIndex >= frameData.length - 1) {
        currentFrameIndex = 0;
      }
      startFramePlayback();
    }
  });

  $("#frame-scrubber").addEventListener("input", function () {
    stopFramePlayback();
    currentFrameIndex = parseInt(this.value);
    updateFrameDisplay();
  });

  $("#playback-speed").addEventListener("change", function () {
    playbackSpeed = parseFloat(this.value);
  });

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  async function init() {
    schema = await api("GET", "/api/schema");
    await loadTemplates();
  }

  init();

})();
