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
      $$(".env-card").forEach(function (c) { c.classList.remove("selected"); });
      card.classList.add("selected");
      selectedEnv = card.dataset.env;
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
      btn.innerHTML = c.name + '<span class="env-type">' + c.env_type + "</span>";
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

      Object.keys(sectionData).forEach(function (key) {
        var spec = sectionData[key];
        var prefillVal = prefillEnv && prefillEnv[section] ? prefillEnv[section][key] : null;
        var val = prefillVal != null ? prefillVal : spec.value;
        var inputEl = createInput("env." + section + "." + key, spec, val);
        group.appendChild(inputEl);
      });

      container.appendChild(group);
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
    var prefillTrain = (prefill || currentConfig).training || null;

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
    var prefillEval = (prefill || currentConfig).evaluation || null;
    Object.keys(evalFields).forEach(function (key) {
      var spec = evalFields[key];
      var prefillVal = prefillEval ? prefillEval[key] : null;
      var val = prefillVal != null ? prefillVal : spec.value;
      var inputEl = createInput("evaluation." + key, spec, val);
      evalGroup.appendChild(inputEl);
    });
    container.appendChild(evalGroup);
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
    currentConfig = {
      experiment_name: $("#cfg-experiment-name").value || "experiment",
      seed: _seed,
      environment: { type: selectedEnv, seed: _seed },
      training: {},
      evaluation: {},
      output: { base_dir: "outputs" },
    };

    $$(".cfg-input").forEach(function (input) {
      var path = input.dataset.path;
      if (!path) return;

      var val;
      if (input.type === "checkbox") {
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

  async function refreshRuns() {
    var runs = await api("GET", "/api/train/runs");
    var sel = $("#active-run-select");
    sel.innerHTML = "";
    if (runs.length === 0) {
      sel.innerHTML = '<option value="">No runs</option>';
      return;
    }
    runs.forEach(function (r) {
      var opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = r.run_id + " (" + r.experiment + ") - " + r.status;
      sel.appendChild(opt);
    });
    if (activeRunId) sel.value = activeRunId;
    else activeRunId = runs[runs.length - 1].run_id;
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
    if (activeRunId) startPolling();
  });

  async function pollStatus() {
    if (!activeRunId) return;
    var data = await api("GET", "/api/train/status/" + activeRunId);
    if (data.error) return;

    // Status badge
    var badge = $("#run-status-badge");
    badge.textContent = data.status;
    badge.className = "badge " + data.status;

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
  }

  async function applyTuning(key, value) {
    if (!activeRunId) { toast("No active run", "error"); return; }
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
      var cps = o.checkpoints.map(function (c) {
        var cls = c === "final_model.zip" ? "checkpoint-tag final" : "checkpoint-tag";
        return '<span class="' + cls + '">' + c + "</span>";
      }).join("");
      item.innerHTML =
        "<h3>" + o.experiment + " / " + o.seed + "</h3>" +
        '<div class="meta">' + o.path + "</div>" +
        '<div class="checkpoint-list">' + cps + "</div>";
      container.appendChild(item);
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
