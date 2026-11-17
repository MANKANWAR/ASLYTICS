/**
 * ASLytics Accessible Group Conversation Client
 * Connects to the local NVIDIA Parakeet On-Device Pipeline API.
 */

const state = {
  scenarioData: null,
  displayedTurnCount: 0,
  showAslGloss: true,
  showWordTimings: true,
  activeSpeakerId: null,
};

const els = {
  visualAlertBanner: document.getElementById("visualAlertBanner"),
  visualAlertText: document.getElementById("visualAlertText"),
  dismissAlertBtn: document.getElementById("dismissAlertBtn"),
  engineBackendLabel: document.getElementById("engineBackendLabel"),
  fontScaleSelect: document.getElementById("fontScaleSelect"),
  contrastToggleBtn: document.getElementById("contrastToggleBtn"),
  activeDirectionPill: document.getElementById("activeDirectionPill"),
  radarBeamLayer: document.getElementById("radarBeamLayer"),
  radarSpeakersLayer: document.getElementById("radarSpeakersLayer"),
  speakerLegendList: document.getElementById("speakerLegendList"),
  modelSelect: document.getElementById("modelSelect"),
  targetLangSelect: document.getElementById("targetLangSelect"),
  scenarioSelect: document.getElementById("scenarioSelect"),
  watchwordsInput: document.getElementById("watchwordsInput"),
  toggleAslGloss: document.getElementById("toggleAslGloss"),
  toggleWordTimings: document.getElementById("toggleWordTimings"),
  scenarioTitleHeading: document.getElementById("scenarioTitleHeading"),
  scenarioDescText: document.getElementById("scenarioDescText"),
  stepTurnBtn: document.getElementById("stepTurnBtn"),
  playAllBtn: document.getElementById("playAllBtn"),
  transcriptStream: document.getElementById("transcriptStream"),
  addTurnForm: document.getElementById("addTurnForm"),
  customSpeakerName: document.getElementById("customSpeakerName"),
  customSpeakerAngle: document.getElementById("customSpeakerAngle"),
  customSpeakerLang: document.getElementById("customSpeakerLang"),
  customUtteranceText: document.getElementById("customUtteranceText"),
  micCaptureBtn: document.getElementById("micCaptureBtn"),
  dhhQuickReplyInput: document.getElementById("dhhQuickReplyInput"),
  speakToGroupBtn: document.getElementById("speakToGroupBtn"),
};

function polarToCartesian(cx, cy, radius, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180.0;
  return {
    x: cx + radius * Math.cos(rad),
    y: cy + radius * Math.sin(rad),
  };
}

function renderRadar(speakers, activeTurn = null) {
  els.radarBeamLayer.innerHTML = "";
  els.radarSpeakersLayer.innerHTML = "";
  els.speakerLegendList.innerHTML = "";

  if (activeTurn) {
    const spk = activeTurn.speaker;
    const target = polarToCartesian(170, 170, 122, spk.azimuth_deg);
    const beam = document.createElementNS("http://www.w3.org/2000/svg", "line");
    beam.setAttribute("x1", "170");
    beam.setAttribute("y1", "170");
    beam.setAttribute("x2", String(target.x));
    beam.setAttribute("y2", String(target.y));
    beam.setAttribute("stroke", spk.color_hex);
    beam.setAttribute("stroke-width", "4");
    beam.setAttribute("stroke-linecap", "round");
    els.radarBeamLayer.appendChild(beam);

    const shiftNote =
      activeTurn.diarization.spatial_shift_deg > 25
        ? ` (Turn head ${activeTurn.diarization.spatial_shift_deg}°)`
        : "";
    els.activeDirectionPill.textContent = `Look: ${spk.compass_direction}${shiftNote}`;
    els.activeDirectionPill.style.borderColor = spk.color_hex;
    els.activeDirectionPill.style.color = spk.color_hex;
  }

  speakers.forEach((spk) => {
    const pos = polarToCartesian(170, 170, 118, spk.azimuth_deg);
    const isCurrent = activeTurn && activeTurn.speaker.speaker_id === spk.speaker_id;

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    if (isCurrent) {
      const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      halo.setAttribute("cx", String(pos.x));
      halo.setAttribute("cy", String(pos.y));
      halo.setAttribute("r", "24");
      halo.setAttribute("fill", spk.badge_bg);
      halo.setAttribute("stroke", spk.color_hex);
      halo.setAttribute("stroke-width", "2");
      g.appendChild(halo);
    }

    const node = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    node.setAttribute("cx", String(pos.x));
    node.setAttribute("cy", String(pos.y));
    node.setAttribute("r", isCurrent ? "16" : "13");
    node.setAttribute("fill", "#0f172a");
    node.setAttribute("stroke", spk.color_hex);
    node.setAttribute("stroke-width", isCurrent ? "3.5" : "2");
    g.appendChild(node);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(pos.x));
    label.setAttribute("y", String(pos.y + 4));
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", "#f8fafc");
    label.setAttribute("font-size", "10");
    label.setAttribute("font-weight", "800");
    label.textContent = spk.display_name.slice(0, 3).toUpperCase();
    g.appendChild(label);

    els.radarSpeakersLayer.appendChild(g);

    // Legend item
    const item = document.createElement("div");
    item.className = "speaker-legend-item";
    item.style.borderLeftColor = spk.color_hex;
    item.innerHTML = `
      <div>
        <strong>${spk.display_name}</strong>
        <span style="color: var(--text-muted); font-size: 0.76rem;"> • ${spk.compass_direction}</span>
      </div>
      <span class="meta-pill">${spk.turn_count} turns (${spk.share_pct || 0}%)</span>
    `;
    els.speakerLegendList.appendChild(item);
  });
}

function triggerVisualAlert(turn) {
  const acc = turn.accessibility;
  const diar = turn.diarization;
  if (acc.direct_address_to_user) {
    els.visualAlertText.textContent = `🔔 Watchword Alert (${acc.triggered_watchwords.join(", ").toUpperCase()}): ${turn.speaker.display_name} at ${turn.speaker.compass_direction} is addressing you!`;
    els.visualAlertBanner.classList.remove("hidden");
  } else if (diar.is_overlapping) {
    els.visualAlertText.textContent = `⚠️ Overlapping Speech Detected: ${turn.speaker.display_name} overlapped with ${diar.overlapping_with} — separated cleanly by Sortformer.`;
    els.visualAlertBanner.classList.remove("hidden");
  }
  if (acc.haptic_pattern_ms && acc.haptic_pattern_ms.length && navigator.vibrate) {
    navigator.vibrate(acc.haptic_pattern_ms);
  }
}

function createTurnCard(turn) {
  const spk = turn.speaker;
  const diar = turn.diarization;
  const tx = turn.transcript;
  const ml = turn.multilingual;
  const asl = turn.asl_gloss;
  const acc = turn.accessibility;

  const card = document.createElement("article");
  card.className = `turn-card${acc.direct_address_to_user ? " direct-mention" : ""}`;
  card.style.borderLeftColor = spk.color_hex;

  const badges = [];
  badges.push(`<span class="meta-pill">📍 ${spk.compass_direction}</span>`);
  badges.push(`<span class="meta-pill">🌐 ${ml.source_lang_name} (${ml.source_lang.toUpperCase()})</span>`);
  badges.push(`<span class="meta-pill">${acc.tone_icon} ${acc.tone_label}</span>`);
  badges.push(`<span class="meta-pill">⏱ ${acc.pacing_status}</span>`);

  if (diar.is_overlapping) {
    badges.push(`<span class="meta-pill overlap-pill">⚠️ OVERLAP w/ ${diar.overlapping_with}</span>`);
  }
  if (acc.triggered_watchwords.length > 0) {
    badges.push(`<span class="meta-pill watchword-pill">🔔 MENTION: ${acc.triggered_watchwords.join(", ").toUpperCase()}</span>`);
  }
  acc.acoustic_events.forEach((ev) => {
    if (ev !== "OVERLAPPING SPEECH") {
      badges.push(`<span class="meta-pill">🔊 [${ev}]</span>`);
    }
  });

  const primaryText = ml.is_translated ? ml.translated_text : tx.original_text;
  const originalSub = ml.is_translated
    ? `<div class="utterance-original">Spoken (${ml.source_lang_name}): “${tx.original_text}”</div>`
    : "";

  const nmmHtml = (asl.non_manual_markers || [])
    .map((m) => `<span class="nmm-badge">👀 ${m}</span>`)
    .join(" ");

  const aslHtml = state.showAslGloss
    ? `<div class="asl-gloss-box">
         <div>
           <span style="font-size:0.75rem; font-weight:700; color:var(--text-secondary);">🤟 ASL GLOSS: </span>
           <span class="asl-gloss-text">${asl.gloss_text}</span>
         </div>
         <div>${nmmHtml}</div>
       </div>`
    : "";

  const wordsHtml =
    state.showWordTimings && tx.words && tx.words.length
      ? `<div class="word-timing-strip">
           ${tx.words
             .map(
               (w) =>
                 `<span class="word-chip" title="Confidence ${(w.confidence * 100).toFixed(0)}%">${w.word} <small>${w.start_s.toFixed(2)}s</small></span>`
             )
             .join("")}
         </div>`
      : "";

  card.innerHTML = `
    <div class="turn-top-bar">
      <div class="speaker-badge-group">
        <span class="speaker-pill" style="background:${spk.badge_bg}; color:${spk.color_hex}; border:1px solid ${spk.color_hex};">
          ${spk.display_name}
        </span>
        ${badges.join("")}
      </div>
      <span class="meta-pill">${turn.timing.start_s.toFixed(1)}s – ${turn.timing.end_s.toFixed(1)}s • ${turn.timing.inference_latency_ms.toFixed(1)}ms</span>
    </div>
    <div class="utterance-primary">${primaryText}</div>
    ${originalSub}
    ${aslHtml}
    ${wordsHtml}
  `;
  return card;
}

function renderTurnsUpTo(count) {
  if (!state.scenarioData) return;
  const turns = state.scenarioData.turns.slice(0, count);
  els.transcriptStream.innerHTML = "";
  turns.forEach((t) => {
    els.transcriptStream.appendChild(createTurnCard(t));
  });
  els.transcriptStream.scrollTop = els.transcriptStream.scrollHeight;

  const latestTurn = turns.length > 0 ? turns[turns.length - 1] : null;
  renderRadar(state.scenarioData.group_summary.speakers, latestTurn);
  if (latestTurn) {
    triggerVisualAlert(latestTurn);
  }
}

async function loadScenario(scenarioId, showAll = true) {
  const targetLang = els.targetLangSelect.value;
  const res = await fetch(
    `/api/scenario?id=${encodeURIComponent(scenarioId)}&target_lang=${encodeURIComponent(targetLang)}`
  );
  const data = await res.json();
  state.scenarioData = data;
  els.scenarioTitleHeading.textContent = data.scenario.title;
  els.scenarioDescText.textContent = `${data.scenario.setting} — ${data.scenario.description}`;
  els.engineBackendLabel.textContent = `On-Device: ${data.engine_status.active_model.name} [${data.engine_status.active_backend.toUpperCase()}]`;

  state.displayedTurnCount = showAll ? data.turns.length : 1;
  renderTurnsUpTo(state.displayedTurnCount);
}

// Event Listeners
els.dismissAlertBtn.addEventListener("click", () => {
  els.visualAlertBanner.classList.add("hidden");
});

els.fontScaleSelect.addEventListener("change", (e) => {
  document.body.setAttribute("data-font-scale", e.target.value);
});

els.contrastToggleBtn.addEventListener("click", () => {
  const cur = document.body.getAttribute("data-theme");
  document.body.setAttribute(
    "data-theme",
    cur === "high-contrast-dark" ? "high-contrast-light" : "high-contrast-dark"
  );
});

els.scenarioSelect.addEventListener("change", (e) => {
  loadScenario(e.target.value, true);
});

els.targetLangSelect.addEventListener("change", () => {
  loadScenario(els.scenarioSelect.value, true);
});

els.modelSelect.addEventListener("change", async (e) => {
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ asr_model: e.target.value }),
  });
  loadScenario(els.scenarioSelect.value, true);
});

els.toggleAslGloss.addEventListener("change", (e) => {
  state.showAslGloss = e.target.checked;
  renderTurnsUpTo(state.displayedTurnCount);
});

els.toggleWordTimings.addEventListener("change", (e) => {
  state.showWordTimings = e.target.checked;
  renderTurnsUpTo(state.displayedTurnCount);
});

els.stepTurnBtn.addEventListener("click", () => {
  if (!state.scenarioData) return;
  if (state.displayedTurnCount >= state.scenarioData.turns.length) {
    state.displayedTurnCount = 1;
  } else {
    state.displayedTurnCount += 1;
  }
  renderTurnsUpTo(state.displayedTurnCount);
});

els.playAllBtn.addEventListener("click", () => {
  if (!state.scenarioData) return;
  state.displayedTurnCount = state.scenarioData.turns.length;
  renderTurnsUpTo(state.displayedTurnCount);
});

els.addTurnForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = els.customUtteranceText.value.trim();
  if (!text) return;

  const watchwords = els.watchwordsInput.value
    .split(",")
    .map((w) => w.trim())
    .filter(Boolean);

  const res = await fetch("/api/transcribe_turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      speaker_id: `spk_${els.customSpeakerName.value.toLowerCase().replace(/\W+/g, "_")}`,
      display_name: els.customSpeakerName.value.trim(),
      azimuth_deg: parseInt(els.customSpeakerAngle.value, 10),
      language_hint: els.customSpeakerLang.value,
      target_language: els.targetLangSelect.value,
      watchwords,
      text,
    }),
  });
  const payload = await res.json();
  if (state.scenarioData) {
    state.scenarioData.turns.push(payload.turn);
    state.scenarioData.group_summary = payload.group_summary;
    state.displayedTurnCount = state.scenarioData.turns.length;
    renderTurnsUpTo(state.displayedTurnCount);
  }
  els.customUtteranceText.value = "";
});

// Live Browser Microphone Recognition hook
els.micCaptureBtn.addEventListener("click", () => {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    els.customUtteranceText.value =
      "Alex, can everyone see the live on-device Parakeet captions clearly?";
    els.customUtteranceText.focus();
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = els.customSpeakerLang.value;
  recognition.interimResults = false;
  els.micCaptureBtn.textContent = "🔴 Listening…";
  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    els.customUtteranceText.value = transcript;
    els.micCaptureBtn.textContent = "🎙️ Live Mic Input";
  };
  recognition.onerror = () => {
    els.micCaptureBtn.textContent = "🎙️ Live Mic Input";
  };
  recognition.onend = () => {
    els.micCaptureBtn.textContent = "🎙️ Live Mic Input";
  };
  recognition.start();
});

// Deaf/HoH Quick Reply / Visual & Spoken Floor Request
els.speakToGroupBtn.addEventListener("click", async () => {
  const msg = els.dhhQuickReplyInput.value.trim();
  if (!msg) return;

  if ("speechSynthesis" in window) {
    const utter = new SpeechSynthesisUtterance(msg);
    window.speechSynthesis.speak(utter);
  }

  const res = await fetch("/api/transcribe_turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      speaker_id: "spk_user_dhh",
      display_name: "You (D/HoH Participant)",
      azimuth_deg: 180,
      language_hint: "en",
      target_language: els.targetLangSelect.value,
      text: msg,
      acoustic_events: ["VISUAL FLOOR REQUEST"],
    }),
  });
  const payload = await res.json();
  if (state.scenarioData) {
    state.scenarioData.turns.push(payload.turn);
    state.scenarioData.group_summary = payload.group_summary;
    state.displayedTurnCount = state.scenarioData.turns.length;
    renderTurnsUpTo(state.displayedTurnCount);
  }
  els.dhhQuickReplyInput.value = "";
});

// Initialize on load
loadScenario("multilingual_standup", true);
