/**
 * RoboComposer UI: prompt + MIDI playback with piano + robot hands.
 */

import * as Tone from "https://esm.sh/tone@14.7.77";
import { Midi } from "https://esm.sh/@tonejs/midi@2.0.28";

const START_MIDI = 48;
const END_MIDI = 84;
const HAND_SPLIT = 66;

const el = {
  prompt: document.getElementById("prompt"),
  btnGenerate: document.getElementById("btn-generate"),
  generateStatus: document.getElementById("generate-status"),
  midiFile: document.getElementById("midi-file"),
  keyboard: document.getElementById("keyboard"),
  keyboardWrap: document.getElementById("keyboard-wrap"),
  handLeft: document.getElementById("hand-left"),
  handRight: document.getElementById("hand-right"),
  btnPlay: document.getElementById("btn-play"),
  btnStop: document.getElementById("btn-stop"),
  btnSaveMidi: document.getElementById("btn-save-midi"),
  playbackStatus: document.getElementById("playback-status"),
};

/** @type {ArrayBuffer | null} */
let midiBuffer = null;
/** Filename used when saving the current buffer to disk. */
let midiDownloadName = "robocomposer.mid";
/** @type {Midi | null} */
let parsedMidi = null;
/** @type {Tone.PolySynth | null} */
let activeSynth = null;
/** @type {Tone.Part | null} */
let activePart = null;
/** @type {ReturnType<typeof setTimeout> | null} */
let finishTimer = null;
let playbackGeneration = 0;
const keyElements = new Map();

/**
 * One octave pattern
 * @returns {{ offset: number, type: 'white'|'black' }[]}
 */
function octavePattern() {
  return [
    { offset: 0, type: "white" },
    { offset: 1, type: "black" },
    { offset: 2, type: "white" },
    { offset: 3, type: "black" },
    { offset: 4, type: "white" },
    { offset: 5, type: "white" },
    { offset: 6, type: "black" },
    { offset: 7, type: "white" },
    { offset: 8, type: "black" },
    { offset: 9, type: "white" },
    { offset: 10, type: "black" },
    { offset: 11, type: "white" },
  ];
}

function buildKeyboard() {
  el.keyboard.innerHTML = "";
  keyElements.clear();

  for (let base = START_MIDI; base <= END_MIDI; base += 12) {
    for (const step of octavePattern()) {
      const midi = base + step.offset;
      if (midi < START_MIDI || midi > END_MIDI) continue;

      const div = document.createElement("div");
      div.className = `key ${step.type}`;
      div.dataset.midi = String(midi);
      div.setAttribute("role", "presentation");

      const lab = document.createElement("span");
      lab.className = "label-mini";
      lab.textContent = Tone.Frequency(midi, "midi").toNote().replace("#", "♯");
      div.appendChild(lab);

      el.keyboard.appendChild(div);
      keyElements.set(midi, div);
    }
  }
}

function setGenerateStatus(text, isWarn = false) {
  el.generateStatus.textContent = text;
  el.generateStatus.classList.toggle("warn", isWarn);
}

function setPlaybackStatus(text, isWarn = false) {
  el.playbackStatus.textContent = text;
  el.playbackStatus.classList.toggle("warn", isWarn);
}

function positionHandForMidi(midi, pressing) {
  const key = keyElements.get(midi);
  if (!key) return;

  const wrap = el.keyboardWrap;
  const wrapRect = wrap.getBoundingClientRect();
  const keyRect = key.getBoundingClientRect();
  const cxPct = ((keyRect.left + keyRect.width / 2 - wrapRect.left) / wrapRect.width) * 100;

  const hand = midi < HAND_SPLIT ? el.handLeft : el.handRight;
  hand.style.left = `${Math.min(98, Math.max(2, cxPct))}%`;
  hand.classList.toggle("press", pressing);
}

function activateKey(midi) {
  const key = keyElements.get(midi);
  if (key) key.classList.add("active");
  positionHandForMidi(midi, true);
}

function deactivateKey(midi) {
  const key = keyElements.get(midi);
  if (key) key.classList.remove("active");
  positionHandForMidi(midi, false);
}

function stopPlayback() {
  playbackGeneration += 1;

  if (finishTimer) {
    clearTimeout(finishTimer);
    finishTimer = null;
  }

  Tone.Transport.stop();
  Tone.Transport.cancel(0);

  if (activePart) {
    activePart.stop();
    activePart.dispose();
    activePart = null;
  }

  if (activeSynth) {
    activeSynth.releaseAll();
    activeSynth.dispose();
    activeSynth = null;
  }

  Tone.Draw.cancel(0);

  keyElements.forEach((k) => k.classList.remove("active"));
  el.handLeft.classList.remove("press");
  el.handRight.classList.remove("press");
  el.handLeft.style.left = "";
  el.handRight.style.left = "";
  el.btnPlay.disabled = !midiBuffer;
  el.btnStop.disabled = true;
  el.btnSaveMidi.disabled = !midiBuffer;
  setPlaybackStatus("");
}

/**
 * @param {Midi} midi
 */
async function startPlayback(midi) {
  stopPlayback();
  await Tone.start();
  if (Tone.context.state !== "running") {
    await Tone.context.resume();
  }

  const synth = new Tone.PolySynth(Tone.Synth).toDestination();
  synth.set({
    envelope: { attack: 0.02, decay: 0.1, sustain: 0.28, release: 0.4 },
    oscillator: { type: "triangle" },
  });
  synth.volume.value = -10;

  const events = [];
  for (const track of midi.tracks) {
    for (const note of track.notes) {
      events.push({
        time: note.time,
        midi: note.midi,
        duration: note.duration,
        velocity: note.velocity,
      });
    }
  }
  events.sort((a, b) => a.time - b.time);

  if (events.length === 0) {
    setPlaybackStatus("No notes in MIDI.");
    synth.dispose();
    return;
  }

  const session = playbackGeneration;
  activeSynth = synth;

  const timeline = events.map((ev) => [ev.time, ev]);
  const part = new Tone.Part((time, ev) => {
    const name = Tone.Frequency(ev.midi, "midi").toNote();
    synth.triggerAttackRelease(name, ev.duration, time, ev.velocity);
    Tone.Draw.schedule(() => {
      if (session !== playbackGeneration) return;
      activateKey(ev.midi);
    }, time);
    Tone.Draw.schedule(() => {
      if (session !== playbackGeneration) return;
      deactivateKey(ev.midi);
    }, time + ev.duration);
  }, timeline);

  activePart = part;
  part.loop = false;

  Tone.Transport.seconds = 0;
  part.start(0);
  Tone.Transport.start();

  el.btnPlay.disabled = true;
  el.btnStop.disabled = false;
  setPlaybackStatus("Playing…");

  finishTimer = setTimeout(() => {
    finishTimer = null;
    if (session !== playbackGeneration) return;
    if (activeSynth === synth) {
      stopPlayback();
      setPlaybackStatus("Finished.");
    }
  }, midi.duration * 1000 + 500);
}

function sanitizeFilename(name) {
  const trimmed = name.trim().replace(/[/\\?%*:|"<>]/g, "_");
  return trimmed || "robocomposer.mid";
}

function downloadMidiFile() {
  if (!midiBuffer) return;
  const blob = new Blob([midiBuffer], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const name = midiDownloadName.toLowerCase();
  a.download =
    name.endsWith(".mid") || name.endsWith(".midi") ? midiDownloadName : `${midiDownloadName}.mid`;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/**
 * @param {ArrayBuffer} buf
 * @param {string} [filename] suggested download name
 */
async function loadMidiFromBuffer(buf, filename) {
  midiBuffer = buf.slice(0);
  if (filename) {
    midiDownloadName = sanitizeFilename(filename);
  }
  parsedMidi = new Midi(midiBuffer);
  el.btnPlay.disabled = false;
  el.btnSaveMidi.disabled = false;
  setPlaybackStatus(`Ready (${parsedMidi.duration.toFixed(1)}s).`);
}

el.btnGenerate.addEventListener("click", async () => {
  const text = el.prompt.value.trim();
  if (!text) {
    setGenerateStatus("Enter a prompt first.", true);
    return;
  }
  setGenerateStatus("Requesting…");
  try {
    const res = await fetch("/api/compose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("audio/midi") || ct.includes("application/octet-stream")) {
      const buf = await res.arrayBuffer();
      await loadMidiFromBuffer(buf, "generated.mid");
      setGenerateStatus("MIDI received. Press Play.");
    } else {
      const data = await res.json().catch(() => ({}));
      if (data.midiUrl) {
        const r2 = await fetch(data.midiUrl);
        if (!r2.ok) throw new Error("midiUrl fetch failed");
        await loadMidiFromBuffer(await r2.arrayBuffer(), "generated.mid");
        setGenerateStatus("MIDI loaded. Press Play.");
      } else {
        setGenerateStatus("Server returned no MIDI yet. Upload a file.", true);
      }
    }
  } catch {
    setGenerateStatus("No /api/compose yet — use file upload + Play.", true);
  }
});

el.midiFile.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  stopPlayback();
  setPlaybackStatus("Loading file…");
  try {
    await loadMidiFromBuffer(await file.arrayBuffer(), file.name);
    setPlaybackStatus(`Loaded: ${file.name}`);
  } catch {
    setPlaybackStatus("Could not read MIDI.", true);
  }
});

el.btnPlay.addEventListener("click", async () => {
  if (!parsedMidi) return;
  try {
    await Tone.start();
    if (Tone.context.state !== "running") {
      await Tone.context.resume();
    }
    await startPlayback(parsedMidi);
  } catch (err) {
    setPlaybackStatus(`Playback error: ${err?.message || err}`);
  }
});

el.btnStop.addEventListener("click", () => {
  stopPlayback();
  setPlaybackStatus("Stopped.");
});

el.btnSaveMidi.addEventListener("click", () => {
  downloadMidiFile();
});

buildKeyboard();
