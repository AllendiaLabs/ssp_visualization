/**
 * @file audio_ssp.js
 * @brief Front-end logic for uploading an audio sample, converting it into an SSP, exponentiating it,
 *        and allowing users to audibly compare the original and exponentiated versions.
 *
 * This script relies on the global Complex, customFFT, customIFFT, and powerSSP helpers that are
 * defined in ssp.js.  Ensure that ssp.js is loaded before this file.
 *
 * The high-level workflow is:
 *   1.  The user selects an audio file from disk.
 *   2.  We decode it to a Web Audio AudioBuffer (mono – first channel only).
 *   3.  The raw waveform is normalised and treated as an SSP of dimension D = N (samples).
 *   4.  When the user clicks "Process", we exponentiate the SSP with the slider-controlled value.
 *   5.  Both buffers can be played back on demand for A/B auditory comparison.
 *
 * All public symbols are namespaced under the global `audioSSP` object to avoid cluttering the
 * global scope further.
 */

'use strict';

// ---------------------------------------------------------------------------
// Private helpers & state
// ---------------------------------------------------------------------------
/** Single shared AudioContext (lazy-initialised to satisfy browser autoplay policies). */
let _audioCtx = null;
function getAudioContext() {
  if (!_audioCtx) {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return _audioCtx;
}

/**
 * Normalise a Float32Array / Number[] to the range [-1, 1].
 *
 * @param {Array<number>} signal – Raw audio samples.
 * @return {Array<number>}      – Normalised signal.
 */
function normaliseSignal(signal) {
  let maxAbs = 0;
  for (let i = 0; i < signal.length; i++) {
    const absVal = Math.abs(signal[i]);
    if (absVal > maxAbs) maxAbs = absVal;
  }
  if (maxAbs === 0) return signal.slice();
  return signal.map((v) => v / maxAbs);
}

/**
 * Convert a Number[] into a Web Audio AudioBuffer.
 *
 * @param {Array<number>} samples  – PCM samples in the range [-1, 1].
 * @param {number}        sr       – Sample rate in Hz.
 * @return {AudioBuffer}           – Resulting mono buffer.
 */
function arrayToBuffer(samples, sr) {
  const ctx = getAudioContext();
  const buf = ctx.createBuffer(1, samples.length, sr);
  buf.copyToChannel(Float32Array.from(samples), 0);
  return buf;
}

/** Global state variables. */
let originalBuffer = null;         // AudioBuffer of the uploaded sample (mono)
let exponentiatedBuffer = null;    // AudioBuffer after SSP exponentiation

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const fileInput            = document.getElementById('audioUpload');
const exponentSlider       = document.getElementById('audioExponent');
const exponentValueLabel   = document.getElementById('audioExponentValue');
const processBtn           = document.getElementById('processAudioButton');
const playOriginalBtn      = document.getElementById('playOriginalButton');
const playExponentiatedBtn = document.getElementById('playExponentiatedButton');

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------
fileInput.addEventListener('change', handleFileSelect);
exponentSlider.addEventListener('input', () => {
  exponentValueLabel.textContent = exponentSlider.value;
});
processBtn.addEventListener('click', processAudio);
playOriginalBtn.addEventListener('click', () => playBuffer(originalBuffer));
playExponentiatedBtn.addEventListener('click', () => playBuffer(exponentiatedBuffer));

/**
 * Handles the <input type="file"> change event.
 */
function handleFileSelect() {
  const files = fileInput.files;
  if (!files || files.length === 0) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const arrayBuffer = e.target.result;
    const ctx = getAudioContext();

    ctx.decodeAudioData(arrayBuffer)
      .then((buffer) => {
        // Keep only the first channel for simplicity.
        originalBuffer = ctx.createBuffer(1, buffer.length, buffer.sampleRate);
        const tmp = new Float32Array(buffer.length);
        buffer.copyFromChannel(tmp, 0);
        originalBuffer.copyToChannel(tmp, 0);

        // Enable UI controls.
        processBtn.disabled = false;
        playOriginalBtn.disabled = false;
        playExponentiatedBtn.disabled = true;
      })
      .catch((err) => {
        console.error('Failed to decode audio:', err);
        alert('Could not decode the selected audio file.');
      });
  };
  reader.readAsArrayBuffer(files[0]);
}

/**
 * Perform SSP exponentiation on the uploaded audio sample.
 */
function processAudio() {
  if (!originalBuffer) return;

  // Extract mono signal.
  const channelData = new Float32Array(originalBuffer.length);
  originalBuffer.copyFromChannel(channelData, 0);
  const samples = Array.from(channelData);

  // Treat waveform as an SSP: normalise and exponentiate.
  const normalised = normaliseSignal(samples);
  const exponent = parseFloat(exponentSlider.value);

  // powerSSP expects Number[], returns Number[].
  let powered;
  try {
    powered = powerSSP(normalised, exponent);
  } catch (e) {
    console.error('Error exponentiating signal:', e);
    alert('Failed to exponentiate the signal. See console for details.');
    return;
  }

  // Prevent clipping after iFFT by re-normalising.
  const poweredNorm = normaliseSignal(powered);

  // Convert back to AudioBuffer.
  exponentiatedBuffer = arrayToBuffer(poweredNorm, originalBuffer.sampleRate);
  playExponentiatedBtn.disabled = false;
}

/**
 * Play an AudioBuffer through the shared AudioContext.
 *
 * @param {AudioBuffer|null} buf – Buffer to play.
 */
function playBuffer(buf) {
  if (!buf) return;
  const ctx = getAudioContext();
  const source = ctx.createBufferSource();
  source.buffer = buf;
  source.connect(ctx.destination);
  source.start();
}

// ---------------------------------------------------------------------------
// Expose minimal API for debugging.
// ---------------------------------------------------------------------------
window.audioSSP = {
  get originalBuffer() { return originalBuffer; },
  get exponentiatedBuffer() { return exponentiatedBuffer; },
  processAudio,
}; 