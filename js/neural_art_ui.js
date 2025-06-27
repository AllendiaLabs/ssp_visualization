/**
 * @file neural_art_ui.js
 * @author AI
 * @brief UI bindings for the RandomArt neural-network generator.
 *
 * This script wires up the HTML sliders/controls that configure a brand-new
 * piece of randomly initialised neural art rendered by `RandomArt.generate`.
 * All public symbols are attached to the global scope so that the inline
 * `onclick` attribute in index.html (see `generateNewArt()`) can invoke it.
 *
 * Sliders & controls:
 *   – #colorMode   : integer [0-4] mapping to ['rgb','hsv','hsl','bw','cmyk']
 *   – #numLayers   : integer [1-10] (# fully-connected hidden layers)
 *   – #numNeurons  : integer [5-100] (neurons per hidden layer)
 *   – #z1Input     : float   [-10..10]
 *   – #z2Input     : float   [-10..10]
 *   – #generateArtButton : button that triggers a (re)render.
 *
 * The canvas holding the artwork is #artCanvas.
 *
 * Doxygen style documentation is provided for all public functions.
 */

'use strict';

/** Available colour spaces understood by `RandomArt`.  Index == slider value. */
const COLOR_MODES = ['rgb', 'hsv', 'hsl', 'bw', 'cmyk'];

/** Global SSP base vector reused across frames (initialised on demand). */
let BASE_SSP = null;

/** Off-screen canvas for double buffering to prevent white flashes */
let offscreenCanvas = null;
let offscreenCtx = null;

/** Animation state vars */
let animTimer = null;
let animIndex = 0; // Current position in the oscillation cycle
let lastRenderTime = 0;
const MIN_FRAME_INTERVAL = 20; // Minimum 100ms between renders
let isGenerating = false; // Prevent overlapping generations

/** Debounce timer for automatic regeneration */
let regenerateTimer = null;

/** Cached network configuration to avoid regenerating weights unnecessarily */
let cachedNetworkConfig = null;

/** Animation state flag */
let animRunning = false;

/**
 * Update the textual value displays next to each slider.
 * This function is invoked on every `input` event.
 * @private
 */
function updateSliderDisplays() {
  // Colour mode is a special case – we map 0-4 to a string.
  const colorIdx = parseInt(document.getElementById('colorMode').value, 10);
  document.getElementById('colorModeValue').textContent = COLOR_MODES[colorIdx];

  // Direct numeric displays.
  document.getElementById('numLayersValue').textContent = document.getElementById('numLayers').value;
  document.getElementById('numNeuronsValue').textContent = document.getElementById('numNeurons').value;
  document.getElementById('z1Value').textContent = document.getElementById('z1Input').value;
  document.getElementById('z2Value').textContent = document.getElementById('z2Input').value;

  // Exponent & speed slider displays (if present)
  const expEl = document.getElementById('exponent');
  if (expEl) {
    document.getElementById('exponentValue').textContent = expEl.value;
  }
}

/**
 * Check if network architecture has changed (requires new weights).
 * @returns {boolean} True if network needs to be regenerated.
 * @private
 */
function hasNetworkArchitectureChanged() {
  const numLayers = parseInt(document.getElementById('numLayers').value, 10);
  const numNeurons = parseInt(document.getElementById('numNeurons').value, 10);
  const newConfig = { layers: numLayers, neurons: numNeurons };
  
  if (!cachedNetworkConfig || 
      cachedNetworkConfig.layers !== newConfig.layers || 
      cachedNetworkConfig.neurons !== newConfig.neurons) {
    cachedNetworkConfig = newConfig;
    return true;
  }
  return false;
}

/**
 * Construct an options object compatible with `RandomArt.generate` by reading
 * the current state of all UI controls.
 * @param {boolean} forceNewNetwork Whether to force new network generation
 * @returns {Object} Options to forward to `RandomArt.generate`.
 * @private
 */
function collectOptions(forceNewNetwork = false) {
  const colorMode = COLOR_MODES[parseInt(document.getElementById('colorMode').value, 10)];
  const numLayers = parseInt(document.getElementById('numLayers').value, 10);
  const numNeurons = parseInt(document.getElementById('numNeurons').value, 10);
  const z1 = parseFloat(document.getElementById('z1Input').value);
  const z2 = parseFloat(document.getElementById('z2Input').value);

  return {
    // Canvas size – will be set explicitly by RandomArt.generate.
    width: 700,
    height: 500,
    // Network topology
    layers: Array(numLayers).fill(numNeurons),
    // Colour space
    colorMode,
    // Z-inputs controlling trig features
    z1,
    z2,
    // Force new network generation only if architecture changed or explicitly requested
    forceNewNetwork: forceNewNetwork || hasNetworkArchitectureChanged()
  };
}

/**
 * Generate a new artwork and paint it onto the #artCanvas element.
 * Exposed globally so that the HTML `onclick` attribute can refer to it.
 * @function generateNewArt
 * @returns {void}
 */
function generateNewArt() {
  // New art button forces creation of new network and restarts animation.
  const exponent = parseFloat(document.getElementById('exponent').value);
  generateArtForExponent(exponent, true);
  startAnimation();
}

/**
 * Debounced regeneration function to avoid excessive updates during slider changes.
 * @private
 */
function debouncedRegenerate() {
  if (regenerateTimer) {
    clearTimeout(regenerateTimer);
  }
  regenerateTimer = setTimeout(() => {
    // Regenerate art with the current exponent value (animation may be running or stopped)
    const currentExp = parseFloat(document.getElementById('exponent').value);
    const forceNetwork = hasNetworkArchitectureChanged();
    generateArtForExponent(currentExp, forceNetwork);
  }, 500); // 500ms delay for less frequent updates
}

/** Helper – fetch current exponent range min/max as numbers. */
function getExponentRange() {
  const minExp = parseFloat(document.getElementById('exponentMin').value);
  const maxExp = parseFloat(document.getElementById('exponentMax').value);
  return { minExp, maxExp };
}

/** Compute or reuse base SSP with dimensionality matching #dimensions slider. */
function getBaseSSP() {
  const dims = parseInt(document.getElementById('dimensions').value, 10);
  if (!BASE_SSP || BASE_SSP.length !== dims) {
    // Util functions are provided by ssp.js which is loaded afterwards but
    // available by the time this function is first invoked.
    BASE_SSP = makeGoodUnitary(dims);
  }
  return BASE_SSP;
}

/** Render neural art for a specific exponent value. */
function generateArtForExponent(exponent, forceNewNetwork = false) {
  if (isGenerating) return; // Skip if already generating
  isGenerating = true;

  // Persist current exponent slider to reflect animation.
  const expSlider = document.getElementById('exponent');
  if (expSlider) {
    expSlider.value = exponent.toFixed(3);
  }

  updateSliderDisplays();

  const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('artCanvas'));
  
  // Build const feature vector – exponentiated SSP.
  const baseSSP = getBaseSSP();
  const sspExp = powerSSP(baseSSP, exponent);

  const opts = collectOptions(forceNewNetwork);
  opts.constFeatures = sspExp;

  // Update global marker for vertical line and refresh chart
  window.currentExponentMarker = exponent;
  if (window.similarityChart && typeof window.similarityChart.update === 'function') {
    window.similarityChart.update('none'); // 'none' for no animation
  }

  // Generate to off-screen canvas first, but let RandomArt set the size
  if (!offscreenCanvas) {
    offscreenCanvas = document.createElement('canvas');
    offscreenCtx = offscreenCanvas.getContext('2d');
  }

  RandomArt.generate(offscreenCanvas, opts).then(() => {
    // Ensure display canvas matches off-screen canvas size
    if (canvas.width !== offscreenCanvas.width || canvas.height !== offscreenCanvas.height) {
      canvas.width = offscreenCanvas.width;
      canvas.height = offscreenCanvas.height;
    }
    
    // Copy the generated image to display canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreenCanvas, 0, 0);
  }).catch((err) => {
    console.error('Failed to generate neural art:', err);
  }).finally(() => {
    isGenerating = false;
  });
}

/** Start or restart the exponent animation using slider-defined speed. */
function startAnimation() {
  // Stop any running loop first
  stopAnimation();

  animRunning = true;
  animIndex = 0;

  const expSlider = document.getElementById('exponent');
  if (!expSlider) return;

  const frameLoop = () => {
    if (!animRunning) return;

    if (!isGenerating) {
      const now = Date.now();
      if (now - lastRenderTime >= MIN_FRAME_INTERVAL) {
        // Fetch latest animation parameters each frame so changes take effect immediately.
        const { minExp, maxExp } = getExponentRange();
        const numPts = parseInt(document.getElementById('numPoints').value, 10);

        // Compute exponent for current frame.
        const t = (animIndex / numPts) * 2 * Math.PI;
        const sineValue = Math.sin(t);
        const currentExp = minExp + (sineValue + 1) * (maxExp - minExp) / 2;

        // Advance animation index, wrapping around numPts.
        animIndex = (animIndex + 1) % numPts;

        lastRenderTime = now;
        generateArtForExponent(currentExp, false);
      }
    }

    requestAnimationFrame(frameLoop);
  };

  requestAnimationFrame(frameLoop);
}

/** Halt the ongoing exponent animation (if any). */
function stopAnimation() {
  animRunning = false;
  if (animTimer) {
    clearInterval(animTimer);
    animTimer = null;
  }
}

// ---------------------------------------------------------------------------
// Initialisation – executed once DOM is ready.
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  // Initialize cached network config with current values
  const numLayers = parseInt(document.getElementById('numLayers').value, 10);
  const numNeurons = parseInt(document.getElementById('numNeurons').value, 10);
  cachedNetworkConfig = { layers: numLayers, neurons: numNeurons };

  // Hook up live slider display updates and automatic regeneration.
  ['colorMode', 'numLayers', 'numNeurons', 'z1Input', 'z2Input'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', () => {
        updateSliderDisplays();
        debouncedRegenerate(); // Automatically regenerate art when sliders change
      });
    }
  });

  // Additional sliders controlling exponent, number of points, and bounds
  ['exponent', 'numPoints', 'exponentMin', 'exponentMax', 'dimensions'].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;

    if (id === 'exponent') {
      el.addEventListener('input', () => {
        updateSliderDisplays();
        stopAnimation(); // Manual change halts animation
        // Debounce manual exponent changes
        if (regenerateTimer) clearTimeout(regenerateTimer);
        regenerateTimer = setTimeout(() => {
          const val = parseFloat(el.value);
          generateArtForExponent(val, false);
        }, 200);
      });
    } else if (id === 'numPoints') {
      el.addEventListener('input', () => {
        updateSliderDisplays();
        // Reset index and restart animation if it's running, else regenerate current frame
        animIndex = 0;
        if (animRunning) {
          startAnimation();
        } else {
          const val = parseFloat(document.getElementById('exponent').value);
          generateArtForExponent(val, true);
        }
      });
    } else if (id === 'exponentMin' || id === 'exponentMax') {
      el.addEventListener('input', () => {
        updateSliderDisplays();
        // Reset animation index when parameters change
        animIndex = 0;
        
        // Restart animation for bounds changes
        if (animRunning) {
          startAnimation(); // Restart with new parameters
        } else {
          // If animation stopped, regenerate art with current exponent value
          const val = parseFloat(document.getElementById('exponent').value);
          generateArtForExponent(val, true);
        }
      });
    } else if (!animRunning) {
      // For other changes (like dimensions), regenerate if animation stopped or restart animation
      el.addEventListener('input', () => {
        updateSliderDisplays();
        animIndex = 0;
        if (animRunning) {
          startAnimation();
        } else {
          const val = parseFloat(document.getElementById('exponent').value);
          generateArtForExponent(val, true);
        }
      });
    }
  });

  // Perform an initial display update.
  updateSliderDisplays();

  // The art canvas will be initialised by `generateArtForExponent` below.

  // Initial render; user may start animation via the Play button when ready.
  generateArtForExponent(parseFloat(document.getElementById('exponent').value), true);
});

// Attach to global scope so inline HTML can access it.
window.generateNewArt = generateNewArt;
window.startArtAnimation = startAnimation;
window.stopArtAnimation = stopAnimation;
window.generateArtForExponent = generateArtForExponent; 