/**
 * @file random_art.js
 * @author AI
 * @brief JavaScript re-implementation of the "neural-net-random-art" generator.
 *
 * This module recreates the core functionality of the original Python implementation
 * (see python/neural-net-random-art/network.py & artgen.py) using TensorFlow.js.
 *
 * The library is browser-first but can also run under Node.js when `@tensorflow/tfjs-node`
 * is installed. All public functions are namespaced under `RandomArt`.
 *
 * The main entry point is `RandomArt.generate(canvas, opts)` which renders a brand-new
 * piece of neural art to a `<canvas>` element.
 *
 * Example (browser):
 * -----------------------------------------------------------
 * <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
 * <script src="js/random_art.js"></script>
 * <canvas id="art"></canvas>
 * <script>
 *   RandomArt.generate(document.getElementById('art'), {
 *     width: 700,
 *     height: 500,
 *     layers: [10, 10, 10, 10, 10],
 *     activation: 'tanh',
 *     colorMode: 'hsv',
 *     alpha: true
 *   });
 * </script>
 * -----------------------------------------------------------
 *
 * Dependencies:
 *   • TensorFlow.js >= 4.x (loaded globally as `tf`)
 *
 * Notes:
 *   • All random weights are drawn from 𝒩(0,1) for kernels and 𝒩(0,0.1) for biases, matching
 *     the original implementation.
 *   • Color conversions (HSV/HSL) follow the reference algorithms identical to the Python code.
 *   • For brevity, only the activations present in the original code are exposed (tanh, sigmoid,
 *     relu, softsign, sin, cos).
 *   • The module does *not* perform any training – it simply forwards a random input grid through
 *     a randomly initialised feed-forward network to create deterministic, yet randomly looking,
 *     patterns.
 */

/* global tf */

// Guard: TensorFlow.js must be available at runtime.
if (typeof tf === 'undefined') {
  throw new Error('TensorFlow.js is required but not found. Please include it before loading random_art.js');
}

/**
 * Namespace for the random art generator.
 * @namespace RandomArt
 */
// eslint-disable-next-line no-var
var RandomArt = (() => {
  'use strict';

  /**
   * @typedef {Object} GenerateOptions
   * @brief Configuration object for `RandomArt.generate`.
   * @property {number} width           Canvas width in pixels. Default 700.
   * @property {number} height          Canvas height in pixels. Default 500.
   * @property {boolean} symmetry       Whether to square x & y coordinates. Default false.
   * @property {boolean} trig           Whether to apply sine/cosine transformation for z1/z2. Default true.
   * @property {number} z1              Constant/coeff. for z1 component.
   * @property {number} z2              Constant/coeff. for z2 component.
   * @property {boolean} noise          Toggle Gaussian input noise. Default false.
   * @property {number} noiseStd        Std-dev for input noise. Default 0.01.
   * @property {number[]} layers        Hidden layer dimensions. Default [10,10,10,10,10].
   * @property {string} activation      Activation fnc ('tanh','sigmoid','relu','softsign','sin','cos'). Default 'tanh'.
   * @property {string} colorMode       Output color space ('rgb','bw','cmyk','hsv','hsl'). Default 'rgb'.
   * @property {boolean} alpha          Enable alpha channel (mapped to 0.25–1.0). Default true.
   * @property {boolean} forceNewNetwork Force creation of new network weights. Default false.
   */

  /** Default options applied when none are provided. */
  const DEFAULT_OPTS = /** @type {GenerateOptions} */ ({
    width: 700,
    height: 500,
    symmetry: false,
    trig: false,
    z1: 0.0,
    z2: 0.0,
    noise: false,
    noiseStd: 0.01,
    layers: [10, 10, 10, 10, 10],
    activation: 'tanh',
    colorMode: 'rgb',
    alpha: true,
    forceNewNetwork: false
  });

  /** Cached model to avoid regenerating weights unnecessarily */
  let cachedModel = null;
  let cachedModelConfig = null;

  /**
   * Merge user options with defaults (shallow).
   * @param {Partial<GenerateOptions>} usr
   * @returns {GenerateOptions}
   */
  function withDefaults(usr) {
    return Object.assign({}, DEFAULT_OPTS, usr);
  }

  /**
   * Creates the input tensor `[N,5]` corresponding to (x, y, r, z1, z2).
   * @param {number} width Image width in pixels.
   * @param {number} height Image height in pixels.
   * @param {boolean} symmetry Square x & y values (mirror symmetry).
   * @param {boolean} trig Apply trig transformations for z1 & z2.
   * @param {number} z1 Constant or coefficient for z1 channel.
   * @param {number} z2 Constant or coefficient for z2 channel.
   * @param {boolean} noise Add Gaussian noise.
   * @param {number} noiseStd Standard deviation for the noise.
   * @returns {tf.Tensor2D} Flattened tensor of shape `[width*height, 5]`.
   */
  function initData(width, height, symmetry, trig, z1, z2, noise, noiseStd) {
    const factor = Math.min(width, height);

    // Normalised coordinate grids in range [-1, 1].
    // x = height coordinates, y = width coordinates (like Python)
    const x = Array.from({ length: height }, (_, i) => ((i / factor) - 0.5) * 2);
    const y = Array.from({ length: width }, (_, j) => ((j / factor) - 0.5) * 2);

    // Create meshgrid exactly like Python: np.meshgrid(x, y)
    // This creates xv with shape (width, height) and yv with shape (width, height)
    const xv = new Float32Array(width * height);
    const yv = new Float32Array(width * height);
    let ptr = 0;
    for (let j = 0; j < width; ++j) {
      for (let i = 0; i < height; ++i) {
        xv[ptr] = x[i]; // height coordinate
        yv[ptr] = y[j]; // width coordinate
        ++ptr;
      }
    }

    // Apply symmetry (square x and y values)
    if (symmetry) {
      for (let i = 0; i < xv.length; ++i) {
        xv[i] = xv[i] ** 2;
        yv[i] = yv[i] ** 2;
      }
    }

    // Radius r = sqrt(x^2 + y^2)
    const radius = new Float32Array(xv.length);
    for (let i = 0; i < radius.length; ++i) {
      radius[i] = Math.hypot(xv[i], yv[i]);
    }

    // z1, z2 components
    const z1Arr = new Float32Array(xv.length);
    const z2Arr = new Float32Array(xv.length);

    if (trig) {
      for (let i = 0; i < z1Arr.length; ++i) {
        z1Arr[i] = Math.cos(z1 * xv[i]);
        z2Arr[i] = Math.sin(z2 * yv[i]);
      }
    } else {
      z1Arr.fill(z1);
      z2Arr.fill(z2);
    }

    // Stack into tf.Tensor2D - transpose like Python (.T.flatten())
    // Python: np.concatenate([x_, y_, r_, z1_, z2_], axis=1)
    // The .T.flatten() means we need to transpose the meshgrid before flattening
    const data = new Float32Array(xv.length * 5);
    for (let i = 0; i < height; ++i) {
      for (let j = 0; j < width; ++j) {
        const srcIdx = j * height + i; // Transposed indexing
        const dstIdx = (i * width + j) * 5; // Normal row-major indexing
        data[dstIdx] = xv[srcIdx];
        data[dstIdx + 1] = yv[srcIdx];
        data[dstIdx + 2] = radius[srcIdx];
        data[dstIdx + 3] = z1Arr[srcIdx];
        data[dstIdx + 4] = z2Arr[srcIdx];
      }
    }

    let tensor = tf.tensor2d(data, [width * height, 5]);

    if (noise) {
      const noiseTensor = tf.randomNormal(tensor.shape, 0, noiseStd, 'float32');
      tensor = tf.add(tensor, noiseTensor);
    }

    return tensor;
  }

  /**
   * Map human-readable activation to tf.js-compatible function.
   * @param {string} act
   * @returns {string|function(tf.Tensor):tf.Tensor}
   */
  function resolveActivation(act) {
    switch (act.toLowerCase()) {
      case 'tanh':
      case 'sigmoid':
      case 'relu':
      case 'softsign':
        return act.toLowerCase();
      case 'sin':
      case 'cos':
        console.warn(`Activation '${act}' is not natively supported in tf.js layers. Falling back to 'tanh'.`);
        return 'tanh';
      default:
        console.warn(`Unknown activation '${act}' – falling back to tanh.`);
        return 'tanh';
    }
  }

  /**
   * Build a fully-connected feed-forward model with the provided specification.
   * @param {number[]} layers
   * @param {string} activation
   * @param {number} outNodes
   * @returns {tf.LayersModel}
   */
  function createModel(layers, activation, outNodes, inputDim = 5) {
    const model = tf.sequential();

    const actStr = resolveActivation(activation);

    // First hidden layer
    model.add(tf.layers.dense({
      inputShape: [inputDim],
      units: layers[0],
      activation: actStr,
      kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 1 }),
      biasInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 0.1 })
    }));

    for (let i = 1; i < layers.length; ++i) {
      model.add(tf.layers.dense({
        units: layers[i],
        activation: actStr,
        kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 1 }),
        biasInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 0.1 })
      }));
    }

    model.add(tf.layers.dense({
      units: outNodes,
      activation: 'sigmoid',
      kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 1 }),
      biasInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 0.1 })
    }));

    return model;
  }

  /**
   * Determine number of output channels based on color configuration.
   * @param {string} colorMode
   * @param {boolean} alpha
   * @returns {number}
   */
  function computeOutNodes(colorMode, alpha) {
    switch (colorMode.toLowerCase()) {
      case 'rgb':
      case 'hsv':
      case 'hsl':
        return alpha ? 4 : 3;
      case 'cmyk':
        return alpha ? 5 : 4;
      case 'bw':
        return alpha ? 2 : 1;
      default:
        throw new Error(`Unsupported color mode '${colorMode}'.`);
    }
  }

  /**
   * Get the maximum number of output nodes needed for any color mode.
   * @returns {number}
   */
  function getMaxOutNodes() {
    return 5; // CMYK + alpha is the maximum
  }

  /**
   * Convert HSV (all in [0,1]) to RGB.
   * Reference: identical algorithm to artgen.py.
   * @param {number} h
   * @param {number} s
   * @param {number} v
   * @returns {[number, number, number]}
   */
  function hsvToRgb(h, s, v) {
    h *= 6;
    const i = Math.floor(h);
    const f = h - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    const mod = i % 6;
    // Exactly like Python: [v, q, p, p, t, v][mod], [t, v, v, q, p, p][mod], [p, p, t, v, v, q][mod]
    const r = [v, q, p, p, t, v][mod];
    const g = [t, v, v, q, p, p][mod];
    const b = [p, p, t, v, v, q][mod];
    return [r, g, b];
  }

  /**
   * Helper for the HSL hue-to-rgb step.
   * @param {number} p
   * @param {number} q
   * @param {number} t
   * @returns {number}
   */
  function hueToRgb(p, q, t) {
    if (t < 0 || t > 1) return p;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  }

  /**
   * Convert HSL (all in [0,1]) to RGB.
   * @param {number} h
   * @param {number} s
   * @param {number} l
   * @returns {[number, number, number]}
   */
  function hslToRgb(h, s, l) {
    if (s === 0) {
      return [l, l, l];
    }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    return [hueToRgb(p, q, h + 1 / 3), hueToRgb(p, q, h), hueToRgb(p, q, h - 1 / 3)];
  }

  /**
   * Transform raw network outputs into RGBA pixels (reshaped version).
   * @param {Float32Array} raw Raw network output in [height, width, channels] format.
   * @param {number} height
   * @param {number} width
   * @param {number} channels
   * @param {string} colorMode
   * @param {boolean} alphaEnabled
   * @returns {Uint8ClampedArray} Pixels ready for `ImageData`.
   */
  function transformColorsReshaped(raw, height, width, channels, colorMode, alphaEnabled) {
    const pixelCount = height * width;
    const rgba = new Uint8ClampedArray(pixelCount * 4);

    colorMode = colorMode.toLowerCase();

    for (let i = 0; i < height; ++i) {
      for (let j = 0; j < width; ++j) {
        const pixelIdx = i * width + j;
        const rawIdx = pixelIdx * channels;
        
        let r, g, b, a;

        switch (colorMode) {
          case 'rgb': {
            r = raw[rawIdx];
            g = raw[rawIdx + 1];
            b = raw[rawIdx + 2];
            break;
          }
          case 'bw': {
            const v = raw[rawIdx];
            r = g = b = v;
            break;
          }
          case 'cmyk': {
            const c = raw[rawIdx];
            const m = raw[rawIdx + 1];
            const y = raw[rawIdx + 2];
            const k = raw[rawIdx + 3];
            r = (1 - c) * k;
            g = (1 - m) * k;
            b = (1 - y) * k;
            break;
          }
          case 'hsv': {
            const h = raw[rawIdx];
            const s = raw[rawIdx + 1];
            const v = raw[rawIdx + 2];
            [r, g, b] = hsvToRgb(h, s, v);
            break;
          }
          case 'hsl': {
            const h = raw[rawIdx];
            const s = raw[rawIdx + 1];
            const l = raw[rawIdx + 2];
            [r, g, b] = hslToRgb(h, s, l);
            break;
          }
          default:
            throw new Error(`Unsupported color mode '${colorMode}'.`);
        }

        // Alpha handling exactly like Python: a = 1 - abs(2 * alpha - 1), then 0.25 + 0.75 * a
        if (alphaEnabled) {
          const alphaRaw = raw[rawIdx + channels - 1]; // Last channel is alpha
          const aLinear = 1 - Math.abs(2 * alphaRaw - 1);
          a = 0.25 + 0.75 * aLinear;
        } else {
          a = 1.0;
        }

        const outOff = pixelIdx * 4;
        rgba[outOff] = Math.round(r * 255);
        rgba[outOff + 1] = Math.round(g * 255);
        rgba[outOff + 2] = Math.round(b * 255);
        rgba[outOff + 3] = Math.round(a * 255);
      }
    }

    return rgba;
  }

  /**
   * Transform raw network outputs into RGBA pixels.
   * @param {Float32Array} raw Raw network output flattened `[N * channels]`.
   * @param {number} width
   * @param {number} height
   * @param {string} colorMode
   * @param {boolean} alphaEnabled
   * @returns {Uint8ClampedArray} Pixels ready for `ImageData`.
   */
  function transformColors(raw, width, height, colorMode, alphaEnabled) {
    const pixelCount = width * height;
    const channels = computeOutNodes(colorMode, alphaEnabled);
    const rgba = new Uint8ClampedArray(pixelCount * 4);

    colorMode = colorMode.toLowerCase();

    for (let i = 0; i < height; ++i) {
      for (let j = 0; j < width; ++j) {
        const pixelIdx = i * width + j;
        const rawIdx = pixelIdx * channels;
        
        let r, g, b, a;

        switch (colorMode) {
          case 'rgb': {
            r = raw[rawIdx];
            g = raw[rawIdx + 1];
            b = raw[rawIdx + 2];
            break;
          }
          case 'bw': {
            const v = raw[rawIdx];
            r = g = b = v;
            break;
          }
          case 'cmyk': {
            const c = raw[rawIdx];
            const m = raw[rawIdx + 1];
            const y = raw[rawIdx + 2];
            const k = raw[rawIdx + 3];
            r = (1 - c) * k;
            g = (1 - m) * k;
            b = (1 - y) * k;
            break;
          }
          case 'hsv': {
            const h = raw[rawIdx];
            const s = raw[rawIdx + 1];
            const v = raw[rawIdx + 2];
            [r, g, b] = hsvToRgb(h, s, v);
            break;
          }
          case 'hsl': {
            const h = raw[rawIdx];
            const s = raw[rawIdx + 1];
            const l = raw[rawIdx + 2];
            [r, g, b] = hslToRgb(h, s, l);
            break;
          }
          default:
            throw new Error(`Unsupported color mode '${colorMode}'.`);
        }

        // Alpha handling exactly like Python: a = 1 - abs(2 * alpha - 1), then 0.25 + 0.75 * a
        if (alphaEnabled) {
          const alphaRaw = raw[rawIdx + channels - 1]; // Last channel is alpha
          const aLinear = 1 - Math.abs(2 * alphaRaw - 1);
          a = 0.25 + 0.75 * aLinear;
        } else {
          a = 1.0;
        }

        const outOff = pixelIdx * 4;
        rgba[outOff] = Math.round(r * 255);
        rgba[outOff + 1] = Math.round(g * 255);
        rgba[outOff + 2] = Math.round(b * 255);
        rgba[outOff + 3] = Math.round(a * 255);
      }
    }

    return rgba;
  }

  /**
   * Get the channel mapping for a specific color mode.
   * @param {string} colorMode
   * @param {boolean} alphaEnabled
   * @returns {Object} Mapping with channels array and alphaChannel index
   */
  function getChannelMapping(colorMode, alphaEnabled) {
    colorMode = colorMode.toLowerCase();
    
    switch (colorMode) {
      case 'rgb':
        return {
          channels: [0, 1, 2], // R, G, B
          alphaChannel: alphaEnabled ? 4 : null
        };
      case 'hsv':
        return {
          channels: [0, 1, 2], // H, S, V
          alphaChannel: alphaEnabled ? 4 : null
        };
      case 'hsl':
        return {
          channels: [0, 1, 2], // H, S, L
          alphaChannel: alphaEnabled ? 4 : null
        };
      case 'cmyk':
        return {
          channels: [0, 1, 2, 3], // C, M, Y, K
          alphaChannel: alphaEnabled ? 4 : null
        };
      case 'bw':
        return {
          channels: [0], // Single channel
          alphaChannel: alphaEnabled ? 1 : null
        };
      default:
        throw new Error(`Unsupported color mode '${colorMode}'.`);
    }
  }

  /**
   * Transform raw network outputs into RGBA pixels with proper channel mapping.
   * @param {Float32Array} raw Raw network output with mapped channels.
   * @param {Float32Array} originalRaw Original raw network output.
   * @param {number} width
   * @param {number} height
   * @param {Object} channelMapping Channel mapping object
   * @param {string} colorMode
   * @param {boolean} alphaEnabled
   * @returns {Uint8ClampedArray} Pixels ready for `ImageData`.
   */
  function transformColorsWithMapping(raw, originalRaw, width, height, channelMapping, colorMode, alphaEnabled) {
    const pixelCount = width * height;
    const rgba = new Uint8ClampedArray(pixelCount * 4);

    colorMode = colorMode.toLowerCase();

    for (let i = 0; i < height; ++i) {
      for (let j = 0; j < width; ++j) {
        const pixelIdx = i * width + j;
        const rawIdx = pixelIdx * channelMapping.channels.length;
        
        let r, g, b, a;

        switch (colorMode) {
          case 'rgb': {
            r = raw[rawIdx];
            g = raw[rawIdx + 1];
            b = raw[rawIdx + 2];
            break;
          }
          case 'bw': {
            const v = raw[rawIdx];
            r = g = b = v;
            break;
          }
          case 'cmyk': {
            const c = raw[rawIdx];
            const m = raw[rawIdx + 1];
            const y = raw[rawIdx + 2];
            const k = raw[rawIdx + 3];
            r = (1 - c) * k;
            g = (1 - m) * k;
            b = (1 - y) * k;
            break;
          }
          case 'hsv': {
            const h = raw[rawIdx];
            const s = raw[rawIdx + 1];
            const v = raw[rawIdx + 2];
            [r, g, b] = hsvToRgb(h, s, v);
            break;
          }
          case 'hsl': {
            const h = raw[rawIdx];
            const s = raw[rawIdx + 1];
            const l = raw[rawIdx + 2];
            [r, g, b] = hslToRgb(h, s, l);
            break;
          }
          default:
            throw new Error(`Unsupported color mode '${colorMode}'.`);
        }

        // Alpha handling - get alpha from the original raw data using channelMapping.alphaChannel
        if (alphaEnabled && channelMapping.alphaChannel !== null) {
          const alphaRaw = originalRaw[pixelIdx * getMaxOutNodes() + channelMapping.alphaChannel];
          const aLinear = 1 - Math.abs(2 * alphaRaw - 1);
          a = 0.25 + 0.75 * aLinear;
        } else {
          a = 1.0;
        }

        const outOff = pixelIdx * 4;
        rgba[outOff] = Math.round(r * 255);
        rgba[outOff + 1] = Math.round(g * 255);
        rgba[outOff + 2] = Math.round(b * 255);
        rgba[outOff + 3] = Math.round(a * 255);
      }
    }

    return rgba;
  }

  /**
   * Render a random neural-network-generated artwork to the provided canvas.
   * @param {HTMLCanvasElement} canvas Target canvas.
   * @param {Partial<GenerateOptions>=} options Optional render configuration.
   * @returns {Promise<void>} Resolves when rendering is complete.
   */
  async function generate(canvas, options = {}) {
    const opts = withDefaults(options);
    const { width, height } = opts;

    canvas.width = width;
    canvas.height = height;

    // 1. Prepare input data (shape [N,5]).
    let input = initData(width, height, opts.symmetry, opts.trig, opts.z1, opts.z2, opts.noise, opts.noiseStd);

    // -------------------------------------------------------------------
    // Optional constant feature augmentation (e.g., exponentiated SSP).
    // When `opts.constFeatures` is provided, replicate the vector across
    // all pixels and concatenate it to the default [x,y,r,z1,z2] inputs.
    // -------------------------------------------------------------------
    let extraDim = 0;
    if (opts.constFeatures && typeof opts.constFeatures.length === 'number' && opts.constFeatures.length > 0) {
      const extra = tf.tensor2d(opts.constFeatures, [1, opts.constFeatures.length]);
      const tiled = tf.tile(extra, [width * height, 1]);
      input = tf.concat([input, tiled], 1);
      extraDim = opts.constFeatures.length;

      // Dispose helper tensors – `input` keeps its own memory reference.
      tf.dispose([extra, tiled]);
    }

    // 2. Create or reuse neural network model (always with max output nodes).
    const maxOutNodes = getMaxOutNodes();
    const totalInputDim = 5 + extraDim;
    const modelConfig = {
      layers: opts.layers,
      activation: opts.activation,
      inputDim: totalInputDim
    };
    
    let model;
    if (opts.forceNewNetwork || !cachedModel || !cachedModelConfig || 
        JSON.stringify(cachedModelConfig) !== JSON.stringify(modelConfig)) {
      // Create new model with maximum output nodes
      model = createModel(opts.layers, opts.activation, maxOutNodes, totalInputDim);
      cachedModel = model;
      cachedModelConfig = modelConfig;
    } else {
      // Reuse cached model
      model = cachedModel;
    }

    // 3. Forward pass.
    const prediction = model.predict(input, { batchSize: 2048 });
    const raw = await prediction.data(); // Float32Array

    // 4. Map output channels based on color mode
    const channelMapping = getChannelMapping(opts.colorMode, opts.alpha);
    const mappedRaw = new Float32Array(width * height * channelMapping.channels.length);
    
    for (let i = 0; i < width * height; ++i) {
      for (let j = 0; j < channelMapping.channels.length; ++j) {
        mappedRaw[i * channelMapping.channels.length + j] = raw[i * maxOutNodes + channelMapping.channels[j]];
      }
    }

    // 5. Convert to RGBA - the input was flattened in row-major order, so output should match
    const pixels = transformColorsWithMapping(mappedRaw, raw, width, height, channelMapping, opts.colorMode, opts.alpha);

    // 6. Draw to canvas.
    const ctx = canvas.getContext('2d');
    const imgData = new ImageData(pixels, width, height);
    ctx.putImageData(imgData, 0, 0);

    // Clean up.
    tf.dispose([input, prediction]);
  }

  // Export public API.
  return {
    generate
  };

})(); 