/**
 * @file ssp.js
 * @brief Stand-alone JavaScript bundle for SSP similarity visualization.
 *
 * This file was automatically extracted from the inline <script> section
 * that originally resided in index.html.  No functional changes were made.
 * Moving the code here keeps index.html clean and allows for browser caching.
 *
 * Usage: ensure index.html includes
 *   <script src="js/ssp.js"></script>
 * just before the closing </body> tag (after Chart.js has loaded).
 *
 * The file intentionally attaches its symbols to the global scope so that
 * existing inline event‐handler attributes (e.g. onclick="generateNewSSP()")
 * continue to work without modification.
 */

'use strict';

// ---------------------------------------------------------------------------
// Global variables
// ---------------------------------------------------------------------------
let chart = null;
let isComputing = false;
let cachedSSP = null;
let cachedDimensions = null;
// Store the latest heatmap data for responsive redraws
let lastHeatmapMatrix = null;

// Global tracking variable for current exponent used by neural_art_ui.js
window.currentExponentMarker = null;

// ---------------------------------------------------------------------------
// Complex number helper class
// ---------------------------------------------------------------------------
class Complex {
  constructor(real, imag = 0) {
    this.real = real;
    this.imag = imag;
  }

  add(other) {
    return new Complex(this.real + other.real, this.imag + other.imag);
  }

  subtract(other) {
    return new Complex(this.real - other.real, this.imag - other.imag);
  }

  multiply(other) {
    if (typeof other === 'number') {
      return new Complex(this.real * other, this.imag * other);
    }
    return new Complex(
      this.real * other.real - this.imag * other.imag,
      this.real * other.imag + this.imag * other.real,
    );
  }

  conjugate() {
    return new Complex(this.real, -this.imag);
  }

  magnitude() {
    return Math.sqrt(this.real * this.real + this.imag * this.imag);
  }

  power(exponent) {
    if (exponent === 0) return new Complex(1, 0);
    if (exponent === 1) return new Complex(this.real, this.imag);

    const magnitude = this.magnitude();
    const phase = Math.atan2(this.imag, this.real);

    const newMagnitude = Math.pow(magnitude, exponent);
    const newPhase = phase * exponent;

    return new Complex(
      newMagnitude * Math.cos(newPhase),
      newMagnitude * Math.sin(newPhase),
    );
  }
}

// ---------------------------------------------------------------------------
// FFT & IFFT (recursive, Cooley-Tukey for power-of-two lengths)
// ---------------------------------------------------------------------------
function customFFT(x) {
  const N = x.length;
  if (N <= 1) return x;

  // Fallback to direct DFT when length is not power of two.
  if ((N & (N - 1)) !== 0) {
    return directDFT(x);
  }

  const even = [];
  const odd = [];

  for (let i = 0; i < N; i += 2) {
    even.push(x[i]);
    if (i + 1 < N) odd.push(x[i + 1]);
  }

  const evenFFT = customFFT(even);
  const oddFFT = customFFT(odd);

  const result = new Array(N);
  const halfN = N / 2;

  for (let k = 0; k < halfN; k++) {
    const t = new Complex(
      Math.cos((-2 * Math.PI * k) / N),
      Math.sin((-2 * Math.PI * k) / N),
    ).multiply(oddFFT[k]);
    result[k] = evenFFT[k].add(t);
    result[k + halfN] = evenFFT[k].subtract(t);
  }

  return result;
}

function directDFT(x) {
  const N = x.length;
  const result = new Array(N);

  for (let k = 0; k < N; k++) {
    result[k] = new Complex(0, 0);
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      const twiddle = new Complex(Math.cos(angle), Math.sin(angle));
      result[k] = result[k].add(x[n].multiply(twiddle));
    }
  }
  return result;
}

function customIFFT(x) {
  const N = x.length;
  const conjugated = x.map((c) => c.conjugate());
  const fftResult = customFFT(conjugated);
  return fftResult.map((c) => c.conjugate().multiply(1 / N));
}

// ---------------------------------------------------------------------------
// SSP helpers
// ---------------------------------------------------------------------------
function makeGoodUnitary(D, eps = 1e-3) {
  const halfD = Math.floor((D - 1) / 2);
  const a = new Array(halfD);
  const phi = new Array(halfD);

  for (let i = 0; i < halfD; i++) {
    a[i] = Math.random();
    const sign = Math.random() < 0.5 ? -1 : 1;
    phi[i] = sign * Math.PI * (eps + a[i] * (1 - 2 * eps));
  }

  const fv = new Array(D).fill(null).map(() => new Complex(0, 0));
  fv[0] = new Complex(1, 0);

  for (let i = 1; i <= halfD; i++) {
    fv[i] = new Complex(Math.cos(phi[i - 1]), Math.sin(phi[i - 1]));
  }
  for (let i = 1; i <= halfD; i++) {
    fv[D - i] = fv[i].conjugate();
  }
  if (D % 2 === 0) {
    fv[D / 2] = new Complex(1, 0);
  }

  const v = customIFFT(fv);
  return v.map((c) => c.real);
}

function powerSSP(ssp, exponent) {
  const sspComplex = ssp.map((x) => new Complex(x, 0));
  const sspFFT = customFFT(sspComplex);
  const sspPowered = sspFFT.map((c) => c.power(exponent));
  const sspPoweredReal = customIFFT(sspPowered);
  return sspPoweredReal.map((c) => c.real);
}

// ---------------------------------------------------------------------------
// Vector utilities
// ---------------------------------------------------------------------------
function dotProduct(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

function norm(v) {
  return Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
}

function cosineSimilarity(a, b) {
  return dotProduct(a, b) / (norm(a) * norm(b));
}

function linspace(start, end, num) {
  const result = [];
  const step = (end - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    result.push(start + i * step);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Core analysis routine
// ---------------------------------------------------------------------------
async function analyzeSSPSimilarity(N, D, minExp, maxExp, numPoints) {
  const exponents = linspace(minExp, maxExp, numPoints);

  // Generate N unitary SSPs only when dimension changes
  const ssps = [];
  if (cachedSSP === null || cachedDimensions !== D) {
    for (let i = 0; i < N; i++) {
      ssps.push(makeGoodUnitary(D));
    }
    cachedSSP = ssps[0];
    cachedDimensions = D;
  } else {
    ssps.push(cachedSSP);
  }

  const averageDotProducts = [];
  const averageCosineSimilarities = [];
  const sspNorms = [];
  const sspMeans = [];
  const sspStds = [];
  const sspMatrix = [];

  const normsOriginal = ssps.map((ssp) => norm(ssp));

  for (let i = 0; i < exponents.length; i++) {
    const exponent = exponents[i];

    const sspsPowered = ssps.map((ssp) => powerSSP(ssp, exponent));

    const dotProducts = [];
    const cosineSimilarities = [];
    const poweredNorms = [];
    const poweredMeans = [];
    const poweredStds = [];

    for (let j = 0; j < N; j++) {
      const dotProd = dotProduct(ssps[j], sspsPowered[j]);
      const normPowered = norm(sspsPowered[j]);
      const cosSim = dotProd / (normsOriginal[j] * normPowered);

      dotProducts.push(dotProd);
      cosineSimilarities.push(cosSim);
      poweredNorms.push(normPowered);
      poweredMeans.push(sspsPowered[j].reduce((s, v) => s + v, 0) / sspsPowered[j].length);
      poweredStds.push(
        Math.sqrt(
          sspsPowered[j].reduce((s, v) => s + (v - poweredMeans[j]) ** 2, 0) /
            sspsPowered[j].length,
        ),
      );
    }

    const avgDotProduct = dotProducts.reduce((s, v) => s + v, 0) / N;
    const avgCosineSimilarity = cosineSimilarities.reduce((s, v) => s + v, 0) / N;
    const avgNorm = poweredNorms.reduce((s, v) => s + v, 0) / N;
    const avgMean = poweredMeans.reduce((s, v) => s + v, 0) / N;
    const avgStd = poweredStds.reduce((s, v) => s + v, 0) / N;

    averageDotProducts.push(avgDotProduct);
    averageCosineSimilarities.push(avgCosineSimilarity);
    sspNorms.push(avgNorm);
    sspMeans.push(avgMean);
    sspStds.push(avgStd);
    sspMatrix.push(sspsPowered[0]);
  }

  return {
    exponents,
    dotProducts: averageDotProducts,
    cosineSimilarities: averageCosineSimilarities,
    sspNorms,
    sspMeans,
    sspStds,
    sspMatrix,
  };
}

// ---------------------------------------------------------------------------
// Chart.js initialisation
// ---------------------------------------------------------------------------
function initChart() {
  const ctx = document.getElementById('similarityChart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Dot Product',
          data: [],
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
        },
        {
          label: 'Cosine Similarity',
          data: [],
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
        },
        {
          label: 'Norm',
          data: [],
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          hidden: true,
        },
        {
          label: 'Mean',
          data: [],
          borderColor: 'rgb(255, 159, 64)',
          backgroundColor: 'rgba(255, 159, 64, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          hidden: true,
        },
        {
          label: 'Standard Deviation',
          data: [],
          borderColor: 'rgb(153, 102, 255)',
          backgroundColor: 'rgba(153, 102, 255, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          hidden: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      layout: { padding: { right: 20 } },
      plugins: {
        title: { display: true, text: 'SSP Analysis vs. Exponent' },
        legend: { display: true, position: 'top' },
      },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Exponent' },
          ticks: {
            autoSkip: false,
            callback(value, index, values) {
              const minExp = parseFloat(document.getElementById('exponentMin').value);
              const maxExp = parseFloat(document.getElementById('exponentMax').value);

              if (index === 0) return minExp.toFixed(1);
              if (index === values.length - 1) return maxExp.toFixed(1);

              const step = (maxExp - minExp) / (values.length - 1);

              if (0 > minExp && 0 < maxExp) {
                const nearestZeroIndex = Math.round((0 - minExp) / step);
                if (index === nearestZeroIndex) return '0';
              }
              if (1 > minExp && 1 < maxExp) {
                const nearestOneIndex = Math.round((1 - minExp) / step);
                if (index === nearestOneIndex) return '1';
              }
              return '';
            },
          },
        },
        y: { display: true, title: { display: true, text: 'Value' } },
      },
      interaction: { intersect: false, mode: 'index' },
    },
  });

  // Expose chart globally for other scripts (e.g., neural_art_ui.js)
  window.similarityChart = chart;
}

// ---------------------------------------------------------------------------
// Main update routine
// ---------------------------------------------------------------------------
async function updateVisualization() {
  if (isComputing) return;
  isComputing = true;

  const N = 1;
  const D = parseInt(document.getElementById('dimensions').value, 10);
  const minExp = parseFloat(document.getElementById('exponentMin').value);
  const maxExp = parseFloat(document.getElementById('exponentMax').value);
  const numPoints = parseInt(document.getElementById('numPoints').value, 10);

  try {
    const results = await analyzeSSPSimilarity(N, D, minExp, maxExp, numPoints);

    chart.data.labels = results.exponents.map((x) => x.toFixed(2));
    chart.data.datasets[0].data = results.dotProducts;
    chart.data.datasets[1].data = results.cosineSimilarities;
    chart.data.datasets[2].data = results.sspNorms;
    chart.data.datasets[3].data = results.sspMeans;
    chart.data.datasets[4].data = results.sspStds;
    chart.update();

    drawHeatmap(results.sspMatrix);
  } catch (err) {
    console.error('Error during analysis:', err);
    document.getElementById('status').innerHTML =
      '<span style="color: red;">Error during computation. Check console for details.</span>';
  }
  isComputing = false;
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function updateSliderValues() {
  document.getElementById('dimensionsValue').textContent = document.getElementById('dimensions').value;
  document.getElementById('exponentRangeValue').textContent =
    `${document.getElementById('exponentMin').value} to ${document.getElementById('exponentMax').value}`;
  document.getElementById('numPointsValue').textContent = document.getElementById('numPoints').value;
}

function handleRangeSlider() {
  const minSlider = document.getElementById('exponentMin');
  const maxSlider = document.getElementById('exponentMax');
  const rangeFill = document.getElementById('rangeFill');

  const minVal = parseInt(minSlider.value, 10);
  const maxVal = parseInt(maxSlider.value, 10);

  if (minVal > maxVal) {
    if (this === minSlider) {
      maxSlider.value = minVal;
    } else {
      minSlider.value = maxVal;
    }
  }

  const range = 200;
  const minPercent = ((minVal + 100) / range) * 100;
  const maxPercent = ((maxVal + 100) / range) * 100;

  rangeFill.style.left = `${minPercent}%`;
  rangeFill.style.width = `${maxPercent - minPercent}%`;

  updateSliderValues();
  debouncedUpdate();
}

function generateNewSSP() {
  cachedSSP = null;
  cachedDimensions = null;
  updateVisualization();
}

let updateTimeout = null;
function debouncedUpdate() {
  if (updateTimeout) clearTimeout(updateTimeout);
  updateTimeout = setTimeout(updateVisualization, 300);
}

// ---------------------------------------------------------------------------
// Heatmap drawing
// ---------------------------------------------------------------------------
function drawHeatmap(matrix) {
  const canvas = document.getElementById('heatmapCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const container = canvas.parentElement;
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;

  const numCols = matrix.length;
  const numRows = matrix[0].length;

  canvas.width = containerWidth;
  canvas.height = containerHeight;

  const leftPad = 60;
  const rightPad = 20;
  const cellWidth = (canvas.width - leftPad - rightPad) / numCols;
  const cellHeight = canvas.height / numRows;

  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let c = 0; c < numCols; c++) {
    for (let r = 0; r < numRows; r++) {
      const val = matrix[c][r];
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }
  }
  const range = maxVal - minVal || 1;

  for (let c = 0; c < numCols; c++) {
    for (let r = 0; r < numRows; r++) {
      const val = matrix[c][r];
      const t = (val - minVal) / range;
      const hue = (1 - t) * 240;
      ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
      ctx.fillRect(leftPad + c * cellWidth, r * cellHeight, cellWidth, cellHeight);
    }
  }
  lastHeatmapMatrix = matrix;
}

// ---------------------------------------------------------------------------
// Bootstrap everything on DOM ready
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  updateSliderValues();

  handleRangeSlider();

  const sliders = ['dimensions', 'numPoints'];
  sliders.forEach((id) => {
    document.getElementById(id).addEventListener('input', () => {
      updateSliderValues();
      debouncedUpdate();
    });
  });

  document.getElementById('exponentMin').addEventListener('input', handleRangeSlider);
  document.getElementById('exponentMax').addEventListener('input', handleRangeSlider);

  updateVisualization();

  window.addEventListener('resize', () => {
    if (lastHeatmapMatrix) drawHeatmap(lastHeatmapMatrix);
  });
});

// Chart.js plugin to draw vertical line at current exponent
const verticalLinePlugin = {
  id: 'verticalLine',
  afterDraw(chart) {
    const marker = window.currentExponentMarker;
    if (marker === null || marker === undefined) return;

    const xScale = chart.scales.x;
    if (!xScale) return;

    // Determine min and max exponents from current sliders (fallback to labels)
    let minExp = parseFloat(document.getElementById('exponentMin')?.value);
    let maxExp = parseFloat(document.getElementById('exponentMax')?.value);
    if (isNaN(minExp) || isNaN(maxExp)) {
      // Fallback: derive from labels
      const first = parseFloat(chart.data.labels[0]);
      const last = parseFloat(chart.data.labels[chart.data.labels.length - 1]);
      minExp = first;
      maxExp = last;
    }
    if (maxExp === minExp) return;

    const { left, right } = xScale;
    const ratio = (marker - minExp) / (maxExp - minExp);
    const xPixel = left + ratio * (right - left);

    const { ctx, chartArea: { top, bottom } } = chart;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(xPixel, top);
    ctx.lineTo(xPixel, bottom);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(0,0,0,0.7)';
    ctx.stroke();

    // Draw label below x-axis
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(marker.toFixed(2), xPixel, bottom + 4);
    ctx.restore();
  }
};

// Register plugin once Chart.js is available
if (typeof Chart !== 'undefined' && Chart.register) {
  Chart.register(verticalLinePlugin);
} 