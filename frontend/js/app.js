// frontend/js/app.js
// Stylish frontend with Chart.js semicircle gauge
const FEATURES = ["PM2.5","PM10","NO2","SO2","CO","O3","NH3"];
const PREDICT_URL = "/api/predict";
const METRICS_URL = "/api/metrics";

function buildForm() {
  const container = document.getElementById("inputs");
  container.innerHTML = "";
  FEATURES.forEach(f => {
    const div = document.createElement("div");
    div.className = "input-row";
    div.innerHTML = `<label for="${f}">${f}</label><input id="${f}" type="number" step="any" value="0">`;
    container.appendChild(div);
  });
}

async function fetchMetrics() {
  try {
    const res = await fetch(METRICS_URL);
    if (!res.ok) throw new Error("No metrics");
    const m = await res.json();
    document.getElementById("metricsText").innerText = JSON.stringify(m, null, 2);
  } catch (e) {
    document.getElementById("metricsText").innerText = "Metrics not available. Train the model first.";
  }
}

let gaugeChart = null;
function createGauge() {
  const ctx = document.getElementById('gaugeChart').getContext('2d');

  // initial with zero
  const data = {
    labels: ['AQI', 'empty'],
    datasets: [{
      data: [0, 500],
      backgroundColor: ['#fbbf24', '#eee'],
      borderWidth: 0,
      cutout: '70%'
    }]
  };

  const options = {
    rotation: -Math.PI,
    circumference: Math.PI,
    plugins: { tooltip: { enabled:false }, legend:{ display:false } },
    animation: { animateRotate: true, animateScale: false },
  };

  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data,
    options
  });
}

function updateGauge(aqi) {
  if (!gaugeChart) return;
  const value = Math.max(0, Math.min(aqi, 500));
  gaugeChart.data.datasets[0].data[0] = value;
  gaugeChart.data.datasets[0].data[1] = 500 - value;

  // pick color depending on bucket
  let color = '#16a34a';
  if (value <= 50) color = '#16a34a';
  else if (value <= 100) color = '#84cc16';
  else if (value <= 200) color = '#fbbf24';
  else if (value <= 300) color = '#fb923c';
  else if (value <= 400) color = '#ef4444';
  else color = '#7f1d1d';

  gaugeChart.data.datasets[0].backgroundColor[0] = color;
  gaugeChart.update();
}

function showResult(res) {
  document.getElementById("aqiValue").innerText = res.aqi_value;
  document.getElementById("aqiCategory").innerText = res.aqi_bucket;
  document.getElementById("aqiHealth").innerText = res.health_message || "-";
  updateGauge(res.aqi_value);
}

async function predict() {
  const payload = {};
  FEATURES.forEach(f => {
    const el = document.getElementById(f);
    payload[f] = el ? parseFloat(el.value || 0) : 0;
  });

  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.innerText = "Predicting...";

  try {
    const r = await fetch(PREDICT_URL, {
      method: "POST", headers: { "Content-Type":"application/json" }, body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (data.error) alert(data.error);
    else showResult(data);
  } catch (e) {
    alert("Prediction request failed. Check server console.");
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.innerText = "Predict";
  }
}

function fillSample(values) {
  Object.keys(values).forEach(k => {
    const el = document.getElementById(k);
    if (el) el.value = values[k];
  });
}

// presets
const PRESETS = {
  good: {"PM2.5":10,"PM10":20,"NO2":15,"SO2":5,"CO":0.2,"O3":30,"NH3":10},
  satisfactory: {"PM2.5":40,"PM10":60,"NO2":25,"SO2":10,"CO":0.5,"O3":60,"NH3":12},
  moderate: {"PM2.5":120,"PM10":160,"NO2":50,"SO2":20,"CO":1.2,"O3":80,"NH3":20},
  poor: {"PM2.5":220,"PM10":280,"NO2":90,"SO2":40,"CO":1.5,"O3":100,"NH3":30},
  verypoor: {"PM2.5":330,"PM10":400,"NO2":130,"SO2":60,"CO":2.0,"O3":130,"NH3":40}
};

document.addEventListener("DOMContentLoaded", () => {
  buildForm();
  createGauge();
  fetchMetrics();

  document.getElementById("predictBtn").addEventListener("click", predict);
  document.getElementById("fillSampleGood").addEventListener("click", () => fillSample(PRESETS.good));
  document.getElementById("fillSampleSat").addEventListener("click", () => fillSample(PRESETS.satisfactory));
  document.getElementById("fillSampleMod").addEventListener("click", () => fillSample(PRESETS.moderate));
  document.getElementById("fillSamplePoor").addEventListener("click", () => fillSample(PRESETS.poor));
  document.getElementById("fillSampleVP").addEventListener("click", () => fillSample(PRESETS.verypoor));
  document.getElementById("refreshMetrics").addEventListener("click", fetchMetrics);
});
