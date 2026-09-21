let routePoints = [];
let leafletMap = null;
let marker = null;
let mainPolyline = null;
let passedPolyline = null;
let combinedChart = null;
let animIndex = 0;
let animTimer = null;
let isPlaying = false;  // ← добавлен явный флаг состояния
let startTimeMs = null;
let playbackSpeed = 1;
const BASE_INTERVAL_MS = 300;

// ─────────────────────────────────────────
// GAUGE
// ─────────────────────────────────────────
const GAUGE_V_MIN = 0;
const GAUGE_V_MAX = 90;
const GAUGE_CX = 100, GAUGE_CY = 108, GAUGE_R = 80;
const GAUGE_START_DEG = 210;

function degToRad(d) { return d * Math.PI / 180; }

function arcPath(cx, cy, r, startDeg, endDeg) {
  const s = degToRad(startDeg);
  const e = degToRad(endDeg);
  const x1 = cx + r * Math.cos(s);
  const y1 = cy + r * Math.sin(s);
  const x2 = cx + r * Math.cos(e);
  const y2 = cy + r * Math.sin(e);
  const large = (endDeg - startDeg) > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

function initGauge() {
  document.getElementById("gauge-bg")
    .setAttribute("d", arcPath(GAUGE_CX, GAUGE_CY, GAUGE_R, GAUGE_START_DEG, GAUGE_START_DEG + 240));
}

function updateGauge(speed) {
  const arc   = document.getElementById("gauge-arc");
  const label = document.getElementById("gauge-value");
  const pct   = Math.min(1, Math.max(0, (speed - GAUGE_V_MIN) / (GAUGE_V_MAX - GAUGE_V_MIN)));
  arc.setAttribute("d", arcPath(GAUGE_CX, GAUGE_CY, GAUGE_R, GAUGE_START_DEG, GAUGE_START_DEG + pct * 240));
  let color;
  if (speed <= 50)      color = "#437a22";
  else if (speed <= 65) color = "#d19900";
  else                  color = "#a13544";
  arc.setAttribute("stroke", color);
  label.textContent = speed.toFixed(1);
  label.setAttribute("fill", color);
}

// ─────────────────────────────────────────
// DATA LOADING
// ─────────────────────────────────────────
async function loadRoute() {
  const resp = await fetch("/api/route");
  if (!resp.ok) throw new Error("Не удалось загрузить маршрут: " + resp.statusText);
  const data = await resp.json();
  routePoints = data.points || [];
}

// ─────────────────────────────────────────
// MAP
// ─────────────────────────────────────────
function initMap() {
  if (!routePoints.length) return;
  leafletMap = L.map("map");
  const latlngs = routePoints.map(p => [p.lat, p.lon]);
  mainPolyline   = L.polyline(latlngs, { color: "#01696f" }).addTo(leafletMap);
  passedPolyline = L.polyline([], { color: "#da7101" }).addTo(leafletMap);
  leafletMap.fitBounds(mainPolyline.getBounds());
  marker = L.marker(latlngs[0]).addTo(leafletMap);
}

// ─────────────────────────────────────────
// CURSOR PLUGIN — declared BEFORE initCombinedChart
// ─────────────────────────────────────────
const cursorPlugin = {
  id: "cursor",
  afterDraw(chart) {
    if (animIndex <= 0 || !routePoints.length) return;
    const meta = chart.getDatasetMeta(0);
    if (!meta.data || !meta.data[animIndex]) return;
    const x   = meta.data[animIndex].x;
    const ctx  = chart.ctx;
    const yTop = chart.chartArea.top;
    const yBot = chart.chartArea.bottom;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, yTop);
    ctx.lineTo(x, yBot);
    ctx.strokeStyle = "rgba(161,53,68,0.8)";
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.restore();
  },
};

// ─────────────────────────────────────────
// COMBINED CHART
// ─────────────────────────────────────────
function initCombinedChart() {
  const ctx = document.getElementById("chart-combined").getContext("2d");

  const labels    = routePoints.map((_, i) => i);
  const elevData  = routePoints.map(p => p.elevation ?? 0);
  const gradeData = routePoints.map(p => p.grade ?? 0);

  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0,   "rgba(150,100,50,0.7)");
  gradient.addColorStop(0.5, "rgba(80,140,60,0.5)");
  gradient.addColorStop(1,   "rgba(80,140,60,0.05)");

  combinedChart = new Chart(ctx, {
    data: {
      labels,
      datasets: [
        {
          type: "line",
          label: "Высота, м",
          data: elevData,
          borderColor: "#6d5a3a",
          backgroundColor: gradient,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          yAxisID: "yElev",
          order: 2,
        },
        {
          type: "bar",
          label: "Уклон, %",
          data: gradeData,
          backgroundColor: gradeData.map(g => g >= 0 ? "rgba(67,122,34,0.7)" : "rgba(0,100,148,0.7)"),
          yAxisID: "yGrade",
          order: 1,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { display: false },
        yElev:  { position: "left",  title: { display: true, text: "м" }, grid: { drawOnChartArea: true  } },
        yGrade: { position: "right", title: { display: true, text: "%" }, grid: { drawOnChartArea: false } },
      },
      plugins: {
        legend: { display: true, position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } },
      },
    },
    plugins: [cursorPlugin],
  });
}

// ─────────────────────────────────────────
// SIDEBAR METRICS
// ─────────────────────────────────────────
function updateSidebar(point, elapsedSeconds) {
  document.getElementById("metric-time").textContent     = elapsedSeconds.toFixed(0) + " с";
  document.getElementById("metric-distance").textContent = ((point.distance ?? 0) / 1000).toFixed(2) + " км";
  document.getElementById("metric-grade").textContent    = (point.grade ?? 0).toFixed(2) + " %";
  updateGauge(point.speed_optimal ?? 0);
}

// ─────────────────────────────────────────
// ANIMATION
// ─────────────────────────────────────────
function applyIndex(idx) {
  if (idx < 0) idx = 0;
  if (idx >= routePoints.length) idx = routePoints.length - 1;
  animIndex = idx;

  const point = routePoints[animIndex];
  marker.setLatLng([point.lat, point.lon]);
  passedPolyline.setLatLngs(routePoints.slice(0, animIndex + 1).map(p => [p.lat, p.lon]));

  const elapsed = startTimeMs ? (performance.now() - startTimeMs) / 1000 : 0;
  updateSidebar(point, elapsed);
  document.getElementById("progress-slider").value = animIndex;
  combinedChart.update("none");
}

function stepAnimation() {
  if (animIndex >= routePoints.length - 1) { pauseAnim(); return; }
  applyIndex(animIndex + 1);
}

function pauseAnim() {
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  isPlaying = false;  // ← сбрасываем флаг
  document.getElementById("btn-playpause").textContent = "▶";
}

function startAnim() {
  if (animTimer) clearInterval(animTimer);
  animTimer = setInterval(stepAnimation, BASE_INTERVAL_MS / playbackSpeed);
  isPlaying = true;  // ← устанавливаем флаг
  document.getElementById("btn-playpause").textContent = "⏸";
}

// ─────────────────────────────────────────
// CONTROLS
// ─────────────────────────────────────────
function setupControls() {
  const slider = document.getElementById("progress-slider");
  slider.max   = routePoints.length - 1;

  document.getElementById("btn-playpause").addEventListener("click", () => {
    if (isPlaying) {           // ← проверяем флаг, а не animTimer
      pauseAnim();
    } else {
      if (animIndex >= routePoints.length - 1) animIndex = 0;
      if (!startTimeMs) startTimeMs = performance.now();
      startAnim();
    }
  });

  document.getElementById("btn-rewind").addEventListener("click", () => {
    pauseAnim(); startTimeMs = null; applyIndex(0);
  });
  document.getElementById("btn-back").addEventListener("click",    () => applyIndex(animIndex - 10));
  document.getElementById("btn-forward").addEventListener("click", () => applyIndex(animIndex + 10));

  slider.addEventListener("input", () => {
    pauseAnim();
    applyIndex(parseInt(slider.value, 10));
  });

  document.querySelectorAll(".speed-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      playbackSpeed = parseFloat(btn.dataset.speed);
      document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      if (isPlaying) { pauseAnim(); startAnim(); }  // ← тоже через флаг
    });
  });
}

// ─────────────────────────────────────────
// BOOTSTRAP
// ─────────────────────────────────────────
async function bootstrap() {
  try {
    await loadRoute();
  } catch (err) {
    console.error(err);
    alert("Ошибка загрузки маршрута. См. консоль браузера.");
    return;
  }
  if (!routePoints.length) { alert("Маршрут пустой."); return; }

  initGauge();
  initMap();
  initCombinedChart();
  setupControls();
  applyIndex(0);
}

document.addEventListener("DOMContentLoaded", bootstrap);
