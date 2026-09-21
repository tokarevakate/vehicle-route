let routePoints = [];
let leafletMap = null;
let marker = null;
let mainPolyline = null;
let passedPolyline = null;
let chartElevation = null;
let chartGrade = null;
let animIndex = 0;
let animTimer = null;
let startTimeMs = null;

// Simulated current speed with smoothed noise around optimal
let currentSpeedSimulated = null;

/** Return a simulated "actual" speed: optimal ± random noise clamped to [v-8, v+8]. */
function simulateCurrentSpeed(optimalSpeed) {
  if (currentSpeedSimulated === null) {
    currentSpeedSimulated = optimalSpeed;
  }
  // Random walk: nudge toward optimal, add small noise
  const noise = (Math.random() - 0.5) * 4.0;       // ±2 km/h per step
  const pull  = (optimalSpeed - currentSpeedSimulated) * 0.15; // drift toward optimal
  currentSpeedSimulated = currentSpeedSimulated + pull + noise;
  // Clamp to ±8 km/h around optimal
  const lo = Math.max(20, optimalSpeed - 8);
  const hi = optimalSpeed + 8;
  currentSpeedSimulated = Math.max(lo, Math.min(hi, currentSpeedSimulated));
  return currentSpeedSimulated;
}

async function loadRoute() {
  const resp = await fetch("/api/route");
  if (!resp.ok) {
    throw new Error("Не удалось загрузить маршрут: " + resp.statusText);
  }
  const data = await resp.json();
  routePoints = data.points || [];
}

function initMap() {
  if (!routePoints.length) return;

  leafletMap = L.map("map");

  const latlngs = routePoints.map((p) => [p.lat, p.lon]);
  mainPolyline = L.polyline(latlngs, { color: "#01696f" }).addTo(leafletMap);
  passedPolyline = L.polyline([], { color: "#da7101" }).addTo(leafletMap);

  leafletMap.fitBounds(mainPolyline.getBounds());

  marker = L.marker(latlngs[0]).addTo(leafletMap);
}

function initCharts() {
  const ctxElev = document
    .getElementById("chart-elevation")
    .getContext("2d");
  const ctxGrade = document.getElementById("chart-grade").getContext("2d");

  chartElevation = new Chart(ctxElev, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Высота, м",
          data: [],
          borderColor: "#006494",
          tension: 0.25,
          pointRadius: 0,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      scales: {
        x: { display: false },
        y: { title: { display: true, text: "м" } },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });

  chartGrade = new Chart(ctxGrade, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Уклон, %",
          data: [],
          borderColor: "#a13544",
          tension: 0.25,
          pointRadius: 0,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      scales: {
        x: { display: false },
        y: { title: { display: true, text: "%" } },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

function updateSidebar(point, elapsedSeconds) {
  const speedSpan    = document.getElementById("metric-speed");
  const timeSpan     = document.getElementById("metric-time");
  const distSpan     = document.getElementById("metric-distance");
  const gradeSpan    = document.getElementById("metric-grade");
  const optSpeedSpan = document.getElementById("optimal-speed-value");

  const optSpeed = point.speed_optimal ?? 0;
  const curSpeed = simulateCurrentSpeed(optSpeed);

  const distKm = (point.distance ?? 0) / 1000.0;
  const grade  = point.grade ?? 0;

  // Current (simulated) speed
  speedSpan.textContent = curSpeed.toFixed(1) + " км/ч";
  timeSpan.textContent  = elapsedSeconds.toFixed(0) + " с";
  distSpan.textContent  = distKm.toFixed(2) + " км";
  gradeSpan.textContent = grade.toFixed(2) + " %";

  // Optimal speed from model
  optSpeedSpan.textContent = optSpeed.toFixed(1);

  // Visual hint: color the current speed red/green vs optimal
  const diff = curSpeed - optSpeed;
  if (Math.abs(diff) <= 2) {
    speedSpan.style.color = "var(--color-success, #437a22)";
  } else if (diff > 2) {
    speedSpan.style.color = "var(--color-notification, #a13544)";
  } else {
    speedSpan.style.color = "var(--color-warning, #964219)";
  }
}

function updateCharts(point) {
  chartElevation.data.labels.push("");
  chartElevation.data.datasets[0].data.push(point.elevation ?? 0);
  chartElevation.update();

  chartGrade.data.labels.push("");
  chartGrade.data.datasets[0].data.push(point.grade ?? 0);
  chartGrade.update();
}

function resetCharts() {
  chartElevation.data.labels = [];
  chartElevation.data.datasets[0].data = [];
  chartElevation.update();

  chartGrade.data.labels = [];
  chartGrade.data.datasets[0].data = [];
  chartGrade.update();
}

function stepAnimation() {
  if (animIndex >= routePoints.length) {
    clearInterval(animTimer);
    animTimer = null;
    return;
  }

  const point  = routePoints[animIndex];
  const latlng = [point.lat, point.lon];

  marker.setLatLng(latlng);

  const passedLatLngs = routePoints
    .slice(0, animIndex + 1)
    .map((p) => [p.lat, p.lon]);
  passedPolyline.setLatLngs(passedLatLngs);

  const now            = performance.now();
  const elapsedSeconds = (now - startTimeMs) / 1000.0;

  updateSidebar(point, elapsedSeconds);
  updateCharts(point);

  animIndex += 1;
}

function setupControls() {
  const btnStart = document.getElementById("btn-start");
  const btnPause = document.getElementById("btn-pause");

  btnStart.addEventListener("click", () => {
    if (!routePoints.length) return;
    if (animTimer) return; // already running

    animIndex = 0;
    currentSpeedSimulated = null; // reset simulation
    resetCharts();
    startTimeMs = performance.now();
    animTimer = setInterval(stepAnimation, 300); // 0.3s per point
  });

  btnPause.addEventListener("click", () => {
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }
  });
}

async function bootstrap() {
  try {
    await loadRoute();
  } catch (err) {
    console.error(err);
    alert("Ошибка загрузки маршрута. См. консоль браузера.");
    return;
  }

  if (!routePoints.length) {
    alert("Маршрут пустой или данные не найдены.");
    return;
  }

  initMap();
  initCharts();
  setupControls();

  // Инициализировать значения на панели для первой точки
  updateSidebar(routePoints[0], 0);
}

document.addEventListener("DOMContentLoaded", bootstrap);
