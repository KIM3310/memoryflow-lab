const fmt = (value, digits = 2) => Number(value).toLocaleString("en-US", {
  minimumFractionDigits: digits,
  maximumFractionDigits: digits,
});

const shortPolicy = (name) => {
  if (name.includes("HBM")) return "HBM only";
  if (name.includes("stress")) return "Near-memory stress";
  if (name.includes("Near")) return "Near-memory";
  return "CXL tiered";
};

const renderSelected = (result, allResults) => {
  document.querySelector("#metric-feasible").textContent = result.feasible ? "Feasible" : "Rejected";
  document.querySelector("#metric-feasible").className = result.feasible ? "yes" : "no";
  document.querySelector("#metric-reason").textContent = result.rejection_reason || result.bottleneck;
  document.querySelector("#metric-latency").textContent = result.feasible ? `${fmt(result.mean_decode_latency_ms)} ms` : "Capacity fail";
  document.querySelector("#metric-throughput").textContent = result.feasible ? fmt(result.throughput_tokens_s) : "0";
  document.querySelector("#metric-traffic").textContent = result.feasible ? `${fmt(result.total_remote_read_gib)} GiB` : "0 GiB";

  const maximums = {
    latency: Math.max(...allResults.map((item) => item.mean_decode_latency_ms), 1),
    throughput: Math.max(...allResults.map((item) => item.throughput_tokens_s), 1),
    traffic: Math.max(...allResults.map((item) => item.total_remote_read_gib), 1),
  };
  const rows = [
    ["Mean decode latency", result.mean_decode_latency_ms, maximums.latency, "ms"],
    ["Throughput", result.throughput_tokens_s, maximums.throughput, "token/s"],
    ["Remote traffic", result.total_remote_read_gib, maximums.traffic, "GiB"],
  ];
  document.querySelector("#comparison-chart").innerHTML = rows.map(([label, value, max, unit]) => `
    <div class="bar-row">
      <span class="bar-label">${label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max((value / max) * 100, value ? 2 : 0)}%"></div></div>
      <span class="bar-value">${fmt(value)} ${unit}</span>
    </div>
  `).join("");
};

const renderCopyMeasurement = (measurement) => {
  const baseline = measurement.models.bandwidth_only;
  const affine = measurement.models.affine;
  const baselinePoints = new Map(
    baseline.validation.points.map((point) => [point.size_bytes, point]),
  );

  document.querySelector("#copy-baseline-mape").textContent =
    `${fmt(baseline.validation.mean_absolute_percentage_error_pct)}%`;
  document.querySelector("#copy-affine-mape").textContent =
    `${fmt(affine.validation.mean_absolute_percentage_error_pct)}%`;
  document.querySelector("#copy-bandwidth").textContent =
    `${fmt(affine.bandwidth_gbps)} GB/s`;
  document.querySelector("#copy-base-latency").textContent =
    `${fmt(affine.base_latency_us)} µs`;
  document.querySelector("#copy-table").innerHTML = affine.validation.points.map((point) => {
    const baselinePoint = baselinePoints.get(point.size_bytes);
    return `
      <tr>
        <td>${fmt(point.size_bytes / (1024 ** 2), 0)} MiB</td>
        <td>${fmt(point.measured_ms, 4)} ms</td>
        <td>${fmt(baselinePoint.predicted_ms, 4)} ms</td>
        <td>${fmt(point.predicted_ms, 4)} ms</td>
        <td>${fmt(point.absolute_percentage_error_pct)}%</td>
      </tr>
    `;
  }).join("");
  document.querySelector("#copy-boundary").textContent =
    `${measurement.scope.measured}. Not measured: ${measurement.scope.not_measured}.`;
};

const renderAttentionMeasurement = (measurement) => {
  const environment = measurement.environment;
  const roofline = measurement.models.independent_roofline;
  const calibrated = measurement.models.attention_affine;
  const rooflinePoints = new Map(
    roofline.validation.points.map((point) => [point.context_tokens, point]),
  );

  document.querySelector("#measurement-device").textContent =
    `${environment.device_label} · ${environment.device_backend.toUpperCase()} · PyTorch ${environment.torch_version}`;
  document.querySelector("#attention-roofline-mape").textContent =
    `${fmt(roofline.validation.mean_absolute_percentage_error_pct)}%`;
  document.querySelector("#attention-affine-mape").textContent =
    `${fmt(calibrated.validation.mean_absolute_percentage_error_pct)}%`;
  document.querySelector("#attention-max-error").textContent =
    `${fmt(calibrated.validation.max_absolute_percentage_error_pct)}%`;
  document.querySelector("#attention-effective-rate").textContent =
    `${fmt(calibrated.effective_stream_gbps)} GB/s`;
  document.querySelector("#attention-table").innerHTML = calibrated.validation.points.map((point) => {
    const rooflinePoint = rooflinePoints.get(point.context_tokens);
    return `
      <tr>
        <td>${fmt(point.context_tokens, 0)} tokens</td>
        <td>${fmt(point.measured_ms, 4)} ms</td>
        <td>${fmt(rooflinePoint.predicted_ms, 4)} ms</td>
        <td>${fmt(point.predicted_ms, 4)} ms</td>
        <td>${fmt(point.absolute_percentage_error_pct)}%</td>
      </tr>
    `;
  }).join("");
  document.querySelector("#attention-boundary").textContent =
    `${measurement.scope.measured}. Not measured: ${measurement.scope.not_measured}.`;
};

const render = (payload) => {
  const results = payload.results;
  const selector = document.querySelector("#policy-selector");
  selector.innerHTML = results.map((result, index) => `
    <button type="button" role="tab" aria-selected="${index === 1}" data-index="${index}">${shortPolicy(result.policy_name)}</button>
  `).join("");
  selector.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    selector.querySelectorAll("button").forEach((item) => item.setAttribute("aria-selected", "false"));
    button.setAttribute("aria-selected", "true");
    renderSelected(results[Number(button.dataset.index)], results);
  });

  document.querySelector("#results-table").innerHTML = results.map((result) => `
    <tr>
      <td>${result.policy_name}</td>
      <td class="${result.feasible ? "yes" : "no"}">${result.feasible ? "Yes" : "No"}</td>
      <td>${result.feasible ? `${fmt(result.mean_decode_latency_ms)} ms` : "-"}</td>
      <td>${result.feasible ? fmt(result.throughput_tokens_s) : "-"}</td>
      <td>${result.feasible ? `${fmt(result.total_remote_read_gib)} GiB` : "-"}</td>
      <td>${result.bottleneck}</td>
    </tr>
  `).join("");
  document.querySelector("#disclaimer").textContent = payload.disclaimer;
  document.querySelector("#scenario-hash").textContent = `Scenario ${payload.scenario_sha256.slice(0, 12)}`;
  renderAttentionMeasurement(payload.attention_measurement);
  renderCopyMeasurement(payload.measurement);
  renderSelected(results[1], results);
};

fetch("./results.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(render)
  .catch((error) => {
    document.querySelector("#metric-feasible").textContent = "Evidence unavailable";
    document.querySelector("#metric-reason").textContent = error.message;
  });
