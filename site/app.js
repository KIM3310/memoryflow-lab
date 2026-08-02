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
