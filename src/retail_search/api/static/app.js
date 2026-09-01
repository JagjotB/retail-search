const $ = (selector) => document.querySelector(selector);
const formatScore = (value) => Number(value).toFixed(4);

function resultRow(item) {
  const movement = item.rank_movement > 0 ? `<span class="movement">↑${item.rank_movement}</span>` : "";
  return `<li class="result-item"><span class="rank">${String(item.rank).padStart(2, "0")}</span><div><div class="product-title">${escapeHtml(item.title)}</div><div class="product-meta">${escapeHtml(item.brand || "Unknown brand")} · ${escapeHtml(item.product_id)} ${movement}</div><div class="product-scores">dense ${formatScore(item.retrieval_score)} · ranker ${formatScore(item.reranker_score)}</div></div><span class="label label-${item.relevance_label}">${item.relevance_label}</span></li>`;
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

async function loadComparison(queryId) {
  const response = await fetch(`/compare/${encodeURIComponent(queryId)}`);
  if (!response.ok) throw new Error("Unable to load benchmark comparison");
  const data = await response.json();
  $("#active-query").textContent = `“${data.query}”`;
  $("#baseline-ndcg").textContent = formatScore(data.baseline_ndcg_at_10);
  $("#reranker-ndcg").textContent = formatScore(data.reranker_ndcg_at_10);
  const delta = (data.relative_gain * 100).toFixed(1);
  $("#query-delta").textContent = `${delta >= 0 ? "+" : ""}${delta}% NDCG`;
  $("#baseline-results").innerHTML = data.baseline.map(resultRow).join("");
  $("#reranked-results").innerHTML = data.reranked.map(resultRow).join("");
}

async function initializeQueries() {
  const response = await fetch("/demo/queries");
  const queries = await response.json();
  const select = $("#query-select");
  select.innerHTML = queries.map((item) => `<option value="${escapeHtml(item.query_id)}">${escapeHtml(item.query)}</option>`).join("");
  if (queries.length) await loadComparison(queries[0].query_id);
  select.addEventListener("change", () => loadComparison(select.value));
}

async function runSearch(query) {
  if (!query) return;
  $("#search-results").innerHTML = "<p>Ranking candidates…</p>";
  const response = await fetch("/search", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({query, top_k: 6})});
  const data = await response.json();
  if (!response.ok) { $("#search-results").textContent = data.detail || "Search failed"; return; }
  $("#search-timing").textContent = `${data.timing_ms.retrieval.toFixed(1)} ms retrieval · ${data.timing_ms.reranking.toFixed(1)} ms reranking · ${data.timing_ms.total.toFixed(1)} ms total`;
  $("#search-results").innerHTML = data.results.map((item) => `<article class="search-product"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.brand || "Unknown brand")} · ${escapeHtml(item.product_id)}</small><small>#${item.rank} · dense ${formatScore(item.retrieval_score)} · ranker ${formatScore(item.reranker_score)} · ${item.rank_movement >= 0 ? "+" : ""}${item.rank_movement} positions</small></article>`).join("");
}

$("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await runSearch($("#search-input").value.trim());
});

async function initializeDemo() {
  await initializeQueries();

  const initialQuery = new URLSearchParams(window.location.search).get("q")?.trim();
  if (initialQuery) {
    $("#search-input").value = initialQuery;
    await runSearch(initialQuery);
  }

  const anchorId = window.location.hash.slice(1);
  if (anchorId) document.getElementById(anchorId)?.scrollIntoView();
}

initializeDemo().catch((error) => {
  $("#active-query").textContent = error.message;
  $("#search-results").textContent = error.message;
});
