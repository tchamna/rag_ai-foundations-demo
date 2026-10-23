const form = document.querySelector("#compare-form");
const input = document.querySelector("#compare-input");
const send = document.querySelector("#compare-send");
const results = document.querySelector("#compare-results");
const citations = document.querySelector("#compare-citations");
const ready = document.querySelector("#compare-ready");
const citationModal = document.querySelector("#citation-modal");
const questionBank = document.querySelector("#question-bank");
const questionFilter = document.querySelector("#question-filter");
const shell = document.querySelector("[data-compare-endpoint]");
const compareEndpoint = shell?.dataset.compareEndpoint || "/api/compare";
const statusEndpoint = shell?.dataset.statusEndpoint || "/api/status";
const questionsEndpoint = shell?.dataset.questionsEndpoint || "/api/questions";
const loadingLabels = (shell?.dataset.loadingLabels || "OpenAI,Claude,Ollama").split(",");
const providers = (shell?.dataset.providers || "")
  .split(",")
  .map((provider) => provider.trim())
  .filter(Boolean);
const citationStores = new Map();
let sharedCitations = new Map();
let answerCounter = 0;
let suggestedQuestions = [];

async function loadStatus() {
  const response = await fetch(statusEndpoint);
  const status = await response.json();
  ready.textContent = status.ready ? "Ready" : "No index";
  ready.classList.toggle("warn", !status.ready);
}

async function loadQuestions() {
  if (!questionBank) return;
  let endpoint = questionsEndpoint;
  if (questionsEndpoint === '/api/generic/questions') {
    let batch = 0;
    try {
      batch = (Number(localStorage.getItem('rag-idea-batch')) || 0) % 8;
      localStorage.setItem('rag-idea-batch', String((batch + 1) % 8));
    } catch { batch = Math.floor(Math.random() * 8); }
    endpoint += `?batch=${batch}`;
  }
  const response = await fetch(endpoint);
  if (!response.ok) throw new Error('Could not load question ideas. Please try again.');
  const data = await response.json();
  suggestedQuestions = shuffleArray(data.questions ?? []);
  renderQuestions();
}

document.querySelector('#new-question-ideas')?.addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = 'Finding ideas…';
  try {
    if (questionFilter) questionFilter.value = '';
    await loadQuestions();
  } catch (error) {
    questionBank.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = 'New ideas';
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || send.disabled) return;

  send.disabled = true;
  results.innerHTML = "";
  citations.innerHTML = "";
  citationStores.clear();
  renderLoadingCards();

  try {
    const payload = { message, top_k: 8 };
    if (providers.length) payload.providers = providers;
    if (shell.dataset.openaiModels) payload.openai_models = (document.querySelector('#model-pair')?.value || shell.dataset.openaiModels).split(',');
    const data = await (shell.dataset.openaiModels ? postJson(compareEndpoint, payload) : postJsonWithRetry(compareEndpoint, payload));
    sharedCitations = citationMap(data.citations);
    renderResults(data.results);
    renderCacheNotice(data.cache);
    if (data.comparison_budget?.context_reduced) {
      const notice = document.createElement('p'); notice.className = 'compare-cache-notice';
      notice.textContent = `To stay within the $0.05 budget, all models received the same ${data.comparison_budget.included_passages} selected passage(s) out of ${data.comparison_budget.original_passages}.` + (data.comparison_budget.passage_clipped ? ' A long passage was shortened.' : '') + ' This comparison may not cover every meaning in the full document.';
      results.prepend(notice);
    }
    renderCitations(data.citations);
  } catch (error) {
    results.innerHTML = `<article class="compare-card error-card"><h3>Compare failed</h3><p>${escapeHtml(error.message)}</p></article>`;
  } finally {
    send.disabled = false;
  }
});

async function postJsonWithRetry(url, payload) {
  try {
    return await postJson(url, payload);
  } catch (error) {
    if (!String(error.message).includes("Failed to fetch")) throw error;
    await new Promise((resolve) => setTimeout(resolve, 700));
    return postJson(url, payload);
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(typeof detail.detail === 'string' ? detail.detail : `Request failed: ${response.status}`);
  }
  return response.json();
}

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionFilter?.addEventListener("input", renderQuestions);

questionBank?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (!button) return;
  input.value = button.dataset.question;
  input.focus();
  form.requestSubmit();
});

function renderLoadingCards() {
  for (const provider of (shell.dataset.openaiModels ? (document.querySelector('#model-pair')?.value || shell.dataset.openaiModels).split(',') : loadingLabels)) {
    const card = document.createElement("article");
    card.className = "compare-card";
    card.innerHTML = `<h3>${provider}</h3><p class="muted">Running retrieval-grounded synthesis...</p>`;
    results.appendChild(card);
  }
}

function renderResults(items) {
  results.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("article");
    const answerId = `compare-${++answerCounter}`;
    card.className = item.ok ? "compare-card" : "compare-card error-card";
    card.dataset.answerId = answerId;
    citationStores.set(answerId, sharedCitations);
    card.innerHTML = `
      <div class="model-head">
        <div>
          <p class="eyebrow">${escapeHtml(item.mode)}</p>
          <h3>${escapeHtml(item.model)}</h3>
          <p class="model-cost">${modelCostLabel(item)}</p>
        </div>
        <span>${formatSeconds(item.elapsed_seconds ?? item.elapsed_ms / 1000)}</span>
      </div>
      <div class="answer">${item.ok ? formatAnswer(item.answer, answerId) : escapeHtml(item.error)}</div>
    `;
    results.appendChild(card);
  }
}

function renderCacheNotice(cache) {
  if (!cache?.hit) return;
  const notice = document.createElement("article");
  notice.className = "compare-cache-notice";
  notice.textContent = `Served from cache. Saved about ${Number(cache.tokens_saved || 0).toLocaleString()} tokens.`;
  results.prepend(notice);
}

function renderCitations(items) {
  citations.innerHTML = "";
  for (const citation of items) {
    const item = document.createElement("div");
    item.className = "citation";
    const location = citation.page ? `page ${citation.page}` : citation.section;
    item.innerHTML = `
      <strong>[${citation.rank}] ${escapeHtml(citation.source)}</strong>
      <p>${escapeHtml(location || "")} - score ${citation.score}</p>
      <p>${escapeHtml(citation.preview)}</p>
    `;
    item.addEventListener("click", () => openCitation(citation));
    citations.appendChild(item);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatSeconds(value) {
  const seconds = Number(value || 0);
  return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
}

function modelCostLabel(item) {
  const pricing = item.pricing || {};
  const tokens = item.tokens || {};
  const inputPrice = formatMoney(pricing.input_per_million ?? 0);
  const outputPrice = formatMoney(pricing.output_per_million ?? 0);
  const inputTokens = formatInteger(tokens.input ?? 0);
  const outputTokens = formatInteger(tokens.output ?? 0);
  const estimated = tokens.estimated ? " est." : "";
  const cost = item.estimated_cost_usd == null ? "Cost unavailable" : `Estimated answer cost: $${Number(item.estimated_cost_usd).toFixed(6)}`;
  return `${cost} · ${inputPrice}/${outputPrice} per 1M · in: ${inputTokens}; out: ${outputTokens}${estimated}`;
}

function formatMoney(value) {
  const number = Number(value || 0);
  if (number === 0) return "$0";
  return `$${number.toFixed(number >= 1 ? 2 : 2).replace(/\.00$/, "")}`;
}

function formatInteger(value) {
  return Number(value || 0).toLocaleString();
}

function formatAnswer(value, answerId = "") {
  const lines = String(value).split(/\r?\n/);
  const html = [];
  let listOpen = false;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      if (listOpen) {
        html.push("</ul>");
        listOpen = false;
      }
      continue;
    }

    const heading = line.match(/^#{1,3}\s+(.+)$/);
    if (heading) {
      if (listOpen) {
        html.push("</ul>");
        listOpen = false;
      }
      html.push(`<h4>${inlineFormat(heading[1], answerId)}</h4>`);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${inlineFormat(bullet[1], answerId)}</li>`);
      continue;
    }

    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
    html.push(`<p>${inlineFormat(line, answerId)}</p>`);
  }

  if (listOpen) html.push("</ul>");
  return html.join("");
}

function inlineFormat(value, answerId = "") {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([0-9]+)\]/g, `<button class="cite-ref" type="button" data-answer-id="${answerId}" data-cite="$1">[$1]</button>`);
}

results.addEventListener("click", (event) => {
  const target = event.target.closest("[data-cite]");
  if (!target) return;
  const answerId = target.dataset.answerId || target.closest("[data-answer-id]")?.dataset.answerId;
  const citation = citationStores.get(answerId)?.get(Number(target.dataset.cite)) || sharedCitations.get(Number(target.dataset.cite));
  if (citation) {
    openCitation(citation);
  } else {
    openMissingCitation(target.dataset.cite);
  }
});

citationModal.addEventListener("click", (event) => {
  if (event.target.closest("[data-close-modal]")) closeCitation();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !citationModal.hidden) closeCitation();
});

function citationMap(items) {
  return new Map(items.map((citation) => [Number(citation.rank), citation]));
}

function openCitation(citation) {
  document.querySelector("#modal-kicker").textContent = `Reference [${citation.rank}]`;
  document.querySelector("#modal-title").textContent = citation.source;
  document.querySelector("#modal-location").textContent = citation.page ? `Page ${citation.page}` : citation.section || "";
  document.querySelector("#modal-text").textContent = citation.text || citation.preview || "";
  citationModal.hidden = false;
}

function openMissingCitation(rank) {
  document.querySelector("#modal-kicker").textContent = `Reference [${rank}]`;
  document.querySelector("#modal-title").textContent = "Citation unavailable";
  document.querySelector("#modal-location").textContent = "";
  document.querySelector("#modal-text").textContent =
    "This model cited a reference number that was not returned by retrieval. Try rerunning the comparison or increasing the retrieved context.";
  citationModal.hidden = false;
}

function closeCitation() {
  citationModal.hidden = true;
}

loadStatus().catch(() => {
  ready.textContent = "Offline";
  ready.classList.add("warn");
});

loadQuestions().catch(() => {
  if (questionBank) questionBank.innerHTML = '<p class="muted">Question ideas unavailable.</p>';
});

function renderQuestions() {
  const filter = questionFilter?.value.trim().toLowerCase() || "";
  const visible = suggestedQuestions
    .filter((item) => !filter || item.question.toLowerCase().includes(filter) || item.topic.toLowerCase().includes(filter))
    .slice(0, 24);
  questionBank.innerHTML = "";
  for (const item of visible) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "question-item";
    button.dataset.question = item.question;
    button.innerHTML = `
      <span>${escapeHtml(item.question)}</span>
      <small>${escapeHtml(item.source)}${item.page ? ` · page ${item.page}` : ""}</small>
    `;
    questionBank.appendChild(button);
  }
}

function shuffleArray(items) {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}
