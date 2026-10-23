const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const send = document.querySelector("#send");
const reindex = document.querySelector("#reindex");
const newChat = document.querySelector("#new-chat");
const userTemplate = document.querySelector("#user-template");
const assistantTemplate = document.querySelector("#assistant-template");
const fileCount = document.querySelector("#file-count");
const chunkCount = document.querySelector("#chunk-count");
const sources = document.querySelector("#sources");
const readyPill = document.querySelector("#ready-pill");
const questionBank = document.querySelector("#question-bank");
const questionFilter = document.querySelector("#question-filter");
const questionCount = document.querySelector("#question-count");
const dropZone = document.querySelector("#drop-zone");
const fileInput = document.querySelector("#file-input");
const uploadStatus = document.querySelector("#upload-status");
const citationModal = document.querySelector("#citation-modal");
const citationStores = new Map();
const pageConfig = document.body.dataset;
const apiPrefix = pageConfig.apiPrefix || "/api";
const welcomeMessage =
  pageConfig.welcomeMessage ||
  "Hi! Drop documents into the sidebar, then ask anything about the indexed content. Answers are synthesized from retrieved evidence with citations.";
const thinkingLabel = pageConfig.thinkingLabel || "your documents";
let answerCounter = 0;
let suggestedQuestions = [];
let busy = false;
let conversationHistory = [];

async function refreshStatus() {
  const response = await fetch(`${apiPrefix}/status`);
  const status = await response.json();
  renderStatus(status);
}

async function loadQuestions() {
  const response = await fetch(`${apiPrefix}/questions`);
  if (!response.ok) throw new Error("Question suggestions are unavailable. Please try refreshing.");
  const data = await response.json();
  document.body.dataset.questionGeneration = data.generation || "";
  document.body.dataset.questionNotice = data.notice || "";
  suggestedQuestions = shuffleArray(data.questions ?? []);
  renderQuestions();
}

function renderStatus(status) {
  fileCount.textContent = status.files ?? 0;
  chunkCount.textContent = status.chunks ?? 0;
  sources.innerHTML = "";
  for (const [index, source] of (status.sources ?? []).entries()) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `source-chip${index === 0 ? " active" : ""}`;
    item.innerHTML = `<span class="doc-icon">D</span><span>${escapeHtml(source)}</span>`;
    sources.appendChild(item);
  }
  readyPill.innerHTML = `<span></span>${status.ready ? "Ready" : "No index"}`;
  readyPill.classList.toggle("warn", !status.ready);
}

function addUserMessage(text) {
  const node = userTemplate.content.cloneNode(true);
  node.querySelector(".bubble").textContent = text;
  messages.appendChild(node);
  scrollToBottom();
}

function addAssistantMessage(answer, citations = [], model = "", mode = "", warning = "") {
  const node = assistantTemplate.content.cloneNode(true);
  const article = node.querySelector("article");
  const answerId = `answer-${++answerCounter}`;
  article.dataset.answerId = answerId;
  citationStores.set(answerId, citationMap(citations));
  node.querySelector(".answer").innerHTML = formatAnswer(answer, answerId);
  const list = node.querySelector(".citation-list");
  if (model || warning) {
    const meta = document.createElement("div");
    meta.className = warning ? "answer-meta warn" : "answer-meta";
    meta.textContent = warning ? `Fallback answer: ${warning}` : `Synthesized by ${model} (${mode})`;
    list.appendChild(meta);
  }
  for (const citation of citations) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "citation";
    const location = citation.page ? `page ${citation.page}` : citation.section;
    item.innerHTML = `
      <strong>[${citation.rank}] ${escapeHtml(citation.source)}</strong>
      <p>${escapeHtml(location || "")} - score ${citation.score}</p>
      <p>${escapeHtml(citation.preview)}</p>
    `;
    item.addEventListener("click", () => openCitation(citation));
    list.appendChild(item);
  }
  messages.appendChild(node);
  scrollToBottom();
}

function addThinkingMessage() {
  const node = assistantTemplate.content.cloneNode(true);
  const article = node.querySelector("article");
  const source = sources.querySelector(".source-chip.active span:last-child")?.textContent || thinkingLabel;
  article.dataset.thinking = "true";
  node.querySelector(".answer").innerHTML = `<span class="typing-dots"><i></i><i></i><i></i></span> Retrieving evidence from <strong>${escapeHtml(source)}</strong> ...`;
  messages.appendChild(node);
  scrollToBottom();
}

function removeThinkingMessage() {
  const thinking = messages.querySelector("[data-thinking='true']");
  if (thinking) thinking.remove();
}

function addCacheNotice(tokensSaved = 0) {
  const node = assistantTemplate.content.cloneNode(true);
  node.querySelector(".answer").innerHTML = `<p><strong>Served from cache.</strong> Saved about ${Number(tokensSaved || 0).toLocaleString()} tokens.</p>`;
  node.querySelector(".citation-list").remove();
  messages.appendChild(node);
  scrollToBottom();
}

async function ask(message) {
  if (busy) return;
  busy = true;
  newChat.disabled = true;
  addUserMessage(message);
  addThinkingMessage();
  send.disabled = true;
  try {
    const response = await fetch(`${apiPrefix}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, top_k: 8, history: conversationHistory.slice(-10) }),
    });
    if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(typeof error.detail === "string" ? error.detail : "Unable to answer right now. Please try again."); }
    const data = await response.json();
    conversationHistory.push({role:'user', content:message.slice(0,4000)}, {role:'assistant', content:String(data.answer || '').slice(0,4000)});
    conversationHistory = conversationHistory.slice(-10);
    removeThinkingMessage();
    addAssistantMessage(data.answer, data.citations, data.model, data.mode, data.warning);
    if (data.interpretation?.mode === 'llm' && data.retrieval_query !== message && data.mode !== 'clarification') {
      const notice = document.createElement('p'); notice.className = 'intent-notice';
      notice.textContent = `Understood as: ${data.retrieval_query}`;
      messages.lastElementChild.querySelector('.bubble').prepend(notice);
    }
    if (data.meaning_coverage?.limited) {
      const notice = document.createElement('p'); notice.className = 'answer-meta warn';
      notice.textContent = 'Many matching passages were found. This answer covers a limited set; narrow the entry or document to check the rest.';
      messages.lastElementChild.querySelector('.bubble').append(notice);
    }
    if (data.cache?.hit) addCacheNotice(data.cache.tokens_saved);
    renderStatus(data.status);
  } catch (error) {
    removeThinkingMessage();
    addAssistantMessage(`I could not complete the request: ${error.message}`, []);
  } finally {
    send.disabled = false;
    busy = false;
    newChat.disabled = false;
    input.focus();
  }
}

function resetChat() {
  conversationHistory = [];
  citationStores.clear();
  messages.innerHTML = `
    <article class="message assistant">
      <div class="avatar">AI</div>
      <div class="bubble">
        <p>${escapeHtml(welcomeMessage)}</p>
      </div>
    </article>
  `;
  input.value = "";
  input.style.height = "auto";
  input.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || busy) return;
  input.value = "";
  input.style.height = "auto";
  ask(message);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
});

questionFilter.addEventListener("input", renderQuestions);

questionBank.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (!button) return;
  input.value = button.dataset.question;
  input.focus();
  form.requestSubmit();
});

newChat.addEventListener("click", resetChat);

reindex.addEventListener("click", async () => {
  reindex.disabled = true;
  readyPill.innerHTML = "<span></span>Indexing";
  try {
    const response = await fetch(`${apiPrefix}/reindex`, { method: "POST" });
    if (!response.ok) throw new Error("Could not refresh the document index.");
    renderStatus(await response.json());
    await loadQuestions();
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    reindex.disabled = false;
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  if (event.dataTransfer.files.length) uploadFiles(event.dataTransfer.files);
});

async function uploadFiles(files) {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);
  uploadStatus.textContent = "Uploading and indexing...";
  try {
    const response = await fetch(`${apiPrefix}/upload`, { method: "POST", body: formData });
    if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
    const data = await response.json();
    renderStatus(data.status);
    await loadQuestions();
    const ok = data.uploaded.filter((item) => item.ok).length;
    const failed = data.uploaded.length - ok;
    uploadStatus.textContent = failed ? `${ok} uploaded, ${failed} skipped.` : `${ok} uploaded and indexed.`;
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    fileInput.value = "";
  }
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayUnicode(value) {
  // Decode only numeric character references as text, then escape before rendering.
  return String(value).replace(/&#(x[0-9a-f]+|[0-9]+);/gi, (entity, code) => {
    const point = code[0].toLowerCase() === 'x' ? parseInt(code.slice(1), 16) : parseInt(code, 10);
    return point > 0 && point <= 0x10ffff && !(point >= 0xd800 && point <= 0xdfff) ? String.fromCodePoint(point) : entity;
  }).normalize('NFC');
}

function formatAnswer(value, answerId = "") {
  const lines = displayUnicode(value).split(/\r?\n/);
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
      html.push(`<h3>${inlineFormat(heading[1], answerId)}</h3>`);
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

messages.addEventListener("click", (event) => {
  const target = event.target.closest("[data-cite]");
  if (!target) return;
  const answerId = target.dataset.answerId || target.closest("[data-answer-id]")?.dataset.answerId;
  const citation = citationStores.get(answerId)?.get(Number(target.dataset.cite));
  if (citation) openCitation(citation);
});

citationModal.addEventListener("click", (event) => {
  if (event.target.closest("[data-close-modal]")) closeCitation();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !citationModal.hidden) closeCitation();
});

function citationMap(citations) {
  return new Map(citations.map((citation) => [Number(citation.rank), citation]));
}

function openCitation(citation) {
  document.querySelector("#modal-kicker").textContent = `Reference [${citation.rank}]`;
  document.querySelector("#modal-title").textContent = citation.source;
  document.querySelector("#modal-location").textContent = citation.page ? `Page ${citation.page}` : citation.section || "";
  document.querySelector("#modal-text").textContent = displayUnicode(citation.text || citation.preview || "");
  citationModal._previousFocus = document.activeElement;
  citationModal.hidden = false;
  citationModal.querySelector(".modal-close").focus();
}

function closeCitation() {
  citationModal.hidden = true;
  citationModal._previousFocus?.focus();
}

refreshStatus().catch(() => {
  readyPill.innerHTML = "<span></span>Offline";
  readyPill.classList.add("warn");
});

loadQuestions().catch(() => {
  questionBank.innerHTML = '<p class="muted">Question ideas unavailable.</p>';
});

function renderQuestions() {
  const filter = questionFilter.value.trim().toLowerCase();
  const visible = suggestedQuestions
    .filter((item) => !filter || item.question.toLowerCase().includes(filter) || item.topic.toLowerCase().includes(filter))
    .slice(0, 100);
  questionCount.textContent = String(visible.length);
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
