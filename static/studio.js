// Workspace-specific interactions; no document content is stored in localStorage.
const initialWelcome = messages.innerHTML;
const baseRenderStatus = renderStatus;
renderStatus = function(status) {
  baseRenderStatus(status);
  readyPill.innerHTML = `<span></span>${status.ready ? 'Ready to explore' : 'Add your first document'}`;
  for (const chip of [...sources.children]) {
    const name = chip.querySelector('span:last-child').textContent;
    const row = document.createElement('div'); row.className = 'document-row';
    chip.replaceWith(row); row.append(chip);
    chip.title = name;
    const remove = document.createElement('button'); remove.type = 'button';
    remove.className = 'remove-document'; remove.textContent = '×'; remove.setAttribute('aria-label', `Remove ${name}`);
    remove.addEventListener('click', async () => {
      if (!confirm(`Remove "${name}" from this workspace?`)) return;
      remove.disabled = true;
      try {
        const response = await fetch(`${apiPrefix}/documents/${encodeURIComponent(name)}`, {method:'DELETE'});
        if (!response.ok) throw new Error('Could not remove document. Please try again.');
        resetChat(); renderStatus(await response.json()); await loadQuestions();
        uploadStatus.textContent = 'Document removed. Start a fresh conversation with your remaining files.';
      } catch(error) { uploadStatus.textContent = error.message; remove.disabled = false; }
    }); row.append(remove);
  }
};
const baseResetChat = resetChat;
resetChat = function() { baseResetChat(); messages.innerHTML = initialWelcome; bindWelcome(); showDocumentQuestions(); };
function bindWelcome() { messages.querySelectorAll('[data-prompt]').forEach(button => { button.onclick = () => { input.value = button.dataset.prompt; input.focus(); }; }); }
const baseAsk = ask;
ask = async function(message) { if (busy) return; messages.querySelector('.welcome')?.remove(); await baseAsk(message); };
function createUploadProgress(files) {
  uploadStatus.replaceChildren();
  const card = document.createElement('section'); card.className = 'upload-progress';
  card.setAttribute('aria-label', 'Document preparation progress');
  const title = document.createElement('strong'); title.textContent = `Preparing ${files.length} document${files.length === 1 ? '' : 's'}`;
  const detail = document.createElement('p'); detail.className = 'progress-detail';
  const steps = [['uploading','Uploading files'], ['extracting','Reading document text'], ['chunking','Creating passages'], ['embedding','Creating semantic embeddings'], ['indexing','Building FAISS index'], ['questions','Generating AI questions']];
  const rows = steps.map(([key, label]) => {
    const row = document.createElement('div'); row.className = 'progress-step';
    const icon = document.createElement('span'); icon.className = 'step-icon'; icon.textContent = '·';
    const text = document.createElement('span'); text.textContent = label;
    row.append(icon, text); return row;
  });
  const bar = document.createElement('progress'); bar.max = 100; bar.value = 0;
  bar.setAttribute('aria-label', 'File transfer progress');
  card.append(title, ...rows, bar, detail); uploadStatus.append(card);
  let current = -1;
  return {
    update(stage, message) {
      const next = steps.findIndex(([key]) => key === stage);
      if (next >= 0) current = Math.max(current, next);
      rows.forEach((row, i) => { row.className = `progress-step ${i < current ? 'done' : i === current ? 'active' : ''}`; row.firstChild.textContent = i < current ? '✓' : i === current ? '' : '·'; });
      detail.textContent = message;
      if (current > 0) { bar.value = 100; bar.hidden = true; }
    },
    transfer(percent) { bar.value = percent; detail.textContent = `Uploading files · ${percent}% transferred`; },
    finish(message, warning = false) {
      rows.forEach(row => { row.className = 'progress-step done'; row.firstChild.textContent = '✓'; });
      if (warning) { rows[5].className = 'progress-step caution'; rows[5].firstChild.textContent = '!'; }
      title.textContent = warning ? 'Documents ready · check suggestions' : 'Ready to explore';
      detail.textContent = message; bar.hidden = true;
    },
    fail(message) { card.classList.add('failed'); title.textContent = 'Preparation needs attention'; detail.textContent = message; rows.forEach(row => row.classList.remove('active')); bar.hidden = true; }
  };
}

function uploadWithProgress(data, progress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); let offset = 0, pending = '', result;
    xhr.open('POST', `${apiPrefix}/upload/stream`);
    xhr.upload.onprogress = event => { if (event.lengthComputable) progress.transfer(Math.round(event.loaded / event.total * 100)); };
    xhr.upload.onload = () => progress.update('extracting', 'Upload complete. Waiting for document processing...');
    function consume() {
      pending += xhr.responseText.slice(offset); offset = xhr.responseText.length;
      const lines = pending.split('\n'); pending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let event; try { event = JSON.parse(line); } catch { continue; }
        if (event.stage === 'heartbeat') continue;
        if (event.stage === 'error') { reject(new Error(event.detail)); continue; }
        if (event.result) result = event.result;
        else progress.update(event.stage, event.detail);
      }
    }
    xhr.onprogress = consume;
    xhr.onload = () => {
      consume();
      if (xhr.status < 200 || xhr.status >= 300) { let detail; try { detail = JSON.parse(xhr.responseText).detail; } catch {} reject(new Error(typeof detail === 'string' ? detail : 'Upload failed. Please try again.')); }
      else if (!result) reject(new Error('Processing did not complete. Please try again.'));
      else resolve(result);
    };
    xhr.onerror = () => reject(new Error('Connection interrupted. Refresh to check your documents before retrying.'));
    xhr.onabort = () => reject(new Error('Upload cancelled.'));
    xhr.send(data);
  });
}

uploadFiles = async function(files) {
  if (fileInput.disabled) return;
  const list = [...files];
  if (list.length > 10) { uploadStatus.textContent = 'Choose up to 10 files at a time.'; return; }
  if (list.some(file => file.size > 20*1024*1024 || file.size === 0)) { uploadStatus.textContent = 'Each file must be nonempty and no larger than 20 MB.'; return; }
  fileInput.disabled = true; reindex.disabled = true;
  const progress = createUploadProgress(list);
  progress.update('uploading', 'Transferring your files...');
  const data = new FormData(); list.forEach(file => data.append('files',file));
  let previousSources = null;
  try {
    try {
      const baseline = await fetch(`${apiPrefix}/status`);
      if (baseline.ok) previousSources = (await baseline.json()).sources;
    } catch {}
    const result = await uploadWithProgress(data, progress);
    renderStatus(result.status);
    const accepted = result.uploaded.filter(file => file.ok);
    const summary = result.uploaded.map(file => file.ok ? `${file.name}: ready` : `${file.name}: ${file.error}`).join('\n');
    if (!accepted.length) { progress.fail(summary); return; }
    progress.update('questions', 'AI is reading the indexed passages and creating useful questions. This may take a moment.');
    try {
      await loadQuestions();
      showDocumentQuestions(accepted.map(file => file.name), true);
      progress.finish(summary + (document.body.dataset.questionNotice ? '\n' + document.body.dataset.questionNotice : ''), document.body.dataset.questionGeneration !== 'llm');
    } catch(error) { progress.finish('Your documents are indexed and ready to chat. Question suggestions could not be loaded. Refresh to retry.', true); }
  } catch(error) {
    progress.update('indexing', 'Connection lost. Checking whether your documents finished processing…');
    let recovered = false;
    for (let attempt = 0; attempt < 3 && !recovered; attempt++) {
      try {
        const response = await fetch(`${apiPrefix}/status`);
        if (!response.ok) throw new Error('Status unavailable');
        const status = await response.json();
        renderStatus(status);
        // Verify every requested file, rather than using unrelated old suggestions.
        if (status.ready && previousSources && list.every(file => !previousSources.includes(file.name) && status.sources.includes(file.name))) {
          recovered = true;
          try { await loadQuestions(); showDocumentQuestions(list.map(file => file.name), true); } catch {}
          progress.finish('Documents verified in the saved FAISS index. You can start chatting.');
        }
      } catch {}
      if (!recovered && attempt < 2) await new Promise(resolve => setTimeout(resolve, 2000));
    }
    if (!recovered) progress.fail('The connection was interrupted and this upload could not be confirmed. Check your document list before retrying.');
  }
  finally { fileInput.disabled = false; reindex.disabled = false; fileInput.value = ''; }
};
// Keep focus inside the source dialog, and return it to the invoking control.
citationModal.addEventListener('keydown', event => { if (event.key === 'Tab') { event.preventDefault(); citationModal.querySelector('.modal-close').focus(); } });


function showDocumentQuestions(names = null, announce = false) {
  const ideas = suggestedQuestions.filter(item => !names || names.includes(item.source)).slice(0, 8);
  if (!ideas.length) { messages.querySelectorAll(".document-questions").forEach(node => node.remove()); return; }
  messages.querySelector('.welcome')?.remove();
  messages.querySelectorAll('.document-questions').forEach(node => node.remove());
  const panel = document.createElement('section');
  panel.className = 'document-questions';
  panel.setAttribute('aria-label', 'Suggested questions from your documents');
  const title = document.createElement('h3');
  title.textContent = announce ? 'Your documents are ready. What would you like to know?' : 'Explore your documents';
  panel.append(title);
  const description = document.createElement('p');
  description.textContent = (document.body.dataset.questionGeneration === 'llm' ? 'AI-generated questions based on your document. ' : '') + 'Choose a question to get an answer with supporting sources, or write your own below.' + (document.body.dataset.questionNotice ? ' ' + document.body.dataset.questionNotice : '');
  panel.append(description);
  for (const item of ideas) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'document-question';
    const label = document.createElement('strong'); label.textContent = item.question;
    const source = document.createElement('small'); source.textContent = item.source;
    button.append(label, source);
    button.addEventListener('click', async () => {
      if (busy) return;
      const buttons = panel.querySelectorAll('button'); buttons.forEach(b => b.disabled = true);
      try { await ask(item.question); }
      finally { buttons.forEach(b => b.disabled = false); }
    });
    panel.append(button);
  }
  messages.append(panel);
  if (announce) panel.scrollIntoView({block: "start", behavior: "smooth"});
}
const baseLoadQuestions = loadQuestions;
loadQuestions = async function() {
  await baseLoadQuestions();
  if (messages.querySelector('.welcome') || messages.querySelector('.document-questions')) showDocumentQuestions();
};
// The initial request may already be in flight when this script loads.
const baseRenderQuestions = renderQuestions;
renderQuestions = function() {
  baseRenderQuestions();
  if (messages.querySelector('.welcome') || messages.querySelector('.document-questions')) showDocumentQuestions();
};
// Handle cached/fast initial responses that completed before this script ran.
showDocumentQuestions();
