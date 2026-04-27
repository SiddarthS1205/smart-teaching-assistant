/**
 * Smart Teaching Assistant — Frontend Application
 * Handles: auth guard, file upload, chat, summary, logs modal, status polling
 */

const API_BASE = "http://127.0.0.1:8000";

// ── Auth guard ────────────────────────────────────────────────────────────────
const currentUser = sessionStorage.getItem("sta_user");
if (!currentUser) {
  window.location.href = "index.html";
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadArea      = document.getElementById("uploadArea");
const fileInput       = document.getElementById("fileInput");
const uploadProgress  = document.getElementById("uploadProgress");
const progressFill    = document.getElementById("progressFill");
const progressText    = document.getElementById("progressText");
const docInfo         = document.getElementById("docInfo");
const docName         = document.getElementById("docName");
const docMeta         = document.getElementById("docMeta");
const clearDocBtn     = document.getElementById("clearDocBtn");
const summaryBtn      = document.getElementById("summaryBtn");
const clearChatBtn    = document.getElementById("clearChatBtn");
const statusDot       = document.getElementById("statusDot");
const statusLabel     = document.getElementById("statusLabel");
const statusDoc       = document.getElementById("statusDoc");
const statusChunks    = document.getElementById("statusChunks");
const userName        = document.getElementById("userName");
const logoutBtn       = document.getElementById("logoutBtn");
const welcomeScreen   = document.getElementById("welcomeScreen");
const messages        = document.getElementById("messages");
const queryInput      = document.getElementById("queryInput");
const sendBtn         = document.getElementById("sendBtn");
const sendBtnIcon     = document.getElementById("sendBtnIcon");
const sendSpinner     = document.getElementById("sendSpinner");
const charCount       = document.getElementById("charCount");
const agentBadge      = document.getElementById("agentBadge");
const logsBtn         = document.getElementById("logsBtn");
const logsModal       = document.getElementById("logsModal");
const closeLogsBtn    = document.getElementById("closeLogsBtn");
const logsBody        = document.getElementById("logsBody");
const sidebar         = document.getElementById("sidebar");
const sidebarToggle   = document.getElementById("sidebarToggle");
const sidebarToggleMobile = document.getElementById("sidebarToggleMobile");

// ── State ─────────────────────────────────────────────────────────────────────
let documentLoaded = false;
let conversationHistory = [];

// ── Init ──────────────────────────────────────────────────────────────────────
userName.textContent = currentUser;
pollStatus();
setInterval(pollStatus, 15000);

// ── Sidebar toggle ────────────────────────────────────────────────────────────
[sidebarToggle, sidebarToggleMobile].forEach(btn => {
  btn.addEventListener("click", () => sidebar.classList.toggle("open"));
});

// Close sidebar on outside click (mobile)
document.addEventListener("click", (e) => {
  if (window.innerWidth <= 768 && sidebar.classList.contains("open")) {
    if (!sidebar.contains(e.target) && e.target !== sidebarToggleMobile) {
      sidebar.classList.remove("open");
    }
  }
});

// ── Logout ────────────────────────────────────────────────────────────────────
logoutBtn.addEventListener("click", () => {
  sessionStorage.clear();
  window.location.href = "index.html";
});

// ── Status polling ────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error("Server error");
    const data = await res.json();

    statusDot.className = "status-dot online";
    statusLabel.textContent = "Online";
    documentLoaded = data.document_loaded;

    if (data.document_loaded) {
      statusDoc.textContent = truncate(data.document_name, 20);
      statusChunks.textContent = data.chunk_count;
      summaryBtn.disabled = false;
      sendBtn.disabled = queryInput.value.trim().length < 3;
    } else {
      statusDoc.textContent = "None";
      statusChunks.textContent = "0";
      summaryBtn.disabled = true;
      sendBtn.disabled = true;
    }
  } catch {
    statusDot.className = "status-dot offline";
    statusLabel.textContent = "Offline";
  }
}

// ── File upload ───────────────────────────────────────────────────────────────
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
});

async function handleFileUpload(file) {
  // Client-side validation
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Only PDF files are accepted.", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast("File exceeds the 10 MB limit.", "error");
    return;
  }
  if (file.size === 0) {
    showToast("The selected file is empty.", "error");
    return;
  }

  // Show progress UI
  uploadArea.classList.add("hidden");
  docInfo.classList.add("hidden");
  uploadProgress.classList.remove("hidden");
  animateProgress(0, 30, 400);

  const formData = new FormData();
  formData.append("file", file);

  try {
    animateProgress(30, 70, 800);
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
    const data = await res.json();

    animateProgress(70, 100, 300);
    await sleep(400);

    if (res.ok && data.status === "success") {
      uploadProgress.classList.add("hidden");
      docInfo.classList.remove("hidden");
      docName.textContent = data.stats.filename;
      docMeta.textContent = `${data.stats.pages} pages · ${data.stats.chunks} chunks · ${data.stats.processing_time_ms}ms`;
      documentLoaded = true;
      summaryBtn.disabled = false;
      sendBtn.disabled = queryInput.value.trim().length < 3;
      showToast(`✅ "${data.stats.filename}" processed successfully!`, "success");
      addSystemMessage(`Document uploaded: **${data.stats.filename}** (${data.stats.pages} pages, ${data.stats.chunks} chunks). You can now ask questions!`);
      pollStatus();
    } else {
      uploadProgress.classList.add("hidden");
      uploadArea.classList.remove("hidden");
      showToast(data.error || "Upload failed. Please try again.", "error");
    }
  } catch (err) {
    uploadProgress.classList.add("hidden");
    uploadArea.classList.remove("hidden");
    showToast("Network error. Is the backend running?", "error");
  }

  // Reset file input
  fileInput.value = "";
}

function animateProgress(from, to, duration) {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const val = from + (to - from) * t;
    progressFill.style.width = val + "%";
    progressText.textContent = `Processing… ${Math.round(val)}%`;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Clear document ────────────────────────────────────────────────────────────
clearDocBtn.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/document`, { method: "DELETE" });
    docInfo.classList.add("hidden");
    uploadArea.classList.remove("hidden");
    documentLoaded = false;
    summaryBtn.disabled = true;
    sendBtn.disabled = true;
    pollStatus();
    addSystemMessage("Document removed. Upload a new PDF to continue.");
    showToast("Document cleared.", "success");
  } catch {
    showToast("Failed to clear document.", "error");
  }
});

// ── Chat input ────────────────────────────────────────────────────────────────
queryInput.addEventListener("input", () => {
  // Auto-resize
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";

  // Char count
  const len = queryInput.value.length;
  charCount.textContent = `${len} / 1000`;
  charCount.style.color = len > 900 ? "var(--warning)" : "var(--text-muted)";

  // Enable send only if document loaded and query long enough
  sendBtn.disabled = !documentLoaded || queryInput.value.trim().length < 3;
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query || query.length < 3) return;

  // Add user message
  addMessage("user", query);
  queryInput.value = "";
  queryInput.style.height = "auto";
  charCount.textContent = "0 / 1000";
  sendBtn.disabled = true;

  // Show typing indicator
  const typingId = addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/ask?q=${encodeURIComponent(query)}`);
    const data = await res.json();

    removeTypingIndicator(typingId);

    if (res.ok && data.status === "success") {
      addMessage("assistant", data.answer, {
        agent: data.agent_used,
        time: data.response_time_ms,
      });
      updateAgentBadge(data.agent_used);
    } else {
      addMessage("error", data.error || "An error occurred. Please try again.");
    }
  } catch {
    removeTypingIndicator(typingId);
    addMessage("error", "Network error. Please check your connection and try again.");
  } finally {
    sendBtn.disabled = !documentLoaded || queryInput.value.trim().length < 3;
  }
}

// ── Summary ───────────────────────────────────────────────────────────────────
summaryBtn.addEventListener("click", async () => {
  summaryBtn.disabled = true;
  summaryBtn.textContent = "⏳ Generating…";

  const typingId = addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/summary`);
    const data = await res.json();

    removeTypingIndicator(typingId);

    if (res.ok && data.status === "success") {
      addMessage("assistant", data.summary, {
        agent: "SummarizerAgent",
        time: data.response_time_ms,
      });
      updateAgentBadge("summarizer");
    } else {
      addMessage("error", data.error || "Summary generation failed.");
    }
  } catch {
    removeTypingIndicator(typingId);
    addMessage("error", "Network error during summary generation.");
  } finally {
    summaryBtn.disabled = false;
    summaryBtn.textContent = "📝 Generate Summary";
  }
});

// ── Clear chat ────────────────────────────────────────────────────────────────
clearChatBtn.addEventListener("click", () => {
  messages.innerHTML = "";
  conversationHistory = [];
  welcomeScreen.style.display = "flex";
  showToast("Chat cleared.", "success");
});

// ── Logs modal ────────────────────────────────────────────────────────────────
logsBtn.addEventListener("click", async () => {
  logsModal.classList.remove("hidden");
  logsBody.innerHTML = '<p class="loading-text">Loading logs…</p>';

  try {
    const res = await fetch(`${API_BASE}/logs`);
    const data = await res.json();

    if (data.queries.length === 0) {
      logsBody.innerHTML = '<p class="empty-text">No queries logged yet.</p>';
      return;
    }

    logsBody.innerHTML = data.queries
      .slice()
      .reverse()
      .map(entry => `
        <div class="log-entry">
          <div class="log-entry-header">
            <span class="log-agent">🤖 ${entry.agent}</span>
            <span class="log-time">${formatTimestamp(entry.timestamp)}</span>
          </div>
          <p class="log-query"><strong>Q:</strong> ${escapeHtml(entry.query)}</p>
          <p class="log-response"><strong>A:</strong> ${escapeHtml(entry.response_preview)}…</p>
          <p class="log-duration">⏱ ${entry.duration_ms}ms</p>
        </div>
      `)
      .join("");
  } catch {
    logsBody.innerHTML = '<p class="empty-text">Failed to load logs.</p>';
  }
});

closeLogsBtn.addEventListener("click", () => logsModal.classList.add("hidden"));
logsModal.addEventListener("click", (e) => {
  if (e.target === logsModal) logsModal.classList.add("hidden");
});

// ── Message helpers ───────────────────────────────────────────────────────────
function addMessage(role, text, meta = {}) {
  // Hide welcome screen on first message
  if (welcomeScreen.style.display !== "none") {
    welcomeScreen.style.display = "none";
  }

  const div = document.createElement("div");
  div.className = `message ${role}`;

  const avatar = role === "user" ? "👤" : role === "error" ? "⚠️" : "🤖";
  const agentTag = meta.agent
    ? `<span class="agent-tag">${formatAgentName(meta.agent)}</span>`
    : "";
  const timeTag = meta.time
    ? `<span>${meta.time}ms</span>`
    : `<span>${new Date().toLocaleTimeString()}</span>`;

  div.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-bubble">${escapeHtml(text)}</div>
      <div class="message-meta">${agentTag}${timeTag}</div>
    </div>
  `;

  messages.appendChild(div);
  scrollToBottom();
  conversationHistory.push({ role, text, meta });
}

function addSystemMessage(text) {
  if (welcomeScreen.style.display !== "none") {
    welcomeScreen.style.display = "none";
  }
  const div = document.createElement("div");
  div.className = "message assistant";
  div.innerHTML = `
    <div class="message-avatar">ℹ️</div>
    <div class="message-content">
      <div class="message-bubble" style="background:rgba(14,165,233,0.1);border-color:rgba(14,165,233,0.3);">${escapeHtml(text)}</div>
    </div>
  `;
  messages.appendChild(div);
  scrollToBottom();
}

let typingCounter = 0;
function addTypingIndicator() {
  const id = `typing-${++typingCounter}`;
  const div = document.createElement("div");
  div.className = "message assistant typing-indicator";
  div.id = id;
  div.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-content">
      <div class="message-bubble">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>
  `;
  messages.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function updateAgentBadge(agentName) {
  agentBadge.textContent = `🤖 ${formatAgentName(agentName)}`;
}

function formatAgentName(name) {
  const map = {
    qa: "QA Agent",
    summarizer: "Summarizer Agent",
    QAAgent: "QA Agent",
    SummarizerAgent: "Summarizer Agent",
    RouterAgent: "Router Agent",
  };
  return map[name] || name;
}

// ── Toast notifications ───────────────────────────────────────────────────────
function showToast(message, type = "success") {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: ${type === "error" ? "rgba(239,68,68,0.9)" : "rgba(34,197,94,0.9)"};
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    font-size: 0.875rem;
    font-weight: 500;
    z-index: 9999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    animation: toast-in 0.3s ease;
    max-width: 360px;
  `;
  toast.textContent = message;

  const style = document.createElement("style");
  style.textContent = `
    @keyframes toast-in { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
  `;
  document.head.appendChild(style);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  const container = document.getElementById("chatContainer");
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function truncate(str, maxLen) {
  if (!str) return "";
  return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
}

function formatTimestamp(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
