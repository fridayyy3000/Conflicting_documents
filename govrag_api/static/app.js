(function () {
  "use strict";

  const STORAGE_KEY = "docuinsight_corpus";
  const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];

  const el = {
    errorBanner: document.getElementById("error-banner"),
    newCollectionBtn: document.getElementById("new-collection-btn"),
    createCard: document.getElementById("create-collection-card"),
    createForm: document.getElementById("create-collection-form"),
    collectionNameInput: document.getElementById("collection-name-input"),
    workspace: document.getElementById("workspace"),
    collectionNameDisplay: document.getElementById("collection-name-display"),
    collectionMeta: document.getElementById("collection-meta"),
    dropzone: document.getElementById("dropzone"),
    browseBtn: document.getElementById("browse-btn"),
    fileInput: document.getElementById("file-input"),
    loadExampleBtn: document.getElementById("load-example-btn"),
    uploadStatus: document.getElementById("upload-status"),
    documentList: document.getElementById("document-list"),
    questionInput: document.getElementById("question-input"),
    resolveBtn: document.getElementById("resolve-btn"),
    resolveHint: document.getElementById("resolve-hint"),
    results: document.getElementById("results"),
    resultAnswer: document.getElementById("result-answer"),
    resultConfidenceBadge: document.getElementById("result-confidence-badge"),
    conflictFields: document.getElementById("conflict-fields"),
    resultConflictSource: document.getElementById("result-conflict-source"),
    resultConflict: document.getElementById("result-conflict"),
    resultNumConflicts: document.getElementById("result-num-conflicts"),
    simpleFields: document.getElementById("simple-fields"),
    resultSimpleSource: document.getElementById("result-simple-source"),
    resultNoConflict: document.getElementById("result-no-conflict"),
    reasonLabel: document.getElementById("reason-label"),
    resultReason: document.getElementById("result-reason"),
    evidenceSectionConflict: document.getElementById("evidence-section-conflict"),
    evidenceTableBody: document.getElementById("evidence-table-body"),
    evidenceSectionSimple: document.getElementById("evidence-section-simple"),
    simpleEvidenceList: document.getElementById("simple-evidence-list"),
    loadingOverlay: document.getElementById("loading-overlay"),
    loadingMessage: document.getElementById("loading-message"),
  };

  let state = {
    corpusId: null,
    corpusName: null,
    expiresAt: null,
    documents: [],
  };

  // ---------- utilities ----------

  function showError(message) {
    el.errorBanner.textContent = message;
    el.errorBanner.classList.remove("hidden");
    window.clearTimeout(showError._t);
    showError._t = window.setTimeout(() => el.errorBanner.classList.add("hidden"), 6000);
  }

  function showLoading(message) {
    el.loadingMessage.textContent = message || "Working…";
    el.loadingOverlay.classList.remove("hidden");
  }

  function hideLoading() {
    el.loadingOverlay.classList.add("hidden");
  }

  async function apiFetch(path, options) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (networkErr) {
      throw new Error("Could not reach the server. Check your connection and try again.");
    }

    let body = null;
    const text = await response.text();
    if (text) {
      try {
        body = JSON.parse(text);
      } catch (parseErr) {
        body = null;
      }
    }

    if (!response.ok) {
      const detail = (body && body.detail) || `Request failed (HTTP ${response.status})`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    return body;
  }

  function persistState() {
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        corpusId: state.corpusId,
        corpusName: state.corpusName,
        expiresAt: state.expiresAt,
      })
    );
  }

  function clearState() {
    state = { corpusId: null, corpusName: null, expiresAt: null, documents: [] };
    sessionStorage.removeItem(STORAGE_KEY);
  }

  // ---------- rendering ----------

  function renderCollectionView() {
    const hasCorpus = Boolean(state.corpusId);
    el.createCard.classList.toggle("hidden", hasCorpus);
    el.workspace.classList.toggle("hidden", !hasCorpus);
    el.newCollectionBtn.classList.toggle("hidden", !hasCorpus);

    if (!hasCorpus) {
      return;
    }

    el.collectionNameDisplay.textContent = state.corpusName || "Collection";
    const expiresText = state.expiresAt
      ? `Expires ${new Date(state.expiresAt).toLocaleString()}`
      : "";
    el.collectionMeta.textContent = expiresText;

    renderDocumentList();
    updateResolveAvailability();
  }

  function renderDocumentList() {
    el.documentList.innerHTML = "";
    state.documents.forEach((doc) => {
      const li = document.createElement("li");
      const icon = document.createElement("span");
      icon.className = "doc-icon";
      icon.textContent = "\u2713";
      const label = document.createElement("span");
      label.textContent = doc.filename;
      li.appendChild(icon);
      li.appendChild(label);
      el.documentList.appendChild(li);
    });
  }

  function updateResolveAvailability() {
    const hasDocs = state.documents.length > 0;
    el.resolveBtn.disabled = !hasDocs;
    el.resolveHint.classList.toggle("hidden", hasDocs);
  }

  function confidenceBadgeClass(confidence) {
    const normalized = (confidence || "").toLowerCase();
    if (normalized === "high") return "badge badge--high";
    if (normalized === "medium") return "badge badge--medium";
    if (normalized === "low") return "badge badge--low";
    return "badge badge--unknown";
  }

  function renderResult(result) {
    el.results.classList.remove("hidden");

    const conflict = Boolean(result.conflict_detected);

    el.resultAnswer.textContent = result.answer || "Cannot be determined";
    el.resultConfidenceBadge.textContent = (result.confidence || "unknown").toUpperCase();
    el.resultConfidenceBadge.className = confidenceBadgeClass(result.confidence);
    el.resultReason.textContent = result.reason || "—";

    el.conflictFields.classList.toggle("hidden", !conflict);
    el.simpleFields.classList.toggle("hidden", conflict);
    el.resultNoConflict.classList.toggle("hidden", conflict);
    el.reasonLabel.textContent = conflict ? "Why This Source Was Selected" : "Why this answer is supported";

    if (conflict) {
      el.resultConflictSource.textContent = result.selected_source || "—";
      el.resultConflict.textContent = "Yes";
      el.resultNumConflicts.textContent = String(result.num_conflicts ?? 0);
    } else {
      el.resultSimpleSource.textContent = result.selected_source || "—";
    }

    el.evidenceSectionConflict.classList.toggle("hidden", !conflict);
    el.evidenceSectionSimple.classList.toggle("hidden", conflict);

    const sources = result.top_sources || [];

    if (conflict) {
      el.evidenceTableBody.innerHTML = "";
      sources.forEach((src) => {
        const tr = document.createElement("tr");
        if (src.selected) {
          tr.classList.add("is-selected");
        }

        const cells = [
          src.filename,
          src.claim,
          src.status,
          src.document_type,
          formatScore(src.semantic_score),
          formatScore(src.authority_score),
          formatScore(src.scope_score),
          formatScore(src.final_score),
          src.selected ? "Yes" : "No",
          src.supporting_sentence,
        ];

        cells.forEach((value) => {
          const td = document.createElement("td");
          td.textContent = value === null || value === undefined ? "—" : String(value);
          tr.appendChild(td);
        });

        el.evidenceTableBody.appendChild(tr);
      });
    } else {
      el.simpleEvidenceList.innerHTML = "";
      sources.forEach((src) => {
        const li = document.createElement("li");
        const sourceLabel = document.createElement("strong");
        sourceLabel.textContent = src.filename;
        const sentence = document.createElement("span");
        sentence.textContent = src.supporting_sentence ? ` — ${src.supporting_sentence}` : "";
        li.appendChild(sourceLabel);
        li.appendChild(sentence);
        el.simpleEvidenceList.appendChild(li);
      });
    }
  }

  function formatScore(value) {
    if (typeof value !== "number") return "—";
    return value.toFixed(3);
  }

  // ---------- API actions ----------

  async function createCollection(name) {
    const created = await apiFetch("/demo/corpora", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });

    state.corpusId = created.corpus_id;
    state.corpusName = created.name;
    state.expiresAt = created.expires_at;
    state.documents = [];
    persistState();
    renderCollectionView();
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;

    const accepted = [];
    const rejected = [];
    files.forEach((file) => {
      const lower = file.name.toLowerCase();
      const isSupported = ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
      if (isSupported) {
        accepted.push(file);
      } else {
        rejected.push(file.name);
      }
    });

    if (rejected.length > 0) {
      showUploadStatus(
        `Skipped unsupported file(s): ${rejected.join(", ")}`,
        "error"
      );
    }

    if (accepted.length === 0) return;

    const formData = new FormData();
    accepted.forEach((file) => formData.append("files", file));

    showLoading("Uploading documents…");
    try {
      const response = await apiFetch(
        `/demo/corpora/${encodeURIComponent(state.corpusId)}/documents`,
        { method: "POST", body: formData }
      );

      (response.uploaded || []).forEach((doc) => {
        state.documents.push({ filename: doc.filename, document_id: doc.document_id });
      });

      renderDocumentList();
      updateResolveAvailability();
      showUploadStatus(
        `Uploaded ${response.uploaded.length} file(s). Total: ${response.document_count}.`,
        "success"
      );
    } catch (err) {
      showUploadStatus(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  function showUploadStatus(message, kind) {
    el.uploadStatus.textContent = message;
    el.uploadStatus.classList.remove("hidden", "upload-status--success", "upload-status--error");
    el.uploadStatus.classList.add(kind === "error" ? "upload-status--error" : "upload-status--success");
  }

  async function loadExampleDataset() {
    showLoading("Loading 180 example documents\u2026");
    try {
      const response = await apiFetch(
        `/demo/corpora/${encodeURIComponent(state.corpusId)}/load_example`,
        { method: "POST" }
      );

      (response.uploaded || []).forEach((doc) => {
        state.documents.push({ filename: doc.filename, document_id: doc.document_id });
      });

      renderDocumentList();
      updateResolveAvailability();
      showUploadStatus(
        `Loaded ${response.uploaded.length} example document(s). Total: ${response.document_count}.`,
        "success"
      );
    } catch (err) {
      showUploadStatus(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  async function askQuestion() {
    const question = el.questionInput.value.trim();
    if (!question) {
      showError("Enter a question before resolving.");
      return;
    }

    el.resolveBtn.disabled = true;
    showLoading("Resolving conflicts and reasoning over evidence…");

    try {
      const result = await apiFetch(
        `/demo/corpora/${encodeURIComponent(state.corpusId)}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question,
            project_id: "project-79920195-9e86-44ea-8c9",
            region: "us-central1",
            use_llm: true,
            top_k: 8,
          }),
        }
      );
      renderResult(result);
    } catch (err) {
      showError(err.message);
    } finally {
      hideLoading();
      updateResolveAvailability();
    }
  }

  async function startNewCollection() {
    if (state.corpusId) {
      try {
        await apiFetch(`/demo/corpora/${encodeURIComponent(state.corpusId)}`, {
          method: "DELETE",
        });
      } catch (err) {
        // Ignore delete failures; the corpus will expire on its own regardless.
      }
    }

    clearState();
    el.questionInput.value = "";
    el.uploadStatus.classList.add("hidden");
    el.results.classList.add("hidden");
    renderCollectionView();
  }

  async function restoreSession() {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return;

    let saved;
    try {
      saved = JSON.parse(raw);
    } catch (err) {
      sessionStorage.removeItem(STORAGE_KEY);
      return;
    }

    if (!saved || !saved.corpusId) return;

    try {
      const metadata = await apiFetch(`/demo/corpora/${encodeURIComponent(saved.corpusId)}`);
      const docsResponse = await apiFetch(
        `/demo/corpora/${encodeURIComponent(saved.corpusId)}/documents`
      );

      state.corpusId = metadata.corpus_id;
      state.corpusName = metadata.name;
      state.expiresAt = metadata.expires_at;
      state.documents = docsResponse.documents || [];
      renderCollectionView();
    } catch (err) {
      // Corpus expired or was deleted elsewhere; start fresh silently.
      clearState();
    }
  }

  // ---------- event wiring ----------

  el.createForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = el.collectionNameInput.value.trim();
    if (!name) return;

    showLoading("Creating collection…");
    try {
      await createCollection(name);
      el.collectionNameInput.value = "";
    } catch (err) {
      showError(err.message);
    } finally {
      hideLoading();
    }
  });

  el.browseBtn.addEventListener("click", () => el.fileInput.click());

  el.loadExampleBtn.addEventListener("click", loadExampleDataset);

  el.fileInput.addEventListener("change", (event) => {
    uploadFiles(event.target.files);
    event.target.value = "";
  });

  ["dragover", "dragenter"].forEach((evtName) => {
    el.dropzone.addEventListener(evtName, (event) => {
      event.preventDefault();
      el.dropzone.classList.add("dropzone--active");
    });
  });

  ["dragleave", "dragend"].forEach((evtName) => {
    el.dropzone.addEventListener(evtName, () => {
      el.dropzone.classList.remove("dropzone--active");
    });
  });

  el.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    el.dropzone.classList.remove("dropzone--active");
    uploadFiles(event.dataTransfer.files);
  });

  el.resolveBtn.addEventListener("click", askQuestion);
  el.newCollectionBtn.addEventListener("click", startNewCollection);

  restoreSession();
})();
