// Virdixt Phase 3 — Audit Dashboard & Multi-Lane Explainability
// Vanilla JS + Alpine.js reactive state
// No build step. Calls FastAPI endpoints on 127.0.0.1

document.addEventListener('alpine:init', () => {

  Alpine.data('virdixt', () => ({
    /* ── Theme ── */
    theme: localStorage.getItem('vx-theme') || 'dark',

    /* ── Tab state ── */
    activeTab: 'file',

    /* ── File input state ── */
    selectedFile: null,
    dragOver: false,
    fileError: '',

    /* ── Text input ── */
    rawText: '',

    /* ── Sample state ── */
    samples: [],
    selectedSample: null,

    /* ── Exposure ── */
    exposure: 0,

    /* ── Processing state ── */
    analyzing: false,
    traceLogs: [],
    showTrace: false,
    errorMsg: '',

    /* ── Result ── */
    result: null,

    // ── Init ──────────────────────────────────────────────────
    async init() {
      this.applyTheme();
      await this.fetchSamples();
    },

    // ── Theme ─────────────────────────────────────────────────
    applyTheme() {
      document.documentElement.setAttribute('data-theme', this.theme);
    },
    toggleTheme() {
      this.theme = this.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('vx-theme', this.theme);
      this.applyTheme();
    },
    get themeIcon()  { return this.theme === 'dark' ? '☀️' : '🌙'; },
    get themeLabel() { return this.theme === 'dark' ? 'Light mode' : 'Dark mode'; },

    // ── Samples ───────────────────────────────────────────────
    async fetchSamples() {
      try {
        const res = await fetch('/api/samples');
        this.samples = await res.json();
      } catch {
        this.samples = [];
      }
    },
    selectSample(s) {
      this.selectedSample = this.selectedSample?.filename === s.filename ? null : s;
    },

    // ── File drag & drop ──────────────────────────────────────
    onDragOver(e)  { e.preventDefault(); this.dragOver = true; },
    onDragLeave()  { this.dragOver = false; },
    onDrop(e) {
      e.preventDefault(); this.dragOver = false;
      const file = e.dataTransfer.files[0];
      if (file) this.setFile(file);
    },
    onFileChange(e) {
      const file = e.target.files[0];
      if (file) this.setFile(file);
    },
    setFile(file) {
      this.fileError = '';
      const MAX = 100 * 1024 * 1024;
      if (file.size > MAX) {
        this.fileError = `File too large (${(file.size/1024/1024).toFixed(1)} MB). Max 100 MB.`;
        return;
      }
      const allowed = ['.txt','.pdf','.csv','.docx','.doc','.xlsx'];
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (!allowed.includes(ext)) {
        this.fileError = `Unsupported format "${ext}". Accepted: ${allowed.join(', ')}`;
        return;
      }
      this.selectedFile = file;
    },
    removeFile() {
      this.selectedFile = null; this.fileError = '';
      const inp = document.getElementById('file-input');
      if (inp) inp.value = '';
    },
    get fileSizeLabel() {
      if (!this.selectedFile) return '';
      const kb = this.selectedFile.size / 1024;
      return kb > 1024 ? `${(kb/1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`;
    },

    // ── Validation ─────────────────────────────────────────────
    get canAnalyze() {
      if (this.analyzing) return false;
      if (this.activeTab === 'file')   return !!this.selectedFile;
      if (this.activeTab === 'text')   return this.rawText.trim().length > 10;
      if (this.activeTab === 'sample') return !!this.selectedSample;
      return false;
    },

    // ── Trace ─────────────────────────────────────────────────
    pushTrace(stage, message, timestamp = null) {
      const ts = timestamp ?? this.nowTs();
      this.traceLogs.push({ ts, stage, message });
      this.$nextTick(() => {
        const el = document.getElementById('trace-console');
        if (el) el.scrollTop = el.scrollHeight;
      });
    },
    nowTs() {
      const d = new Date();
      return `${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}.${String(d.getMilliseconds()).padStart(3,'0')}`;
    },

    // ── Main Analyze ──────────────────────────────────────────
    async analyze() {
      this.analyzing = true; this.showTrace = true;
      this.traceLogs = []; this.errorMsg = ''; this.result = null;
      const t0 = performance.now();

      try {
        this.pushTrace('REQUEST', 'Sending document to Virdixt forensic engine...');
        const form = new FormData();
        form.append('exposure', this.exposure || 0);

        if (this.activeTab === 'file' && this.selectedFile) {
          this.pushTrace('INGESTION', `Reading file: ${this.selectedFile.name} (${this.fileSizeLabel})`);
          form.append('file', this.selectedFile);
        } else if (this.activeTab === 'text') {
          this.pushTrace('INGESTION', `Processing raw text input (${this.rawText.trim().length} chars)`);
          form.append('text', this.rawText.trim());
        } else if (this.activeTab === 'sample') {
          this.pushTrace('INGESTION', `Loading sample report: ${this.selectedSample.filename}`);
          form.append('sample_name', this.selectedSample.filename);
        }

        const res = await fetch('/api/analyze', { method: 'POST', body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || 'Analysis failed');
        }

        const data = await res.json();
        const elapsed = (performance.now() - t0).toFixed(1);

        if (data.trace_logs?.length) {
          for (const log of data.trace_logs) {
            this.pushTrace(log.stage, log.message, log.timestamp);
          }
        }
        this.pushTrace('SYNTHESIS', `Complete — total round-trip: ${elapsed}ms`);
        this.result = data;

        this.$nextTick(() => {
          const el = document.getElementById('result-section');
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });

      } catch (err) {
        this.pushTrace('ERROR', err.message);
        this.errorMsg = err.message;
      } finally {
        this.analyzing = false;
      }
    },

    // ══════════════════════════════════════════════════════════
    // PHASE 3 — Computed Getters for Dashboard Rendering
    // ══════════════════════════════════════════════════════════

    get verdictGradeClass() {
      return 'grade-' + (this.result?.verdict?.rating ?? 'MONITOR');
    },

    get distressScore() {
      const s = this.result?.nlp_sentiment?.distress_score;
      return s != null ? Number(s).toFixed(1) : '—';
    },

    chipClass(status) {
      if (!status) return 'chip-neutral';
      const s = status.toUpperCase();
      if (s.includes('DISTRESS') || s.includes('HIGH') || s.includes('WEAK') || s.includes('MANIPULATION')) return 'chip-danger';
      if (s.includes('GREY') || s.includes('MODERATE') || s.includes('CAUTION')) return 'chip-warning';
      if (s.includes('SAFE') || s.includes('STRONG') || s.includes('MINIMAL')) return 'chip-success';
      return 'chip-neutral';
    },

    fmtScore(val) {
      if (val == null) return 'N/A';
      if (typeof val === 'number') return val.toFixed(2);
      return String(val);
    },

    get toneClass() {
      const tone = this.result?.nlp_sentiment?.tone ?? '';
      if (['NEGATIVE','DEFENSIVE_DISTRESS'].includes(tone.toUpperCase())) return 'grade-CRITICAL';
      if (tone.toUpperCase() === 'NEUTRAL') return 'grade-MONITOR';
      return 'grade-MINIMAL';
    },

    get priorityFormatted() {
      const p = this.result?.priority_index;
      if (!p || p === 0) return null;
      return '$' + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    },

    get actionGradeClass() {
      return 'grade-' + (this.result?.verdict?.rating ?? 'MONITOR');
    },

  }));

});
