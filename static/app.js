// Virdixt Phase 3 — Audit Dashboard & Multi-Lane Explainability
// Vanilla JS + Alpine.js reactive state
// No build step. Calls FastAPI endpoints on 127.0.0.1

document.addEventListener('alpine:init', () => {

  Alpine.data('virdixt', () => ({
    /* ── Theme ── */
    theme: localStorage.getItem('vx-theme') || 'dark',

    /* ── Tab state ── */
    activeTab: 'file',

    /* ── Sidebar state ── */
    sidebarCollapsed: JSON.parse(localStorage.getItem('vx-sidebar') || 'false'),
    toggleSidebar() {
      this.sidebarCollapsed = !this.sidebarCollapsed;
      localStorage.setItem('vx-sidebar', JSON.stringify(this.sidebarCollapsed));
    },

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
    _cachedText: '',           // raw text (for text/file sources)
    _cachedSampleName: '',     // sample filename (for sample source)

    /* ── Feedback Loop ── */
    feedbackModalOpen: false,
    feedbackSubmitting: false,
    feedbackForm: {
      expected_rating: 'MINIMAL_RISK',
      feedback_text: ''
    },
    openFeedbackModal() {
      this.feedbackModalOpen = true;
      this.feedbackForm.expected_rating = 'MINIMAL_RISK';
      this.feedbackForm.feedback_text = '';
    },
    closeFeedbackModal() {
      this.feedbackModalOpen = false;
    },
    async submitFeedback() {
      this.feedbackSubmitting = true;
      try {
        const payload = {
          document_id: this.result?.file_meta?.name || 'unknown_document',
          expected_rating: this.feedbackForm.expected_rating,
          actual_rating: this.result?.verdict?.rating || 'unknown',
          feedback_text: this.feedbackForm.feedback_text
        };
        const res = await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Failed to submit feedback');
        alert('Feedback submitted successfully. The 2x penalty weight has been logged for the RL pipeline.');
        this.closeFeedbackModal();
      } catch (e) {
        alert('Error: ' + e.message);
      } finally {
        this.feedbackSubmitting = false;
      }
    },

    // ── Init ──────────────────────────────────────────────────
    async init() {
      this.applyTheme();
      if (window.location.pathname.includes('report.html')) {
        const stored = sessionStorage.getItem('virdixt_report');
        if (!stored) {
          window.location.href = '/';
          return;
        }
        this.result = JSON.parse(stored);
        this.seedSimulator();
        this.initSidebarObserver();
      } else {
        await this.fetchSamples();
      }
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
    get themeIcon()  { return ''; },
    get themeLabel() { return this.theme === 'dark' ? 'Light Mode' : 'Dark Mode'; },

    // ── Sidebar Observer ──────────────────────────────────────
    initSidebarObserver() {
      this.$nextTick(() => {
        const sections = document.querySelectorAll('.panel, .verdict-banner');
        const navLinks = document.querySelectorAll('.sidebar-link');
        const observer = new IntersectionObserver((entries) => {
          let visibleIds = [];
          entries.forEach(entry => {
            if (entry.isIntersecting) visibleIds.push(entry.target.id);
          });
          if (visibleIds.length > 0) {
            const activeId = visibleIds[0];
            navLinks.forEach(link => {
              link.classList.remove('active');
              if (link.getAttribute('href') === '#' + activeId) {
                link.classList.add('active');
              }
            });
          }
        }, { root: document.querySelector('.report-content'), threshold: 0.2, rootMargin: "-10% 0px -40% 0px" });

        sections.forEach(sec => { if (sec.id) observer.observe(sec); });
      });
    },

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
          this._cachedText = await this.selectedFile.text().catch(() => '');
          this._cachedSampleName = '';
        } else if (this.activeTab === 'text') {
          this.pushTrace('INGESTION', `Processing raw text input (${this.rawText.trim().length} chars)`);
          form.append('text', this.rawText.trim());
          this._cachedText = this.rawText.trim();
          this._cachedSampleName = '';
        } else if (this.activeTab === 'sample') {
          this.pushTrace('INGESTION', `Loading sample report: ${this.selectedSample.filename}`);
          form.append('sample_name', this.selectedSample.filename);
          this._cachedSampleName = this.selectedSample.filename;
          this._cachedText = '';
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
        sessionStorage.setItem('virdixt_report', JSON.stringify(data));
        window.location.href = '/report.html';

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

    // ══════════════════════════════════════════════════════════
    // PHASE 4 — Multi-Variable In-Memory What-If Simulator
    // Calls /api/simulate_facts (pure arithmetic ~2ms, zero ML)
    // ══════════════════════════════════════════════════════════

    simResult:          null,
    simulating:         false,
    simLastTs:          '',
    _simDebounceTimer:  null,
    _simBasePriority:   null,
    _simBaseRating:     null,
    _frozenTextSignals: null,
    simExposure:        0,
    simFacts:           {},      // live copy of financial_facts being nudged
    simFactsMeta:       [],      // [{ key, label, value, min, max, step }]

    // ── Formatters ─────────────────────────────────────────
    get simExposureFmt() {
      const v = this.simExposure;
      if (v === 0) return '$0';
      if (v >= 1000000)  return '$' + (v / 1000000).toFixed(1) + 'M';
      if (v >= 1000)     return '$' + (v / 1000).toFixed(0) + 'K';
      return '$' + v;
    },

    fmtMoney(v) {
      if (v == null || isNaN(v)) return 'N/A';
      const abs = Math.abs(v), sign = v < 0 ? '-' : '';
      if (abs >= 1e9)  return sign + '$' + (abs / 1e9).toFixed(2)  + 'B';
      if (abs >= 1e6)  return sign + '$' + (abs / 1e6).toFixed(1)  + 'M';
      if (abs >= 1e3)  return sign + '$' + (abs / 1e3).toFixed(0)  + 'K';
      return sign + '$' + abs.toFixed(0);
    },

    get simGradeClass()     { return 'grade-' + (this.simResult?.verdict?.rating ?? 'MONITOR'); },
    get simBaseGradeClass() { return 'grade-' + (this._simBaseRating ?? 'MONITOR'); },

    get simPriorityFmt() {
      const p = this.simResult?.priority_index;
      if (p == null) return '—';
      if (p === 0)   return '$0';
      return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 0 });
    },

    get simPriorityDelta() {
      if (!this.simResult || this._simBasePriority == null) return '';
      const delta = (this.simResult.priority_index ?? 0) - this._simBasePriority;
      if (Math.abs(delta) < 1) return '';
      const sign = delta > 0 ? '▲ +' : '▼ ';
      return sign + '$' + Math.abs(delta).toLocaleString('en-US', { maximumFractionDigits: 0 });
    },

    get simPriorityDeltaClass() {
      if (!this.simResult || this._simBasePriority == null) return 'same';
      const delta = (this.simResult.priority_index ?? 0) - this._simBasePriority;
      if (delta > 1)  return 'up';
      if (delta < -1) return 'down';
      return 'same';
    },

    get simGradeChanged() {
      return this.simResult && this._simBaseRating &&
             this.simResult.verdict.rating !== this._simBaseRating;
    },

    get simAltmanFmt() {
      const z = this.simResult?.forensic_scores?.altman_z?.score;
      return (z != null && !isNaN(z)) ? Number(z).toFixed(2) : '—';
    },
    get simAltmanZone()  { return this.simResult?.forensic_scores?.altman_z?.status ?? ''; },
    get simAltmanClass() {
      const s = this.simAltmanZone;
      if (s.includes('DISTRESS')) return 'chip-danger';
      if (s.includes('GREY'))     return 'chip-warning';
      if (s.includes('SAFE'))     return 'chip-success';
      return 'chip-neutral';
    },

    get simHasFinancialFacts() { return this.simFactsMeta.length > 0; },

    get simOverrideText() {
      const ors = this.simResult?.override_reasons ?? [];
      return ors.length ? 'Forensic override: ' + ors.join(', ') : '';
    },

    // ── Seed simulator when analysis result arrives ────────
    seedSimulator() {
      if (!this.result) return;
      const snap = this.result.simulator_snapshot;
      this._simBasePriority   = this.result.priority_index ?? 0;
      this._simBaseRating     = this.result.verdict?.rating ?? null;
      this._frozenTextSignals = snap?.frozen_text_signals ?? null;
      this.simExposure        = this.exposure || 0;
      this.simResult          = null;

      // Build dynamic slider metadata from financial_facts
      const rawFacts = snap?.financial_facts ?? {};
      this.simFacts     = {};
      this.simFactsMeta = [];

      const LABELS = {
        'revenue':           'Revenue / Sales',   'sales':             'Revenue / Sales',
        'operating income':  'EBIT / Op. Income', 'ebit':              'EBIT / Op. Income',
        'net income':        'Net Income',         'net profit':        'Net Income',
        'total assets':      'Total Assets',       'assets':            'Total Assets',
        'total debt':        'Total Debt',         'debt':              'Total Debt',
        'liabilities':       'Total Liabilities',
        'cash':              'Cash & Equivalents',
        'working capital':   'Working Capital',
        'retained earnings': 'Retained Earnings',
        'equity':            'Equity',
      };

      const seen = new Set();
      for (const [rawKey, rawVal] of Object.entries(rawFacts)) {
        if (rawKey.startsWith('_') || rawVal == null || typeof rawVal !== 'number') continue;
        const kLow = rawKey.toLowerCase().replace(/_/g,' ').replace(/-/g,' ');
        let label = rawKey;
        for (const [frag, lbl] of Object.entries(LABELS)) {
          if (kLow.includes(frag)) { label = lbl; break; }
        }
        if (seen.has(label)) continue;
        seen.add(label);

        const absVal   = Math.abs(rawVal);
        const magnitude = Math.pow(10, Math.floor(Math.log10(Math.max(absVal, 1))));
        const step     = Math.max(magnitude / 10, 1);
        const minV     = Math.round(rawVal < 0 ? rawVal * 3 : 0);
        const maxV     = Math.round(rawVal < 0 ? 0 : rawVal * 3);

        this.simFacts[rawKey] = rawVal;
        this.simFactsMeta.push({ key: rawKey, label, value: rawVal, min: minV, max: maxV, step });
      }
    },

    // ── Input handlers ─────────────────────────────────────
    onSimInput()  {
      clearTimeout(this._simDebounceTimer);
      this._simDebounceTimer = setTimeout(() => this.simulateFacts(), 200);
    },
    onSimSlider() { this.onSimInput(); },   // legacy alias

    // ── Core simulation call ───────────────────────────────
    async simulateFacts() {
      if (!this._frozenTextSignals) return;
      this.simulating = true;
      const t0 = performance.now();
      try {
        const payload = {
          financial_facts:     this.simFacts,
          frozen_text_signals: this._frozenTextSignals,
          exposure:            this.simExposure,
        };
        const res = await fetch('/api/simulate_facts', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(payload),
        });
        if (!res.ok) throw new Error('Simulation failed');
        this.simResult = await res.json();
        const ms = (performance.now() - t0).toFixed(0);
        this.simLastTs = this.nowTs() + ' (' + ms + 'ms)';
      } catch (e) {
        console.error('Simulator error:', e);
      } finally {
        this.simulating = false;
      }
    },

    // Legacy simulate() kept for backward compat with old tests
    async simulate() { await this.simulateFacts(); },

  }));

});
