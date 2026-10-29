const BASE = 'http://localhost:8000/api/v1';

// ── Helpers ───────────────────────────────────────────────────────────────────

function apiKey() { return document.getElementById('apiKey').value.trim(); }

function authHeaders() {
  return { 'Content-Type': 'application/json', 'X-API-Key': apiKey() };
}

const LEVEL_STYLES = {
  CRITICAL: 'bg-red-50 text-red-700 border border-red-200',
  HIGH:     'bg-orange-50 text-orange-700 border border-orange-200',
  MEDIUM:   'bg-amber-50 text-amber-700 border border-amber-200',
  NORMAL:   'bg-green-50 text-green-700 border border-green-200',
  REJECTED: 'bg-purple-50 text-purple-700 border border-purple-200',
};

const ACTION_LABELS = {
  immediate_response: 'Immediate Response',
  urgent_review:      'Urgent Review',
  clinician_review:   'Clinician Review',
  observation:        'Observation',
};

function levelBadge(level) {
  const cls = LEVEL_STYLES[level] || LEVEL_STYLES.NORMAL;
  return `<span class="text-xs font-bold px-2.5 py-1 rounded-lg ${cls}">${level}</span>`;
}

function scoreColor(s) {
  return ['#bbf7d0','#fde68a','#fed7aa','#fecaca'][s] || '#e2e8f0';
}

// ── Navigation ────────────────────────────────────────────────────────────────

function enterApp() {
  document.getElementById('landing').style.display = 'none';
  document.getElementById('app').style.display = 'flex';
  checkHealth();
  loadTestCases();
}

function goBack() {
  document.getElementById('app').style.display = 'none';
  document.getElementById('landing').style.display = 'flex';
}

function switchTab(name) {
  document.querySelectorAll('.toptab').forEach(t => t.classList.remove('active-tab'));
  document.querySelectorAll('.tabpanel').forEach(p => p.style.display = 'none');
  document.querySelector(`[data-tab="${name}"]`).classList.add('active-tab');
  document.getElementById(`tab-${name}`).style.display = 'block';
  if (name === 'history') loadAlerts();
}

// ── Health dot ────────────────────────────────────────────────────────────────

async function checkHealth() {
  const dot = document.getElementById('health-dot');
  try {
    const res = await fetch(`${BASE}/health`, { headers: authHeaders() });
    const d = await res.json();
    dot.className = d.status === 'healthy'
      ? 'w-2.5 h-2.5 rounded-full bg-green-500 transition-colors'
      : 'w-2.5 h-2.5 rounded-full bg-red-500 transition-colors';
    dot.title = d.status === 'healthy' ? 'All systems healthy' : JSON.stringify(d);
  } catch {
    dot.className = 'w-2.5 h-2.5 rounded-full bg-red-500 transition-colors';
    dot.title = 'Cannot reach server';
  }
}

// ── EHR file picker ───────────────────────────────────────────────────────────

function handleEhrSelect(input) {
  const el = document.getElementById('ehr-file-name');
  if (input.files[0]) {
    el.textContent = '📄 ' + input.files[0].name;
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

// ── Alert result card ─────────────────────────────────────────────────────────

function renderAlert(data, vitals) {
  const n = data.news2 || {};
  const q = data.qsofa || {};
  const level = data.alert_level;

  const rows = [
    { label: 'SBP',  val: vitals ? `${vitals.sbp} mmHg`  : '—', score: n.sbp_score },
    { label: 'DBP',  val: vitals ? `${vitals.dbp} mmHg`  : '—', score: n.dbp_score },
    { label: 'HR',   val: vitals ? `${vitals.hr} bpm`    : '—', score: n.heart_rate_score },
    { label: 'SpO₂', val: vitals ? `${vitals.spo2}%`     : '—', score: n.spo2_score },
  ].filter(r => r.score !== undefined);

  const news2Rows = rows.map(r => `
    <div class="flex items-center gap-3 text-sm">
      <span class="w-10 text-slate-400 font-medium text-xs">${r.label}</span>
      <span class="w-20 text-slate-600 text-xs">${r.val}</span>
      <div class="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div class="h-full rounded-full" style="width:${(r.score/3)*100}%;background:${scoreColor(r.score)}"></div>
      </div>
      <span class="w-4 text-right text-xs font-bold text-slate-600">${r.score}</span>
    </div>`).join('');

  const qFlag = (set, label) => `
    <div class="flex items-center gap-2 text-xs">
      <span class="w-4 h-4 rounded-full flex items-center justify-center text-xs font-bold ${set ? 'bg-orange-100 text-orange-600' : 'bg-green-100 text-green-600'}">${set ? '✓' : '✗'}</span>
      <span class="text-slate-500">${label}</span>
    </div>`;

  // LLM reasoning
  let reasoning = '';
  if (data.rule_based_only) {
    const why = level === 'CRITICAL'
      ? 'CRITICAL vitals trigger an immediate safety bypass — the LLM is intentionally skipped.'
      : 'LLM output was unavailable or rejected by output guardrails.';
    reasoning = `
      <div class="border-t border-slate-100 p-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Assessment</span>
          <span class="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-500">Rule-based only</span>
        </div>
        <p class="text-sm text-slate-400 italic">${why}</p>
      </div>`;
  } else if (data.reasoning_summary) {
    const factors = (data.contributing_factors || []).map(f =>
      `<span class="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">${f}</span>`
    ).join('');
    const conf = data.llm_confidence != null
      ? `<p class="text-xs text-slate-400 mt-2">Confidence: <strong class="text-slate-600">${(data.llm_confidence*100).toFixed(0)}%</strong></p>` : '';
    reasoning = `
      <div class="border-t border-slate-100 p-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">MedGemma Reasoning</span>
          <span class="text-xs font-semibold px-2 py-0.5 rounded bg-purple-100 text-purple-600">LLM</span>
        </div>
        <p class="text-sm text-slate-700 leading-relaxed">${data.reasoning_summary}</p>
        ${factors ? `<div class="flex flex-wrap gap-1.5 mt-3">${factors}</div>` : ''}
        ${conf}
        ${data.llm_limitations ? `<p class="text-xs text-slate-400 mt-2 italic">${data.llm_limitations}</p>` : ''}
      </div>`;
  }

  return `
    <div class="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-slate-100">
        <div class="flex items-center gap-3">
          ${levelBadge(level)}
          <span class="text-sm font-semibold">${data.patient_id}</span>
        </div>
        <span class="text-xs text-slate-400">${new Date(data.created_at).toLocaleTimeString()}</span>
      </div>

      <!-- Scores -->
      <div class="grid grid-cols-2 gap-0 divide-x divide-slate-100">
        <div class="p-4">
          <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">NEWS2 Breakdown</p>
          <div class="flex flex-col gap-2">${news2Rows}</div>
          <p class="text-xs font-bold text-slate-600 mt-3 pt-3 border-t border-slate-100">
            Total: ${n.total_score ?? '—'}/12
            ${n.any_single_critical ? '<span class="text-red-500 ml-1">· single-3 critical</span>' : ''}
          </p>
        </div>
        <div class="p-4">
          <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">qSOFA Sepsis Screen</p>
          <div class="flex flex-col gap-2">
            ${qFlag(q.sbp_flag,  `SBP ≤ 100 (${vitals?.sbp ?? data.vitals?.sbp ?? '—'} mmHg)`)}
            ${qFlag(q.hr_flag,   `HR > 90 (${vitals?.hr ?? data.vitals?.hr ?? '—'} bpm)`)}
            ${qFlag(q.spo2_flag, `SpO₂ < 94% (${vitals?.spo2 ?? data.vitals?.spo2 ?? '—'}%)`)}
          </div>
          <p class="text-xs font-bold text-slate-600 mt-3 pt-3 border-t border-slate-100">
            Score: ${q.score ?? '—'}/3
            ${q.high_risk ? '<span class="text-red-500 ml-1">· high sepsis risk</span>' : ''}
          </p>
        </div>
      </div>

      <!-- Reasoning -->
      ${reasoning}

      <!-- Action -->
      <div class="flex items-center justify-between px-4 py-3 bg-slate-50 border-t border-slate-100">
        <div>
          <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-0.5">Recommended Action</p>
          <p class="text-sm font-semibold text-slate-700">${ACTION_LABELS[data.recommended_action] || data.recommended_action}</p>
        </div>
        ${data.rule_based_only && level === 'CRITICAL'
          ? '<span class="text-xs px-2 py-1 rounded bg-amber-100 text-amber-700 font-semibold">LLM bypassed · safety design</span>' : ''}
      </div>

      <!-- Disclaimer -->
      <p class="px-4 py-2 text-xs text-slate-400 italic border-t border-slate-100 leading-relaxed">
        ${data.disclaimer || ''}
      </p>
    </div>`;
}

function fmtDetail(detail) {
  if (Array.isArray(detail)) {
    return detail.map(e => `<strong>${(e.loc || []).slice(-1)[0] ?? 'field'}</strong>: ${e.msg}`).join('<br>');
  }
  return detail || 'Unknown error';
}

function renderError(msg) {
  return `<div class="bg-red-50 border border-red-200 rounded-xl p-4">
    <p class="text-sm font-semibold text-red-700 mb-1">Error / Rejected</p>
    <p class="text-sm text-slate-700">${msg}</p>
  </div>`;
}

function loadingHtml(msg) {
  return `<div class="flex items-center gap-3 text-sm text-slate-400 py-4">
    <div class="w-4 h-4 border-2 border-slate-200 border-t-blue-500 rounded-full animate-spin"></div>
    ${msg}
  </div>`;
}

// ── Manual Entry ──────────────────────────────────────────────────────────────

async function submitVitals() {
  const el = document.getElementById('vitals-result');
  el.classList.remove('hidden');
  el.innerHTML = loadingHtml('Ingesting EHR context and calling MedGemma…');

  const patientId = document.getElementById('v-patient').value.trim();
  const sbp  = parseFloat(document.getElementById('v-sbp').value);
  const dbp  = parseFloat(document.getElementById('v-dbp').value);
  const hr   = parseFloat(document.getElementById('v-hr').value);
  const spo2 = parseFloat(document.getElementById('v-spo2').value);
  const ehrFile = document.getElementById('v-ehr-file').files[0];

  try {
    if (ehrFile) {
      const form = new FormData();
      form.append('patient_id', patientId);
      form.append('file', ehrFile);
      await fetch(`${BASE}/ehr/ingest`, {
        method: 'POST', headers: { 'X-API-Key': apiKey() }, body: form,
      });
    }

    const res = await fetch(`${BASE}/vitals/snapshot`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ patient_id: patientId, systolic_bp: sbp, diastolic_bp: dbp, heart_rate: hr, spo2 }),
    });
    const data = await res.json();

    el.innerHTML = res.ok
      ? renderAlert(data, { sbp, dbp, hr, spo2 })
      : renderError(fmtDetail(data.detail) || JSON.stringify(data));
  } catch (e) {
    el.innerHTML = renderError(`Network error: ${e.message}`);
  }
}

// ── Test Cases ────────────────────────────────────────────────────────────────

async function loadTestCases() {
  const loading = document.getElementById('tc-loading');
  const list = document.getElementById('tc-list');
  loading.style.display = 'flex';
  list.innerHTML = '';

  try {
    const res = await fetch(`${BASE}/mock/test-cases`, { headers: authHeaders() });
    const cases = await res.json();
    loading.style.display = 'none';
    list.innerHTML = cases.map(tc => tcCard(tc)).join('');
  } catch (e) {
    loading.style.display = 'none';
    list.innerHTML = renderError(`Could not load test cases: ${e.message}`);
  }
}

function tcCard(tc) {
  const v = tc.vitals || {};
  const expectedCls = LEVEL_STYLES[tc.expected_alert] || LEVEL_STYLES.NORMAL;

  const vitalPill = (label, val, unit) => `
    <div class="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-center min-w-[64px]">
      <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-0.5">${label}</p>
      <p class="text-xl font-bold text-slate-800 leading-none">${val ?? '—'}</p>
      <p class="text-xs text-slate-400 mt-0.5">${unit}</p>
    </div>`;

  return `
    <div class="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm" id="tc-card-${tc.id}">
      <!-- Header -->
      <div class="flex items-start justify-between px-4 py-3 bg-slate-50 border-b border-slate-100">
        <div>
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">${tc.id}</span>
          <h3 class="text-sm font-bold text-slate-800 mt-0.5">${tc.name}</h3>
        </div>
        <span class="text-xs font-bold px-2.5 py-1 rounded-lg ${expectedCls} flex-shrink-0 ml-3">
          Expected: ${tc.expected_alert}
        </span>
      </div>

      <!-- Body -->
      <div class="p-4">
        <!-- Narrative -->
        <p class="text-sm text-slate-500 leading-relaxed mb-4">${tc.clinical_narrative || tc.notes}</p>

        <!-- Vitals being sent -->
        <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Vitals Being Sent</p>
        <div class="flex gap-2 flex-wrap mb-4">
          ${vitalPill('SBP',  v.systolic_bp,  'mmHg')}
          ${vitalPill('DBP',  v.diastolic_bp, 'mmHg')}
          ${vitalPill('HR',   v.heart_rate,   'bpm')}
          ${vitalPill('SpO₂', v.spo2,         '%')}
        </div>

        <!-- Meta + Run -->
        <div class="flex items-center justify-between flex-wrap gap-3">
          <div class="flex items-center gap-2 text-xs text-slate-400 flex-wrap">
            <span>${tc.patient_id}</span>
            <span>·</span>
            ${tc.ehr_file
              ? `<span class="px-2 py-0.5 rounded bg-blue-50 text-blue-600 border border-blue-200 font-semibold">EHR: ${tc.ehr_file}</span>`
              : `<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-400 font-medium">No EHR</span>`}
          </div>
          <button id="btn-${tc.id}" onclick="runTestCase('${tc.id}')"
            class="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors">
            Run Test
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </button>
        </div>
        <p class="text-xs text-slate-400 italic mt-2">${tc.notes}</p>
      </div>

      <!-- Result -->
      <div id="tc-result-${tc.id}" class="hidden border-t border-slate-100 p-4"></div>
    </div>`;
}

async function runTestCase(id) {
  const resultEl = document.getElementById(`tc-result-${id}`);
  const btn = document.getElementById(`btn-${id}`);
  resultEl.classList.remove('hidden');
  resultEl.innerHTML = loadingHtml('Ingesting EHR if needed, then calling MedGemma…');
  btn.disabled = true;
  btn.textContent = 'Running…';

  try {
    const res = await fetch(`${BASE}/mock/run/${id}`, {
      method: 'POST', headers: authHeaders(),
    });
    const data = await res.json();
    resultEl.innerHTML = res.ok ? renderAlert(data) : renderError(fmtDetail(data.detail) || JSON.stringify(data));
  } catch (e) {
    resultEl.innerHTML = renderError(`Network error: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Run Test <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ── ABG Interpreter ───────────────────────────────────────────────────────────

const ABG_LEVEL_STYLES = {
  CRITICAL: 'bg-red-50 text-red-700 border border-red-200',
  HIGH:     'bg-orange-50 text-orange-700 border border-orange-200',
  MEDIUM:   'bg-amber-50 text-amber-700 border border-amber-200',
  NORMAL:   'bg-green-50 text-green-700 border border-green-200',
};

const DISORDER_LABELS = {
  none:                  'None',
  respiratory_acidosis:  'Respiratory Acidosis',
  respiratory_alkalosis: 'Respiratory Alkalosis',
  metabolic_acidosis:    'Metabolic Acidosis',
  metabolic_alkalosis:   'Metabolic Alkalosis',
  mixed:                 'Mixed Disorder',
};

function abgLevelBadge(level) {
  const cls = ABG_LEVEL_STYLES[level] || ABG_LEVEL_STYLES.NORMAL;
  return `<span class="text-xs font-bold px-2.5 py-1 rounded-lg ${cls}">${level}</span>`;
}

function switchAbgTab(name) {
  ['manual', 'testcases', 'history'].forEach(t => {
    document.getElementById(`abg-panel-${t}`).style.display = t === name ? 'block' : 'none';
    const btn = document.getElementById(`abg-tab-${t}`);
    if (t === name) {
      btn.classList.add('text-blue-600', 'border-blue-600');
      btn.classList.remove('text-slate-500', 'border-transparent');
    } else {
      btn.classList.remove('text-blue-600', 'border-blue-600');
      btn.classList.add('text-slate-500', 'border-transparent');
    }
  });
  if (name === 'testcases') loadABGTestCases();
  if (name === 'history')   loadABGHistory();
}

function renderABGResult(data) {
  const resp = data.respiratory || {};
  const ab   = data.acid_base   || {};
  const comp = data.compensation || {};
  const met  = data.metabolic   || {};

  // Respiratory row
  const rfType = resp.type === 'type1' ? 'Type 1 RF' : resp.type === 'type2' ? 'Type 2 RF' : 'None';
  const pfBadge = resp.pf_ratio
    ? `<span class="text-xs px-2 py-0.5 rounded ${resp.pf_ratio < 100 ? 'bg-red-100 text-red-700' : resp.pf_ratio < 200 ? 'bg-orange-100 text-orange-700' : 'bg-amber-100 text-amber-700'}">P/F ${resp.pf_ratio}</span>`
    : '';

  // Acid-base
  const disorderLabel = DISORDER_LABELS[ab.primary_disorder] || ab.primary_disorder || '—';
  const acuteChronic = ab.acute_vs_chronic ? `<span class="text-xs text-slate-400 ml-1">(${ab.acute_vs_chronic})</span>` : '';

  // Compensation
  const compLabel = comp.compensated
    ? comp.compensated.replace(/_/g, ' ')
    : '—';

  // Anion gap
  const agDisplay = met.anion_gap_corrected != null
    ? `${met.anion_gap_corrected} mmol/L (${met.gap_type || ''})`
    : met.anion_gap != null ? `${met.anion_gap} mmol/L` : '—';

  // Clinical alerts
  const alertsHtml = data.clinical_alerts && data.clinical_alerts.length
    ? data.clinical_alerts.map(a =>
        `<div class="text-xs px-3 py-1.5 rounded-lg bg-red-50 text-red-700 border border-red-200">${a}</div>`
      ).join('')
    : '<span class="text-xs text-slate-400">No urgent alerts</span>';

  // MedGemma reasoning
  let reasoning = '';
  if (data.rule_based_only) {
    reasoning = `
      <div class="border-t border-slate-100 p-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Assessment</span>
          <span class="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-500 font-semibold">Rule-based only</span>
        </div>
        <p class="text-sm text-slate-400 italic">MedGemma reasoning unavailable — LLM offline or timed out.</p>
      </div>`;
  } else if (data.clinical_significance) {
    const factors = (data.contributing_factors || [])
      .map(f => `<span class="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">${f}</span>`).join('');
    const conf = data.llm_confidence != null
      ? `<p class="text-xs text-slate-400 mt-2">Confidence: <strong class="text-slate-600">${(data.llm_confidence * 100).toFixed(0)}%</strong></p>` : '';
    reasoning = `
      <div class="border-t border-slate-100 p-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">MedGemma Reasoning</span>
          <span class="text-xs px-2 py-0.5 rounded bg-purple-100 text-purple-600 font-semibold">LLM</span>
        </div>
        ${data.primary_concern ? `<p class="text-xs font-semibold text-slate-500 mb-2">Primary concern: <span class="text-slate-700">${data.primary_concern}</span></p>` : ''}
        <p class="text-sm text-slate-700 leading-relaxed">${data.clinical_significance}</p>
        ${factors ? `<div class="flex flex-wrap gap-1.5 mt-3">${factors}</div>` : ''}
        ${conf}
        ${data.llm_limitations ? `<p class="text-xs text-slate-400 mt-2 italic">${data.llm_limitations}</p>` : ''}
      </div>`;
  }

  return `
    <div class="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-slate-100">
        <div class="flex items-center gap-3">
          ${abgLevelBadge(data.alert_level)}
          <span class="text-sm font-semibold">${data.patient_id}</span>
        </div>
        <span class="text-xs text-slate-400">${new Date(data.created_at).toLocaleTimeString()}</span>
      </div>

      <!-- Summary -->
      <div class="px-4 py-3 bg-slate-50 border-b border-slate-100">
        <p class="text-sm text-slate-700 leading-relaxed">${data.summary}</p>
      </div>

      <!-- Grid: Respiratory + Acid-Base -->
      <div class="grid grid-cols-2 gap-0 divide-x divide-slate-100">
        <div class="p-4">
          <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Respiratory</p>
          <div class="flex items-center gap-2 mb-2">
            <span class="text-sm font-semibold text-slate-700">${rfType}</span>
            ${pfBadge}
          </div>
          <div class="flex flex-col gap-1 mt-2">
            ${(resp.explanation || []).map(e => `<p class="text-xs text-slate-500">• ${e}</p>`).join('')}
          </div>
        </div>
        <div class="p-4">
          <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Acid-Base</p>
          <div class="flex items-center gap-1 mb-2">
            <span class="text-sm font-semibold text-slate-700">${disorderLabel}</span>
            ${acuteChronic}
          </div>
          <div class="flex flex-col gap-1 mt-2">
            ${(ab.explanation || []).map(e => `<p class="text-xs text-slate-500">• ${e}</p>`).join('')}
          </div>
        </div>
      </div>

      <!-- Grid: Compensation + Metabolic -->
      <div class="grid grid-cols-2 gap-0 divide-x divide-slate-100 border-t border-slate-100">
        <div class="p-4">
          <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Compensation</p>
          <p class="text-sm font-semibold text-slate-700 capitalize mb-1">${compLabel}</p>
          ${comp.expected_value != null ? `<p class="text-xs text-slate-400">Expected: ${comp.expected_value} · Measured: ${comp.measured_value}</p>` : ''}
          <div class="flex flex-col gap-1 mt-1">
            ${(comp.explanation || []).map(e => `<p class="text-xs text-slate-500">• ${e}</p>`).join('')}
          </div>
        </div>
        <div class="p-4">
          <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Anion Gap</p>
          <p class="text-sm font-semibold text-slate-700 mb-1">${agDisplay}</p>
          ${met.likely_causes && met.likely_causes.length
            ? met.likely_causes.map(c => `<p class="text-xs text-slate-500 mt-1">• ${c}</p>`).join('')
            : ''}
        </div>
      </div>

      <!-- Clinical Alerts -->
      <div class="px-4 py-3 border-t border-slate-100">
        <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Clinical Alerts</p>
        <div class="flex flex-wrap gap-2">${alertsHtml}</div>
      </div>

      <!-- MedGemma Reasoning -->
      ${reasoning}

      <!-- Disclaimer -->
      <p class="px-4 py-2 text-xs text-slate-400 italic border-t border-slate-100 leading-relaxed">
        ${data.disclaimer || ''}
      </p>
    </div>`;
}

// ── ABG PDF Upload & form pre-fill ────────────────────────────────────────────

async function handleABGPdfUpload(input) {
  if (!input.files[0]) return;
  const file    = input.files[0];
  const statusEl = document.getElementById('abg-pdf-status');
  statusEl.classList.remove('hidden');
  statusEl.innerHTML = loadingHtml(`Parsing <strong>${file.name}</strong>…`);

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch(`${BASE}/abg/upload`, {
      method: 'POST',
      headers: { 'X-API-Key': apiKey() },
      body: form,
    });
    const data = await res.json();

    if (!res.ok) {
      statusEl.innerHTML = `<div class="text-sm text-red-600 font-medium">
        Parse failed: ${fmtDetail(data.detail) || JSON.stringify(data)}
      </div>`;
      return;
    }

    // Pre-fill form fields
    const fieldMap = {
      ph:          'abg-ph',
      pao2_mmhg:   'abg-pao2',
      paco2_mmhg:  'abg-paco2',
      hco3:        'abg-hco3',
      base_excess: 'abg-be',
      lactate:     'abg-lactate',
      fio2:        'abg-fio2',
      na:          'abg-na',
      cl:          'abg-cl',
      albumin:     'abg-albumin',
    };

    const extracted  = data.extracted_fields || [];
    const uncertain  = data.uncertain_fields || [];
    let filledCount  = 0;

    for (const [key, elId] of Object.entries(fieldMap)) {
      const el = document.getElementById(elId);
      if (!el) continue;
      if (data[key] != null) {
        el.value = data[key];
        filledCount++;
        // Highlight uncertain fields in amber
        if (uncertain.includes(key)) {
          el.style.color = '#d97706';
          el.title = 'Value extracted with lower confidence — please verify';
        } else {
          el.style.color = '';
          el.title = '';
        }
      }
    }

    // Boolean fields
    if (data.chronic_copd) document.getElementById('abg-copd').checked = true;
    if (data.ketones)      document.getElementById('abg-ketones').checked = true;

    // Patient ID
    if (data.patient_id) {
      document.getElementById('abg-patient').value = data.patient_id;
    }

    // Status message
    const missing = data.missing_core_fields || [];
    const warningHtml = missing.length
      ? `<p class="text-xs text-amber-600 mt-1">Could not extract: <strong>${missing.join(', ')}</strong> — please enter manually.</p>`
      : '';

    statusEl.innerHTML = `
      <div class="flex items-start gap-2 p-3 rounded-lg ${data.parse_success ? 'bg-green-50 border border-green-200' : 'bg-amber-50 border border-amber-200'}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="${data.parse_success ? '#16a34a' : '#d97706'}" stroke-width="2.5" stroke-linecap="round" class="mt-0.5 flex-shrink-0"><path d="${data.parse_success ? 'M20 6L9 17l-5-5' : 'M12 9v4M12 17h.01'}"/></svg>
        <div>
          <p class="text-sm font-semibold ${data.parse_success ? 'text-green-700' : 'text-amber-700'}">
            ${data.parse_success ? `${filledCount} values extracted from ${file.name}` : `Partial extraction — ${filledCount} values found`}
          </p>
          ${warningHtml}
          ${uncertain.length ? `<p class="text-xs text-amber-600 mt-0.5">Highlighted in amber — please verify before submitting.</p>` : ''}
        </div>
      </div>`;

  } catch (e) {
    statusEl.innerHTML = `<div class="text-sm text-red-600">Network error: ${e.message}</div>`;
  }
}

// Drag-and-drop support for ABG PDF
document.addEventListener('DOMContentLoaded', () => {
  const zone = document.getElementById('abg-pdf-drop-zone');
  if (!zone) return;
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('border-blue-500'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('border-blue-500'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('border-blue-500');
    const file = e.dataTransfer?.files[0];
    if (file && file.type === 'application/pdf') {
      const input = document.getElementById('abg-pdf-file');
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      handleABGPdfUpload(input);
    }
  });
});

async function submitABG() {
  const el = document.getElementById('abg-result');
  el.classList.remove('hidden');
  el.innerHTML = loadingHtml('Running ABG interpretation and calling MedGemma…');

  const payload = {
    patient_id:   document.getElementById('abg-patient').value.trim() || 'manual_entry',
    pH:           parseFloat(document.getElementById('abg-ph').value),
    pao2_mmhg:    parseFloat(document.getElementById('abg-pao2').value),
    paco2_mmhg:   parseFloat(document.getElementById('abg-paco2').value),
    hco3:         parseFloat(document.getElementById('abg-hco3').value),
    base_excess:  parseFloat(document.getElementById('abg-be').value),
    lactate:      parseFloat(document.getElementById('abg-lactate').value),
    fio2:         parseFloat(document.getElementById('abg-fio2').value) || 0.21,
    chronic_copd: document.getElementById('abg-copd').checked,
    ketones:      document.getElementById('abg-ketones').checked || undefined,
  };

  const na      = parseFloat(document.getElementById('abg-na').value);
  const cl      = parseFloat(document.getElementById('abg-cl').value);
  const albumin = parseFloat(document.getElementById('abg-albumin').value);
  if (!isNaN(na))      payload.na      = na;
  if (!isNaN(cl))      payload.cl      = cl;
  if (!isNaN(albumin)) payload.albumin = albumin;

  try {
    const res  = await fetch(`${BASE}/abg/analyze`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    el.innerHTML = res.ok ? renderABGResult(data) : renderError(fmtDetail(data.detail) || JSON.stringify(data));
  } catch (e) {
    el.innerHTML = renderError(`Network error: ${e.message}`);
  }
}

async function loadABGTestCases() {
  const loading = document.getElementById('abg-tc-loading');
  const list    = document.getElementById('abg-tc-list');
  loading.style.display = 'flex';
  list.innerHTML = '';

  try {
    const res   = await fetch(`${BASE}/abg/test-cases`, { headers: authHeaders() });
    const cases = await res.json();
    loading.style.display = 'none';
    list.innerHTML = cases.map(tc => abgTcCard(tc)).join('');
  } catch (e) {
    loading.style.display = 'none';
    list.innerHTML = renderError(`Could not load ABG test cases: ${e.message}`);
  }
}

function abgTcCard(tc) {
  const v = tc.values || {};
  const expCls = ABG_LEVEL_STYLES[tc.expected_alert] || ABG_LEVEL_STYLES.NORMAL;

  const pill = (label, val, unit) => `
    <div class="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-center min-w-[72px]">
      <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-0.5">${label}</p>
      <p class="text-xl font-bold text-slate-800 leading-none">${val ?? '—'}</p>
      <p class="text-xs text-slate-400 mt-0.5">${unit}</p>
    </div>`;

  return `
    <div class="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm" id="abg-card-${tc.id}">
      <div class="flex items-start justify-between px-4 py-3 bg-slate-50 border-b border-slate-100">
        <div>
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">${tc.id}</span>
          <h3 class="text-sm font-bold text-slate-800 mt-0.5">${tc.name}</h3>
        </div>
        <span class="text-xs font-bold px-2.5 py-1 rounded-lg ${expCls} flex-shrink-0 ml-3">
          Expected: ${tc.expected_alert}
        </span>
      </div>
      <div class="p-4">
        <p class="text-sm text-slate-500 leading-relaxed mb-4">${tc.description || tc.notes}</p>
        <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Values</p>
        <div class="flex gap-2 flex-wrap mb-4">
          ${pill('pH',    v.pH,          '')}
          ${pill('PaO₂',  v.pao2_mmhg,  'mmHg')}
          ${pill('PaCO₂', v.paco2_mmhg, 'mmHg')}
          ${pill('HCO₃',  v.hco3,       'mmol/L')}
          ${pill('Lactate', v.lactate,  'mmol/L')}
        </div>
        <div class="flex items-center justify-between flex-wrap gap-3">
          <div class="text-xs text-slate-400">
            <span>${tc.patient_id}</span>
            ${v.chronic_copd ? '<span class="ml-2 px-2 py-0.5 rounded bg-blue-50 text-blue-600 border border-blue-200 font-semibold">COPD</span>' : ''}
            ${v.ketones ? '<span class="ml-2 px-2 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-200 font-semibold">Ketones+</span>' : ''}
          </div>
          <button id="abg-btn-${tc.id}" onclick="runABGTestCase('${tc.id}')"
            class="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors">
            Run Test
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </button>
        </div>
        <p class="text-xs text-slate-400 italic mt-2">${tc.notes}</p>
      </div>
      <div id="abg-tc-result-${tc.id}" class="hidden border-t border-slate-100 p-4"></div>
    </div>`;
}

async function runABGTestCase(id) {
  const resultEl = document.getElementById(`abg-tc-result-${id}`);
  const btn      = document.getElementById(`abg-btn-${id}`);
  resultEl.classList.remove('hidden');
  resultEl.innerHTML = loadingHtml('Interpreting ABG and calling MedGemma…');
  btn.disabled = true;
  btn.textContent = 'Running…';

  try {
    const res  = await fetch(`${BASE}/abg/run/${id}`, { method: 'POST', headers: authHeaders() });
    const data = await res.json();
    resultEl.innerHTML = res.ok ? renderABGResult(data) : renderError(fmtDetail(data.detail) || JSON.stringify(data));
  } catch (e) {
    resultEl.innerHTML = renderError(`Network error: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Run Test <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

async function loadABGHistory() {
  const list    = document.getElementById('abg-history-list');
  const patient = document.getElementById('abg-h-patient').value.trim();
  list.innerHTML = loadingHtml('Loading ABG history…');

  try {
    const params = new URLSearchParams();
    if (patient) params.set('patient_id', patient);
    const res  = await fetch(`${BASE}/abg/history?${params}`, { headers: authHeaders() });
    const data = await res.json();
    if (!res.ok) { list.innerHTML = renderError(JSON.stringify(data)); return; }
    if (!data.length) {
      list.innerHTML = '<p class="text-sm text-slate-400 py-4">No ABG results found.</p>';
      return;
    }
    list.innerHTML = data.map(a => `
      <div class="bg-white border border-slate-200 rounded-xl px-4 py-3 flex items-center justify-between gap-4 shadow-sm">
        <div class="flex items-center gap-3">
          ${abgLevelBadge(a.alert_level)}
          <span class="text-sm font-semibold">${a.patient_id}</span>
        </div>
        <div class="flex items-center gap-4 text-xs text-slate-400 flex-wrap">
          <span class="text-slate-600 font-medium">${DISORDER_LABELS[a.primary_disorder] || a.primary_disorder}</span>
          <span class="max-w-xs truncate">${a.summary}</span>
          ${a.rule_based_only ? '<span class="text-slate-300">rule-based only</span>' : ''}
          <span>${new Date(a.created_at).toLocaleString()}</span>
        </div>
      </div>`).join('');
  } catch (e) {
    list.innerHTML = renderError(`Network error: ${e.message}`);
  }
}

// ── Alert History ─────────────────────────────────────────────────────────────

async function loadAlerts() {
  const list = document.getElementById('alerts-list');
  list.innerHTML = loadingHtml('Loading alerts…');

  const patient  = document.getElementById('h-patient').value.trim();
  const severity = document.getElementById('h-severity').value;
  const params   = new URLSearchParams();
  if (patient)  params.set('patient_id', patient);
  if (severity) params.set('severity', severity);

  try {
    const res = await fetch(`${BASE}/alerts?${params}`, { headers: authHeaders() });
    const data = await res.json();
    if (!res.ok) { list.innerHTML = renderError(JSON.stringify(data)); return; }
    if (!data.length) {
      list.innerHTML = '<p class="text-sm text-slate-400 py-4">No alerts found.</p>';
      return;
    }
    list.innerHTML = data.map(a => `
      <div class="bg-white border border-slate-200 rounded-xl px-4 py-3 flex items-center justify-between gap-4 shadow-sm">
        <div class="flex items-center gap-3">
          ${levelBadge(a.alert_level)}
          <span class="text-sm font-semibold">${a.patient_id}</span>
        </div>
        <div class="flex items-center gap-4 text-xs text-slate-400">
          <span>NEWS2: <strong class="text-slate-600">${a.news2_score}</strong></span>
          <span>qSOFA: <strong class="text-slate-600">${a.qsofa_score}</strong></span>
          <span>${ACTION_LABELS[a.recommended_action] || a.recommended_action}</span>
          ${a.rule_based_only ? '<span class="text-slate-300">rule-based only</span>' : ''}
          <span>${new Date(a.created_at).toLocaleString()}</span>
        </div>
      </div>`).join('');
  } catch (e) {
    list.innerHTML = renderError(`Network error: ${e.message}`);
  }
}
