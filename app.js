import * as C from './src/calibrate.js';

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const STORE = 'jill-calibrate-v1';
const K_TOP = 10, TARGET_Q = 6, CONFIDENT = 0.85;

const S = { data: null, results: null, answers: [], skipped: [], versions: [], status: {}, q: '', filter: 'all', sel: 0, post: null, view: null, auto: false };

function load() { try { const s = JSON.parse(localStorage.getItem(STORE) || '{}'); Object.assign(S, { answers: s.answers || [], skipped: s.skipped || [], versions: s.versions || [], status: s.status || {} }); } catch { /* fresh */ } }
function save() { localStorage.setItem(STORE, JSON.stringify({ answers: S.answers, skipped: S.skipped, versions: S.versions, status: S.status })); }

// ------------------------------------------------------------------ model state
const scores = () => S.data.candidates.map((c) => c.scores);
function rebuild() {
  const d = S.data;
  S.post = C.createPosterior({ w0: d.briefPriorities, ...d.model, seed: 7 });
  for (const a of S.answers) C.update(S.post, scores()[a.w], scores()[a.l]);
  derive();
}
function derive() {
  const sc = scores();
  S.mean = C.posteriorMean(S.post);
  S.rank = C.rankBy(sc, S.mean);
  S.base = C.rankBy(sc, S.data.briefPriorities);
  S.mem = C.topKMembership(S.post, sc, K_TOP, 600);
  S.conf = C.expectedPrecision(S.mem, S.rank.slice(0, K_TOP));
  S.ci = C.weightQuantiles(S.post, [0.1, 0.9]);
  const asked = new Set([...S.answers, ...S.skipped].map((a) => C.pairKey(a.w ?? a.a, a.l ?? a.b)));
  S.question = C.nextQuestion(S.post, sc, { asked, pool: 16, sample: 500 });
  S.pos = new Map(S.rank.map((id, i) => [id, i]));
  S.basePos = new Map(S.base.map((id, i) => [id, i]));
}
function fitOf(i) { const p = S.mem[i]; if (p >= 0.5) return ['excellent', 'Excellent']; if (S.pos.get(i) < 30) return ['good', 'Good']; return ['fair', 'Fair']; }

// ------------------------------------------------------------------ helpers
const PALETTE = ['#ebe9e1', '#f3e6d8', '#e2ebdf', '#e4e7f1', '#f2e0e0', '#ece4f1'];
function avatar(c, cls = 'av') { let h = 0; for (const ch of c.name) h = (h * 31 + ch.charCodeAt(0)) >>> 0; const ini = c.name.split(' ').map((x) => x[0]).join(''); return `<span class="${cls}" style="background:${PALETTE[h % PALETTE.length]}" aria-hidden="true">${ini}</span>`; }
const crit = () => S.data.criteria;
function summary(c) {
  const s = c.scores, out = [];
  const tops = [...s.keys()].sort((a, b) => s[b] - s[a]);
  const lines = {
    0: `Shipped LLM features to production at ${c.company}`,
    1: `Designed systems that carry real load, ${c.years} years in`,
    2: c.archetype === 'Founder-engineer' ? 'Built a product from zero as a founder' : 'Took a product from zero to first customers',
    3: 'Works directly with customers and owns the problem, not just the ticket',
    4: c.archetype === 'Engineering lead' ? `Has hired and led a team of ${3 + (c.years % 7)} engineers` : 'Mentors engineers and has run hiring loops',
  };
  for (const k of tops.slice(0, 3)) if (s[k] >= 0.45 || out.length < 2) out.push(lines[k]);
  return out;
}
function why(c, i) {
  const s = c.scores, tops = [...s.keys()].sort((a, b) => S.mean[b] * s[b] - S.mean[a] * s[a]).slice(0, 2).map((k) => crit()[k].label.toLowerCase());
  const moved = S.answers.length ? ` After your ${S.answers.length} picks, ${c.name.split(' ')[0]} ranks #${S.pos.get(i) + 1} for you (was #${S.basePos.get(i) + 1}).` : '';
  return `${c.name.split(' ')[0]} is a ${c.archetype.toLowerCase()} whose strongest evidence is on what you weight most: ${tops[0]} and ${tops[1]}.${moved}`;
}
const GOALS = { 'Applied ML engineer': 'Wants to own a model-backed product end to end, not just the model.', 'Infrastructure engineer': 'Looking for an early team where reliability work ships to customers.', 'Founder-engineer': 'Wants founder-level ownership again without the fundraising.', 'Product engineer': 'Wants to work closer to users and AI-native products.', 'Engineering lead': 'Wants to build the first engineering team at a company with traction.', 'Forward deployed engineer': 'Wants to stay close to customers while going deeper on the product.' };
function toast(m) { const t = $('#toast'); t.textContent = m; t.classList.add('on'); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove('on'), 2400); }
function download(name, text, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
const ICON = {
  prev: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>',
  next: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>',
  send: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4Z"/></svg>',
  dots: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="5" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="12" cy="19" r="1.6"/></svg>',
  x: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  bars: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ada598" stroke-width="2.2" stroke-linecap="round"><path d="M6 20v-4M12 20v-8M18 20V8"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  dl: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>',
};

// ------------------------------------------------------------------ boot + routing
async function boot() {
  load();
  try {
    const [d, r] = await Promise.all([fetch('data/role.json'), fetch('data/results.json')]);
    if (!d.ok || !r.ok) throw new Error(`HTTP ${d.status}/${r.status}`);
    S.data = await d.json(); S.results = await r.json();
  } catch (e) {
    $('#body').innerHTML = `<div class="card empty"><b>Couldn't load this role</b>${esc(e.message)}<div style="margin-top:12px"><button class="btn" onclick="location.reload()">Retry</button></div></div>`;
    return;
  }
  rebuild();
  $('#roleList').innerHTML = [['Founding AI Engineer', 101, true], ['Founding Designer', 64], ['Account Executive', 88]].map(([t, n, on]) => `<a class="role-link ${on ? 'on' : ''}" href="${on ? '#/candidates' : '#'}" data-role="${esc(t)}">${t}<span class="n">${n}</span></a>`).join('');
  document.querySelectorAll('[data-role]').forEach((a) => { if (!a.classList.contains('on')) a.onclick = (e) => { e.preventDefault(); toast('This demo has one role: Founding AI Engineer'); }; });
  document.querySelectorAll('.side .nav').forEach((b) => { b.onclick = () => { const n = b.dataset.nav; if (n === 'roles') location.hash = '#/candidates'; else if (n === 'pipeline') location.hash = '#/pipeline'; else toast(`${b.textContent} isn't part of this concept demo`); }; });
  $('#kb').onclick = () => $('#help').showModal();
  document.addEventListener('keydown', onKey);
  document.addEventListener('mousemove', (e) => { const t = e.target.closest('[data-tip]'); const tip = $('#tip'); if (!t) { tip.style.display = 'none'; return; } tip.textContent = t.dataset.tip; tip.style.display = 'block'; tip.style.left = Math.min(innerWidth - tip.offsetWidth - 10, e.clientX + 12) + 'px'; tip.style.top = e.clientY + 16 + 'px'; });
  addEventListener('hashchange', route);
  route();
}

const TABS = [['candidates', 'Candidates'], ['brief', 'Brief'], ['pitch', 'Pitch'], ['pipeline', 'Pipeline'], ['configure', 'Configure']];
function route() {
  const [, view, arg] = (location.hash || '#/candidates').split('/');
  S.view = view;
  document.querySelectorAll('.side .nav').forEach((b) => b.classList.toggle('on', b.dataset.nav === (view === 'pipeline' ? 'pipeline' : 'roles')));
  const r = S.data.role;
  if (view === 'c') {
    $('#phead').innerHTML = ''; $('#tabs').innerHTML = ''; $('#tabs').style.display = 'none';
    return renderCandidate(arg);
  }
  $('#tabs').style.display = '';
  $('#phead').innerHTML = `<div><h1>${esc(r.title)}</h1><div class="meta">${esc(r.company)} · ${esc(r.location)} · ${esc(r.type)} · ${esc(r.band)}</div></div>
    <div class="actions"><a class="btn ghost" href="#/report">How Calibrate was tested</a></div>`;
  $('#tabs').innerHTML = TABS.map(([k, l]) => `<a class="tab ${view === k || (k === 'candidates' && !TABS.some((t) => t[0] === view) && view !== 'report') ? 'on' : ''}" href="#/${k}">${l}${k === 'candidates' ? ` <span class="n">${S.data.candidates.length}</span>` : ''}${k === 'brief' && S.versions.length ? ' <span class="new">Calibrated</span>' : ''}</a>`).join('');
  if (view === 'brief') renderBrief(); else if (view === 'pitch') renderPitch(); else if (view === 'pipeline') renderPipeline(); else if (view === 'configure') renderConfigure(); else if (view === 'report') renderReport(); else renderCandidates();
}

function onKey(e) {
  if ((e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') && e.key !== 'Escape') return;
  if (e.key === '?') return $('#help').showModal();
  if (e.key === 'Escape') { if (document.activeElement.tagName === 'INPUT') return document.activeElement.blur(); if (S.view === 'c') location.hash = '#/candidates'; return; }
  if (S.view === 'c') { if (e.key === '[') $('#prev')?.click(); if (e.key === ']') $('#next')?.click(); return; }
  if (e.key === '/') { const s = $('#q'); if (s) { e.preventDefault(); s.focus(); } return; }
  if (e.key === 'ArrowLeft') return $('#pickA')?.click();
  if (e.key === 'ArrowRight') return $('#pickB')?.click();
  if (e.key === 's') return $('#skip')?.click();
  const rows = [...document.querySelectorAll('.row')];
  if ((e.key === 'j' || e.key === 'k') && rows.length) { S.sel = Math.max(0, Math.min(rows.length - 1, S.sel + (e.key === 'j' ? 1 : -1))); rows.forEach((r, i) => r.classList.toggle('sel', i === S.sel)); rows[S.sel].scrollIntoView({ block: 'nearest' }); }
  if (e.key === 'Enter' && rows[S.sel]) location.hash = rows[S.sel].dataset.href;
}

// ------------------------------------------------------------------ Candidates + Calibrate
function renderCandidates() {
  $('#body').innerHTML = `<div class="cols"><div>
      <div class="toolbar">
        <label class="search">${ICON.search}<span class="sr">Search candidates</span><input id="q" placeholder="Search candidates" value="${esc(S.q)}"><span class="kbd">/</span></label>
        ${[['all', 'All'], ['excellent', 'Excellent'], ['moved', 'Moved up']].map(([k, l]) => `<button class="chip ${S.filter === k ? 'on' : ''}" data-f="${k}">${l}</button>`).join('')}
        <span style="flex:1"></span>
        <button class="btn plain" id="csv" title="Export ranking as CSV">${ICON.dl} Export</button>
      </div>
      <div class="list" id="list"></div></div>
    <aside class="cal" id="cal" aria-label="Calibrate Jill"></aside></div>`;
  const q = $('#q'); q.oninput = () => { S.q = q.value; drawList(); };
  document.querySelectorAll('[data-f]').forEach((b) => { b.onclick = () => { S.filter = b.dataset.f; renderCandidates(); }; });
  $('#csv').onclick = exportCsv;
  drawList(); drawCal();
}

function drawList() {
  const c = S.data.candidates, q = S.q.trim().toLowerCase();
  let ids = S.rank.filter((i) => !q || (c[i].name + ' ' + c[i].title + ' ' + c[i].company).toLowerCase().includes(q));
  if (S.filter === 'excellent') ids = ids.filter((i) => fitOf(i)[0] === 'excellent');
  if (S.filter === 'moved') ids = ids.filter((i) => S.basePos.get(i) - S.pos.get(i) > 0);
  ids = ids.slice(0, 40);
  if (!ids.length) { $('#list').innerHTML = `<div class="card empty"><b>No candidates match</b>Clear the search or pick another filter.</div>`; return; }
  const imp = S.mean.map((w, k) => [w, k]).sort((a, b) => b[0] - a[0]).slice(0, 2).map((x) => x[1]);
  $('#list').innerHTML = ids.map((i, n) => {
    const cd = c[i], [cls, lab] = fitOf(i), d = S.basePos.get(i) - S.pos.get(i);
    const st = S.status[cd.id];
    return `<div class="row ${n === S.sel ? 'sel' : ''}" data-href="#/c/${cd.id}" tabindex="0" role="link" aria-label="${esc(cd.name)}, ${lab}">
      <span class="rank">${S.pos.get(i) + 1}</span>${avatar(cd)}
      <div class="who"><b>${esc(cd.name)}</b>${st ? ` <span class="status ${st === 'intro' ? 'intro' : st === 'passed' ? 'passed' : ''}" style="font-size:11px;padding:1px 7px">${st === 'intro' ? 'Intro requested' : st === 'passed' ? 'Passed' : 'Saved'}</span>` : ''}<small>${esc(cd.title)} · ${esc(cd.company)} · ${cd.years} yrs</small></div>
      <div class="bars5" aria-hidden="true">${cd.scores.map((s, k) => `<i class="${imp.includes(k) ? 'top' : ''}" data-tip="${esc(crit()[k].label)}: ${pct(s)}${imp.includes(k) ? ' (you weight this most)' : ''}"><b style="height:${100 * s}%"></b></i>`).join('')}</div>
      <div style="text-align:right"><span class="fit ${cls}" data-tip="Chance ${esc(cd.name.split(' ')[0])} is in your real top 10: ${pct(S.mem[i])}">${lab}</span>${S.answers.length && d !== 0 ? `<span class="mv ${d > 0 ? 'up' : 'down'}">${d > 0 ? '▲' : '▼'} ${Math.abs(d)}</span>` : ''}</div></div>`;
  }).join('');
  document.querySelectorAll('.row').forEach((r, n) => { r.onclick = () => { location.hash = r.dataset.href; }; r.onkeydown = (e) => { if (e.key === 'Enter') location.hash = r.dataset.href; }; r.onfocus = () => { S.sel = n; }; });
}

function optCard(i, id, side) {
  const c = S.data.candidates[i];
  return `<button class="opt" id="${id}" aria-label="Meet ${esc(c.name)}">
    <div class="top">${avatar(c)}<div style="min-width:0"><b>${esc(c.name)}</b><small>${esc(c.title)} · ${esc(c.company)}</small></div></div>
    <ul>${summary(c).slice(0, 2).map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
    <div class="sig"><div class="bars5" aria-hidden="true">${c.scores.map((s, k) => `<i data-tip="${esc(crit()[k].label)}: ${pct(s)}"><b style="height:${100 * s}%"></b></i>`).join('')}</div><span>${c.years} yrs · £${Math.round(c.salary / 1000)}k</span></div>
    <div class="meet">Meet ${esc(c.name.split(' ')[0])} <span class="kbd">${side}</span></div></button>`;
}

function drawCal() {
  const n = S.answers.length, done = n >= TARGET_Q, q = S.question;
  const brief = S.data.briefPriorities;
  const order = brief.map((_, k) => k);
  $('#cal').innerHTML = `<div class="card">
      <h3><img src="assets/jill-avatar.png" alt="">Help Jill calibrate <span class="new">New</span></h3>
      <p class="sub" style="margin-top:6px">${done ? 'Your ranking is calibrated. Keep going if a few picks felt like coin flips.' : `Six quick picks teach Jill what you actually weigh. Who would you rather meet first?`}</p>
      ${q ? `<div class="pair">${optCard(q.a, 'pickA', '←')}${optCard(q.b, 'pickB', '→')}</div>
        <div class="actions" style="justify-content:space-between;margin-top:12px"><div class="dots" aria-label="${n} of ${TARGET_Q} picks">${Array.from({ length: TARGET_Q }, (_, k) => `<i class="${k < n ? 'on' : ''}"></i>`).join('')}</div>
        <div class="actions"><button class="btn plain" id="skip">Not sure <span class="kbd">s</span></button>${n < TARGET_Q ? `<button class="btn plain" id="auto" title="Answers six picks as a hiring manager who cares most about product ownership and building from zero">Demo answers</button>` : ''}</div></div>`
        : `<div class="empty"><b>No more useful questions</b>Jill has what she needs.</div>`}
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:baseline"><span class="cap">Ranking confidence</span><span class="mono" style="font-size:13px">${pct(S.conf)}</span></div>
      <div class="meter ${S.conf >= CONFIDENT ? 'ok' : ''}"><i style="width:${100 * S.conf}%"></i></div>
      <p class="note" style="margin-top:8px">Expected share of the top ${K_TOP} shown here that are in your real top ${K_TOP}, from Jill's posterior over your priorities.${S.conf >= CONFIDENT ? '' : ` Jill is confident at ${pct(CONFIDENT)}.`}</p>
      ${S.conf >= CONFIDENT || done ? `<div class="done">✓ <div>${S.answers.length} picks moved ${S.rank.slice(0, K_TOP).filter((i) => S.basePos.get(i) >= K_TOP).length} new people into your top ${K_TOP}.</div></div>` : ''}
    </div>
    <div class="card">
      <span class="cap">What you value</span>
      <div class="wts">${order.map((k) => `<div class="wt"><span class="lab">${esc(crit()[k].label)}</span><span class="mono" style="font-size:12px;text-align:right">${pct(S.mean[k])}</span>
        <div class="trk" data-tip="Brief ${pct(brief[k])} · learned ${pct(S.mean[k])} (80% interval ${pct(S.ci[k][0])} to ${pct(S.ci[k][1])})"><span class="ci" style="left:${100 * S.ci[k][0] / 0.7}%;width:${100 * (S.ci[k][1] - S.ci[k][0]) / 0.7}%"></span><span class="br" style="left:${100 * brief[k] / 0.7}%"></span><span class="pm" style="left:calc(${100 * S.mean[k] / 0.7}% - 1px)"></span></div></div>`).join('')}</div>
      <div class="legend"><span><i style="background:var(--coral)"></i>Brief priority</span><span><i style="background:var(--ink)"></i>Learned</span><span><i style="background:var(--clay-300)"></i>80% interval</span></div>
      <div class="actions" style="margin-top:14px"><button class="btn" id="apply" ${n ? '' : 'disabled'}>Update brief priorities</button><button class="btn ghost" id="reset" ${n || S.skipped.length ? '' : 'disabled'}>Reset</button><button class="btn plain" id="exp">${ICON.dl} JSON</button></div>
    </div>`;
  const pick = (w, l) => { if (S.busy) return; S.busy = true; S.answers.push({ w, l }); save(); C.update(S.post, scores()[w], scores()[l]); derive(); drawList(); drawCal(); S.busy = false; };
  if (q) {
    $('#pickA').onclick = () => pick(q.a, q.b);
    $('#pickB').onclick = () => pick(q.b, q.a);
    $('#skip').onclick = () => { S.skipped.push({ a: q.a, b: q.b }); save(); derive(); drawCal(); };
    const au = $('#auto'); if (au) au.onclick = autoAnswer;
  }
  $('#apply').onclick = () => {
    S.versions.push({ at: Date.now(), priorities: S.mean.map((x) => +x.toFixed(4)), picks: S.answers.length });
    save(); toast(`Brief updated: v${S.versions.length + 1} from ${S.answers.length} picks`); route();
  };
  $('#reset').onclick = () => { S.answers = []; S.skipped = []; save(); rebuild(); renderCandidates(); toast('Calibration reset'); };
  $('#exp').onclick = () => download('jill-calibration.json', JSON.stringify({ role: S.data.role.title, criteria: crit().map((c) => c.label), brief: S.data.briefPriorities, learned: S.mean, interval80: S.ci, picks: S.answers.map((a) => ({ preferred: S.data.candidates[a.w].id, over: S.data.candidates[a.l].id })), top10: S.rank.slice(0, 10).map((i) => S.data.candidates[i].id), confidence: S.conf }, null, 2), 'application/json');
}

async function autoAnswer() {
  const dm = S.data.demoManager, sc = scores();
  const u = (i) => dm.reduce((s, w, k) => s + w * sc[i][k], 0);
  while (S.answers.length < TARGET_Q && S.question && S.view !== 'c') {
    const q = S.question, aWins = u(q.a) >= u(q.b);
    const el = $(aWins ? '#pickA' : '#pickB'); if (el) { el.style.boxShadow = 'inset 0 0 0 2px var(--ink)'; }
    await new Promise((r) => setTimeout(r, 450));
    S.answers.push({ w: aWins ? q.a : q.b, l: aWins ? q.b : q.a }); save();
    C.update(S.post, sc[aWins ? q.a : q.b], sc[aWins ? q.b : q.a]); derive();
    if (S.view === 'candidates' || !S.view || !['brief', 'pitch', 'pipeline', 'configure', 'report'].includes(S.view)) { drawList(); drawCal(); }
  }
}

function exportCsv() {
  const c = S.data.candidates;
  const lines = ['rank,brief_rank,id,name,title,company,fit,p_top10,' + crit().map((x) => `"${x.label}"`).join(',')];
  S.rank.forEach((i, n) => lines.push([n + 1, S.basePos.get(i) + 1, c[i].id, `"${c[i].name}"`, `"${c[i].title}"`, `"${c[i].company}"`, fitOf(i)[1], S.mem[i].toFixed(3), ...c[i].scores].join(',')));
  download('jill-ranking.csv', lines.join('\n'), 'text/csv'); toast('Exported ranking (101 candidates)');
}

// ------------------------------------------------------------------ candidate detail
function renderCandidate(id) {
  const cs = S.data.candidates, i = cs.findIndex((c) => c.id === id);
  if (i < 0) { $('#body').innerHTML = `<div class="card empty"><b>Candidate not found</b><a href="#/candidates" style="text-decoration:underline">Back to candidates</a></div>`; return; }
  const c = cs[i], n = S.pos.get(i), st = S.status[c.id];
  const prevId = n > 0 ? cs[S.rank[n - 1]].id : null, nextId = n < cs.length - 1 ? cs[S.rank[n + 1]].id : null;
  $('#body').innerHTML = `<div class="dhead">
      <div class="pager"><button id="prev" aria-label="Previous candidate" ${prevId ? '' : 'disabled'}>${ICON.prev}</button><span class="mono" style="font-family:var(--font)">${n + 1} / ${cs.length}</span><button id="next" aria-label="Next candidate" ${nextId ? '' : 'disabled'}>${ICON.next}</button></div>
      <div class="dact">${st ? `<span class="status ${st === 'intro' ? 'intro' : st === 'passed' ? 'passed' : ''}">${st === 'intro' ? 'Intro requested' : st === 'passed' ? 'Passed' : 'Saved'}</span>` : ''}
        <button class="btn plain" id="pass">Pass</button><button class="btn plain" id="intro">${ICON.send} Request intro</button>
        <button class="btn plain" id="save" aria-label="Save" title="Save">${ICON.dots}</button><a class="btn plain" href="#/candidates" aria-label="Close">${ICON.x}</a></div></div>
    <div class="dstack">
      <div class="card idcard">${avatar(c)}<h2>${esc(c.name)}</h2><div class="links"><span>CV +</span><span>LinkedIn ↗</span></div></div>
      <div class="card sec"><div class="cap">Summary</div><ul>${summary(c).map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>
      <div class="card sec"><div class="cap"><img src="assets/jill-avatar.png" alt="">Why Jill shortlisted</div><p>${esc(why(c, i))}</p></div>
      <div class="card sec"><div class="cap">Goals</div><p>${esc(GOALS[c.archetype])} Expects about £${Math.round(c.salary / 1000)}k base; ${esc(c.location)}.</p></div>
      <div class="sec" style="padding-left:0"><div class="cap">Criteria</div><div class="crit">${c.scores.map((s, k) => `<div><span class="ic">${ICON.bars}</span>${esc(crit()[k].label)}<small>${pct(s)} · weight ${pct(S.mean[k])}</small></div>`).join('')}</div></div>
    </div>`;
  const go = (cid) => { if (cid) location.hash = `#/c/${cid}`; };
  $('#prev').onclick = () => go(prevId); $('#next').onclick = () => go(nextId);
  const set = (v, msg) => { S.status[c.id] = v; save(); toast(msg); renderCandidate(id); };
  $('#pass').onclick = () => set('passed', `Passed on ${c.name.split(' ')[0]}`);
  $('#intro').onclick = () => set('intro', `Jill is asking ${c.name.split(' ')[0]}'s Jack about an intro`);
  $('#save').onclick = () => set('saved', `Saved ${c.name.split(' ')[0]}`);
}

// ------------------------------------------------------------------ other tabs
function renderBrief() {
  const brief = S.data.briefPriorities, cur = S.versions.length ? S.versions[S.versions.length - 1].priorities : brief;
  $('#body').innerHTML = `<div class="grid2">
    <div class="card"><div class="cap" style="margin-bottom:10px">Evaluation criteria</div>
      <div class="prio" style="color:var(--ink-40);font-size:12px;text-transform:uppercase;letter-spacing:.05em"><span>Criterion</span><span style="text-align:right">Written</span><span style="text-align:right">In use</span></div>
      ${crit().map((c, k) => `<div class="prio"><span>${esc(c.label)}</span><span class="v">${pct(brief[k])}</span><span class="v" style="color:${Math.abs(cur[k] - brief[k]) > 0.04 ? 'var(--ink)' : 'var(--ink-60)'}">${pct(cur[k])}</span></div>`).join('')}
      <p class="note" style="margin-top:12px">${S.versions.length ? 'Jill ranks with the calibrated priorities; your written brief is kept for reference.' : 'Calibrate on the Candidates tab to replace these guesses with priorities learned from your picks.'}</p></div>
    <div class="card"><div class="cap">Brief history</div><div class="hist">
      ${[...S.versions].reverse().map((v, j) => `<div><span>v${S.versions.length - j + 1} · calibrated from ${v.picks} picks</span><span class="sub">${new Date(v.at).toLocaleString()}</span></div>`).join('')}
      <div><span>v1 · written with Jill</span><span class="sub">setup conversation</span></div></div>
      ${S.versions.length ? '<div class="actions" style="margin-top:12px"><button class="btn ghost" id="restore">Restore v1</button></div>' : ''}</div>
    <div class="card"><div class="cap" style="margin-bottom:10px">Role summary</div><p style="font-size:15px;line-height:1.6">First AI engineer at a 14-person clinical-documentation startup. Owns the LLM pipeline from prototype to production and sits with clinicians weekly.</p></div>
    <div class="card"><div class="cap" style="margin-bottom:10px">Compensation and logistics</div><p style="font-size:15px;line-height:1.6">${esc(S.data.role.band)} · ${esc(S.data.role.type)} · London · visa sponsorship available.</p></div></div>`;
  const r = $('#restore'); if (r) r.onclick = () => { S.versions = []; save(); toast('Restored v1'); route(); };
}
function renderPitch() {
  $('#body').innerHTML = `<div class="card" style="max-width:760px"><div class="cap" style="margin-bottom:10px">Candidate pitch</div><p style="font-size:16px;line-height:1.65">Northwind Health turns clinic conversations into notes doctors trust. You'd be the first AI engineer: own the LLM pipeline end to end, sit with clinicians every week, and help hire the next three engineers. Seed-funded, growing 20% a month.</p><p class="note" style="margin-top:12px">This is what Jack shares with candidates. They never see the brief.</p></div>`;
}
function renderPipeline() {
  const c = S.data.candidates, by = (v) => c.filter((x) => S.status[x.id] === v);
  const lanes = [['Saved', by('saved')], ['Intro requested', by('intro')], ['Intro made', []], ['Archived', by('passed')]];
  $('#body').innerHTML = `<div class="board">${lanes.map(([t, xs]) => `<div class="lane"><h4>${t}<span class="mono">${xs.length}</span></h4>${xs.map((x) => `<a class="mini" href="#/c/${x.id}">${avatar(x)}${esc(x.name)}</a>`).join('') || '<p class="note" style="padding:4px">Nothing here yet.</p>'}</div>`).join('')}</div>`;
}
function renderConfigure() {
  $('#body').innerHTML = `<div class="card" style="max-width:640px"><div class="cap" style="margin-bottom:12px">Intro emails</div><label class="sub" for="cal-link">Scheduling link</label>
    <div class="search" style="margin-top:6px"><input id="cal-link" placeholder="https://cal.com/you/intro" value="${esc(localStorage.getItem(STORE + '-link') || '')}"></div>
    <div class="actions" style="margin-top:12px"><button class="btn" id="savelink">Save</button></div></div>`;
  $('#savelink').onclick = () => { localStorage.setItem(STORE + '-link', $('#cal-link').value); toast('Saved'); };
}

function renderReport() {
  const R = S.results, b = R.strategies.bald.meanPrecisionAt10, rd = R.strategies.random.meanPrecisionAt10, nz = R.strategies.bald_noisy.meanPrecisionAt10;
  const W = 620, H = 260, L = 40, Rm = 12, T = 12, B = 30, n = b.length - 1;
  const x = (q) => L + (q / n) * (W - L - Rm), y = (v) => T + (1 - (v - 0.5) / 0.4) * (H - T - B);
  const line = (a, col, dash = '') => `<path d="${a.map((v, q) => `${q ? 'L' : 'M'}${x(q)},${y(v)}`).join('')}" fill="none" stroke="${col}" stroke-width="2.4" ${dash ? `stroke-dasharray="${dash}"` : ''}/>${a.map((v, q) => `<circle cx="${x(q)}" cy="${y(v)}" r="3" fill="${col}" data-tip="${q} picks: ${pct(v, 1)}"/>`).join('')}`;
  $('#body').innerHTML = `<div class="grid2" style="grid-template-columns:minmax(0,7fr) minmax(0,5fr)">
    <div class="card"><div class="cap">Precision@10 vs number of picks</div><p class="sub" style="margin:6px 0 12px">${R.managers} simulated hiring managers whose real priorities differ from what the brief says (weights drawn around the brief, concentration ${R.model.concentration}). Same 101-person pool.</p>
      <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision at 10 by number of picks">
        ${[0.5, 0.6, 0.7, 0.8, 0.9].map((v) => `<line x1="${L}" x2="${W - Rm}" y1="${y(v)}" y2="${y(v)}" stroke="#ebe9e1"/><text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${v * 100}%</text>`).join('')}
        ${b.map((_, q) => `<text x="${x(q)}" y="${H - 10}" text-anchor="middle">${q}</text>`).join('')}
        ${line(b, '#100f0f')}${line(rd, '#ada598', '5 4')}${line(nz, '#ff9463', '2 3')}
      </svg>
      <div class="legend"><span><i style="background:#100f0f"></i>Information-gain picks</span><span><i style="background:#ada598"></i>Random pairs</span><span><i style="background:#ff9463"></i>Information-gain, noisier manager</span></div></div>
    <div class="card"><div class="cap" style="margin-bottom:8px">Headline</div>
      <table class="t"><tr><td>Brief priorities only</td><td class="num">${pct(R.headline.brief, 1)}</td></tr><tr><td>After 6 information-gain picks</td><td class="num"><b>${pct(R.headline.after6, 1)}</b></td></tr><tr><td>After 6 random pairs</td><td class="num">${pct(R.headline.after6Random, 1)}</td></tr><tr><td>After 6 picks, manager twice as noisy as modelled</td><td class="num">${pct(R.headline.after6Noisy, 1)}</td></tr><tr><td>After 10 picks</td><td class="num">${pct(R.headline.after10, 1)}</td></tr></table>
      <div class="cap" style="margin:16px 0 8px">Is the confidence meter honest?</div>
      <table class="t"><tr><th>Predicted</th><th style="text-align:right">Realised</th><th style="text-align:right">Sessions</th></tr>${R.calibration.filter((c) => c.n).map((c) => `<tr><td class="mono" style="font-size:13px">${pct(c.predicted, 1)}</td><td class="num">${pct(c.realised, 1)}</td><td class="num">${c.n}</td></tr>`).join('')}</table>
      <div class="cap" style="margin:16px 0 8px">If managers stray further from the brief</div>
      <table class="t"><tr><th>Concentration</th><th style="text-align:right">Brief</th><th style="text-align:right">6 picks</th><th style="text-align:right">6 random</th></tr>${R.sensitivity.map((s) => `<tr><td class="mono" style="font-size:13px">${s.concentration}</td><td class="num">${pct(s.brief, 1)}</td><td class="num">${pct(s.after6, 1)}</td><td class="num">${pct(s.after6Random, 1)}</td></tr>`).join('')}</table>
      <p class="note" style="margin-top:12px">Lower concentration means real priorities stray further from the written brief. Simulated managers follow the answer model; real ones will be noisier.</p></div></div>`;
}

boot();
