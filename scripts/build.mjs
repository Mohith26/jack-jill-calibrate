// Seeds the synthetic role + candidate pool (data/role.json) and runs the simulation study
// (data/results.json). usage: node scripts/build.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as C from '../src/calibrate.js';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

export const CRITERIA = [
  { key: 'llm', label: 'Shipped LLM features to production' },
  { key: 'systems', label: 'Strong systems design' },
  { key: 'zero', label: 'Built something from zero' },
  { key: 'product', label: 'Owns the customer problem' },
  { key: 'lead', label: 'Has hired and led engineers' },
];
export const BRIEF_PRIORITIES = [0.32, 0.26, 0.18, 0.14, 0.10];
export const MODEL = { concentration: 3, beta: 14, particles: 4000 }; // concentration: how closely hiring managers' real weights follow the brief (sensitivity in results.json)

const FIRST = ['Priya', 'Tom', 'Amara', 'Luca', 'Sofia', 'Daniel', 'Mei', 'Kwame', 'Hannah', 'Arjun', 'Chloe', 'Ollie', 'Zara', 'Mateo', 'Grace', 'Yusuf', 'Isla', 'Ravi', 'Freya', 'Jonah', 'Aisha', 'Felix', 'Nadia', 'Sam', 'Elena', 'Kofi', 'Lily', 'Omar', 'Ruby', 'Theo', 'Anya', 'Ben', 'Leila', 'Marcus', 'Ines', 'Callum', 'Yara', 'Joe', 'Maya', 'Harvey', 'Tara', 'Niko', 'Esme', 'Adam', 'Rosa', 'Ivan', 'Alice', 'Dev', 'Noor', 'Finn', 'Clara', 'Hugo', 'Sana', 'Leo', 'Eva', 'Max', 'Jia', 'Rory', 'Asha', 'Seb'];
const LAST = ['Shah', 'Okafor', 'Rossi', 'Kowalski', 'Chen', 'Mensah', 'Fischer', 'Patel', 'Martin', 'Nguyen', 'Haddad', 'Silva', 'Murphy', 'Ibrahim', 'Larsen', 'Costa', 'Novak', 'Clarke', 'Mehta', 'Dubois', 'Adeyemi', 'Brennan', 'Santos', 'Kim', 'Moreau', 'Lindqvist', 'Hughes', 'Rahman', 'Bianchi', 'Walsh'];
const COMPANIES = ['Monzo', 'Wise', 'Google DeepMind', 'Revolut', 'Palantir', 'ElevenLabs', 'Synthesia', 'Cleo', 'GoCardless', 'Deliveroo', 'Faculty', 'Cohere', 'PolyAI', 'Tractable', 'Thought Machine', 'Stripe', 'Meta', 'Ocado Technology', 'Onfido', 'Bloomberg', 'a seed-stage startup (acquired)', 'their own startup'];
// archetype mean criterion scores [llm, systems, zero, product, lead]
const ARCHETYPES = [
  { name: 'Applied ML engineer', titles: ['ML Engineer', 'Senior ML Engineer', 'Applied AI Engineer'], m: [0.85, 0.55, 0.35, 0.40, 0.25] },
  { name: 'Infrastructure engineer', titles: ['Senior Backend Engineer', 'Staff Engineer, Platform', 'Infrastructure Engineer'], m: [0.40, 0.88, 0.40, 0.35, 0.45] },
  { name: 'Founder-engineer', titles: ['Founder & CTO', 'Founding Engineer', 'Co-founder'], m: [0.60, 0.55, 0.90, 0.80, 0.50] },
  { name: 'Product engineer', titles: ['Senior Product Engineer', 'Full-stack Engineer', 'Product Engineer'], m: [0.60, 0.45, 0.60, 0.85, 0.30] },
  { name: 'Engineering lead', titles: ['Engineering Manager', 'Head of Engineering', 'Tech Lead'], m: [0.45, 0.70, 0.45, 0.55, 0.88] },
  { name: 'Forward deployed engineer', titles: ['Forward Deployed Engineer', 'Solutions Engineer', 'Deployed Engineer'], m: [0.70, 0.50, 0.55, 0.80, 0.30] },
];

const clip = (x) => Math.max(0.04, Math.min(0.98, x));
const round2 = (x) => Math.round(x * 100) / 100;

export function seedRole(seed = 20261110, n = 101) {
  const r = C.rng(seed);
  const used = new Set();
  const cands = [];
  for (let i = 0; i < n; i++) {
    const arch = ARCHETYPES[Math.floor(r() * ARCHETYPES.length)];
    let name;
    do { name = `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`; } while (used.has(name));
    used.add(name);
    const general = 0.12 * C.normal(r); // overall strength shared across criteria
    const scores = arch.m.map((m) => round2(clip(m + general + 0.11 * C.normal(r))));
    const years = Math.max(2, Math.round(4 + 6 * r() + (arch.name === 'Engineering lead' ? 4 : 0)));
    // expectations seeded from Jack & Jill's published London AI-engineer range: half expect GBP 70k-120k base
    const salary = Math.round(Math.exp(Math.log(92000) + 0.4 * C.normal(r)) / 1000) * 1000;
    cands.push({
      id: `c${String(i + 1).padStart(3, '0')}`, name, archetype: arch.name,
      title: arch.titles[Math.floor(r() * arch.titles.length)],
      company: COMPANIES[Math.floor(r() * COMPANIES.length)], years, salary,
      location: r() < 0.82 ? 'London' : r() < 0.5 ? 'Remote, UK' : 'Cambridge',
      scores,
    });
  }
  return {
    role: { id: 'founding-ai-engineer', title: 'Founding AI Engineer', company: 'Northwind Health', location: 'London', type: 'Full-time · On-site 4 days', band: '£110k–£150k + 0.4–0.8% equity' },
    criteria: CRITERIA, briefPriorities: BRIEF_PRIORITIES, model: MODEL, candidates: cands,
    source: 'Synthetic candidates (seeded). Salary expectations seeded from jackandjill.ai/roles/ai-engineer/london (half of candidates expect GBP 70k-120k base).',
  };
}

/** One simulated calibration session: returns precision@k after each answer and the confidence trace. */
export function simulate({ data, wTrue, strategy, questions = 12, k = 10, seed = 1, betaTrue }) {
  const scores = data.candidates.map((c) => c.scores);
  const post = C.createPosterior({ w0: data.briefPriorities, ...data.model, particles: 2000, seed });
  const r = C.rng(seed * 7 + 3);
  const truth = C.rankBy(scores, wTrue);
  const asked = new Set();
  const prec = [], conf = [];
  const record = () => {
    const mean = C.posteriorMean(post);
    const list = C.rankBy(scores, mean);
    prec.push(C.precisionAtK(list, truth, k));
    const mem = C.topKMembership(post, scores, k, 300);
    conf.push(C.expectedPrecision(mem, list.slice(0, k)));
  };
  record();
  for (let q = 0; q < questions; q++) {
    const pair = strategy === 'bald' ? C.nextQuestion(post, scores, { asked, sample: 250 }) : C.randomQuestion(post, scores, r, { asked });
    if (!pair) break;
    asked.add(C.pairKey(pair.a, pair.b));
    const pA = C.sigmoid((betaTrue ?? data.model.beta) * (dot(wTrue, scores[pair.a]) - dot(wTrue, scores[pair.b])));
    const aWins = r() < pA;
    C.update(post, scores[aWins ? pair.a : pair.b], scores[aWins ? pair.b : pair.a]);
    record();
  }
  return { prec, conf };
}
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

export function study(data, { managers = 120, questions = 10, seed = 99 } = {}) {
  const r = C.rng(seed);
  const alpha = data.briefPriorities.map((x) => x * data.model.concentration);
  const out = { managers, questions, k: 10, strategies: {} };
  const confPairs = [];
  for (const [name, strategy, betaTrue] of [['bald', 'bald'], ['random', 'random'], ['bald_noisy', 'bald', data.model.beta / 2]]) {
    const curves = [];
    const r2 = C.rng(seed);
    for (let m = 0; m < managers; m++) {
      const wTrue = C.dirichlet(r2, alpha);
      const res = simulate({ data, wTrue, strategy, questions, seed: 1000 + m, betaTrue });
      curves.push(res.prec);
      if (name === 'bald') res.prec.forEach((p, q) => confPairs.push([res.conf[q], p]));
    }
    const mean = Array.from({ length: questions + 1 }, (_, q) => curves.reduce((s, c) => s + (c[q] ?? c[c.length - 1]), 0) / curves.length);
    out.strategies[name] = { meanPrecisionAt10: mean.map((x) => Math.round(x * 1000) / 1000) };
  }
  // confidence calibration: predicted expected precision vs realised precision (well-specified sessions)
  const bins = [[0, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.01]].map(([lo, hi]) => {
    const sel = confPairs.filter(([c]) => c >= lo && c < hi);
    return { lo, hi: Math.min(1, hi), n: sel.length, predicted: sel.length ? sel.reduce((s, x) => s + x[0], 0) / sel.length : null, realised: sel.length ? sel.reduce((s, x) => s + x[1], 0) / sel.length : null };
  });
  out.calibration = bins;
  const b = out.strategies.bald.meanPrecisionAt10, rd = out.strategies.random.meanPrecisionAt10;
  out.headline = { brief: b[0], after6: b[Math.min(6, questions)], after6Random: rd[Math.min(6, questions)], after10: b[questions], after6Noisy: out.strategies.bald_noisy.meanPrecisionAt10[Math.min(6, questions)] };
  void r;
  return out;
}

/** The scripted demo hiring manager whose answers drive "Let Jill simulate". */
export function demoManager() { return [0.14, 0.12, 0.30, 0.34, 0.10]; }

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const t0 = Date.now();
  const data = seedRole();
  data.demoManager = demoManager();
  fs.mkdirSync(path.join(ROOT, 'data'), { recursive: true });
  fs.writeFileSync(path.join(ROOT, 'data/role.json'), JSON.stringify(data));
  const st = study(data);
  const sensitivity = [2, 6].map((c) => { const d2 = { ...data, model: { ...data.model, concentration: c } }; const s2 = study(d2, { managers: 80, questions: 6, seed: 7 }); return { concentration: c, brief: s2.headline.brief, after6: s2.headline.after6, after6Random: s2.headline.after6Random }; });
  st.sensitivity = [{ concentration: data.model.concentration, brief: st.headline.brief, after6: st.headline.after6, after6Random: st.headline.after6Random }, ...sensitivity];
  const results = { generated: 'node scripts/build.mjs', model: MODEL, criteria: CRITERIA.map((c) => c.label), briefPriorities: BRIEF_PRIORITIES, ...st };
  fs.writeFileSync(path.join(ROOT, 'data/results.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ sensitivity: st.sensitivity, headline: st.headline, bald: st.strategies.bald.meanPrecisionAt10, random: st.strategies.random.meanPrecisionAt10, noisy: st.strategies.bald_noisy.meanPrecisionAt10, calibration: st.calibration }, null, 1));
  console.log('built in', Date.now() - t0, 'ms');
}
