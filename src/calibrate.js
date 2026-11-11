// Calibrate: learn what a hiring manager actually values from a few pairwise
// "who would you rather meet?" answers, and re-rank Jill's shortlist.
//
// Model. Each candidate i has Jill's criterion scores s_i in [0,1]^K (the evaluation criteria
// in the hiring brief). The hiring manager ranks by u_i = w . s_i with unknown weights w on the
// simplex. Prior: w ~ Dirichlet(c * w0), w0 = the brief's stated priorities. Answer model
// (Bradley-Terry / logistic): P(A preferred to B | w) = sigmoid(beta * w . (s_A - s_B)).
// Posterior: weighted particles, rejuvenated by Metropolis-Hastings in additive-logistic
// coordinates when the effective sample size drops. Next question: the pair with the largest
// expected information gain about w (BALD), among the current top candidates. Stop when the
// posterior expected precision@k of the shown top-k reaches a target.
//
// Pure functions; used by the app (browser) and by scripts/tests (node).

export function rng(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function normal(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
export function gamma(r, k) { // Marsaglia-Tsang
  if (k < 1) return gamma(r, k + 1) * Math.pow(r(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) { let x, v; do { x = normal(r); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r(); if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v; }
}
export function dirichlet(r, alpha) { const g = alpha.map((a) => gamma(r, a)); const s = g.reduce((x, y) => x + y, 0); return g.map((x) => x / s); }

export const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const H = (p) => (p <= 0 || p >= 1 ? 0 : -(p * Math.log(p) + (1 - p) * Math.log(1 - p)));

/** Create a posterior over weights. opts: {w0, concentration, beta, particles, seed}. */
export function createPosterior(opts) {
  const { w0, concentration = 8, beta = 14, particles = 4000, seed = 1 } = opts;
  const r = rng(seed);
  const alpha = w0.map((x) => Math.max(0.05, x * concentration));
  const W = Array.from({ length: particles }, () => dirichlet(r, alpha));
  return { K: w0.length, alpha, beta, W, logw: new Float64Array(particles), answers: [], r, rejuvenations: 0 };
}

export function weights(post) {
  const m = Math.max(...post.logw);
  const e = Array.from(post.logw, (x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
export function ess(post) { const p = weights(post); return 1 / p.reduce((s, x) => s + x * x, 0); }

function loglik(post, w) {
  let L = 0;
  for (const a of post.answers) L += Math.log(Math.max(1e-300, sigmoid(post.beta * dot(w, a.diff))));
  return L;
}

/** Record an answer: candidate `winner` preferred over `loser` (score vectors). */
export function update(post, winnerScores, loserScores, opts = {}) {
  const diff = winnerScores.map((x, i) => x - loserScores[i]);
  post.answers.push({ diff });
  for (let j = 0; j < post.W.length; j++) post.logw[j] += Math.log(Math.max(1e-300, sigmoid(post.beta * dot(post.W[j], diff))));
  if (ess(post) < post.W.length * (opts.essFrac ?? 0.5)) rejuvenate(post, opts.mhSteps ?? 4);
  return post;
}

// additive-logistic coordinates: w = softmax([t_1..t_{K-1}, 0]); density of t under
// Dirichlet(alpha) is proportional to prod_i w_i^{alpha_i} (Jacobian prod_i w_i).
const toT = (w) => w.slice(0, -1).map((x) => Math.log(x / w[w.length - 1]));
const fromT = (t) => { const e = [...t.map(Math.exp), 1]; const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
const logPriorT = (post, w) => w.reduce((s, x, i) => s + post.alpha[i] * Math.log(Math.max(1e-300, x)), 0);

/** Systematic resampling followed by random-walk Metropolis-Hastings moves (posterior-invariant). */
export function rejuvenate(post, steps = 4, scale = 0.25) {
  const p = weights(post), n = post.W.length, r = post.r;
  const out = []; let u = r() / n, c = p[0], i = 0;
  for (let j = 0; j < n; j++) { const t = u + j / n; while (t > c && i < n - 1) c += p[++i]; out.push(post.W[i]); }
  let accepted = 0;
  post.W = out.map((w) => {
    let t = toT(w), cur = logPriorT(post, w) + loglik(post, w);
    for (let s = 0; s < steps; s++) {
      const t2 = t.map((x) => x + scale * normal(r)), w2 = fromT(t2);
      const prop = logPriorT(post, w2) + loglik(post, w2);
      if (Math.log(r()) < prop - cur) { t = t2; cur = prop; accepted++; }
    }
    return fromT(t);
  });
  post.logw = new Float64Array(n);
  post.rejuvenations++;
  post.acceptRate = accepted / (n * steps);
  return post;
}

export function posteriorMean(post) {
  const p = weights(post), m = new Array(post.K).fill(0);
  post.W.forEach((w, j) => { for (let k = 0; k < post.K; k++) m[k] += p[j] * w[k]; });
  return m;
}
/** Weighted quantile of each weight (for intervals). */
export function weightQuantiles(post, qs = [0.1, 0.9]) {
  const p = weights(post);
  return Array.from({ length: post.K }, (_, k) => {
    const xs = post.W.map((w, j) => [w[k], p[j]]).sort((a, b) => a[0] - b[0]);
    return qs.map((q) => { let c = 0; for (const [x, wt] of xs) { c += wt; if (c >= q) return x; } return xs[xs.length - 1][0]; });
  });
}

export function rankBy(scores, w) {
  return scores.map((s, i) => [dot(w, s), i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map((x) => x[1]);
}

/**
 * Posterior probability that each candidate is in the hiring manager's top-k, from a
 * subsample of particles; and the expected precision@k of a proposed top-k list.
 */
export function topKMembership(post, scores, k, sample = 600) {
  const p = weights(post), n = post.W.length;
  const idx = n <= sample ? [...Array(n).keys()] : systematicIdx(p, sample, post.r);
  const wt = n <= sample ? p : idx.map(() => 1 / sample);
  const prob = new Float64Array(scores.length);
  idx.forEach((j, q) => { for (const i of rankBy(scores, post.W[j]).slice(0, k)) prob[i] += wt[q]; });
  return Array.from(prob);
}
function systematicIdx(p, m, r) {
  const out = []; let u = r() / m, c = p[0], i = 0;
  for (let j = 0; j < m; j++) { const t = u + j / m; while (t > c && i < p.length - 1) c += p[++i]; out.push(i); }
  return out;
}
export function expectedPrecision(membership, list) { return list.reduce((s, i) => s + membership[i], 0) / list.length; }

/**
 * Next question: the pair (among the `pool` best candidates by posterior mean utility) with the
 * largest expected information gain about w: H(E[p]) - E[H(p)] (BALD).
 */
export function nextQuestion(post, scores, opts = {}) {
  const pool = opts.pool ?? 16, sample = opts.sample ?? 500, asked = opts.asked ?? new Set();
  const mean = posteriorMean(post);
  const top = rankBy(scores, mean).slice(0, pool);
  const p = weights(post);
  const idx = systematicIdx(p, Math.min(sample, post.W.length), post.r);
  const U = idx.map((j) => top.map((i) => dot(post.W[j], scores[i])));
  let best = null;
  for (let a = 0; a < top.length; a++) for (let b = a + 1; b < top.length; b++) {
    const key = pairKey(top[a], top[b]);
    if (asked.has(key)) continue;
    let pm = 0, eh = 0;
    for (let q = 0; q < idx.length; q++) { const pr = sigmoid(post.beta * (U[q][a] - U[q][b])); pm += pr; eh += H(pr); }
    pm /= idx.length; eh /= idx.length;
    const gain = H(pm) - eh;
    if (!best || gain > best.gain) best = { a: top[a], b: top[b], gain, pA: pm };
  }
  return best;
}
export const pairKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

export function randomQuestion(post, scores, r, opts = {}) {
  const pool = opts.pool ?? 16, asked = opts.asked ?? new Set();
  const top = rankBy(scores, posteriorMean(post)).slice(0, pool);
  for (let t = 0; t < 200; t++) {
    const a = top[Math.floor(r() * top.length)], b = top[Math.floor(r() * top.length)];
    if (a !== b && !asked.has(pairKey(a, b))) return { a, b };
  }
  return null;
}

/** Precision@k of a ranking against the truth. */
export function precisionAtK(list, truthTop, k) { const t = new Set(truthTop.slice(0, k)); return list.slice(0, k).filter((i) => t.has(i)).length / k; }

/** Kendall tau-b style agreement between two rankings of the same items (top-n). */
export function spearmanTop(listA, listB, n) {
  const pos = new Map(listB.map((x, i) => [x, i]));
  const a = listA.slice(0, n);
  let d2 = 0; a.forEach((x, i) => { const j = pos.get(x); d2 += (i - j) ** 2; });
  return 1 - (6 * d2) / (n * (n * n - 1));
}
