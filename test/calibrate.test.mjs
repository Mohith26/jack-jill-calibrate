import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as C from '../src/calibrate.js';
import { seedRole, study, MODEL } from '../scripts/build.mjs';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b}`);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

test('Dirichlet draws lie on the simplex with the right mean', () => {
  const r = C.rng(1), alpha = [3, 1, 2, 0.5];
  const m = [0, 0, 0, 0], N = 20000;
  for (let i = 0; i < N; i++) { const w = C.dirichlet(r, alpha); close(w.reduce((s, x) => s + x, 0), 1, 1e-12); w.forEach((x, k) => { assert.ok(x >= 0); m[k] += x / N; }); }
  m.forEach((x, k) => close(x, alpha[k] / 6.5, 0.01, `mean ${k}`));
});

test('sigmoid is symmetric and stable', () => {
  close(C.sigmoid(0), 0.5, 1e-15);
  for (const x of [0.3, 5, 40, 800]) close(C.sigmoid(x) + C.sigmoid(-x), 1, 1e-12);
  assert.ok(Number.isFinite(C.sigmoid(-1000)));
});

test('answers move the posterior toward the true weights (randomised)', () => {
  const r = C.rng(5);
  for (let rep = 0; rep < 6; rep++) {
    const w0 = [0.3, 0.25, 0.2, 0.15, 0.1];
    const truth = C.dirichlet(r, w0.map((x) => x * 3));
    const post = C.createPosterior({ w0, concentration: 3, beta: 14, particles: 3000, seed: rep + 1 });
    const err0 = C.posteriorMean(post).reduce((s, x, k) => s + Math.abs(x - truth[k]), 0);
    for (let q = 0; q < 60; q++) {
      const a = Array.from({ length: 5 }, () => r()), b = Array.from({ length: 5 }, () => r());
      const aWins = r() < C.sigmoid(14 * (dot(truth, a) - dot(truth, b)));
      C.update(post, aWins ? a : b, aWins ? b : a);
    }
    const err = C.posteriorMean(post).reduce((s, x, k) => s + Math.abs(x - truth[k]), 0);
    assert.ok(err < err0 * 0.8 || err < 0.12, `rep ${rep}: ${err0} -> ${err}`);
  }
});

test('Metropolis-Hastings rejuvenation leaves the posterior mean unchanged (within MC error)', () => {
  const post = C.createPosterior({ w0: [0.4, 0.3, 0.2, 0.1], concentration: 4, beta: 10, particles: 20000, seed: 3 });
  const r = C.rng(11);
  for (let q = 0; q < 4; q++) { const a = [r(), r(), r(), r()], b = [r(), r(), r(), r()]; C.update(post, a, b, { essFrac: 0 }); }
  const before = C.posteriorMean(post);
  C.rejuvenate(post, 6);
  const after = C.posteriorMean(post);
  before.forEach((x, k) => close(after[k], x, 0.012, `weight ${k}`));
  assert.ok(post.acceptRate > 0.2 && post.acceptRate < 0.95, `accept ${post.acceptRate}`);
  close(C.ess(post), 20000, 1e-6, 'weights reset');
});

test('BALD picks the pair with the largest expected information gain, never a repeat', () => {
  const data = seedRole(1, 30);
  const scores = data.candidates.map((c) => c.scores);
  const post = C.createPosterior({ w0: data.briefPriorities, concentration: 3, beta: 14, particles: 800, seed: 2 });
  const q = C.nextQuestion(post, scores, { pool: 8, sample: 800 });
  assert.ok(q.gain >= 0);
  // brute force over the same pool with all particles
  const top = C.rankBy(scores, C.posteriorMean(post)).slice(0, 8);
  const p = C.weights(post);
  const H = (x) => (x <= 0 || x >= 1 ? 0 : -(x * Math.log(x) + (1 - x) * Math.log(1 - x)));
  let best = -1;
  for (let i = 0; i < top.length; i++) for (let j = i + 1; j < top.length; j++) {
    let pm = 0, eh = 0;
    post.W.forEach((w, k) => { const pr = C.sigmoid(14 * (dot(w, scores[top[i]]) - dot(w, scores[top[j]]))); pm += p[k] * pr; eh += p[k] * H(pr); });
    best = Math.max(best, H(pm) - eh);
  }
  close(q.gain, best, 0.02, 'gain near brute-force maximum (systematic resample of 800)');
  const q2 = C.nextQuestion(post, scores, { pool: 8, asked: new Set([C.pairKey(q.a, q.b)]) });
  assert.notEqual(C.pairKey(q2.a, q2.b), C.pairKey(q.a, q.b));
});

test('property: information gain is non-negative for random posteriors and pairs', () => {
  const r = C.rng(8);
  for (let rep = 0; rep < 40; rep++) {
    const scores = Array.from({ length: 10 }, () => Array.from({ length: 4 }, () => r()));
    const post = C.createPosterior({ w0: [0.25, 0.25, 0.25, 0.25], concentration: 1 + 5 * r(), beta: 5 + 20 * r(), particles: 300, seed: rep + 1 });
    const q = C.nextQuestion(post, scores, { pool: 10 });
    assert.ok(q.gain >= -1e-12 && q.gain <= Math.log(2) + 1e-12);
  }
});

test('top-k membership sums to k and expected precision is 1 for a point-mass posterior', () => {
  const data = seedRole(4, 40);
  const scores = data.candidates.map((c) => c.scores);
  const post = C.createPosterior({ w0: data.briefPriorities, concentration: 3, beta: 14, particles: 500, seed: 4 });
  const mem = C.topKMembership(post, scores, 10, 500);
  close(mem.reduce((s, x) => s + x, 0), 10, 1e-9);
  const point = { ...post, W: post.W.map(() => [...data.briefPriorities]), logw: new Float64Array(post.W.length) };
  const memP = C.topKMembership(point, scores, 10, 500);
  close(C.expectedPrecision(memP, C.rankBy(scores, data.briefPriorities).slice(0, 10)), 1, 1e-9);
});

test('ranking metrics have the right values', () => {
  close(C.precisionAtK([1, 2, 3, 4], [4, 3, 9, 8], 2), 0, 1e-12);
  close(C.precisionAtK([1, 2, 3, 4], [2, 1, 9, 8], 2), 1, 1e-12);
  close(C.spearmanTop([1, 2, 3], [1, 2, 3], 3), 1, 1e-12);
  close(C.spearmanTop([1, 2, 3], [3, 2, 1], 3), -1, 1e-12);
});

test('seeded candidate pool is deterministic and well-formed', () => {
  const a = seedRole(), b = seedRole();
  assert.deepEqual(a, b);
  assert.equal(a.candidates.length, 101);
  for (const c of a.candidates) { assert.equal(c.scores.length, 5); c.scores.forEach((x) => assert.ok(x >= 0.04 && x <= 0.98)); }
  const sal = a.candidates.map((c) => c.salary).sort((x, y) => x - y);
  const q1 = sal[Math.floor(0.25 * sal.length)], q3 = sal[Math.floor(0.75 * sal.length)];
  assert.ok(q1 > 60000 && q1 < 85000 && q3 > 100000 && q3 < 130000, `IQR ${q1}-${q3} vs published GBP 70k-120k`);
});

const results = JSON.parse(fs.readFileSync(new URL('../data/results.json', import.meta.url)));

test('headline: the committed study reproduces exactly and active questions beat random ones', () => {
  const st = study(seedRole());
  assert.deepEqual(st.headline, results.headline);
  assert.ok(st.headline.after6 - st.headline.brief > 0.1, 'six answers add >10 points of precision@10');
  assert.ok(st.headline.after6 > st.headline.after6Random, 'information-gain questions beat random pairs');
  const b = st.strategies.bald.meanPrecisionAt10;
  assert.ok(b[10] >= b[0], 'more answers do not hurt on average');
});

test('confidence is calibrated: predicted expected precision matches realised precision', () => {
  for (const bin of results.calibration.filter((x) => x.n >= 100)) close(bin.predicted, bin.realised, 0.04, `bin ${bin.lo}-${bin.hi}`);
});

test('app data matches the model configuration', () => {
  const role = JSON.parse(fs.readFileSync(new URL('../data/role.json', import.meta.url)));
  assert.deepEqual(role.model, MODEL);
  assert.equal(role.candidates.length, 101);
  close(role.demoManager.reduce((s, x) => s + x, 0), 1, 1e-9);
});
