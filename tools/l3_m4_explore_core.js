// Computational core of the Module 4 explorer (inlined into media/level3/m4-explore.html by tools/build_l3_m4.py,
// and tested in Node against the lab's numbers). Qubit q is bit q of a basis index, as in qsim and Qiskit.
var M4 = (function () {
  // ------------------------------------------------------------ kernels (2 features, 2 qubits)
  function zzState(x, reps) {               // the ZZ feature map of the lab, as [re0, im0, re1, im1, ...]
    var re = [1, 0, 0, 0], im = [0, 0, 0, 0], k, t;
    function h(q) {
      var m = 1 << q, s = Math.SQRT1_2;
      for (k = 0; k < 4; k++) if (!(k & m)) {
        var j = k | m, ar = re[k], ai = im[k], br = re[j], bi = im[j];
        re[k] = s * (ar + br); im[k] = s * (ai + bi); re[j] = s * (ar - br); im[j] = s * (ai - bi);
      }
    }
    function p(q, a) {
      var c = Math.cos(a), s = Math.sin(a), m = 1 << q;
      for (k = 0; k < 4; k++) if (k & m) { t = re[k]; re[k] = c * t - s * im[k]; im[k] = s * t + c * im[k]; }
    }
    function cx(a, b) {                      // control a, target b
      var ma = 1 << a, mb = 1 << b;
      for (k = 0; k < 4; k++) if ((k & ma) && !(k & mb)) {
        var j = k | mb; t = re[k]; re[k] = re[j]; re[j] = t; t = im[k]; im[k] = im[j]; im[j] = t;
      }
    }
    for (var r = 0; r < reps; r++) {
      h(0); h(1); p(0, 2 * x[0]); p(1, 2 * x[1]);
      cx(0, 1); p(1, 2 * (Math.PI - x[0]) * (Math.PI - x[1])); cx(0, 1);
    }
    return [re, im];
  }
  function zzKernel(x, y, reps) {
    var a = zzState(x, reps), b = zzState(y, reps), sr = 0, si = 0;
    for (var k = 0; k < 4; k++) { sr += a[0][k] * b[0][k] + a[1][k] * b[1][k]; si += a[0][k] * b[1][k] - a[1][k] * b[0][k]; }
    return sr * sr + si * si;
  }
  function rbfKernel(x, y, gamma) {
    var d0 = x[0] - y[0], d1 = x[1] - y[1];
    return Math.exp(-gamma * (d0 * d0 + d1 * d1));
  }

  // ------------------------------------------------------------ the 4-spin Ising chain
  var N = 4, D = 16;
  function zbit(k, q) { return (k >> q) & 1 ? -1 : 1; }
  function hamiltonian(J, h) {               // real symmetric 16 x 16: -J sum Z_i Z_(i+1) - h sum X_i
    var H = [], k, q;
    for (k = 0; k < D; k++) { H.push(new Array(D).fill(0)); }
    for (k = 0; k < D; k++) {
      for (q = 0; q < N - 1; q++) H[k][k] += -J * zbit(k, q) * zbit(k, q + 1);
      for (q = 0; q < N; q++) H[k][k ^ (1 << q)] += -h;
    }
    return H;
  }
  function jacobi(A0) {                      // eigenvalues and eigenvectors (columns) of a real symmetric matrix
    var n = A0.length, A = A0.map(function (r) { return r.slice(); }), V = [], i, j, k;
    for (i = 0; i < n; i++) { V.push(new Array(n).fill(0)); V[i][i] = 1; }
    for (var sweep = 0; sweep < 100; sweep++) {
      var off = 0;
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-26) break;
      for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        var th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        var t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        var c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (k = 0; k < n; k++) {
          var akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
        }
        for (k = 0; k < n; k++) {
          var apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
        }
        for (k = 0; k < n; k++) {
          var vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    return { E: A.map(function (r, i) { return r[i]; }), V: V };
  }
  function magOfProbs(pk) { var m = 0; for (var k = 0; k < D; k++) { var z = 0; for (var q = 0; q < N; q++) z += zbit(k, q); m += pk[k] * z / N; } return m; }
  function exactMag(eig, t) {               // M(t) for exp(-iHt)|0000>
    var amp_re = new Array(D).fill(0), amp_im = new Array(D).fill(0), pk = [];
    for (var a = 0; a < D; a++) {
      var w = eig.V[0][a], c = Math.cos(eig.E[a] * t), s = -Math.sin(eig.E[a] * t);
      for (var k = 0; k < D; k++) { amp_re[k] += eig.V[k][a] * c * w; amp_im[k] += eig.V[k][a] * s * w; }
    }
    for (var k2 = 0; k2 < D; k2++) pk.push(amp_re[k2] * amp_re[k2] + amp_im[k2] * amp_im[k2]);
    return magOfProbs(pk);
  }
  // density matrix as two Float64Arrays (re, im), row-major 16 x 16
  function rx(R, I, q, theta) {
    var c = Math.cos(theta / 2), s = Math.sin(theta / 2), m = 1 << q, a, b, k;
    // rows: U rho
    for (b = 0; b < D; b++) for (a = 0; a < D; a++) if (!(a & m)) {
      var a2 = a | m, i0 = a * D + b, i1 = a2 * D + b;
      var r0 = R[i0], m0 = I[i0], r1 = R[i1], m1 = I[i1];
      R[i0] = c * r0 + s * m1; I[i0] = c * m0 - s * r1;   // c x0 - i s x1
      R[i1] = c * r1 + s * m0; I[i1] = c * m1 - s * r0;   // -i s x0 + c x1
    }
    // columns: (U rho) U^dagger
    for (a = 0; a < D; a++) for (k = 0; k < D; k++) if (!(k & m)) {
      var k2 = k | m, j0 = a * D + k, j1 = a * D + k2;
      var p0 = R[j0], n0 = I[j0], p1 = R[j1], n1 = I[j1];
      R[j0] = c * p0 - s * n1; I[j0] = c * n0 + s * p1;   // c y0 + i s y1
      R[j1] = c * p1 - s * n0; I[j1] = c * n1 + s * p0;   // i s y0 + c y1
    }
  }
  function rzz(R, I, q1, q2, theta) {
    for (var a = 0; a < D; a++) for (var b = 0; b < D; b++) {
      var sa = zbit(a, q1) * zbit(a, q2), sb = zbit(b, q1) * zbit(b, q2);
      var ph = -theta / 2 * (sa - sb);
      if (ph === 0) continue;
      var c = Math.cos(ph), s = Math.sin(ph), i = a * D + b, r = R[i], m = I[i];
      R[i] = c * r - s * m; I[i] = s * r + c * m;
    }
  }
  function depol2(R, I, q1, q2, p) {          // rho -> (1 - p) rho + p I/4 (x) Tr_{q1 q2} rho
    if (p <= 0) return;
    var mask = (1 << q1) | (1 << q2), sr = new Float64Array(D * D), si = new Float64Array(D * D), a, b;
    var subs = [0, 1 << q1, 1 << q2, mask];
    for (a = 0; a < D; a++) if (!(a & mask)) for (b = 0; b < D; b++) if (!(b & mask)) {
      var tr = 0, ti = 0;
      for (var m = 0; m < 4; m++) { tr += R[(a | subs[m]) * D + (b | subs[m])]; ti += I[(a | subs[m]) * D + (b | subs[m])]; }
      sr[a * D + b] = tr; si[a * D + b] = ti;
    }
    for (a = 0; a < D; a++) for (b = 0; b < D; b++) {
      var i = a * D + b;
      R[i] *= (1 - p); I[i] *= (1 - p);
      if ((a & mask) === (b & mask)) { var j = (a & ~mask) * D + (b & ~mask); R[i] += p / 4 * sr[j]; I[i] += p / 4 * si[j]; }
    }
  }
  function trotterMag(J, h, t, n, order, p) {  // magnetization after n Trotter steps to time t, depolarizing p on each RZZ
    var R = new Float64Array(D * D), I = new Float64Array(D * D), dt = t / n, q, s;
    R[0] = 1;
    for (s = 0; s < n; s++) {
      if (order === 2) for (q = 0; q < N; q++) rx(R, I, q, -h * dt);
      for (q = 0; q < N - 1; q++) { rzz(R, I, q, q + 1, -2 * J * dt); depol2(R, I, q, q + 1, p); }
      for (q = 0; q < N; q++) rx(R, I, q, order === 2 ? -h * dt : -2 * h * dt);
    }
    var pk = []; for (var k = 0; k < D; k++) pk.push(R[k * D + k]);
    return magOfProbs(pk);
  }
  return { zzKernel: zzKernel, rbfKernel: rbfKernel, hamiltonian: hamiltonian, jacobi: jacobi, exactMag: exactMag, trotterMag: trotterMag };
})();
if (typeof module !== 'undefined') module.exports = M4;
