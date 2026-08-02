/* hmviewer -- a grid of independently orbitable point clouds on ONE WebGL context.
 *
 * A browser keeps only about sixteen live WebGL contexts and silently destroys
 * the oldest beyond that, so a page cannot give every object its own canvas.
 * Instead one canvas is divided with gl.viewport/gl.scissor into cells, each
 * with its own camera. Labels are plain HTML positioned over the canvas, which
 * keeps the text crisp and selectable.
 *
 * Group file layout (little-endian):
 *   "HMR1" | uint32 jsonLen | json (padded to 4) | payload
 * with the JSON naming the colour sets and, per object, its point count and the
 * byte offset of its block:
 *   float32 xyz[3n] | uint8 rgb[set][3n] ...   (padded to 4)
 */
(function (global) {
  'use strict';

  var VS = [
    'attribute vec3 aPos;',
    'attribute vec3 aColA;',
    'attribute vec3 aColB;',
    'uniform mat4 uMVP;',
    'uniform float uSize;',
    'uniform float uMix;',
    'uniform float uGamma;',
    'uniform float uDist;',
    'varying vec3 vCol;',
    'varying float vDepth;',
    'void main(){',
    '  vec4 p = uMVP * vec4(aPos, 1.0);',
    '  gl_Position = p;',
    '  gl_PointSize = uSize / max(p.w, 0.01);',
    '  vec3 c = mix(aColA, aColB, uMix);',
    '  vCol = pow(c, vec3(uGamma));',
    /* 0 at the near side of the object, 1 at the far side */
    '  vDepth = clamp((p.w - uDist + 1.0) * 0.5, 0.0, 1.0);',
    '}'
  ].join('\n');

  /* The paper's own figures are Mitsuba renders, where every point is a lit,
     shadowed sphere. Flat discs on white read as a faint speckle next to that,
     so each point is shaded as a sphere imposter: the normal is reconstructed
     from gl_PointCoord and lit, then the far side of the object is dimmed. No
     per-point normals are needed, which is just as well -- the plys have none. */
  var FS = [
    'precision mediump float;',
    'varying vec3 vCol;',
    'varying float vDepth;',
    'void main(){',
    '  vec2 d = gl_PointCoord * 2.0 - 1.0;',
    '  float r2 = dot(d, d);',
    '  if (r2 > 1.0) discard;',
    '  vec3 n = vec3(d.x, -d.y, sqrt(max(0.0, 1.0 - r2)));',
    '  float lam = max(0.0, dot(n, normalize(vec3(-0.32, 0.46, 0.83))));',
    '  float lit = 0.68 + 0.36 * lam;',
    '  lit *= mix(1.0, 0.87, vDepth);',
    '  gl_FragColor = vec4(vCol * lit, 1.0);',
    '}'
  ].join('\n');

  /* ---------- small matrix helpers ---------- */
  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0];
  }

  function orbitView(dist, yaw, pitch) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    /* camera on a sphere looking at the origin, y up */
    var ex = dist * cp * sy, ey = dist * sp, ez = dist * cp * cy;
    var zx = ex, zy = ey, zz = ez;
    var zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
    var xx = zz, xy = 0, xz = -zx;
    var xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
    var yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return [xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx * ex + xy * ey + xz * ez),
            -(yx * ex + yy * ey + yz * ez),
            -(zx * ex + zy * ey + zz * ez), 1];
  }

  function mul(a, b) {
    var o = new Array(16);
    for (var i = 0; i < 4; i++)
      for (var j = 0; j < 4; j++) {
        var s = 0;
        for (var k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
        o[i * 4 + j] = s;
      }
    return o;
  }

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  /* ---------- parse a group file ---------- */
  function parse(buffer) {
    var dv = new DataView(buffer);
    if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1),
                            dv.getUint8(2), dv.getUint8(3)) !== 'HMR1')
      throw new Error('not a HMR1 group file');
    var jlen = dv.getUint32(4, true);
    var json = JSON.parse(new TextDecoder().decode(
      new Uint8Array(buffer, 8, jlen)));
    json.base = 8 + jlen;
    json.buffer = buffer;
    return json;
  }

  /* ---------- the grid ---------- */
  function Grid(canvas, opts) {
    opts = opts || {};
    var gl = canvas.getContext('webgl', { antialias: true, alpha: true });
    if (!gl) throw new Error('WebGL unavailable');
    this.gl = gl;
    this.canvas = canvas;
    this.cols = opts.cols || 3;
    /* The paper renders each point as a sphere big enough to overlap its
       neighbours, which is what turns 2048 samples into a readable surface
       rather than a speckle. Matching that matters more than it sounds. */
    this.pointSize = opts.pointSize || 19.0;
    this.dist = opts.dist || 2.34;
    this.gamma = opts.gamma || 1.0;
    this.bg = opts.background || null;
    this.sync = !!opts.sync;                 /* one camera for every cell */
    this.mix = 0;
    this.ia = 0; this.ib = 1;
    this.vk = 0;
    this.cells = [];
    this.spin = opts.spin === undefined ? 0.16 : opts.spin;
    this._raf = null;
    this._t = 0;

    var p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(p));
    this.prog = p;
    this.loc = {
      pos: gl.getAttribLocation(p, 'aPos'),
      ca: gl.getAttribLocation(p, 'aColA'),
      cb: gl.getAttribLocation(p, 'aColB'),
      mvp: gl.getUniformLocation(p, 'uMVP'),
      size: gl.getUniformLocation(p, 'uSize'),
      mixv: gl.getUniformLocation(p, 'uMix'),
      gamma: gl.getUniformLocation(p, 'uGamma'),
      dist: gl.getUniformLocation(p, 'uDist')
    };
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.SCISSOR_TEST);
    this._bind();
  }

  /* A cell can hold several variants of the same thing -- the corruption figure
     puts five severities behind one panel -- so objects that share a name are
     collapsed into one cell and setVariant picks between them. A file with no
     variant field simply gets one variant per cell. */
  Grid.prototype.load = function (buffer) {
    var gl = this.gl, g = parse(buffer);
    this.sets = g.sets;
    this.variantLabels = g.variantLabels || null;
    this.vk = 0;

    var cells = [], byName = {};
    g.objects.forEach(function (o) {
      var off = g.base + o.off;
      var xyz = new Float32Array(g.buffer, off, o.n * 3);
      var cbase = off + o.n * 12;
      var vp = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vp);
      gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.STATIC_DRAW);
      var vc = g.sets.map(function (_, k) {
        var b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER,
          new Uint8Array(g.buffer, cbase + k * o.n * 3, o.n * 3), gl.STATIC_DRAW);
        return b;
      });
      var v = { n: o.n, vp: vp, vc: vc };
      var cell = byName[o.name];
      if (!cell) {
        cell = byName[o.name] = { name: o.name, vars: [],
                                  yaw: 0.6, pitch: 0.26, dist: 0, touched: false };
        cells.push(cell);
      }
      cell.vars.push(v);
    });
    this.cells = cells;
    this.render();
    return this;
  };

  Grid.prototype.setVariant = function (k) {
    this.vk = Math.max(0, k | 0);
    this.render();
  };

  Grid.prototype.rows = function () {
    return Math.ceil(this.cells.length / this.cols);
  };

  /* cell rectangle in CSS pixels, origin top-left */
  Grid.prototype.rect = function (i) {
    var w = this.canvas.clientWidth / this.cols;
    var h = this.canvas.clientHeight / this.rows();
    var c = i % this.cols, r = Math.floor(i / this.cols);
    return { x: c * w, y: r * h, w: w, h: h };
  };

  Grid.prototype.hit = function (clientX, clientY) {
    var b = this.canvas.getBoundingClientRect();
    var x = clientX - b.left, y = clientY - b.top;
    if (x < 0 || y < 0 || x > b.width || y > b.height) return -1;
    var c = Math.floor(x / (b.width / this.cols));
    var r = Math.floor(y / (b.height / this.rows()));
    var i = r * this.cols + c;
    return i < this.cells.length ? i : -1;
  };

  Grid.prototype.setMix = function (m) {
    this.mix = Math.max(0, Math.min(1, m));
    this.render();
  };

  Grid.prototype.setPair = function (a, b) {
    this.ia = a; this.ib = b;
    this.render();
  };

  Grid.prototype.render = function () {
    var gl = this.gl, c = this.canvas;
    var dpr = Math.min(global.devicePixelRatio || 1, 2);
    var w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }

    gl.viewport(0, 0, w, h);
    gl.scissor(0, 0, w, h);
    if (this.bg) gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    else gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.cells.length) return;

    gl.useProgram(this.prog);
    gl.uniform1f(this.loc.mixv, this.mix);
    gl.uniform1f(this.loc.gamma, this.gamma);

    var ia = Math.min(this.ia, this.sets.length - 1);
    var ib = Math.min(this.ib, this.sets.length - 1);

    for (var i = 0; i < this.cells.length; i++) {
      var cell = this.cells[i], r = this.rect(i);
      var vx = Math.round(r.x * dpr);
      var vy = Math.round((c.clientHeight - r.y - r.h) * dpr);   /* gl y is up */
      var vw = Math.round(r.w * dpr), vh = Math.round(r.h * dpr);
      gl.viewport(vx, vy, vw, vh);
      gl.scissor(vx, vy, vw, vh);

      var yaw = cell.yaw + (cell.touched ? 0 : this._t * this.spin);
      var proj = perspective(46 * Math.PI / 180, vw / vh, 0.01, 100);
      var mvp = new Float32Array(mul(proj, orbitView(this.dist + cell.dist,
                                                     yaw, cell.pitch)));
      gl.uniformMatrix4fv(this.loc.mvp, false, mvp);
      gl.uniform1f(this.loc.size, this.pointSize * (vh / 240));
      gl.uniform1f(this.loc.dist, this.dist + cell.dist);

      var v = cell.vars[Math.min(this.vk, cell.vars.length - 1)];
      gl.bindBuffer(gl.ARRAY_BUFFER, v.vp);
      gl.enableVertexAttribArray(this.loc.pos);
      gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, v.vc[ia]);
      gl.enableVertexAttribArray(this.loc.ca);
      gl.vertexAttribPointer(this.loc.ca, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, v.vc[ib]);
      gl.enableVertexAttribArray(this.loc.cb);
      gl.vertexAttribPointer(this.loc.cb, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.drawArrays(gl.POINTS, 0, v.n);
    }
  };

  Grid.prototype._schedule = function () {
    if (this._raf) return;
    var self = this;
    this._raf = requestAnimationFrame(function () {
      self._raf = null;
      self.render();
    });
  };

  /* idle drift, so a still grid still reads as three-dimensional */
  Grid.prototype.play = function (on) {
    var self = this;
    if (this._loop) { cancelAnimationFrame(this._loop); this._loop = null; }
    if (!on) return;
    var last = null;
    (function step(now) {
      if (last !== null) self._t += Math.min(0.05, (now - last) / 1000);
      last = now;
      self.render();
      self._loop = requestAnimationFrame(step);
    })(performance.now());
  };

  Grid.prototype._bind = function () {
    var self = this, drag = null;
    var cv = this.canvas;

    cv.addEventListener('pointerdown', function (e) {
      var i = self.hit(e.clientX, e.clientY);
      if (i < 0) return;
      drag = { i: i, x: e.clientX, y: e.clientY };
      cv.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    cv.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      var targets = self.sync ? self.cells : [self.cells[drag.i]];
      targets.forEach(function (c) {
        c.touched = true;
        c.yaw += dx * 0.008;
        c.pitch = Math.max(-1.35, Math.min(1.35, c.pitch + dy * 0.008));
      });
      self._schedule();
    });
    var end = function (e) {
      if (!drag) return;
      try { cv.releasePointerCapture(e.pointerId); } catch (err) {}
      drag = null;
    };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);

    cv.addEventListener('wheel', function (e) {
      var i = self.hit(e.clientX, e.clientY);
      if (i < 0) return;
      e.preventDefault();
      var targets = self.sync ? self.cells : [self.cells[i]];
      targets.forEach(function (c) {
        c.dist = Math.max(-1.4, Math.min(3.0, c.dist + e.deltaY * 0.0016));
      });
      self._schedule();
    }, { passive: false });

    global.addEventListener('resize', function () { self._schedule(); });
  };

  Grid.prototype.reset = function () {
    this.cells.forEach(function (c) {
      c.yaw = 0.6; c.pitch = 0.26; c.dist = 0; c.touched = false;
    });
    this._t = 0;
    this.render();
  };

  global.HMGrid = Grid;
})(window);
