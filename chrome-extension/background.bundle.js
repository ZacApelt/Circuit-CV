(() => {
  var t = {
      589: (t, e, n) => {
        var r;
        !(function t(e, n, r) {
          function o(a, s) {
            if (!n[a]) {
              if (!e[a]) {
                if (i) return i(a, !0);
                throw new Error("Cannot find module '" + a + "'");
              }
              var c = (n[a] = { exports: {} });
              e[a][0].call(
                c.exports,
                function (t) {
                  var n = e[a][1][t];
                  return o(n || t);
                },
                c,
                c.exports,
                t,
                e,
                n,
                r
              );
            }
            return n[a].exports;
          }
          for (var i = void 0, a = 0; a < r.length; a++) o(r[a]);
          return o;
        })(
          {
            1: [
              function (t, e, n) {
                (function (n) {
                  "use strict";
                  function r(t) {
                    return (
                      (t = t || {}),
                      (this.options = {}),
                      o.extend(this.options, this.defaults, t),
                      this.quality(this.options.quality),
                      this
                    );
                  }
                  var o = t("./utils"),
                    i = t("./polyfills"),
                    a = o.isElectron(),
                    s = o.isNW(),
                    c = o.isBrowser(),
                    u = c || a || s;
                  (r.prototype.defaults = {
                    canvas: null,
                    quality: 92,
                    maxQuality: 100,
                    minQuality: 1,
                    bufsize: 4096,
                  }),
                    (r.prototype.loadImageFromMemory = function (t) {
                      var e = this.options,
                        n = (t = t || this.originalImage).width,
                        r = t.height,
                        i = this.__createCanvas(n, r);
                      return (
                        i.getContext("2d").drawImage(t, 0, 0, n, r),
                        (this.canvas = i),
                        (e.imageFormat =
                          e.imageFormat || o.getImageFormat(t.src)),
                        this.originalImage || (this.originalImage = t),
                        this
                      );
                    }),
                    (r.prototype.loadImageFromUrl = function (t, e) {
                      var n = this,
                        r = this.options,
                        i = this.__createImage();
                      (r.imageFormat = r.imageFormat || o.getImageFormat(t)),
                        (i.onload = function () {
                          n.loadImageFromMemory(i), e.call(n);
                        }),
                        (i.src = t);
                    }),
                    (r.prototype.image = function (t, e) {
                      var n = this.options,
                        r = o.type(t);
                      if (
                        "String" !== r &&
                        "Image" !== r &&
                        "HTMLImageElement" !== r
                      )
                        throw new Error("invalid arguments");
                      if ("String" === r) {
                        if (!e)
                          throw new Error(
                            "callback must be specified when load from path"
                          );
                        (n.imageFormat = n.imageFormat || o.getImageFormat(t)),
                          this.loadImageFromUrl(t, function () {
                            e.call(this);
                          });
                      } else if ("Image" === r || "HTMLImageElement" === r)
                        return (
                          (n.imageFormat =
                            n.imageFormat || o.getImageFormat(t.src)),
                          this.loadImageFromMemory(t),
                          e &&
                            "Function" === o.type(e) &&
                            (e.call(this),
                            console.warn(
                              "No need to specify callback when load from memory, please use chain-capable method directly like this: clipper(Image).crop(...).resize(...)"
                            )),
                          this
                        );
                    }),
                    (r.prototype.crop = function (t, e, n, r) {
                      var o = this.canvas
                          .getContext("2d")
                          .getImageData(t, e, n, r),
                        i = this.__createCanvas(n, r),
                        a = i.getContext("2d");
                      return (
                        a.rect(0, 0, n, r),
                        (a.fillStyle = "white"),
                        a.fill(),
                        a.putImageData(o, 0, 0),
                        (this.canvas = i),
                        this
                      );
                    }),
                    (r.prototype.toFile = function (t, e) {
                      var n = this,
                        r = this.options.imageFormat;
                      return (
                        this.toDataURL(function (o) {
                          c
                            ? e.call(n, o)
                            : this.dataUrlToFile(t, o, r, function () {
                                e.call(n);
                              });
                        }),
                        this
                      );
                    }),
                    (r.prototype.dataUrlToFile = function (t, e, r, o) {
                      var a = this,
                        s = e.replace("data:" + r + ";base64,", ""),
                        c = new n(s, "base64");
                      i.writeFile(t, c, function () {
                        o.call(a);
                      });
                    }),
                    (r.prototype.resize = function (t, e) {
                      var n,
                        r,
                        o = this.canvas;
                      if (!arguments.length)
                        throw new Error(
                          "resize() must be specified at least one parameter"
                        );
                      if (1 === arguments.length) {
                        if (!t)
                          throw new Error("resize() inappropriate parameter");
                        (n = t / o.width), (e = o.height * n);
                      } else !t && e && ((r = e / o.height), (t = o.width * r));
                      var i = this.__createCanvas(t, e),
                        a = i.getContext("2d");
                      return (
                        a.drawImage(o, 0, 0, t, e), (this.canvas = i), this
                      );
                    }),
                    (r.prototype.clear = function (t, e, n, r) {
                      var o = this.canvas.getContext("2d");
                      return (
                        o.clearRect(t, e, n, r),
                        (o.fillStyle = "#fff"),
                        o.fillRect(t, e, n, r),
                        this
                      );
                    }),
                    (r.prototype.quality = function (t) {
                      if ("Number" !== o.type(t) && "String" !== o.type(t))
                        throw new Error("Invalid arguments");
                      if (!t) return this;
                      var e = this.options;
                      return (
                        (t = parseFloat(t)),
                        (t = o.rangeNumber(t, e.minQuality, e.maxQuality)),
                        (e.quality = t),
                        this
                      );
                    }),
                    (r.prototype.toDataURL = function (t, e) {
                      var n = this,
                        r = this.options,
                        a = r.quality,
                        s = r.minQuality,
                        c = r.maxQuality,
                        l = r.imageFormat,
                        f = r.bufsize;
                      "string" == typeof t && (t = parseFloat(t)),
                        0 === arguments.length
                          ? (t = a)
                          : 1 === arguments.length
                          ? "number" == typeof t
                            ? (t = o.rangeNumber(t, s, c))
                            : "function" == typeof t && ((e = t), (t = a))
                          : 2 === arguments.length &&
                            (t = o.rangeNumber(t, s, c));
                      var p = this.canvas;
                      if (u) {
                        var h = p.toDataURL(l, t / 100);
                        return e && e.call(this, h), h;
                      }
                      if (!e)
                        throw new Error(
                          "toDataURL(): callback must be specified"
                        );
                      return (
                        i.toDataURL(
                          { canvas: p, imageFormat: l, quality: t, bufsize: f },
                          function (t) {
                            e.call(n, t);
                          }
                        ),
                        this
                      );
                    }),
                    (r.prototype.configure = function (t, e) {
                      var n = this.options;
                      return (
                        o.setter(n, t, e),
                        n.quality && this.quality(n.quality),
                        this
                      );
                    }),
                    (r.prototype.getCanvas = function () {
                      return this.canvas;
                    }),
                    (r.prototype.destroy = function () {
                      return (this.canvas = null), this;
                    }),
                    (r.prototype.reset = function () {
                      return this.destroy().loadImageFromMemory();
                    }),
                    (r.prototype.injectNodeCanvas = function (t) {
                      void 0 !== t && (this.options.canvas = t);
                    }),
                    (r.prototype.__createCanvas = function (t, e) {
                      var n;
                      if (u) {
                        ((n = window.document.createElement("canvas")).width =
                          t),
                          (n.height = e);
                      } else {
                        var r = this.options.canvas;
                        if (!r || !r.createCanvas)
                          throw new Error(
                            "Require node-canvas on the server-side Node.js"
                          );
                        n = r.createCanvas(t, e);
                      }
                      return n;
                    }),
                    (r.prototype.__createImage = function () {
                      var t;
                      if (u) t = window.Image;
                      else {
                        var e = this.options.canvas;
                        if (!e || !e.Image)
                          throw new Error(
                            "Require node-canvas on the server-side Node.js"
                          );
                        t = e.Image;
                      }
                      return new t();
                    }),
                    (r.__configure = function (t, e) {
                      var n = r.prototype.defaults;
                      o.setter(n, t, e),
                        n.quality &&
                          (n.quality = o.rangeNumber(
                            n.quality,
                            n.minQuality,
                            n.maxQuality
                          ));
                    }),
                    (e.exports = r);
                }).call(this, t("buffer").Buffer);
              },
              { "./polyfills": 3, "./utils": 4, buffer: 5 },
            ],
            2: [
              function (o, i, a) {
                "use strict";
                function s(t, e, n) {
                  var r;
                  switch (arguments.length) {
                    case 0:
                      r = new c();
                      break;
                    case 1:
                      "Object" === u.type(t)
                        ? (r = new c(t))
                        : (r = new c()).image(t);
                      break;
                    case 2:
                      (n = e),
                        (e = null),
                        (r = new c()).image(t, function () {
                          n.call(this);
                        });
                      break;
                    default:
                      if ("Object" !== u.type(e))
                        throw new Error("invalid arguments");
                      (r = new c(e)).image(t, function () {
                        n.call(this);
                      });
                  }
                  return r;
                }
                var c = o("./clipper"),
                  u = o("./utils");
                (s.configure = function (t, e) {
                  c.__configure(t, e);
                }),
                  ((i.exports = s).imageClipper = s),
                  void 0 ===
                    (r = function () {
                      return s;
                    }.call(e, n, e, t)) || (t.exports = r);
              },
              { "./clipper": 1, "./utils": 4 },
            ],
            3: [
              function (t, e, n) {
                "use strict";
                var r = t("fs"),
                  o = {
                    writeFile: function (t, e, n) {
                      r.writeFile(t, e, function (t) {
                        if (t) throw t;
                        n();
                      });
                    },
                    toDataURL: function (t, e) {
                      var n = t.canvas,
                        r = t.imageFormat,
                        o = t.quality,
                        i = t.bufsize;
                      "image/jpeg" === r
                        ? n.toDataURL(
                            r,
                            { quality: o, bufsize: i },
                            function (t, n) {
                              if (t) throw t;
                              e(n);
                            }
                          )
                        : n.toDataURL(r, function (t, n) {
                            if (t) throw t;
                            e(n);
                          });
                    },
                  };
                e.exports = o;
              },
              { fs: 5 },
            ],
            4: [
              function (t, e, n) {
                (function (t, n) {
                  "use strict";
                  var r = {
                    isBrowser: function () {
                      var t = r.isElectron(),
                        e = r.isNW();
                      return (
                        !t &&
                        !e &&
                        !(
                          "undefined" == typeof window ||
                          "undefined" == typeof navigator
                        )
                      );
                    },
                    isNode: function () {
                      return !(void 0 === t || !t.platform || !t.versions);
                    },
                    isNW: function () {
                      return (
                        r.isNode() &&
                        !(
                          void 0 === n ||
                          !t.__node_webkit ||
                          !t.versions["node-webkit"]
                        )
                      );
                    },
                    isElectron: function () {
                      return (
                        r.isNode() && !(void 0 === n || !t.versions.electron)
                      );
                    },
                    type: function (t) {
                      return Object.prototype.toString
                        .call(t)
                        .split(" ")[1]
                        .replace("]", "");
                    },
                    rangeNumber: function (t, e, n) {
                      return t > n ? n : t < e ? e : t;
                    },
                    each: function (t, e) {
                      var n = t.length;
                      if (n)
                        for (
                          var r = 0;
                          r < n && !1 !== e.call(t[r], t[r], r);
                          r++
                        );
                      else if (void 0 === n)
                        for (var o in t)
                          if (!1 === e.call(t[o], t[o], o)) break;
                    },
                    extend: function (t) {
                      r.each(arguments, function (e, n) {
                        n > 0 &&
                          r.each(e, function (e, n) {
                            void 0 !== e && (t[n] = e);
                          });
                      });
                    },
                    setter: function (t, e, n) {
                      var o = r.type(e);
                      if ("String" === o) {
                        if (void 0 === t[e])
                          throw new Error("Invalid configuration name.");
                        if (void 0 === n)
                          throw new Error(
                            "Lack of a value corresponding to the name"
                          );
                        "Object" === r.type(n) && "Object" === r.type(t[e])
                          ? r.extend(t[e], n)
                          : (t[e] = n);
                      } else {
                        if ("Object" !== o)
                          throw new Error("Invalid arguments");
                        (n = e), r.extend(t, n);
                      }
                    },
                    getImageFormat: function (t) {
                      var e = t.substr(t.lastIndexOf(".") + 1, t.length);
                      return "image/" + (e = "jpg" === e ? "jpeg" : e);
                    },
                    upperCaseFirstLetter: function (t) {
                      return t.replace(t.charAt(0), function (t) {
                        return t.toUpperCase();
                      });
                    },
                  };
                  e.exports = r;
                }).call(
                  this,
                  t("pBGvAp"),
                  "undefined" != typeof self
                    ? self
                    : "undefined" != typeof window
                    ? window
                    : {}
                );
              },
              { pBGvAp: 6 },
            ],
            5: [function (t, e, n) {}, {}],
            6: [
              function (t, e, n) {
                function r() {}
                var o = (e.exports = {});
                (o.nextTick = (function () {
                  var t = "undefined" != typeof window && window.setImmediate,
                    e =
                      "undefined" != typeof window &&
                      window.postMessage &&
                      window.addEventListener;
                  if (t)
                    return function (t) {
                      return window.setImmediate(t);
                    };
                  if (e) {
                    var n = [];
                    return (
                      window.addEventListener(
                        "message",
                        function (t) {
                          var e = t.source;
                          (e === window || null === e) &&
                            "process-tick" === t.data &&
                            (t.stopPropagation(), n.length > 0) &&
                            n.shift()();
                        },
                        !0
                      ),
                      function (t) {
                        n.push(t), window.postMessage("process-tick", "*");
                      }
                    );
                  }
                  return function (t) {
                    setTimeout(t, 0);
                  };
                })()),
                  (o.title = "browser"),
                  (o.browser = !0),
                  (o.env = {}),
                  (o.argv = []),
                  (o.on = r),
                  (o.addListener = r),
                  (o.once = r),
                  (o.off = r),
                  (o.removeListener = r),
                  (o.removeAllListeners = r),
                  (o.emit = r),
                  (o.binding = function (t) {
                    throw new Error("process.binding is not supported");
                  }),
                  (o.cwd = function () {
                    return "/";
                  }),
                  (o.chdir = function (t) {
                    throw new Error("process.chdir is not supported");
                  });
              },
              {},
            ],
          },
          {},
          [2]
        );
      },
    },
    e = {};
  function n(r) {
    if (e[r]) return e[r].exports;
    var o = (e[r] = { exports: {} });
    return t[r](o, o.exports, n), o.exports;
  }
  (n.n = (t) => {
    var e = t && t.__esModule ? () => t.default : () => t;
    return n.d(e, { a: e }), e;
  }),
    (n.d = (t, e) => {
      for (var r in e)
        n.o(e, r) &&
          !n.o(t, r) &&
          Object.defineProperty(t, r, { enumerable: !0, get: e[r] });
    }),
    (n.o = (t, e) => Object.prototype.hasOwnProperty.call(t, e)),
    (n.p = "/"),
    (() => {
      "use strict";
      n.p, n.p, n.p;
      var t = !0,
        e = !1,
        r = n(589),
        o = n.n(r);
      chrome.storage.sync.set({ openInTab: t }),
        chrome.storage.sync.set({ download: e });
      chrome.browserAction.setTitle({
        title:
          "Hold the Option/Alt key and drag the mouse to create partial screenshots.\nClick the icon to create full-page screenshots.",
      }),
        chrome.browserAction.onClicked.addListener(function () {
          chrome.tabs.captureVisibleTab(async function (t) {
            if (t) {
              chrome.storage.sync.get(
                ["download", "openInTab"],
                async function (e) {
                  if (e.download) {
                    chrome.downloads.download({
                      url: t,
                      filename: new Date().getTime().toString() + ".jpg",
                    });
                  }
                  if (e.openInTab) {
                    chrome.tabs.create({ url: t });
                  }
                  const data = { data: t };
                  const url = `http://localhost:5000/run-script`;
                  const response = await fetch(url, {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                    },
                    body: JSON.stringify(data),
                  });
                  const result = await response.text();
                }
              );
            }
          });
        }),
        chrome.runtime.onMessage.addListener(function (t, e, n) {
            if ("SCREENSHOT_WITH_COORDINATES" === t.msg) {
              var r = t.rect,
                i = t.windowSize;
          
              chrome.tabs.captureVisibleTab(function (t) {
                if (t) {
                  var e = t;
                  new Promise(function (t, n) {
                    var r = new Image();
                    r.onload = function () {
                      t({ w: r.width, h: r.height });
                    };
                    r.src = e;
                  }).then(async function (e) {
                    var n = e.w / i.width,
                      a = Math.floor(r.x * n),
                      s = Math.floor(r.y * n),
                      c = Math.floor(r.width * n),
                      u = Math.floor(r.height * n);
          
                    o()(t, function () {
                      this.crop(a, s, c, u).toDataURL(async function (t) {
                        chrome.storage.sync.get(["download", "openInTab"], async function (e) {
                          if (e.download) {
                            chrome.downloads.download({
                              url: t,
                              filename: new Date().getTime().toString() + ".jpg",
                            });
                          }
                          if (e.openInTab) {
                            chrome.tabs.create({ url: t });
                          }
                          // Send the cropped image data to the localhost server
                          const data = { data: t };
                          const url = `http://localhost:5000/run-script`;
                          const response = await fetch(url, {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(data),
                          });
                          const result = await response.text();
                          console.log(result); // Optionally log the result
                        });
                      });
                    });
                  });
                }
              });
            }
          });
    })();
})();


