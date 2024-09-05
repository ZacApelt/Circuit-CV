(() => {
  "use strict";
  var e = document.body.appendChild(document.createElement("div"));
  (e.id = "screenshot-bbox"),
    (e.style.zIndex = "9999999"),
    (e.style.position = "fixed"),
    (e.style.top = "0px"),
    (e.style.left = "0px"),
    (e.style.width = "0px"),
    (e.style.height = "0px");
  var t = !1,
    i = !1,
    o = void 0,
    n = void 0,
    s = function () {
      e.style.removeProperty("top"),
        e.style.removeProperty("left"),
        e.style.removeProperty("bottom"),
        e.style.removeProperty("right"),
        e.style.removeProperty("width"),
        e.style.removeProperty("height");
    },
    d = function () {
      (t = !1),
        (i = !1),
        (o = void 0),
        (n = void 0),
        document.body.classList.remove("no-select"),
        s(),
        e.classList.remove("active");
    };
  window.addEventListener("mousedown", function (t) {
    t.altKey &&
      (document.body.classList.add("no-select"),
      (i = !0),
      (o = t.clientX),
      (n = t.clientY),
      (e.style.top = n + "px"),
      (e.style.left = o + "px"),
      (e.style.width = "0px"),
      (e.style.height = "0px"),
      e.classList.add("active"));
  });
  var l = function (e, t) {
    return [e < 0 ? 0 : e, t < 0 ? 0 : t];
  };
  window.addEventListener("mousemove", function (t) {
    if (i && void 0 !== o && void 0 !== n) {
      var d = l(t.clientX, t.clientY),
        r = d[0],
        y = d[1],
        c = Math.abs(r - o),
        a = Math.abs(y - n);
      s(),
        n <= y && o <= r
          ? ((e.style.top = n + "px"), (e.style.left = o + "px"))
          : n <= y && o >= r
          ? ((e.style.top = n + "px"),
            (e.style.right = document.body.clientWidth - o + "px"))
          : n >= y && o <= r
          ? ((e.style.bottom = window.innerHeight - n + "px"),
            (e.style.left = o + "px"))
          : n >= y &&
            o >= r &&
            ((e.style.bottom = window.innerHeight - n + "px"),
            (e.style.right = document.body.clientWidth - o + "px")),
        (e.style.width = c + "px"),
        (e.style.height = a + "px");
    }
  }),
    window.addEventListener("mouseup", function (e) {
      if (i && void 0 !== o && void 0 !== n) {
        var t = l(e.clientX, e.clientY),
          s = t[0],
          r = t[1],
          y = {
            x: Math.min(o, s),
            y: Math.min(n, r),
            width: Math.abs(o - s),
            height: Math.abs(n - r),
          },
          c = { width: window.innerWidth, height: window.innerHeight };
        d(),
          setTimeout(function () {
            chrome.runtime.sendMessage({
              msg: "SCREENSHOT_WITH_COORDINATES",
              rect: y,
              windowSize: c,
            });
          }, 1);
      }
    }),
    window.addEventListener("keydown", function (e) {
      "Escape" === e.key
        ? d()
        : e.altKey && ((t = !0), document.body.classList.add("no-select"));
    }),
    window.addEventListener("keyup", function (e) {
      t &&
        !e.altKey &&
        ((t = !1), i || document.body.classList.remove("no-select"));
    });
})();
