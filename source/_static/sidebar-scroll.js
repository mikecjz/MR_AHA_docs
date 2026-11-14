// Keep the left sidebar at the same scroll position when moving between pages.
// Furo reloads the whole page on each link, so the sidebar would otherwise
// start again at the top. The position is kept for this browser tab only.
(function () {
  "use strict";
  var KEY = "mr-aha-sidebar-scroll";

  function sidebar() {
    return document.querySelector(".sidebar-drawer .sidebar-scroll");
  }

  function save() {
    var el = sidebar();
    if (!el) return;
    try {
      sessionStorage.setItem(KEY, String(el.scrollTop));
    } catch (e) {}
  }

  function restore() {
    var el = sidebar();
    if (!el) return;
    var saved = null;
    try {
      saved = sessionStorage.getItem(KEY);
    } catch (e) {}
    if (saved !== null) {
      el.scrollTop = parseInt(saved, 10) || 0;
    }
    // If the current page's entry is still out of view (for example on the
    // first visit, or after following a link from inside a page), bring it in.
    var current = el.querySelector(".current-page > a, a.current");
    if (current) {
      var a = current.getBoundingClientRect();
      var b = el.getBoundingClientRect();
      if (a.top < b.top || a.bottom > b.bottom) {
        el.scrollTop += a.top - b.top - b.height / 3;
      }
    }
    el.addEventListener("click", function (ev) {
      if (ev.target.closest("a")) save();
    });
    window.addEventListener("pagehide", save);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", restore);
  } else {
    restore();
  }
})();
