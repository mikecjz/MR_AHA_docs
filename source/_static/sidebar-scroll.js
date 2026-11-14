// Keep the left sidebar at the same scroll position when moving between pages.
// Furo reloads the whole page on each link, so the sidebar would otherwise
// start again at the top. This file saves the position (for this browser tab
// only); _templates/sidebar/scroll-end.html restores it before the first paint.
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

  function setup() {
    var el = sidebar();
    if (!el) return;
    // No saved position (first visit in this tab): bring the current page's
    // entry into view if it is below the fold.
    if (!el.hasAttribute("data-scroll-restored")) {
      var current = el.querySelector(".current-page > a, a.current");
      if (current) {
        var a = current.getBoundingClientRect();
        var b = el.getBoundingClientRect();
        if (a.top < b.top || a.bottom > b.bottom) {
          el.style.scrollBehavior = "auto"; // no smooth animation
          el.scrollTop += a.top - b.top - b.height / 3;
          el.style.scrollBehavior = "";
        }
      }
    }
    el.addEventListener("click", function (ev) {
      if (ev.target.closest("a")) save();
    });
    window.addEventListener("pagehide", save);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
