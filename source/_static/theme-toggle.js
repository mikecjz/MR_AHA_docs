// Two-state light/dark toggle in place of Furo's three-state auto/light/dark.
// A visit starts in the system's colour scheme. Clicking the toggle switches
// between light and dark, and the choice is kept while moving between pages in
// this browser tab (sessionStorage), so a new visit follows the system again.
//
// Furo's inline script at the top of <body> sets the theme from localStorage
// "theme" before the first paint. This file loads at the end of <body>, sets
// the theme itself, and writes the value to localStorage "theme" so the next
// page starts in it without a flash.
(function () {
  "use strict";
  var KEY = "mr-aha-theme";
  var query = window.matchMedia
    ? window.matchMedia("(prefers-color-scheme: dark)")
    : null;

  function systemTheme() {
    return query && query.matches ? "dark" : "light";
  }

  function chosenTheme() {
    try {
      var t = sessionStorage.getItem(KEY);
      if (t === "light" || t === "dark") return t;
    } catch (e) {}
    return null;
  }

  function apply(theme) {
    try {
      localStorage.setItem("theme", theme);
    } catch (e) {}
    if (document.body) document.body.dataset.theme = theme;
  }

  apply(chosenTheme() || systemTheme());

  // Follow changes to the system scheme until the reader picks one.
  if (query) {
    var onChange = function () {
      if (!chosenTheme()) apply(systemTheme());
    };
    if (query.addEventListener) query.addEventListener("change", onChange);
    else if (query.addListener) query.addListener(onChange);
  }

  // Runs in the capture phase, before Furo's own click handler on the button,
  // and stops the event so Furo does not cycle through "auto".
  document.addEventListener(
    "click",
    function (event) {
      var btn = event.target.closest && event.target.closest(".theme-toggle");
      if (!btn) return;
      event.preventDefault();
      event.stopPropagation();
      var next = document.body.dataset.theme === "dark" ? "light" : "dark";
      try {
        sessionStorage.setItem(KEY, next);
      } catch (e) {}
      apply(next);
    },
    true
  );
})();
