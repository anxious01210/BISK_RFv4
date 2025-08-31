(function () {
    // Inject a single overlay modal once per page load
    function ensureOverlay() {
        var el = document.getElementById("bisk-img-overlay");
        if (el) return el;

        el = document.createElement("div");
        el.id = "bisk-img-overlay";
        el.setAttribute("role", "dialog");
        el.setAttribute("aria-modal", "true");
        el.style.cssText = [
            "position:fixed", "inset:0", "z-index:9999",
            "display:none", "align-items:center", "justify-content:center",
            "background:rgba(0,0,0,.7)"
        ].join(";");
        el.innerHTML =
            '<div id="bisk-img-wrap" style="max-width:90vw;max-height:90vh;position:relative;">' +
            '  <img id="bisk-img-full" alt="Preview" style="max-width:90vw;max-height:90vh;display:block;border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,.4);" />' +
            '  <button id="bisk-img-close" aria-label="Close" ' +
            '          style="position:absolute;top:-12px;right:-12px;cursor:pointer;border:none;border-radius:999px;' +
            '                 width:32px;height:32px;background:white;color:#111;font-weight:700;box-shadow:0 2px 10px rgba(0,0,0,.4);">×</button>' +
            '</div>';
        document.body.appendChild(el);

        // Close interactions
        var img = el.querySelector("#bisk-img-full");
        var wrap = el.querySelector("#bisk-img-wrap");
        var btn = el.querySelector("#bisk-img-close");

        function close() {
            el.style.display = "none";
            img.removeAttribute("src");
        }

        // Click backdrop closes (but not clicks on the image/wrap)
        el.addEventListener("click", function (e) {
            if (!wrap.contains(e.target)) close();
        });
        btn.addEventListener("click", function (e) {
            e.preventDefault();
            close();
        });
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape" && el.style.display !== "none") close();
        });

        return el;
    }

    function openModal(url) {
        var el = ensureOverlay();
        var img = el.querySelector("#bisk-img-full");
        img.src = url;
        el.style.display = "flex";
    }

    // Delegate clicks from any thumbnail link
    document.addEventListener("click", function (e) {
        var a = e.target.closest && e.target.closest("a.bisk-img-modal");
        if (!a) return;
        var url = a.getAttribute("data-url");
        if (!url) return;
        e.preventDefault();
        openModal(url);
    });
})();
