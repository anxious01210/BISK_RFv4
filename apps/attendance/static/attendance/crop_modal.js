// apps/attendance/static/attendance/crop_modal.js
(function () {
    // --- tiny helper: ensure HTMX is available (for hx-* in injected HTML)
    function ensureHtmx() {
        return new Promise((resolve) => {
            if (window.htmx) return resolve();
            const s = document.createElement("script");
            s.src = "https://unpkg.com/htmx.org@1.9.12";
            s.onload = () => resolve();
            document.head.appendChild(s);
        });
    }

    // --- modal skeleton (one overlay reused)
    let overlay;

    function ensureOverlay() {
        if (overlay) return overlay;
        overlay = document.createElement("div");
        overlay.id = "bisk-reenroll-overlay";
        overlay.style.cssText = `
      position: fixed; inset: 0; z-index: 10000; display: none;
      background: rgba(0,0,0,.5);
    `;
        overlay.innerHTML = `
      <div id="bisk-reenroll-modal" style="
        position:absolute; top: 4vh; left: 50%; transform: translateX(-50%);
        width: min(1100px, 94vw); max-height: 92vh; overflow:auto;
        border-radius: 10px; background: var(--body-bg, #111827);
        box-shadow: 0 20px 60px rgba(0,0,0,.35);
      ">
        <div style="display:flex; justify-content:flex-end; padding:8px;">
          <button type="button" id="bisk-reenroll-close" class="button">Close</button>
        </div>
        <div id="bisk-reenroll-body" style="padding:0 12px 12px;"></div>
      </div>
    `;
        document.body.appendChild(overlay);
        overlay.addEventListener("click", (ev) => {
            if (ev.target.id === "bisk-reenroll-overlay") hide();
        });
        overlay.querySelector("#bisk-reenroll-close").addEventListener("click", hide);
        return overlay;
    }

    function show() {
        ensureOverlay().style.display = "block";
        document.body.style.overflow = "hidden";
    }

    function hide() {
        if (!overlay) return;
        overlay.style.display = "none";
        const body = document.getElementById("bisk-reenroll-body");
        if (body) body.innerHTML = "";
        document.body.style.overflow = "";
    }

    async function openInModal(url) {
        show();
        const body = document.getElementById("bisk-reenroll-body");
        body.innerHTML = '<div style="padding:20px;opacity:.7">Loading…</div>';
        try {
            const res = await fetch(url, {headers: {"X-Requested-With": "XMLHttpRequest"}});
            const html = await res.text();
            body.innerHTML = html;

            // Initialize HTMX in the injected content (so #captures-browser will hx-get)
            await ensureHtmx();
            if (window.htmx && typeof window.htmx.process === "function") {
                window.htmx.process(body);
            }
        } catch (e) {
            body.innerHTML = '<div style="padding:20px;color:#f99">Failed to load content.</div>';
            console.error("Re-enroll modal load failed:", e);
        }
    }

    // --- make it callable from inline onclick on the link
    window.BISK_openReEnrollModal = function (url) {
        openInModal(url);
    };

    // --- backup: also intercept clicks via delegation (in case onclick is removed)
    document.addEventListener("click", function (ev) {
        const a = ev.target.closest("a.js-reenroll-modal");
        if (!a) return;
        ev.preventDefault();
        ev.stopPropagation();
        openInModal(a.getAttribute("href"));
    }, {capture: true});
})();
