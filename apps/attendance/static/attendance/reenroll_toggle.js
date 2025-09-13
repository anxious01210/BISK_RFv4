// apps/attendance/static/attendance/reenroll_toggle.js
(function () {
    function init() {
        const root = document.getElementById('captures-browser');
        const selInput = document.getElementById('selected_paths');
        if (!root || !selInput) {
            // DOM not ready yet (or template changed) — bail silently
            return;
        }

        function getSel() {
            try {
                return JSON.parse(selInput.value || '[]');
            } catch {
                return [];
            }
        }

        function setSel(a) {
            selInput.value = JSON.stringify(a);
        }

        function onGridClick(ev) {
            // let the “Open” button do HTMX navigation
            if (ev.target.closest('a.button')) return;

            const card = ev.target.closest('.card.sel');
            if (!card || !root.contains(card)) return;

            const rel = card.getAttribute('data-rel');
            if (!rel) return;

            const set = new Set(getSel());
            const on = card.classList.toggle('is-selected');
            on ? set.add(rel) : set.delete(rel);
            setSel([...set]);
            // debug
            // console.log('[reenroll_toggle] TOGGLED', rel, '=>', on, selInput.value);
        }

        if (!root.dataset.selBound) {
            root.dataset.selBound = '1';
            root.addEventListener('click', onGridClick);
        }

        function clearSel() {
            setSel([]);
            root.querySelectorAll('.card.sel.is-selected').forEach(el => el.classList.remove('is-selected'));
        }

        document.getElementById('btn-clear')?.addEventListener('click', clearSel);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') clearSel();
        });

        // Re-bind after HTMX replaces the grid content
        document.body.addEventListener('htmx:afterSwap', (e) => {
            if (e.target && e.target.id === 'captures-browser') {
                setSel([]);
                if (!e.target.dataset.selBound) {
                    e.target.dataset.selBound = '1';
                    e.target.addEventListener('click', onGridClick);
                }
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();



// Toggle selection on click for image cards only
document.addEventListener('click', function (e) {
    const card = e.target.closest('#captures-browser .card.sel');
    if (!card) return;
    // ignore clicks on "Open" links/folder buttons
    if (e.target.closest('a,button,[role="button"]')) return;

    card.classList.toggle('is-selected');
    updateSelectedField();  // refresh hidden JSON
});
