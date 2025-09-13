// apps/attendance/static/attendance/re_enroll_popup.js
(function () {
    if (window.__reenroll_pop_bound__) return;
    window.__reenroll_pop_bound__ = true;

    let pop = null;

    function openPopup(url) {
        const name = 'reEnrollPop';
        const feat = 'width=1100,height=720,menubar=no,toolbar=no,location=no,status=no,resizable=yes,scrollbars=yes';
        pop = window.open(url, name, feat);
        try {
            pop && pop.focus();
        } catch (e) {
        }
    }

    // capture-phase handler to beat default target="_blank" if it ever sneaks back
    document.addEventListener('click', function (e) {
        const a = e.target.closest('a.js-reenroll-pop');
        if (!a) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        openPopup(a.href);
        return false;
    }, true);
})();

// function updateSelectedField() {
//     const selected = Array.from(document.querySelectorAll('.card.sel.is-selected'))
//         .map(card => card.dataset.rel);
//     const input = document.getElementById('selected-input');
//     if (input) input.value = JSON.stringify(selected);
// }
//
// // Ensure it runs right before the re-enroll form POSTs
// document.addEventListener("htmx:beforeRequest", function (evt) {
//     const form = evt.detail.elt.closest("form");
//     if (form && form.id === "reenroll-form") {
//         updateSelectedField();
//     }
// });


// === Re-enroll: put current selection into the hidden input ===
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('reenroll-form');
    if (!form) return;

    form.addEventListener('submit', function () {
        // Only image cards are selectable: they have class "card sel".
        // We also require the "is-selected" class to be set by your toggle code.
        const selected = Array.from(
            document.querySelectorAll('#captures-browser .card.sel.is-selected, .captures-browser .card.sel.selected, #captures-browser .card.sel.selected')
        ).map(card => card.dataset.rel).filter(Boolean);

        const input = document.getElementById('selected-input'); // <input name="selected">
        if (input) input.value = JSON.stringify(selected);
    });
});

