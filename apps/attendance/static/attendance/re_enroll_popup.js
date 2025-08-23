(function () {
    // Delegate clicks to links with .js-reenroll-pop
    document.addEventListener('click', function (e) {
        const a = e.target.closest && e.target.closest('a.js-reenroll-pop');
        if (!a) return;
        // If the browser blocks popups, the fallback (target=_blank) will still open a tab.
        e.preventDefault();
        const name = 'reEnroll-' + (a.dataset.id || 'win');
        const feat = [
            'noopener', 'noreferrer',
            'toolbar=0', 'menubar=0', 'location=0', 'status=0',
            'scrollbars=1', 'resizable=1',
            'width=1280', 'height=800'
        ].join(',');
        window.open(a.href, name, feat);
    }, {capture: true});
})();
