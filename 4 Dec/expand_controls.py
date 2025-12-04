"""
Utility helpers to expand only the Preview expanders in the app.
"""

import streamlit as st

# Session-state key used to defer expansion until the page is rendered
EXPAND_ALL_FLAG = "_expand_all_pending"

# Small JS snippet that clicks only Preview expanders with a short delay
_EXPAND_ALL_JS = """
<script>
(function() {
    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    // Return a normalized text for the button/header to inspect for "preview"
    function getButtonHeaderText(btn) {
        try {
            // Prefer aria-label or title if provided
            const aria = (btn.getAttribute('aria-label') || btn.title || '').trim();
            if (aria) return aria.toLowerCase();

            // Try to find a nearby expander header element (Streamlit render varies by version)
            const header = btn.closest('[class*=streamlit-expanderHeader], [class*=stExpander], [role="heading"], [data-testid*="expander"]');
            if (header) {
                const txt = (header.innerText || header.textContent || '').trim();
                if (txt) return txt.toLowerCase();
            }

            // Fallback to button text
            return (btn.innerText || btn.textContent || '').trim().toLowerCase();
        } catch (e) {
            return '';
        }
    }

    function isPreviewText(text) {
        if (!text) return false;
        // Normalize whitespace and punctuation, then check word boundaries
        const norm = text.replace(/[:\u2013\u2014,-]/g, ' ').replace(/\s+/g, ' ').trim();
        // Match if starts with "preview" (e.g. "Preview:") or contains the word "preview"
        return /^preview(\b|[:\s-])/i.test(norm) || /\bpreview\b/i.test(norm);
    }

    async function expandAllPreviews() {
        try {
            await delay(200);
            const buttons = Array.from(document.querySelectorAll('button[aria-expanded]'));
            buttons.forEach((b) => {
                try {
                    const expanded = b.getAttribute('aria-expanded');
                    if (expanded !== 'false') return; // only click collapsed expanders

                    const txt = getButtonHeaderText(b);
                    if (isPreviewText(txt)) {
                        b.click();
                    }
                } catch (inner) {
                    // ignore individual failures
                }
            });
        } catch (e) {
            console.warn('Expand all (preview) failed', e);
        }
    }

    expandAllPreviews();
})();
</script>
"""


def queue_expand_all() -> None:
    """
    Mark expand-all-previews to run after the page is fully rendered.
    """
    st.session_state[EXPAND_ALL_FLAG] = True


def fire_expand_all_if_pending() -> None:
    """
    Inject the expand-all-previews script if the pending flag is set.
    """
    if st.session_state.get(EXPAND_ALL_FLAG):
        st.components.v1.html(_EXPAND_ALL_JS, height=0)
        st.session_state[EXPAND_ALL_FLAG] = False


def render_generate_report_button(label: str = "Generate Report", key: str = "btn_generate_report") -> bool:
    """
    Render a button that, when clicked, queues expand-all.
    Returns True if clicked.
    """
    clicked = st.button(label, key=key)
    if clicked:
        queue_expand_all()
    return clicked
