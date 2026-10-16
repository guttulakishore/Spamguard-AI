"""
app.py - SpamGuard AI: Smart Spam Message Detector (Streamlit interface).

This file ONLY handles the user interface and uses the model that was trained
by train_model.py. Run training first:   python train_model.py --download
Then start the app with:                 streamlit run app.py

Privacy: messages are analysed locally on your computer. They are never sent to
any external service, never opened as links, and never saved to disk.
"""

import html
import re
from pathlib import Path

import joblib
import streamlit as st

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "spam_classifier.joblib"
MAX_CHARS = 5000  # longer inputs are rejected with a friendly message

SAMPLES = {
    "🎁 Prize spam": "Congratulations! You have won a free prize. Click this link to claim now!",
    "⚠️ Urgent reward spam": "URGENT! Claim your reward now by visiting this suspicious link.",
    "📚 Friendly message": "Hey, are we meeting at the library tomorrow?",
    "📝 Assignment reminder": "Your assignment is due on Monday. Please submit it on time.",
}

# Rule-based INDICATORS used only for highlighting and explanation.
# They do NOT decide the result - the trained ML model does that.
INDICATOR_RULES = {
    "link": {
        "label": "Link / web address",
        "color": "#ef4444",
        "pattern": r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|info|xyz|top|biz|ly|click|link)\b\S*",
    },
    "urgency": {
        "label": "Urgency / pressure",
        "color": "#f97316",
        "pattern": r"\b(?:urgent(?:ly)?|immediately|act now|hurry|expires?|expiring|limited time|"
                   r"last chance|final notice|today only|within \d+ (?:hours?|minutes?)|asap)\b",
    },
    "prize": {
        "label": "Prize / reward claim",
        "color": "#a855f7",
        "pattern": r"\b(?:congratulations|congrats|you(?:'ve| have)? won|winner|prize|free|reward|"
                   r"claim|lottery|jackpot|voucher|bonus|gift card|cash)\b",
    },
    "sensitive": {
        "label": "Request for sensitive info",
        "color": "#ec4899",
        "pattern": r"\b(?:password|otp|one[- ]time (?:password|code)|pin|cvv|card number|"
                   r"bank account|account number|ssn|verify your account|confirm your (?:account|identity)|"
                   r"kyc|log ?in)\b",
    },
    "money": {
        "label": "Money amount",
        "color": "#22c55e",
        "pattern": r"[$£€₹]\s?\d[\d,\.]*|\b\d[\d,\.]*\s?(?:usd|inr|rs\.?|dollars|rupees|pounds)\b",
    },
    "phone": {
        "label": "Phone number / short code",
        "color": "#3b82f6",
        "pattern": r"\+?\d[\d\s\-]{8,}\d",
    },
}


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_bundle(path_str: str, modified_time: float):
    """Load the saved model. 'modified_time' makes the cache refresh after retraining."""
    return joblib.load(path_str)


def get_model():
    """Return (bundle, error_message). Never raises - the UI shows the message instead."""
    if not MODEL_PATH.exists():
        return None, "missing"
    try:
        return _load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime), None
    except Exception as error:  # corrupted file or scikit-learn version mismatch
        return None, f"Could not load the model file ({error}). Please run train_model.py again."


def show_model_missing(error):
    """Friendly setup instructions when the model cannot be used."""
    if error == "missing":
        st.error("🚫 Model file not found. The model has not been trained yet.")
    else:
        st.error(f"🚫 {error}")
    st.markdown(
        "**Set it up in 2 commands** (run them in the project folder, with your virtual "
        "environment activated):"
    )
    st.code("python train_model.py --download\nstreamlit run app.py", language="bash")
    st.info(
        "If the automatic download fails, download the *SMS Spam Collection* dataset manually, "
        "put the file in the `data/` folder, and run `python train_model.py`. "
        "See `data/README.md` for details."
    )


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------
def find_indicators(text: str):
    """Find suspicious-looking spans. Returns a list of (start, end, category) with no overlaps."""
    spans = []
    for key, rule in INDICATOR_RULES.items():
        for match in re.finditer(rule["pattern"], text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), key))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    cleaned, last_end = [], 0
    for start, end, key in spans:  # drop spans that overlap an earlier one
        if start >= last_end:
            cleaned.append((start, end, key))
            last_end = end
    return cleaned


def highlight_html(text: str, spans) -> str:
    """Return safe HTML with indicator phrases highlighted. Links are NOT clickable."""
    parts, cursor = [], 0
    for start, end, key in spans:
        parts.append(html.escape(text[cursor:start]))
        color = INDICATOR_RULES[key]["color"]
        parts.append(
            f'<mark style="background:{color}33;border-bottom:2px solid {color};'
            f'color:inherit;padding:0 3px;border-radius:3px;">{html.escape(text[start:end])}</mark>'
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def extra_style_notes(text: str):
    """Extra observations about writing style (also only indicators)."""
    notes = []
    words = re.findall(r"[A-Za-z]{4,}", text)
    caps = [w for w in words if w.isupper()]
    if len(caps) >= 2:
        notes.append("Several words are written in ALL CAPS.")
    if text.count("!") >= 3:
        notes.append("Many exclamation marks are used.")
    return notes


def top_model_terms(pipeline, text: str, top_n: int = 6):
    """
    Explain the model's decision: which words pushed it toward SPAM or toward HAM?
    Contribution = TF-IDF weight of the word x (log P(word|spam) - log P(word|ham)).
    """
    tfidf = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["nb"]
    vector = tfidf.transform([text]).tocoo()
    names = tfidf.get_feature_names_out()
    spam_index = list(model.classes_).index(1)
    ham_index = list(model.classes_).index(0)
    log_ratio = model.feature_log_prob_[spam_index] - model.feature_log_prob_[ham_index]

    scored = [(names[col], float(weight * log_ratio[col])) for col, weight in zip(vector.col, vector.data)]
    spam_terms = sorted([t for t in scored if t[1] > 0], key=lambda t: -t[1])[:top_n]
    ham_terms = sorted([t for t in scored if t[1] < 0], key=lambda t: t[1])[:top_n]
    return [t[0] for t in spam_terms], [t[0] for t in ham_terms]


def classify(bundle, text: str) -> dict:
    """Run the trained pipeline on one message and collect everything the UI needs."""
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    spam_score = float(probabilities[classes.index(1)])
    is_spam = spam_score >= 0.5
    spans = find_indicators(text)
    spam_terms, ham_terms = top_model_terms(pipeline, text)
    return {
        "text": text,
        "is_spam": is_spam,
        "spam_score": spam_score,
        "confidence": spam_score if is_spam else 1 - spam_score,
        "spans": spans,
        "spam_terms": spam_terms,
        "ham_terms": ham_terms,
        "style_notes": extra_style_notes(text),
    }


# ----------------------------------------------------------------------------
# Button callbacks (they run BEFORE the page is redrawn, so they can safely
# change the text box value)
# ----------------------------------------------------------------------------
def run_analysis():
    text = st.session_state.get("message_input", "")
    bundle, error = get_model()
    if bundle is None:
        st.session_state["result"] = {"error": "The model is not available. Please train it first."}
        return
    if not isinstance(text, str) or not text.strip():
        st.session_state["result"] = {"error": "Please type or paste a message first."}
        return
    if len(text) > MAX_CHARS:
        st.session_state["result"] = {
            "error": f"This message is too long ({len(text):,} characters). "
                     f"Please keep it under {MAX_CHARS:,} characters."
        }
        return
    st.session_state["result"] = classify(bundle, text.strip())


def use_sample(sample_text: str):
    st.session_state["message_input"] = sample_text
    run_analysis()  # analyse the sample immediately


def clear_all():
    st.session_state["message_input"] = ""
    st.session_state["result"] = None


# ----------------------------------------------------------------------------
# Styling (uses translucent colours so it works in both light and dark themes)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.hero {
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.30), rgba(236,72,153,0.22));
    border: 1px solid rgba(128,128,128,0.25);
}
.hero h1 { margin: 0; font-size: 2.3rem; }
.hero p  { margin: 0.3rem 0 0 0; font-size: 1.1rem; opacity: 0.85; }
.result-card {
    padding: 1.3rem 1.5rem; border-radius: 16px; margin: 0.8rem 0;
    border: 1px solid rgba(128,128,128,0.3);
}
.result-card.spam { background: rgba(239,68,68,0.14); border-left: 8px solid #ef4444; }
.result-card.ham  { background: rgba(34,197,94,0.13); border-left: 8px solid #22c55e; }
.result-card h2 { margin: 0 0 0.3rem 0; }
.msg-box {
    padding: 0.9rem 1.1rem; border-radius: 12px; line-height: 1.7;
    background: rgba(128,128,128,0.12); border: 1px solid rgba(128,128,128,0.25);
    word-break: break-word;
}
.chip {
    display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
    font-size: 0.85rem; border: 1px solid rgba(128,128,128,0.35); background: rgba(128,128,128,0.12);
}
.chip.spam { background: rgba(239,68,68,0.18); border-color: rgba(239,68,68,0.5); }
.chip.ham  { background: rgba(34,197,94,0.18); border-color: rgba(34,197,94,0.5); }
.small-note { font-size: 0.85rem; opacity: 0.75; }
</style>
"""


def render_hero():
    st.markdown(
        '<div class="hero"><h1>🛡️ SpamGuard AI</h1>'
        "<p>Detect suspicious messages in seconds.</p></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def render_result(result: dict):
    """Show the result card, confidence and detection insights."""
    if "error" in result:
        st.warning(f"⚠️ {result['error']}")
        return

    is_spam = result["is_spam"]
    confidence = result["confidence"]

    if is_spam:
        st.markdown(
            '<div class="result-card spam"><h2>🚨 SPAM</h2>'
            "The model finds this message looks similar to spam messages it was trained on. "
            "Be careful: do not click links or share personal information.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-card ham"><h2>✅ NOT SPAM (HAM)</h2>'
            "The model finds this message looks similar to normal messages. "
            "<b>This does not guarantee it is safe</b> - always stay cautious with unexpected "
            "links or requests.</div>",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model confidence:** {confidence:.1%}")
        st.progress(min(max(confidence, 0.0), 1.0))
    with col2:
        st.markdown(f"**Spam score:** {result['spam_score']:.1%}")
        st.progress(min(max(result["spam_score"], 0.0), 1.0))
    st.caption(
        "These are scores from a statistical model, not a guaranteed probability of being correct. "
        "The model can make mistakes."
    )

    st.subheader("🔍 Smart Detection Insights")
    st.caption("Indicators are hints to help you understand the result. They are not proof.")

    spans = result["spans"]
    st.markdown(
        f'<div class="msg-box">{highlight_html(result["text"], spans)}</div>',
        unsafe_allow_html=True,
    )

    found = sorted({key for _, _, key in spans})
    if found:
        legend = " ".join(
            f'<span class="chip" style="border-color:{INDICATOR_RULES[k]["color"]};">'
            f'{INDICATOR_RULES[k]["label"]}</span>'
            for k in found
        )
        st.markdown(f"<p>Indicators found: {legend}</p>", unsafe_allow_html=True)
    else:
        st.markdown("No common suspicious phrases, links or requests were highlighted.")

    for note in result["style_notes"]:
        st.markdown(f"- ✍️ {note}")

    st.markdown("**Why did the model decide this?**")
    if result["spam_terms"]:
        chips = "".join(f'<span class="chip spam">{html.escape(t)}</span>' for t in result["spam_terms"])
        st.markdown(f"Words that pushed it toward **spam**: {chips}", unsafe_allow_html=True)
    if result["ham_terms"]:
        chips = "".join(f'<span class="chip ham">{html.escape(t)}</span>' for t in result["ham_terms"])
        st.markdown(f"Words that pushed it toward **normal (ham)**: {chips}", unsafe_allow_html=True)
    if not result["spam_terms"] and not result["ham_terms"]:
        st.markdown(
            "The model did not recognise many words from this message, so its decision relies "
            "mostly on how common spam is in its training data. Treat this result with extra care."
        )

    if is_spam:
        st.info(
            "The model classified this as spam because its wording resembles known spam "
            "messages (see the words above). Highlighted indicators may add context."
        )
    else:
        st.info(
            "The model classified this as legitimate because its wording resembles normal "
            "messages. A NOT SPAM result is not a safety guarantee."
        )


def page_home():
    render_hero()
    bundle, error = get_model()
    if bundle is None:
        show_model_missing(error)
        return

    st.markdown("### ✉️ Check a message")
    st.text_area(
        "Paste an SMS, WhatsApp message, email text or any suspicious message:",
        key="message_input",
        height=180,
        placeholder="Type or paste your message here...",
    )

    col_analyze, col_clear, _ = st.columns([1, 1, 3])
    col_analyze.button("🔎 Analyze Message", type="primary", on_click=run_analysis, key="btn_analyze")
    col_clear.button("🧹 Clear", on_click=clear_all, key="btn_clear")

    st.markdown("##### 🧪 Try a sample message")
    sample_cols = st.columns(len(SAMPLES))
    for column, (label, sample_text) in zip(sample_cols, SAMPLES.items()):
        column.button(label, on_click=use_sample, args=(sample_text,), key=f"sample_{label}")

    result = st.session_state.get("result")
    if result:
        st.divider()
        render_result(result)

    st.markdown(
        '<p class="small-note">🔒 Your message is analysed locally. It is not sent anywhere, '
        "not opened as a link and not stored.</p>",
        unsafe_allow_html=True,
    )


def page_performance():
    import plotly.express as px
    import plotly.graph_objects as go

    render_hero()
    bundle, error = get_model()
    if bundle is None:
        show_model_missing(error)
        return

    metrics, meta = bundle["metrics"], bundle["meta"]
    st.markdown("### 📊 Model Performance")
    st.caption(
        f"All numbers below were calculated on the held-out test set "
        f"(messages the model never saw while training). Trained: {meta['trained_at']} "
        f"on {meta['dataset_file']}."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision (spam)", f"{metrics['precision']:.2%}")
    c3.metric("Recall (spam)", f"{metrics['recall']:.2%}")
    c4.metric("F1-score (spam)", f"{metrics['f1']:.2%}")

    c5, c6 = st.columns(2)
    c5.metric("Training samples", f"{metrics['n_train']:,}")
    c6.metric("Testing samples", f"{metrics['n_test']:,}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        figure = px.imshow(
            metrics["confusion_matrix"],
            x=["Predicted HAM", "Predicted SPAM"],
            y=["Actual HAM", "Actual SPAM"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figure)
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        st.caption(
            f"{tn} normal messages correctly kept, {fp} normal messages wrongly flagged as spam, "
            f"{fn} spam messages missed, {tp} spam messages caught."
        )
    with right:
        st.markdown("#### Spam vs legitimate messages")
        counts = metrics["class_counts_total"]
        figure = go.Figure(go.Bar(
            x=["Legitimate (ham)", "Spam"],
            y=[counts["ham"], counts["spam"]],
            marker_color=["#22c55e", "#ef4444"],
            text=[counts["ham"], counts["spam"]],
            textposition="auto",
        ))
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Number of messages")
        st.plotly_chart(figure)
        st.caption("The dataset is imbalanced: most messages are legitimate, so accuracy alone can "
                   "look better than the model really is. Check precision and recall too.")

    st.markdown("#### What do these numbers mean?")
    st.markdown(
        "- **Accuracy**: share of all test messages classified correctly.\n"
        "- **Precision**: of the messages flagged as spam, how many really were spam.\n"
        "- **Recall**: of all real spam messages, how many were caught.\n"
        "- **F1-score**: a single number that balances precision and recall."
    )


def page_about():
    render_hero()
    st.markdown("### ℹ️ About SpamGuard AI")
    st.markdown(
        """
**SpamGuard AI** checks whether a message looks like spam using a real machine-learning model.

**How it works (simple version)**
1. The model was trained on thousands of real SMS messages labelled *spam* or *ham* (not spam).
2. **TF-IDF** converts each message into numbers: words that are unusual but meaningful get more weight.
3. **Multinomial Naive Bayes** learns which words appear more in spam than in normal messages.
4. For a new message, it calculates a spam score and picks the more likely class.

**Tech stack:** Python, Streamlit, scikit-learn, pandas, NumPy, Plotly, joblib.

**Privacy**
- Messages are analysed on your own computer.
- Messages are never sent to external services, opened as links, or stored.

**Limitations - please read**
- The model can make mistakes: it may miss spam or flag normal messages.
- It was trained on English SMS messages, so it may do worse on emails, other languages or new scam styles.
- **NOT SPAM does not mean safe.** Never trust unexpected links or requests for money, passwords or OTPs.
- The highlighted indicators are hints, not proof.
        """
    )


# ----------------------------------------------------------------------------
# App entry point
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.session_state.setdefault("message_input", "")
    st.session_state.setdefault("result", None)

    st.sidebar.title("🛡️ SpamGuard AI")
    page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Model Performance", "ℹ️ About"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Detect suspicious messages in seconds.\n\nThe model can make mistakes.")

    if page == "🏠 Home":
        page_home()
    elif page == "📊 Model Performance":
        page_performance()
    else:
        page_about()


if __name__ == "__main__":
    main()
