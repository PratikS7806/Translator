import time

import sentencepiece as spm
import streamlit as st
import torch

from mytransformer.decode import beam_search_decode
from mytransformer.masks import make_pad_mask
from mytransformer.model import Transformer

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="German to English Translator",
    page_icon="🌐",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.2rem; max-width: 820px; }
        [data-testid="stMetricLabel"] { opacity: 0.7; }
        textarea { font-size: 1.05rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

MAX_SRC_TOKENS = 64  # source sentences longer than this are truncated
MAX_OUT_LEN = 50     # fixed maximum length of the translation, in tokens
EXAMPLES = [
    "Ein Mann steht auf einer Leiter und streicht ein Haus.",
    "Zwei Hunde spielen zusammen im Schnee.",
    "Ein kleines Mädchen schaukelt auf einem Spielplatz.",
    "Eine Frau in einem roten Kleid läuft am Strand entlang.",
]


# ----------------------------------------------------------------------------
# Model loading (cached so it only happens once per server session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file="m30k_bpe.model")
    vocab_size = sp.get_piece_size()

    model = Transformer(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        d_model=256,
        heads=4,
        num_layers=3,
        d_ff=1024,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load("translation_model.pt", map_location=device))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, sp, device, n_params


# ----------------------------------------------------------------------------
# Translation helper
# ----------------------------------------------------------------------------
def translate(text, beam_width):
    bos, eos, pad = sp.bos_id(), sp.eos_id(), sp.pad_id()

    src_ids = sp.encode_as_ids(text)
    truncated = len(src_ids) > MAX_SRC_TOKENS
    src_ids = src_ids[:MAX_SRC_TOKENS]

    src_tensor = torch.tensor([bos] + src_ids + [eos]).unsqueeze(0).to(device)
    src_mask = make_pad_mask(src_tensor, pad).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = beam_search_decode(
            model,
            src_tensor,
            src_mask,
            max_len=MAX_OUT_LEN,
            bos_idx=bos,
            eos_idx=eos,
            device=device,
            beam_width=1,
        )
    elapsed = time.perf_counter() - start

    out_list = output_ids.squeeze().tolist()
    if isinstance(out_list, int):
        out_list = [out_list]
    clean = [i for i in out_list if i not in (bos, eos, pad)]

    return {
        "beam": beam_width,
        "translation": sp.decode_ids(clean),
        "src_tokens": len(src_ids),
        "out_tokens": len(clean),
        "seconds": elapsed,
        "truncated": truncated,
    }


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("input_text", "")
st.session_state.setdefault("auto_run", False)


def use_example(sentence):
    st.session_state.input_text = sentence
    st.session_state.auto_run = True


# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
try:
    model, sp, device, n_params = load_model()
except FileNotFoundError as err:
    st.error(
        f"Missing file: {err.filename}. Run the app from the project folder that "
        "contains `m30k_bpe.model` and `translation_model.pt`."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar: settings and model info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    beam_width = st.slider(
        "Beam width",
        min_value=1,
        max_value=8,
        value=3,
        help="1 is greedy decoding. Higher values explore more candidate "
        "translations but run slower.",
    )
    compare = st.toggle(
        "Compare beam widths 1, 3 and 5",
        help="Translate with greedy decoding and two beam sizes side by side.",
    )

    st.divider()
    st.sidebar.markdown("### About the model")
    st.sidebar.markdown(
    """
    - **Encoder-decoder Transformer** built from scratch in PyTorch
    - **9.6M parameters**, 3 layers, 4 heads
    - **8,000-token** shared BPE vocabulary
    - Trained on **Multi30k + OPUS-100** conversational data
    - **sacreBLEU ~41.0** on the validation set
    - Running on **CPU**
    """
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("German to English translator")
st.caption(
    "A Transformer written from scratch in PyTorch. It was trained on short "
    "image captions, so everyday scene descriptions work best."
)

# ----------------------------------------------------------------------------
# Example sentences
# ----------------------------------------------------------------------------
st.markdown("**Try an example**")
cols = st.columns(2)
for i, sentence in enumerate(EXAMPLES):
    cols[i % 2].button(
        sentence,
        key=f"example_{i}",
        on_click=use_example,
        args=(sentence,),
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# Input form (Ctrl+Enter submits)
# ----------------------------------------------------------------------------
with st.form("translate_form", border=False):
    st.text_area(
        "German sentence",
        key="input_text",
        height=110,
        placeholder="Ein Hund rennt über eine Wiese.",
    )
    submitted = st.form_submit_button("Translate", type="primary")

run = submitted or st.session_state.auto_run
st.session_state.auto_run = False

# ----------------------------------------------------------------------------
# Translate and show result
# ----------------------------------------------------------------------------
if run:
    text = st.session_state.input_text.strip()
    if not text:
        st.info("Type a German sentence above, or pick one of the examples.")
    else:
        try:
            widths = [1, 3, 5] if compare else [beam_width]
            with st.spinner("Translating..."):
                results = [translate(text, w) for w in widths]

            if any(r["truncated"] for r in results):
                st.warning(
                    f"The sentence is longer than {MAX_SRC_TOKENS} tokens, "
                    "so only the first part was translated."
                )

            with st.chat_message("user"):
                st.write(text)

            if compare:
                out_cols = st.columns(len(results))
                for col, r in zip(out_cols, results):
                    label = "Greedy" if r["beam"] == 1 else f"Beam {r['beam']}"
                    with col:
                        st.markdown(f"**{label}**")
                        st.success(r["translation"] or "(empty output)")
                        st.caption(f"{r['seconds'] * 1000:.0f} ms")
            else:
                r = results[0]
                with st.chat_message("assistant"):
                    st.success(r["translation"] or "(empty output)")
                    st.code(r["translation"], language=None)  # one-click copy

                m1, m2, m3 = st.columns(3)
                m1.metric("Time", f"{r['seconds'] * 1000:.0f} ms")
                m2.metric("Source tokens", r["src_tokens"])
                m3.metric("Output tokens", r["out_tokens"])
        except Exception as exc:  # keep the app alive on unexpected failures
            st.error(f"Translation failed: {exc}")