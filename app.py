"""
Campus France Interview Trainer
--------------------------------
Ứng dụng Streamlit giúp luyện phỏng vấn Campus France:
  1. Chọn / tải danh sách câu hỏi (JSON), hỗ trợ 2 định dạng:
       a) Dạng mảng:  [["câu hỏi gốc", "ý trả lời bắt buộc"], ...]
       b) Dạng object: [{"original_question": ..., "key_intent": ..., "key_keywords": [...]}, ...]
          (hoặc dùng key tiếng Việt "câu hỏi" / "ý trả lời")
  2. Dùng Gemini API để paraphrase câu hỏi sang tiếng Pháp
  3. Câu hỏi LUÔN bị ẩn/che mờ (blur) mặc định — chỉ hiện khi bấm "Hiện câu hỏi";
     có nút "Nghe câu hỏi" (gTTS) để nghe trước khi xem chữ.
  4. Thu âm câu trả lời bằng st.audio_input (mặc định của Streamlit), có fallback
     tải file .wav/.mp3 nếu môi trường không hỗ trợ thu âm trực tiếp.
     Chấm phát âm bằng mô hình Wav2Vec2 phoneme (so khớp edit-distance ở cấp
     phoneme, giống bộ công cụ "French Pronunciation Checker" gốc) + chấm nội
     dung bằng Gemini.
  5. Nếu lạc đề => Gemini hỏi vặn lại (follow-up) bằng tiếng Pháp.

LƯU Ý QUAN TRỌNG (giới hạn kỹ thuật):
  - So khớp phát âm dùng thuật toán căn chỉnh (alignment) theo khoảng cách
    chỉnh sửa (edit-distance / Levenshtein) giữa chuỗi phoneme dự đoán từ
    audio và chuỗi phoneme "tham chiếu" sinh ra từ transcript do ASR nhận
    dạng (qua phonemizer/espeak-ng). Đây vẫn là một phép xấp xỉ - không phải
    một bộ chấm phát âm "chuẩn nghiên cứu" - chỉ nên dùng để CHỈ RA XU HƯỚNG
    sai, không nên coi là điểm số tuyệt đối chính xác.
  - Cần cài đặt `espeak-ng` ở cấp hệ điều hành (không phải pip) để thư viện
    `phonemizer` hoạt động (xem file HUONG_DAN_CAI_DAT.md). Trên Windows,
    biến môi trường PHONEMIZER_ESPEAK_LIBRARY bên dưới trỏ thẳng tới file
    libespeak-ng.dll để phonemizer tìm thấy thư viện mà không cần cấu hình
    PATH thủ công.
  - NẾU máy chưa cài espeak-ng (PHONEMIZER_AVAILABLE = False), app KHÔNG dừng
    lại: nó tự động chuyển sang chế độ chấm điểm dựa trên so khớp TỪ VỰNG
    (vocabulary) giữa transcript ASR và các từ khóa mong đợi của câu hỏi,
    thay vì so sánh phoneme.
  - Thu âm dùng `st.audio_input`, có trong Streamlit >= 1.36. Nếu phiên bản
    Streamlit đang chạy cũ hơn và không có thuộc tính này, app tự động chuyển
    sang chế độ chỉ dùng `st.file_uploader` để tải file âm thanh lên.
"""

import os
import platform

# Ép phonemizer tìm đúng thư viện eSpeak NG trên Windows (nếu không nằm
# trong PATH mặc định). Phải đặt TRƯỚC khi phonemizer/espeak được import
# hoặc gọi lần đầu.
# CHỈ áp dụng khi thực sự chạy trên Windows VÀ file .dll thực sự tồn tại —
# nếu set biến này vô điều kiện (kể cả khi deploy trên Linux/macOS, ví dụ
# Streamlit Community Cloud), phonemizer có thể bị trỏ nhầm tới một đường
# dẫn Windows không tồn tại và không tự dò tìm espeak-ng đúng cách nữa.
# `setdefault` để không ghi đè nếu người dùng đã tự cấu hình biến này.
if platform.system() == "Windows":
    _WIN_ESPEAK_DLL = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    if os.path.exists(_WIN_ESPEAK_DLL):
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", _WIN_ESPEAK_DLL)

import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import io
import json
import difflib
import re
import random
import inspect
import html as html_lib

import librosa
import numpy as np
import soundfile as sf

# ----------------------------------------------------------------------
# Các thư viện nặng (torch / transformers) được import trong hàm và cache
# bằng st.cache_resource để tránh load lại mỗi lần rerun.
# ----------------------------------------------------------------------

try:
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    PHONEMIZER_AVAILABLE = True
except Exception:
    PHONEMIZER_AVAILABLE = False

st.set_page_config(page_title="Campus France Interview Trainer", page_icon="🇫🇷", layout="centered")

# Phát hiện xem bản Streamlit đang chạy có hỗ trợ `st.columns(...,
# vertical_alignment=...)` hay không (thêm từ Streamlit 1.32). Nếu có, dùng
# nó để căn giữa nút theo chiều dọc với selectbox thay vì phải "vá" bằng
# một <div style='margin-top:...'> — sạch hơn và không phụ thuộc font-size.
_COLUMNS_SUPPORT_VALIGN = "vertical_alignment" in inspect.signature(st.columns).parameters

# =========================================================================
# 0. DỮ LIỆU CÂU HỎI MẶC ĐỊNH
# =========================================================================
DEFAULT_QUESTIONS = [
    ["Pourquoi avez-vous choisi cette formation en France ?",
     "Phải nêu rõ lý do chọn NGÀNH HỌC cụ thể + tại sao nước Pháp (chất lượng đào tạo, chương trình phù hợp mục tiêu nghề nghiệp), không nói chung chung."],
    ["Quel est votre projet professionnel après vos études ?",
     "Phải trình bày một dự định nghề nghiệp RÕ RÀNG, có liên kết logic với ngành học đang xin, tốt nhất có ý định đóng góp khi trở về Việt Nam."],
    ["Pourquoi avez-vous choisi cette université en particulier ?",
     "Phải nêu được điểm đặc trưng của TRƯỜNG/CHƯƠNG TRÌNH cụ thể (không phải lý do chung của nước Pháp), cho thấy đã tìm hiểu kỹ."],
    ["Comment allez-vous financer vos études en France ?",
     "Phải nêu rõ NGUỒN TÀI CHÍNH cụ thể (gia đình, học bổng, tiết kiệm...) và thể hiện kế hoạch tài chính khả thi, minh bạch."],
    ["Quels sont vos points forts et vos points faibles ?",
     "Phải nêu điểm mạnh liên quan đến ngành học/dự án, và điểm yếu đi kèm cách khắc phục cụ thể, tránh trả lời sáo rỗng."],
]

FRENCH_TTS_LANG = "fr"

# Mô hình nhận diện phoneme dùng chung cho cả bộ chấm phát âm (huấn luyện
# trên CommonVoice với nhãn phoneme sinh bởi espeak, nên xuất ra phoneme
# trực tiếp thay vì chữ viết).
MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
TARGET_SR = 16000


def normalize_qa_list(data):
    """
    Chuẩn hóa dữ liệu câu hỏi từ nhiều định dạng JSON khác nhau về CÙNG một
    cấu trúc chung: list các dict {"question": ..., "key_intent": ..., "keywords": [...]}.

    Hỗ trợ:
      1) Dạng mảng:  [["Câu hỏi 1", "Ý trả lời 1"], ...]
                     (phần tử thứ 3 nếu có và là list -> coi là keywords)
      2) Dạng object với các key (không phân biệt hoa/thường, có/không dấu cách):
         - original_question / key_intent / key_keywords
         - "câu hỏi" / "ý trả lời" / "từ khóa" (hoặc "keywords")

    Raise ValueError với thông báo rõ ràng nếu không nhận diện được định dạng,
    để hiển thị lỗi thân thiện cho người dùng thay vì làm crash toàn app.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON phải là một danh sách (list) không rỗng.")

    question_keys = ["originalquestion", "question", "câuhỏi", "cauhoi", "câuhỏigốc", "cauhoigoc"]
    intent_keys = ["keyintent", "ýtrảlời", "ytraloi", "answer", "réponse", "reponse"]
    keyword_keys = ["keykeywords", "keywords", "từkhóa", "tukhoa"]

    normalized = []
    for i, item in enumerate(data):
        question, key_intent, keywords = None, None, []

        if isinstance(item, (list, tuple)):
            if len(item) < 2:
                raise ValueError(
                    f"Phần tử thứ {i + 1} (dạng mảng) cần có ít nhất 2 phần tử: [câu_hỏi, ý_trả_lời]."
                )
            question = str(item[0]).strip()
            key_intent = str(item[1]).strip()
            if len(item) >= 3 and isinstance(item[2], list):
                keywords = [str(k).strip() for k in item[2] if str(k).strip()]

        elif isinstance(item, dict):
            # bản đồ: key đã "chuẩn hóa" (chữ thường, bỏ khoảng trắng/gạch dưới) -> key gốc
            lower_map = {re.sub(r"[\s_]+", "", k.lower()): k for k in item.keys()}

            for k in question_keys:
                if k in lower_map:
                    question = str(item[lower_map[k]]).strip()
                    break
            for k in intent_keys:
                if k in lower_map:
                    key_intent = str(item[lower_map[k]]).strip()
                    break
            for k in keyword_keys:
                if k in lower_map:
                    val = item[lower_map[k]]
                    if isinstance(val, list):
                        keywords = [str(v).strip() for v in val if str(v).strip()]
                    elif isinstance(val, str):
                        keywords = [v.strip() for v in val.split(",") if v.strip()]
                    break

            if question is None or key_intent is None:
                raise ValueError(
                    f"Phần tử thứ {i + 1} (dạng object) thiếu thông tin câu hỏi hoặc ý trả lời. "
                    "Cần một trong các cặp key: original_question/key_intent hoặc câu hỏi/ý trả lời."
                )
        else:
            raise ValueError(f"Phần tử thứ {i + 1} có kiểu dữ liệu không được hỗ trợ: {type(item).__name__}.")

        if not question or not key_intent:
            raise ValueError(f"Phần tử thứ {i + 1} có câu hỏi hoặc ý trả lời rỗng.")

        normalized.append({"question": question, "key_intent": key_intent, "keywords": keywords})

    return normalized


DEFAULT_QUESTIONS_NORMALIZED = normalize_qa_list(DEFAULT_QUESTIONS)

# =========================================================================
# 1. CACHE CÁC MODEL NẶNG
# =========================================================================

@st.cache_resource(show_spinner="Đang tải mô hình nhận diện phoneme (Wav2Vec2)...")
def load_phoneme_model():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Đang tải mô hình nhận dạng giọng nói tiếng Pháp (ASR)...")
def load_asr_model():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    model_name = "facebook/wav2vec2-large-xlsr-53-french"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    return processor, model


# --------------------------------------------------------------------------
# Đọc audio: dùng sf.read / librosa (giống bản "French Pronunciation Checker"
# gốc) thay vì torchaudio.load, để tránh phải cài thêm TorchCodec.
# --------------------------------------------------------------------------
def load_audio_bytes_as_array(audio_bytes: bytes) -> np.ndarray:
    """Đọc bytes audio (như từ st.audio_input hoặc file .wav/.mp3 tải lên)
    thành mảng numpy mono float32, resample về TARGET_SR."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        # Một số file (vd. .mp3 tùy bản libsndfile) sf không đọc được trực
        # tiếp -> thử lại bằng librosa (cần ffmpeg/audioread).
        data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)

    data = np.asarray(data, dtype="float32")
    if data.ndim > 1:  # stereo -> mono
        data = np.mean(data, axis=0 if data.shape[0] < data.shape[1] else 1)

    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

    return data


def run_asr_transcript(audio_bytes):
    """Trả về transcript tiếng Pháp (chữ) từ audio, dùng để làm 'ground truth'
    cho việc chấm nội dung (Gemini) và để sinh phoneme kỳ vọng."""
    import torch
    processor, model = load_asr_model()
    audio_array = load_audio_bytes_as_array(audio_bytes)
    inputs = processor(audio_array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(pred_ids)[0]
    return transcript.strip().lower()


def run_phoneme_recognition(audio_bytes):
    """Trả về chuỗi phoneme (dạng espeak/IPA-like) thực tế đọc từ audio."""
    import torch
    processor, model = load_phoneme_model()
    audio_array = load_audio_bytes_as_array(audio_bytes)
    inputs = processor(audio_array, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes = processor.batch_decode(predicted_ids)[0]
    return phonemes.strip()


def extract_words(text: str) -> list:
    """Tách một câu tiếng Pháp thành danh sách các TỪ riêng biệt (bỏ dấu câu,
    giữ nguyên chữ có dấu và gạch nối/nháy đơn trong từ, vd. "l'université",
    "peut-être"). Dùng để hiển thị + phiên âm TỪNG TỪ một."""
    return re.findall(r"[a-zà-ÿœæ'\-]+", (text or "").lower())


def phonemize_words(words: list, language: str = "fr-fr") -> list:
    """Sinh phiên âm IPA "tham chiếu" cho TỪNG TỪ riêng biệt (giữ nguyên ranh
    giới từ) — thay vì phiên âm cả câu rồi tách lại, ta phiên âm mỗi từ độc
    lập bằng cách truyền cả danh sách từ vào phonemizer cùng lúc (phonemizer
    hỗ trợ nhận list và trả về list kết quả tương ứng theo đúng thứ tự).
    Trả về list các list phoneme, ví dụ [["b", "ɔ̃", "ʒ", "u", "ʁ"], ["m", ...]]."""
    if not PHONEMIZER_AVAILABLE:
        raise RuntimeError(
            "Thư viện 'phonemizer' (và binary hệ thống espeak-ng) chưa sẵn sàng."
        )
    if not words:
        return []
    separator = Separator(phone=" ", word="", syllable="")
    ipa_list = phonemize(
        words,
        language=language,
        backend="espeak",
        separator=separator,
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
    )
    return [ipa_str.split() for ipa_str in ipa_list]


# --------------------------------------------------------------------------
# Căn chỉnh (alignment) — GIỮ NGUYÊN thuật toán edit-distance / Levenshtein ở
# cấp phoneme từ bản "French Pronunciation Checker" gốc. Kết quả align được
# ánh xạ ngược về TỪNG TỪ (xem compute_word_pronunciation_results) để hiển
# thị "từ nào đúng / từ nào sai" thay vì một chuỗi phoneme dính liền.
# --------------------------------------------------------------------------
def align_phonemes(reference: list, hypothesis: list):
    """
    Classic edit-distance alignment (Levenshtein) between two phoneme
    sequences, returning a list of (ref_token_or_None, hyp_token_or_None, tag)
    where tag is one of: 'match', 'sub', 'ins', 'del'.
    """
    n, m = len(reference), len(hypothesis)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion (in reference, missing from hyp)
                dp[i][j - 1] + 1,       # insertion (extra in hyp)
                dp[i - 1][j - 1] + cost,  # match or substitution
            )

    # backtrack
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append((reference[i - 1], hypothesis[j - 1], "match"))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append((reference[i - 1], hypothesis[j - 1], "sub"))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((reference[i - 1], None, "del"))
            i -= 1
        else:
            ops.append((None, hypothesis[j - 1], "ins"))
            j -= 1
    ops.reverse()
    return ops


def compute_word_pronunciation_results(words, word_phoneme_lists, ops):
    """Ánh xạ kết quả align_phonemes() (cấp phoneme, toàn câu) NGƯỢC về từng
    từ trong câu, để biết từ nào đọc đúng / từ nào lệch.

    ops được xây theo đúng THỨ TỰ các phoneme tham chiếu (chỉ 'match'/'sub'/
    'del' mới "tiêu thụ" một vị trí phoneme tham chiếu, 'ins' thì không) —
    nên duyệt tuần tự và cắt theo độ dài phoneme của từng từ là ánh xạ được
    chính xác, không cần thay đổi thuật toán align_phonemes().

    Trả về list[dict]: {"word": str, "ipa": str, "correct": bool,
    "phoneme_tags": list[str]} theo đúng thứ tự xuất hiện trong câu.
    `phoneme_tags` (mỗi phần tử là 'match'/'sub'/'del', theo ĐÚNG thứ tự
    phoneme tham chiếu của từ) được giữ lại để có thể tô màu XẤP XỈ theo
    từng đoạn ký tự bên trong một từ sai (xem split_word_into_chunks /
    render_inline_sentence_html), thay vì chỉ tô nguyên cả từ một màu."""
    ref_tags = [tag for _, _, tag in ops if tag in ("match", "sub", "del")]

    results = []
    pos = 0
    for word, word_phonemes in zip(words, word_phoneme_lists):
        n = len(word_phonemes)
        if n == 0:
            # Từ không phiên âm được (hiếm) -> không đủ căn cứ để chấm sai,
            # coi là "đúng" để không làm giảm điểm oan.
            results.append({"word": word, "ipa": "", "correct": True, "phoneme_tags": []})
            continue
        word_tags = ref_tags[pos: pos + n]
        pos += n
        correct = len(word_tags) == n and all(t == "match" for t in word_tags)
        results.append({
            "word": word,
            "ipa": " ".join(word_phonemes),
            "correct": correct,
            "phoneme_tags": word_tags,
        })
    return results


def split_word_into_chunks(word: str, n_chunks: int) -> list:
    """Chia một từ thành `n_chunks` đoạn ký tự liên tiếp có độ dài xấp xỉ
    bằng nhau, dùng để tô màu GẦN ĐÚNG theo từng phoneme bên trong từ (ví dụ
    từ "concentre" có 1 phoneme sai ở giữa -> tô đỏ đúng đoạn chữ tương ứng
    ở giữa từ thay vì tô đỏ nguyên cả từ).

    LƯU Ý: đây CHỈ là phép chia đều heuristic theo vị trí ký tự, KHÔNG phải
    một bộ ánh xạ grapheme-phoneme (G2P) thực sự — tiếng Pháp có nhiều chữ
    câm / digraph (ph, ch, qu, ...) nên ánh xạ 1-1 tuyệt đối chính xác giữa
    ký tự và phoneme là không khả thi nếu không có bộ G2P chuyên dụng có
    thông tin alignment. Cách chia này chỉ nhằm mục đích trực quan hoá gần
    đúng vị trí lỗi, không phải một phép đo ngữ âm học chuẩn."""
    word = word or ""
    length = len(word)
    if n_chunks <= 1 or length <= 1:
        return [word] if word else [""]
    n_chunks = min(n_chunks, length)  # không chia nhỏ hơn 1 ký tự / đoạn
    base, rem = divmod(length, n_chunks)
    chunks, pos = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        chunks.append(word[pos: pos + size])
        pos += size
    return chunks


def render_inline_sentence_html(word_results) -> str:
    """Render kết quả chấm phát âm dạng VĂN BẢN LIỀN MẠCH (inline), giống
    phong cách "flashcard ngữ âm": dòng 1 là câu tiếng Pháp (tô xanh/đỏ theo
    từng từ, hoặc theo từng đoạn ký tự bên trong từ nếu từ đó chỉ sai một
    phần), dòng 2 ngay bên dưới là phiên âm IPA của TOÀN CÂU, đặt trong dấu
    /.../ và căn theo đúng cột của từng từ phía trên (không dùng box/khung
    viền như bản cũ).

    Kỹ thuật căn cột: mỗi từ + phần IPA của nó được đặt chung trong MỘT cột
    flex (flex-direction: column) để luôn dính liền nhau kể cả khi câu dài
    phải xuống dòng (flex-wrap: wrap) — thay vì tách thành 2 <div> câu văn
    riêng biệt (khi đó xuống dòng ở 2 dòng sẽ lệch cột với nhau)."""
    GREEN = "#1a7f37"
    RED = "#d1242f"

    def slash_column():
        return (
            "<div style='display:flex; flex-direction:column; align-items:center; "
            "justify-content:flex-end;'>"
            "<span style='visibility:hidden; font-size:1.7em; font-weight:700;'>/</span>"
            f"<span style='font-size:1.1em; color:#555; margin-top:4px;'>/</span>"
            "</div>"
        )

    columns = [slash_column()]
    for r in word_results:
        tags = r.get("phoneme_tags") or []
        if tags:
            chunks = split_word_into_chunks(r["word"], len(tags))
            spans = []
            for chunk, tag in zip(chunks, tags):
                color = GREEN if tag == "match" else RED
                spans.append(f"<span style='color:{color};'>{html_lib.escape(chunk)}</span>")
            word_html = "".join(spans)
        else:
            color = GREEN if r["correct"] else RED
            word_html = f"<span style='color:{color};'>{html_lib.escape(r['word'])}</span>"

        ipa_esc = html_lib.escape(r["ipa"]) if r["ipa"] else "?"
        ipa_color = GREEN if r["correct"] else RED
        columns.append(
            "<div style='display:flex; flex-direction:column; align-items:center;'>"
            f"<span style='font-size:1.7em; font-weight:700; white-space:nowrap;'>{word_html}</span>"
            f"<span style='font-size:1.1em; color:{ipa_color}; margin-top:4px; white-space:nowrap;'>{ipa_esc}</span>"
            "</div>"
        )
    columns.append(slash_column())

    return (
        "<div style='display:flex; flex-wrap:wrap; align-items:flex-start; "
        "column-gap:22px; row-gap:16px; font-family:\"Segoe UI\",Helvetica,Arial,sans-serif; "
        "margin: 10px 0 6px 0;'>"
        + "".join(columns) +
        "</div>"
    )


def grade_pronunciation_phoneme(transcript: str, audio_bytes: bytes):
    """Chấm phát âm THEO TỪNG TỪ (đường chính, cần espeak-ng).

    Trả về (word_results, raw_hyp_phonemes, accuracy_percent):
      - word_results: list[{"word", "ipa", "correct"}] theo đúng thứ tự câu,
        dùng để render các box từ-vựng + phiên âm ở giao diện.
      - raw_hyp_phonemes: chuỗi phoneme thô nhận dạng được từ audio (hiển thị
        trong expander gấp gọn, tránh làm rối giao diện chính).
      - accuracy_percent: % số từ đọc đúng / tổng số từ có thể chấm được.
    """
    actual_phonemes = run_phoneme_recognition(audio_bytes)

    words = extract_words(transcript)
    if not words:
        return [], actual_phonemes, None

    word_phoneme_lists = phonemize_words(words)
    flat_ref = [p for word_phonemes in word_phoneme_lists for p in word_phonemes]
    hyp_list = actual_phonemes.split()

    ops = align_phonemes(flat_ref, hyp_list)
    word_results = compute_word_pronunciation_results(words, word_phoneme_lists, ops)

    total = len(word_results)
    correct = sum(1 for r in word_results if r["correct"])
    accuracy = 100.0 * correct / total if total else None
    return word_results, actual_phonemes, accuracy


# --------------------------------------------------------------------------
# CHẤM ĐIỂM DỰ PHÒNG (fallback) khi KHÔNG có espeak-ng / phonemizer:
# thay vì so phoneme, so khớp TỪ VỰNG tiếng Pháp giữa transcript ASR và các
# từ khóa mong đợi của câu hỏi (question_keywords). Không dừng chương trình.
# --------------------------------------------------------------------------
def grade_vocabulary_fallback(transcript: str, keywords):
    """So khớp gần đúng (difflib) từng từ khóa mong đợi với các từ xuất hiện
    trong transcript, để chấp nhận sai lệch chính tả nhỏ do ASR. Trả về
    (matched, missing, score_percent_or_None)."""
    transcript_lower = (transcript or "").lower()
    transcript_words = set(re.findall(r"[a-zà-ÿ'\-]+", transcript_lower))

    matched, missing = [], []
    for kw in keywords or []:
        kw_clean = kw.strip().lower()
        if not kw_clean:
            continue
        found = kw_clean in transcript_lower or any(
            difflib.SequenceMatcher(None, kw_clean, w).ratio() >= 0.8
            for w in transcript_words
        )
        (matched if found else missing).append(kw)

    total = len(matched) + len(missing)
    score = 100.0 * len(matched) / total if total else None
    return matched, missing, score


def render_vocabulary_html(matched, missing):
    spans = [
        f"<span style='color:#1a7f37;font-weight:700;font-size:1.3em'>{html_lib.escape(kw)}</span>" for kw in matched
    ] + [
        f"<span style='color:#d1242f;font-weight:700;font-size:1.3em;text-decoration:underline'>{html_lib.escape(kw)}</span>"
        for kw in missing
    ]
    return " &nbsp;&nbsp; ".join(spans) if spans else "<i>(Không có từ khóa nào để so khớp)</i>"


# =========================================================================
# 2. GEMINI HELPERS
# =========================================================================

def get_gemini_model(model_name="gemini-flash-latest"):
    return genai.GenerativeModel(model_name)


def gemini_paraphrase(question, model_name):
    model = get_gemini_model(model_name)
    prompt = (
        "Tu es un examinateur Campus France. Reformule la question suivante en français "
        "de manière naturelle et fluide à l'oral, comme lors d'un VRAI entretien.\n\n"
        "Directives de reformulation :\n"
        "- Ne cherche PAS à tout paraphraser à tout prix. Tu peux garder les mots clés simples et naturels (ex: 'ville', 'études', 'établissement').\n"
        "- Varie plutôt la structure de la phrase, la tournure de phrase ou l'angle de vue (ex: passer d'une question directe à une question indirecte).\n"
        "- Évite le langage trop lourd, administratif ou exagérément académique. La phrase doit rester naturelle et parlante à l'oral.\n\n"
        "Réponds UNIQUEMENT avec la nouvelle question, sans aucun autre texte.\n\n"
        f"Question originale : {question}"
    )
    resp = model.generate_content(prompt)
    return resp.text.strip().strip('"')


def gemini_evaluate_answer(question, key_intent, transcript, model_name, keywords=None):
    model = get_gemini_model(model_name)
    keywords_block = ""
    if keywords:
        keywords_str = ", ".join(keywords)
        keywords_block = f"\nMots-clés attendus dans une bonne réponse : {keywords_str}\n"
    prompt = f"""
Tu es un examinateur Campus France strict et rigoureux. Voici la question posée au candidat,
l'intention/l'idée obligatoire attendue dans la réponse (en vietnamien), et la transcription
de la réponse orale du candidat (en français, potentiellement imparfaite car issue d'un ASR).

Question : {question}
Idée obligatoire attendue (vietnamien) : {key_intent}{keywords_block}
Transcription de la réponse du candidat : {transcript}

Évalue si la réponse du candidat touche bien l'idée obligatoire attendue (on_topic = true/false).
Si ce n'est pas clair, incomplet, ou hors-sujet, propose UNE question de relance (follow_up_question)
en français, formulée comme un examinateur sérieux qui met le candidat au pied du mur pour qu'il
précise ou corrige sa réponse. Si la réponse est déjà satisfaisante, follow_up_question = null.

Réponds STRICTEMENT en JSON valide, sans texte autour, avec ce format exact :
{{"on_topic": true ou false, "feedback": "commentaire court en vietnamien", "follow_up_question": "..." ou null}}
"""
    resp = model.generate_content(prompt)
    raw = resp.text.strip()
    raw = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"on_topic": None, "feedback": f"(Không phân tích được JSON từ Gemini) {raw}", "follow_up_question": None}
    return data


# =========================================================================
# 3. TTS
# =========================================================================

def text_to_speech_bytes(text, lang=FRENCH_TTS_LANG):
    buf = io.BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# =========================================================================
# 4. SESSION STATE INIT
# =========================================================================

# Key riêng cho widget st.selectbox chọn câu hỏi. Khi widget có `key`, giá
# trị của nó SỐNG trong st.session_state[SELECTBOX_KEY] và đây là nguồn
# chân lý (source of truth) duy nhất mà Streamlit dùng để hiển thị/lưu lựa
# chọn — ta có thể set nó TRƯỚC khi widget render (trong callback của nút
# khác) để "điều khiển" selectbox từ xa mà không bị đè.
SELECTBOX_KEY = "question_selectbox"


def reset_pronunciation_state():
    """Xóa riêng kết quả chấm PHÁT ÂM/TỪ VỰNG (không đụng transcript, đánh
    giá nội dung Gemini, hay câu hỏi vặn) — dùng khi chuẩn bị chấm lại phát
    âm cho MỘT lượt trả lời cụ thể (transcript của lượt đó vẫn cần giữ)."""
    st.session_state.pron_mode = None
    st.session_state.pron_word_results = None
    st.session_state.pron_raw_phonemes = None
    st.session_state.pron_vocab_html = None
    st.session_state.pron_accuracy = None


def reset_grading_state():
    """Xóa các trường liên quan tới KẾT QUẢ CHẤM của lượt trả lời hiện tại
    (transcript ASR, đánh giá nội dung Gemini, câu hỏi vặn, kết quả chấm
    phát âm/từ vựng). KHÔNG đụng tới `paraphrased_question` / `revealed`.

    Dùng chung cho MỌI nơi cần "làm sạch kết quả chấm cũ" — tránh lặp lại
    cùng các dòng gán None ở nhiều chỗ (nút tạo câu hỏi biến thể, luồng chấm
    điểm audio, và reset_question_state bên dưới)."""
    st.session_state.last_transcript = None
    st.session_state.last_evaluation = None
    st.session_state.follow_up_question = None
    reset_pronunciation_state()


def reset_question_state():
    """Xóa sạch TOÀN BỘ dữ liệu của câu hỏi CŨ (câu hỏi biến thể, trạng thái
    hiện/ẩn, và toàn bộ kết quả chấm qua reset_grading_state()). Hàm này
    CHỈ được gọi khi thực sự CHUYỂN sang một câu hỏi khác — tuyệt đối không
    gọi ở những đoạn code chạy lại (rerun) mà không liên quan tới việc đổi
    câu hỏi, nếu không câu hỏi biến thể vừa tạo bằng Gemini sẽ bị xóa oan
    mỗi khi người dùng bấm một nút bất kỳ khác trong app.

    Đồng thời tăng `question_version` — bộ đếm được nhúng vào key của các
    widget thu âm (st.audio_input / file_uploader câu trả lời) để BUỘC
    Streamlit coi đó là widget MỚI toanh mỗi khi chuyển câu hỏi, nhờ vậy
    file ghi âm/tải lên của câu hỏi trước không bị "dính" lại ở câu hỏi
    hiện tại dù chỉ số (idx) không đổi (ví dụ: vừa tạo câu hỏi biến thể, hoặc
    vừa đổi nguồn dữ liệu JSON mà câu hỏi mới trùng vị trí với câu cũ)."""
    st.session_state.paraphrased_question = None
    st.session_state.revealed = False
    reset_grading_state()
    st.session_state.question_version += 1


def go_to_question(new_idx: int):
    """Điều hướng CÓ CHỦ ĐÍCH sang câu hỏi có chỉ số new_idx — dùng cho nút
    "🔀 Ngẫu nhiên" / "➡️ Câu tiếp theo". Phải cập nhật ĐỒNG THỜI:
      - st.session_state.current_index  (biến chỉ số dùng chung trong app)
      - st.session_state[SELECTBOX_KEY] (giá trị thật của widget selectbox)
    TRƯỚC khi widget selectbox render lại — nếu chỉ cập nhật current_index
    mà không cập nhật luôn key của widget, st.selectbox (đã gắn key) sẽ tiếp
    tục hiển thị giá trị CŨ của nó ở lần render kế tiếp và ghi đè lại
    current_index, khiến nút bấm "không có tác dụng"."""
    st.session_state.current_index = new_idx
    st.session_state[SELECTBOX_KEY] = new_idx
    reset_question_state()


def on_select_question_change():
    """Callback on_change của st.selectbox — CHỈ chạy khi người dùng TỰ TAY
    đổi lựa chọn trong dropdown (không chạy ở các rerun khác). Đây là nơi
    DUY NHẤT đồng bộ current_index theo giá trị mới của widget và reset
    state câu hỏi cũ."""
    new_idx = st.session_state[SELECTBOX_KEY]
    if new_idx != st.session_state.current_index:
        st.session_state.current_index = new_idx
        reset_question_state()


def load_new_question_list(qa_list_data):
    """Nạp một danh sách câu hỏi MỚI (mặc định hoặc từ file JSON vừa tải) và
    đưa người dùng về câu hỏi đầu tiên. Chỉ gọi hàm này khi nguồn dữ liệu
    THỰC SỰ thay đổi (xem `_active_source_signature` ở khối Sidebar) — không
    gọi lại ở mỗi lần rerun khi nguồn dữ liệu không đổi, nếu không
    current_index sẽ liên tục bị ép về 0 mỗi khi người dùng bấm bất kỳ nút
    nào khác (kể cả "🔀 Ngẫu nhiên" / "➡️ Câu tiếp theo" / "👁️ Hiện câu hỏi"),
    vì `st.file_uploader` vẫn "nhớ" file đã tải ở mọi lần rerun sau đó."""
    st.session_state.qa_list = qa_list_data
    st.session_state.current_index = 0
    st.session_state[SELECTBOX_KEY] = 0
    reset_question_state()


def init_state():
    defaults = {
        "qa_list": DEFAULT_QUESTIONS_NORMALIZED,
        "current_index": 0,
        SELECTBOX_KEY: 0,
        "question_version": 0,  # tăng mỗi lần reset_question_state() -> làm mới key widget thu âm
        "paraphrased_question": None,
        "revealed": False,  # LUÔN bắt đầu ở trạng thái ẨN / che mờ
        "last_transcript": None,
        "last_evaluation": None,
        "follow_up_question": None,
        "pron_mode": None,          # "phoneme" | "vocab" | None
        "pron_word_results": None,  # list[{"word","ipa","correct"}] khi pron_mode == "phoneme"
        "pron_raw_phonemes": None,  # chuỗi phoneme thô nhận dạng từ audio (hiển thị trong expander)
        "pron_vocab_html": None,
        "pron_accuracy": None,
        # Định danh nguồn dữ liệu ĐANG hiển thị trong qa_list ("default" hoặc
        # "upload:<tên>:<kích thước>"). Dùng để: (a) không xử lý/reset lại
        # mỗi rerun khi nguồn không đổi, NHƯNG (b) vẫn nạp lại đúng nội dung
        # khi người dùng đổi qua đổi lại giữa "Danh sách mặc định" và
        # "Tải file JSON" (kể cả với cùng một file đã tải trước đó).
        "_active_source_signature": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# =========================================================================
# 5. SIDEBAR: CẤU HÌNH
# =========================================================================

st.sidebar.header("⚙️ Cấu hình")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
gemini_model_name = st.sidebar.text_input("Tên model Gemini", value="gemini-flash-latest")
st.sidebar.caption(
    "`gemini-flash-latest` là **alias tự động trỏ tới bản Flash mới nhất** của Google, "
    "giúp tránh lỗi 404 khi Google ngừng hỗ trợ (deprecate) một phiên bản cụ thể "
    "(ví dụ `gemini-1.5-flash`, `gemini-2.0-flash` đã bị ngừng hỗ trợ hoàn toàn; "
    "`gemini-2.5-flash` cũng đang bị Google rút ngắn thời hạn sớm hơn dự kiến). "
    "Nếu vẫn gặp lỗi 404, hãy thử `gemini-2.5-flash-lite` hoặc kiểm tra model mới nhất tại "
    "https://ai.google.dev/gemini-api/docs/models"
)

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Nguồn câu hỏi")
source_choice = st.sidebar.radio("Chọn nguồn", ["Danh sách mặc định", "Tải file JSON"])

if source_choice == "Tải file JSON":
    uploaded = st.sidebar.file_uploader("Tải file .json", type=["json"])
    if uploaded is not None:
        # QUAN TRỌNG: st.file_uploader vẫn TRẢ VỀ cùng một đối tượng file ở
        # MỌI lần rerun sau đó (kể cả khi rerun đó do người dùng bấm một nút
        # hoàn toàn không liên quan, ví dụ "👁️ Hiện câu hỏi" hay
        # "🔀 Ngẫu nhiên"). Nếu không kiểm tra xem file này ĐÃ là nguồn ĐANG
        # hiển thị hay chưa mà cứ nạp lại + reset mỗi rerun, current_index sẽ
        # liên tục bị ép về 0 và state câu hỏi (kể cả câu hỏi biến thể Gemini)
        # liên tục bị xóa.
        # Lưu ý: so sánh với `_active_source_signature` (nguồn ĐANG active)
        # chứ không phải "đã từng xử lý bao giờ chưa" — nhờ vậy nếu người
        # dùng chuyển sang "Danh sách mặc định" rồi quay lại "Tải file JSON"
        # với CÙNG một file, danh sách JSON vẫn được nạp lại đúng thay vì bị
        # bỏ qua vì "đã xử lý trước đó".
        signature = f"upload:{uploaded.name}:{uploaded.size}"
        if signature != st.session_state._active_source_signature:
            try:
                raw_data = json.load(uploaded)
            except Exception as e:
                st.sidebar.error(f"File không phải JSON hợp lệ (lỗi cú pháp). Chi tiết: {e}")
                raw_data = None

            if raw_data is not None:
                try:
                    normalized_data = normalize_qa_list(raw_data)
                    load_new_question_list(normalized_data)
                    st.session_state._active_source_signature = signature
                    st.sidebar.success(f"Đã tải {len(normalized_data)} câu hỏi.")
                except ValueError as e:
                    st.sidebar.error(
                        "File JSON không đúng định dạng được hỗ trợ. Hỗ trợ 2 dạng:\n"
                        "1) [[\"câu hỏi\", \"ý trả lời\"], ...]\n"
                        "2) [{\"original_question\": ..., \"key_intent\": ..., \"key_keywords\": [...]}, ...] "
                        "(hoặc dùng key \"câu hỏi\" / \"ý trả lời\")\n\n"
                        f"Chi tiết lỗi: {e}"
                    )
    else:
        # Chưa chọn file (hoặc vừa bị gỡ) -> không có nguồn upload nào đang
        # active. Đặt lại signature để nếu người dùng tải lại ĐÚNG file này
        # sau đó, nó vẫn được xử lý như một lần nạp mới.
        st.session_state._active_source_signature = None
else:
    default_signature = "default"
    if default_signature != st.session_state._active_source_signature:
        load_new_question_list(DEFAULT_QUESTIONS_NORMALIZED)
        st.session_state._active_source_signature = default_signature

st.sidebar.markdown("---")
if PHONEMIZER_AVAILABLE:
    st.sidebar.caption("✅ Đã phát hiện `phonemizer`/`espeak-ng` — chấm phát âm theo phoneme (đầy đủ).")
else:
    st.sidebar.warning(
        "⚠️ Chưa phát hiện `phonemizer`/`espeak-ng`. App sẽ tự động chuyển sang chấm điểm "
        "theo TỪ VỰNG (so khớp từ khóa mong đợi trong transcript) thay vì phoneme. "
        "Xem HUONG_DAN_CAI_DAT.md để cài đầy đủ."
    )

# =========================================================================
# 6. MAIN UI
# =========================================================================

st.title("🇫🇷 Campus France Interview Trainer")
st.caption("Luyện phỏng vấn Campus France: paraphrase câu hỏi, nghe, trả lời, chấm phát âm & nội dung.")

qa_list = st.session_state.qa_list
if not qa_list:
    st.warning("Chưa có câu hỏi nào. Hãy tải file JSON ở thanh bên trái.")
    st.stop()

options = [item["question"] for item in qa_list]

# An toàn: nếu danh sách câu hỏi vừa thay đổi độ dài (ví dụ tải file JSON có
# ít câu hỏi hơn) mà chỉ số hiện tại đã vượt quá phạm vi, kẹp về câu cuối
# TRƯỚC khi render widget — không sửa trực tiếp trong lúc widget đang hiển
# thị, tránh xung đột với giá trị key đã lưu trong session_state.
if st.session_state[SELECTBOX_KEY] > len(options) - 1:
    st.session_state[SELECTBOX_KEY] = len(options) - 1
    st.session_state.current_index = len(options) - 1

def _pick_random_other_index(current: int, total: int) -> int:
    """Trả về 1 chỉ số ngẫu nhiên khác `current` trong khoảng [0, total).
    Nếu chỉ có 1 câu hỏi, trả về chính current (không có lựa chọn khác)."""
    if total <= 1:
        return current
    candidates = [i for i in range(total) if i != current]
    return random.choice(candidates)


def handle_random_click():
    """on_click của nút '🔀 Ngẫu nhiên'. Đọc trực tiếp từ session_state (thay
    vì đóng gói biến `options` của lần render trước) để luôn dùng đúng độ
    dài danh sách câu hỏi hiện hành."""
    total = len(st.session_state.qa_list)
    new_idx = _pick_random_other_index(st.session_state.current_index, total)
    go_to_question(new_idx)


def handle_next_click():
    """on_click của nút '➡️ Câu tiếp theo' — chuyển vòng tròn 1 → 2 → ... → 1."""
    total = len(st.session_state.qa_list)
    new_idx = (st.session_state.current_index + 1) % total
    go_to_question(new_idx)


if _COLUMNS_SUPPORT_VALIGN:
    col_select, col_random, col_next = st.columns([3, 1, 1], vertical_alignment="bottom")
else:
    col_select, col_random, col_next = st.columns([3, 1, 1])

with col_select:
    # QUAN TRỌNG: widget đã có `key=SELECTBOX_KEY` nên giá trị của nó SỐNG
    # hoàn toàn trong st.session_state[SELECTBOX_KEY] — không truyền thêm
    # `index=...` ở đây nữa (Streamlit sẽ cảnh báo xung đột), và các nút
    # Ngẫu nhiên/Câu tiếp theo chỉ cần set session_state[SELECTBOX_KEY]
    # TRƯỚC khi widget này render là selectbox sẽ tự cập nhật theo, không bị
    # ghi đè ngược lại như khi dùng `index=`.
    st.selectbox(
        "Chọn câu hỏi để luyện tập",
        options=range(len(options)),
        format_func=lambda i: f"Câu hỏi #{i + 1}",  # KHÔNG hiển thị nội dung câu hỏi trong danh sách chọn
        key=SELECTBOX_KEY,
        on_change=on_select_question_change,
    )

with col_random:
    if not _COLUMNS_SUPPORT_VALIGN:
        st.markdown("<div style='margin-top:1.85em'></div>", unsafe_allow_html=True)
    # Dùng on_click thay vì `if st.button(...): ...; st.rerun()`: callback
    # chạy TRƯỚC khi script rerun, nên state đã đồng bộ đúng ngay ở lần
    # render kế tiếp — không cần gọi st.rerun() thủ công nữa.
    st.button("🔀 Ngẫu nhiên", use_container_width=True, on_click=handle_random_click)

with col_next:
    if not _COLUMNS_SUPPORT_VALIGN:
        st.markdown("<div style='margin-top:1.85em'></div>", unsafe_allow_html=True)
    st.button("➡️ Câu tiếp theo", use_container_width=True, on_click=handle_next_click)

# current_index và st.session_state[SELECTBOX_KEY] luôn được giữ đồng bộ
# (qua on_select_question_change / go_to_question), nên có thể lấy thẳng
# một trong hai để dùng cho phần còn lại của trang.
idx = st.session_state.current_index

qa_item = qa_list[idx]
original_question = qa_item["question"]
key_intent = qa_item["key_intent"]
question_keywords = qa_item.get("keywords", [])

with st.expander("📌 Ý trả lời bắt buộc (chỉ để bạn tự chấm, thí sinh thật sự không nên xem trước khi trả lời)"):
    st.write(key_intent)
    if question_keywords:
        st.caption("Từ khóa tham khảo: " + ", ".join(question_keywords))

st.markdown("---")

# ---- Bước 1: Sinh câu hỏi biến thể (paraphrase) ----
st.subheader("1️⃣ Câu hỏi phỏng vấn")

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("🔄 Tạo câu hỏi biến thể bằng Gemini"):
        if not gemini_api_key:
            st.error("Vui lòng nhập Gemini API Key ở thanh bên trái trước.")
        else:
            with st.spinner("Đang tạo câu hỏi biến thể..."):
                try:
                    new_question = gemini_paraphrase(original_question, gemini_model_name)
                    st.session_state.paraphrased_question = new_question
                    st.session_state.revealed = False
                    reset_grading_state()
                    # Nội dung câu hỏi vừa đổi (biến thể mới) -> buộc widget
                    # thu âm reset theo, tránh giữ lại bản ghi âm của câu hỏi
                    # (biến thể) TRƯỚC ĐÓ.
                    st.session_state.question_version += 1
                except Exception as e:
                    st.error(f"Lỗi khi gọi Gemini: {e}")

with col_b:
    if st.button("👁️ Hiện câu hỏi"):
        st.session_state.revealed = True

# active_question = câu hỏi đang dùng để luyện (biến thể nếu có, không thì câu gốc)
active_question = st.session_state.paraphrased_question or original_question

# QUY TẮC BẮT BUỘC: câu hỏi luôn bị che/ẩn cho tới khi người dùng chủ động
# bấm "👁️ Hiện câu hỏi" — áp dụng cho CẢ câu hỏi gốc lẫn câu hỏi biến thể.
if st.session_state.revealed:
    st.markdown(f"### {active_question}")
    if not st.session_state.paraphrased_question:
        st.caption("(Đây là câu hỏi gốc — chưa tạo câu hỏi biến thể.)")
else:
    st.markdown(
        "<div style='filter: blur(8px); font-size:1.3em; font-weight:600; "
        "user-select:none; pointer-events:none;'>"
        f"{active_question}</div>",
        unsafe_allow_html=True,
    )
    st.caption("🙈 Câu hỏi đang bị ẩn. Bấm '🔊 Nghe câu hỏi' để luyện nghe, hoặc '👁️ Hiện câu hỏi' nếu cần xem chữ.")

if st.button("🔊 Nghe câu hỏi"):
    try:
        audio_q = text_to_speech_bytes(active_question)
        st.audio(audio_q, format="audio/mp3")
    except Exception as e:
        st.error(f"Lỗi TTS: {e}")

st.markdown("---")

# ---- Bước 2: Thu âm câu trả lời ----
st.subheader("2️⃣ Thu âm câu trả lời của bạn")

answer_audio_bytes = None
has_native_audio_input = hasattr(st, "audio_input")

if has_native_audio_input:
    audio_value = st.audio_input(
        "🎙️ Thu âm câu trả lời (bấm để ghi, bấm lại để dừng)",
        key=f"audio_input_{idx}_{st.session_state.question_version}",
    )
    if audio_value is not None:
        answer_audio_bytes = audio_value.getvalue()
else:
    st.info(
        "Phiên bản Streamlit hiện tại không hỗ trợ `st.audio_input` (cần Streamlit >= 1.36). "
        "Vui lòng dùng cách tải file âm thanh lên bên dưới."
    )

with st.expander("➕ Hoặc tải file âm thanh lên (.wav / .mp3)", expanded=not has_native_audio_input):
    uploaded_audio = st.file_uploader(
        "Chọn file .wav hoặc .mp3 chứa câu trả lời của bạn",
        type=["wav", "mp3"],
        key=f"upload_answer_{idx}_{st.session_state.question_version}",
    )
    if uploaded_audio is not None:
        answer_audio_bytes = uploaded_audio.read()

if answer_audio_bytes:
    st.audio(answer_audio_bytes)

    if st.button("✅ Chấm câu trả lời này"):
        if not gemini_api_key:
            st.error("Vui lòng nhập Gemini API Key trước khi chấm.")
        else:
            with st.spinner("Đang nhận dạng giọng nói (ASR)..."):
                try:
                    transcript = run_asr_transcript(answer_audio_bytes)
                except Exception as e:
                    st.error(f"Lỗi ASR: {e}")
                    transcript = ""
            st.session_state.last_transcript = transcript

            with st.spinner("Đang phân tích phát âm..."):
                reset_pronunciation_state()

                if PHONEMIZER_AVAILABLE:
                    # Đường chính: so khớp phoneme theo TỪNG TỪ (edit-distance
                    # ở cấp phoneme, ánh xạ ngược về từng từ để hiển thị).
                    try:
                        word_results, raw_phonemes, accuracy = grade_pronunciation_phoneme(
                            transcript, answer_audio_bytes
                        )
                        st.session_state.pron_mode = "phoneme"
                        st.session_state.pron_word_results = word_results
                        st.session_state.pron_raw_phonemes = raw_phonemes
                        st.session_state.pron_accuracy = accuracy
                    except Exception as e:
                        st.warning(
                            f"Không thể chấm phát âm theo phoneme ({e}). "
                            "Chuyển sang chấm điểm theo từ vựng."
                        )

                if st.session_state.pron_mode is None:
                    # Fallback: máy không có espeak-ng, hoặc phonemizer lỗi khi
                    # chạy thực tế -> chấm theo từ vựng thay vì dừng chương trình.
                    matched, missing, score = grade_vocabulary_fallback(transcript, question_keywords)
                    st.session_state.pron_mode = "vocab"
                    st.session_state.pron_vocab_html = render_vocabulary_html(matched, missing)
                    st.session_state.pron_accuracy = score

            with st.spinner("Đang chấm nội dung câu trả lời bằng Gemini..."):
                try:
                    evaluation = gemini_evaluate_answer(
                        active_question, key_intent, transcript, gemini_model_name, keywords=question_keywords
                    )
                except Exception as e:
                    evaluation = {"on_topic": None, "feedback": f"Lỗi Gemini: {e}", "follow_up_question": None}
            st.session_state.last_evaluation = evaluation
            st.session_state.follow_up_question = evaluation.get("follow_up_question")

# ---- Kết quả chấm ----
if st.session_state.last_transcript is not None:
    st.markdown("---")
    st.subheader("3️⃣ Kết quả chấm")

    st.markdown("**Transcript (ASR nhận dạng được):**")
    st.write(st.session_state.last_transcript or "_(không nhận dạng được giọng nói)_")

    if st.session_state.pron_mode == "phoneme":
        st.markdown("**Đánh giá phát âm** _(xanh = đúng, đỏ = lệch/thiếu)_")
        if st.session_state.pron_accuracy is not None:
            st.metric("Tỉ lệ từ đọc đúng", f"{st.session_state.pron_accuracy:.1f}%")

        word_results = st.session_state.pron_word_results or []
        if word_results:
            st.markdown(render_inline_sentence_html(word_results), unsafe_allow_html=True)
        else:
            st.caption("_(Không tách được từ nào từ transcript để chấm.)_")

        st.caption(
            "Ước lượng dựa trên so khớp chuỗi phoneme (edit-distance) theo từng từ; màu bên trong "
            "một từ chỉ là chia xấp xỉ theo vị trí (không phải ánh xạ chữ-âm chuẩn xác), chỉ mang "
            "tính tham khảo, không phải điểm số nghiên cứu chuẩn."
        )
        with st.expander("🔍 Xem chuỗi Phoneme thô"):
            st.caption("Chuỗi phoneme thô nhận dạng được trực tiếp từ audio (chưa tách theo từ):")
            st.code(st.session_state.pron_raw_phonemes or "(trống)")
    elif st.session_state.pron_mode == "vocab":
        st.markdown(
            "**Đánh giá theo từ vựng** (chế độ dự phòng — chưa có `espeak-ng`, "
            "✓ xanh = từ khóa xuất hiện, ✗ đỏ = từ khóa còn thiếu):"
        )
        if st.session_state.pron_accuracy is not None:
            st.metric("Tỉ lệ từ khóa xuất hiện", f"{st.session_state.pron_accuracy:.1f}%")
        st.markdown(st.session_state.pron_vocab_html, unsafe_allow_html=True)
        st.caption(
            "Chế độ dự phòng: so khớp TỪ VỰNG (không phải phoneme) vì thiếu `phonemizer`/`espeak-ng`. "
            "Cài đặt espeak-ng để có chấm phát âm chi tiết theo từng âm."
        )

    ev = st.session_state.last_evaluation
    if ev:
        st.markdown("**Đánh giá nội dung (Gemini):**")
        if ev.get("on_topic") is True:
            st.success("✅ Trả lời đúng trọng tâm.")
        elif ev.get("on_topic") is False:
            st.error("❌ Trả lời chưa đúng trọng tâm / chưa đầy đủ.")
        else:
            st.info("Không xác định được (xem chi tiết bên dưới).")
        st.write(ev.get("feedback", ""))

    if st.session_state.follow_up_question:
        st.markdown("---")
        st.subheader("🔁 Giám khảo hỏi vặn lại")
        st.warning(st.session_state.follow_up_question)
        if st.button("🔊 Nghe câu hỏi vặn"):
            try:
                audio_f = text_to_speech_bytes(st.session_state.follow_up_question)
                st.audio(audio_f, format="audio/mp3")
            except Exception as e:
                st.error(f"Lỗi TTS: {e}")
        st.caption("Hãy thu âm lại câu trả lời mới ở mục '2️⃣ Thu âm câu trả lời' bên trên để trả lời câu hỏi vặn này.")

st.markdown("---")
st.caption(
    "Ghi chú: các mô hình Wav2Vec2 (~1-2GB) sẽ được tải về lần đầu chạy nên có thể mất vài phút. "
    "Chấm phát âm chỉ mang tính tham khảo, không thay thế đánh giá của giáo viên/giám khảo thật."
)