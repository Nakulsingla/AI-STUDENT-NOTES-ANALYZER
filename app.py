
import streamlit as st
import fitz
import re
import random
import math

from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================
# 1. PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="AI Student Notes Analyzer",
    page_icon="📚",
    layout="wide"
)


# =====================================
# PROFESSIONAL BLUE AND WHITE DESIGN
# =====================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: #F3F7FF !important;
    color: #172554 !important;
}

/* Main content */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* All normal text */
.stApp p,
.stApp label,
.stApp li,
.stApp span,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #243B63;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #123B79 !important;
    font-weight: 750 !important;
}

/* Subtitle and captions */
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #526581 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
}

.stButton > button p,
.stDownloadButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1746B0 !important;
    color: #FFFFFF !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    padding: 20px !important;
    border-radius: 14px !important;
    border: 1px solid #D6E4FF !important;
    box-shadow: 0 3px 12px rgba(30, 64, 175, 0.06);
}

div[data-testid="stMetricLabel"] {
    color: #526581 !important;
}

div[data-testid="stMetricValue"] {
    color: #174EA6 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #174EA6 !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 3px solid #2563EB !important;
}

/* Text input and text area */
.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #BDD2F5 !important;
    border-radius: 9px !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #172554 !important;
    border-color: #BDD2F5 !important;
}

/* Radio buttons and checkboxes */
.stRadio label,
.stCheckbox label {
    color: #243B63 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #FFFFFF !important;
    color: #174EA6 !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #D6E4FF !important;
    border-radius: 12px !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #93B4F5 !important;
    border-radius: 14px !important;
    padding: 15px !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* Horizontal divider */
hr {
    border-color: #D6E4FF !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #E8F0FF !important;
}

section[data-testid="stSidebar"] * {
    color: #174EA6 !important;
}

/* Radio option cards */
div[role="radiogroup"] label {
    background: #FFFFFF;
    border: 1px solid #D6E4FF;
    border-radius: 9px;
    padding: 8px;
}

/* Links */
.stApp a {
    color: #2563EB !important;
}

</style>
""",
 unsafe_allow_html=True)


st.title("📚 AI Student Notes Analyzer")
st.write("Your personal AI study and exam preparation assistant.")
st.divider()


# =====================================
# 2. PDF TEXT EXTRACTION
# =====================================

def read_pdf(file_bytes):

    text = ""

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    try:
        for page in pdf:
            text += page.get_text() + "\n"
    finally:
        pdf.close()

    return text


# =====================================
# 3. TEXT CLEANING
# =====================================

def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =====================================
# 4. SENTENCE EXTRACTION
# =====================================

def get_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        s.strip()
        for s in sentences
        if len(s.split()) >= 5
    ]


# =====================================
# 5. KEYWORD EXTRACTION
# =====================================

def get_keywords(text, limit=20):

    sentences = get_sentences(text)

    if not sentences:
        sentences = [text]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=300
    )

    matrix = vectorizer.fit_transform(sentences)

    scores = matrix.sum(axis=0).A1

    words = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(words, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        word for word, score in ranked[:limit]
    ]


# =====================================
# 6. SMART SUMMARY
# =====================================

def make_summary(text, level="Medium"):

    sentences = get_sentences(text)

    if not sentences:
        return "Not enough sentences for a summary."

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(sentences)

        scores = matrix.sum(axis=1).A1

        scores = [
            score / math.sqrt(max(len(s.split()), 1))
            for score, s in zip(scores, sentences)
        ]

        scores = [
            score + (0.05 / (i + 1))
            for i, score in enumerate(scores)
        ]

        if level == "Short":
            count = 3
        elif level == "Detailed":
            count = 10
        else:
            count = 5

        count = min(count, len(sentences))

        ranked = sorted(
            range(len(sentences)),
            key=lambda i: scores[i],
            reverse=True
        )

        selected = sorted(ranked[:count])

        return "\n\n".join(
            "• " + sentences[i]
            for i in selected
        )

    except ValueError:

        return "\n\n".join(sentences[:5])


# =====================================
# 7. FIND ANSWER SENTENCE
# =====================================

def find_answer_sentence(keyword, sentences):

    pattern = re.compile(
        r'(?<!\w)' + re.escape(keyword) + r'(?!\w)',
        re.IGNORECASE
    )

    for sentence in sentences:

        if pattern.search(sentence):
            return sentence

    return None


# =====================================
# 8. GENERATE MCQS
# =====================================

def generate_mcqs(text, keywords, limit=10):

    sentences = get_sentences(text)

    questions = []

    for keyword in keywords:

        sentence = find_answer_sentence(
            keyword,
            sentences
        )

        if not sentence:
            continue

        pattern = re.compile(
            r'(?<!\w)' + re.escape(keyword) + r'(?!\w)',
            re.IGNORECASE
        )

        question_text = pattern.sub(
            "________",
            sentence,
            count=1
        )

        distractors = [
            word for word in keywords
            if word.lower() != keyword.lower()
            and word.lower() not in sentence.lower()
        ]

        if len(distractors) < 3:
            continue

        rng = random.Random(keyword + sentence)

        selected_distractors = rng.sample(
            distractors,
            3
        )

        options = [keyword] + selected_distractors

        rng.shuffle(options)

        questions.append({
            "question": question_text,
            "options": options,
            "answer": keyword,
            "explanation": sentence
        })

        if len(questions) >= limit:
            break

    return questions


# =====================================
# 9. WRITTEN QUESTION GENERATOR
# =====================================

def generate_written_questions(text, keywords):

    sentences = get_sentences(text)

    result = []

    for keyword in keywords:

        sentence = find_answer_sentence(
            keyword,
            sentences
        )

        if sentence:

            result.append({
                "question": (
                    f"What do your notes explain about {keyword}?"
                ),
                "answer": sentence,
                "long_question": (
                    f"Explain {keyword} in detail "
                    "using your study notes."
                )
            })

        if len(result) >= 10:
            break

    return result


# =====================================
# 10. UPLOAD SECTION
# =====================================

st.subheader("📤 Upload Your Study Notes")

uploaded_file = st.file_uploader(
    "Choose PDF or TXT file",
    type=["pdf", "txt"]
)


# =====================================
# 11. ANALYSIS SECTION
# =====================================

if uploaded_file:

    try:

        file_bytes = uploaded_file.getvalue()

        if uploaded_file.name.lower().endswith(".pdf"):

            text = read_pdf(file_bytes)

        else:

            text = file_bytes.decode(
                "utf-8-sig",
                errors="ignore"
            )

        text = clean_text(text)

        if len(text.split()) < 10:

            st.error("Please upload notes containing at least 10 words.")

        else:

            st.success("Notes uploaded successfully!")

            keywords = get_keywords(text)

            mcqs = generate_mcqs(
                text,
                keywords,
                10
            )

            written = generate_written_questions(
                text,
                keywords
            )

            # DASHBOARD METRICS

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Words",
                len(text.split())
            )

            col2.metric(
                "Important Keywords",
                len(keywords)
            )

            col3.metric(
                "Practice MCQs",
                len(mcqs)
            )

            st.divider()

            # TABS

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📖 Summary",
                "🔑 Keywords",
                "📝 MCQ Quiz",
                "✍️ Written Questions",
                "📄 Original Notes"
            ])

            # =================================
            # SUMMARY TAB
            # =================================

            with tab1:

                st.header("AI Smart Summary")

                level = st.selectbox(
                    "Summary length",
                    ["Short", "Medium", "Detailed"]
                )

                summary = make_summary(
                    text,
                    level
                )

                st.markdown(summary)

                st.download_button(
                    "⬇️ Download Summary",
                    summary,
                    file_name="AI_Study_Summary.txt",
                    mime="text/plain"
                )

            # =================================
            # KEYWORDS TAB
            # =================================

            with tab2:

                st.header("Important Keywords")

                for i, word in enumerate(keywords, 1):

                    st.write(f"{i}. **{word}**")

            # =================================
            # MCQ QUIZ TAB
            # =================================

            with tab3:

                st.header("Practice MCQ Quiz")

                if mcqs:

                    with st.form("mcq_form"):

                        user_answers = []

                        for i, question in enumerate(mcqs):

                            st.subheader(
                                f"Question {i + 1}"
                            )

                            st.write(
                                question["question"]
                            )

                            selected = st.radio(
                                "Choose your answer:",
                                question["options"],
                                index=None,
                                key=f"mcq_{i}"
                            )

                            user_answers.append(selected)

                            st.divider()

                        submitted = st.form_submit_button(
                            "Submit My Quiz"
                        )

                    if submitted:

                        score = sum(
                            1 for i, question in enumerate(mcqs)
                            if user_answers[i] == question["answer"]
                        )

                        st.session_state.quiz_score = score
                        st.session_state.quiz_answers = user_answers
                        st.session_state.quiz_submitted = True

                    if st.session_state.get(
                        "quiz_submitted",
                        False
                    ):

                        score = st.session_state.quiz_score

                        answers = st.session_state.quiz_answers

                        percentage = (
                            score / len(mcqs)
                        ) * 100

                        st.divider()

                        st.subheader("📊 Quiz Results")

                        st.metric(
                            "Your Score",
                            f"{score}/{len(mcqs)}"
                        )

                        st.progress(percentage / 100)

                        st.write(
                            f"Accuracy: {percentage:.1f}%"
                        )

                        if percentage >= 80:
                            st.success("Excellent work!")
                        elif percentage >= 50:
                            st.info("Good effort!")
                        else:
                            st.warning("Keep practicing!")

                        st.subheader(
                            "Correct Answers and Explanations"
                        )

                        for i, question in enumerate(mcqs):

                            st.write(
                                f"Question {i + 1}: "
                                + question["question"]
                            )

                            if answers[i] == question["answer"]:
                                st.success("Correct!")
                            else:
                                st.error("Incorrect!")

                            st.write(
                                "Correct answer:",
                                question["answer"]
                            )

                            st.info(
                                "Explanation: "
                                + question["explanation"]
                            )

                        if st.button("Try Again"):

                            st.session_state.quiz_submitted = False

                            for i in range(len(mcqs)):
                                st.session_state.pop(
                                    f"mcq_{i}",
                                    None
                                )

                            st.rerun()

                else:

                    st.warning(
                        "Not enough suitable keywords to generate MCQs."
                    )

            # =================================
            # WRITTEN QUESTIONS TAB
            # =================================

            with tab4:

                st.header("Important Written Questions")

                for i, item in enumerate(written, 1):

                    with st.expander(
                        f"Question {i}: {item['question']}"
                    ):

                        st.write("Suggested Answer:")
                        st.write(item["answer"])

                st.divider()

                st.subheader("Long Answer Questions")

                for i, item in enumerate(written, 1):

                    st.write(
                        f"{i}. {item['long_question']}"
                    )

            # =================================
            # ORIGINAL NOTES TAB
            # =================================

            with tab5:

                st.header("Original Notes")

                st.text_area(
                    "Extracted text",
                    text,
                    height=350
                )

                st.download_button(
                    "Download Original Notes",
                    text,
                    file_name="Original_Notes.txt",
                    mime="text/plain"
                )

    except Exception as error:

        st.error("Unable to analyze your document.")

        st.write(str(error))

else:

    st.info(
        "Upload a PDF or TXT document to begin."
    )

    st.markdown("""
    ### Features

    - 📖 Smart summaries
    - 🔑 Important keywords
    - 📝 MCQ generation
    - ✅ Quiz solving and scoring
    - ✍️ Short and long questions
    - 📄 Download notes
    """)
