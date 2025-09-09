import os
import re
import streamlit as st
import PyPDF2
from docx import Document
import spacy
import pandas as pd
from datetime import datetime
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# After all import statements, near the top of your file
import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logger = logging.getLogger(__name__)

def load_granite_model_safe(model_name, hf_token):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token)
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        logger.warning(f"Failed to load Granite model {model_name}: {e}")
        # fallback no-op lambda
        return lambda x, max_new_tokens=150: [{"generated_text": "[Granite model unavailable]"}]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Hugging Face token from environment variable before running
HF_TOKEN = os.getenv("HF_TOKEN")

# Load IBM Granite clause simplification model pipeline
try:
    model_name = "ibm-granite/granite-3.2-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)
    simplifier_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    logger.warning(f"Failed to load Granite model: {e}")
    simplifier_pipeline = None


class DocumentProcessor:
    def extract_text_from_pdf(self, file):
        try:
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf.pages:
                if (pt := page.extract_text()) is not None:
                    text += pt + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extract error: {e}")
            return ""

    def extract_text_from_docx(self, file):
        try:
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            logger.error(f"DOCX extract error: {e}")
            return ""

    def extract_text_from_txt(self, file):
        try:
            return file.read().decode("utf-8").strip()
        except Exception as e:
            logger.error(f"TXT extract error: {e}")
            return ""

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r'[^\w\s\.,;:\(\)\-\$%]', '', text)
        return text.strip()

    def process_document(self, file, file_type):
        file_type = file_type.lower()
        if file_type == "pdf":
            return self.extract_text_from_pdf(file)
        elif file_type == "docx":
            return self.extract_text_from_docx(file)
        elif file_type == "txt":
            return self.extract_text_from_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")


class ClauseSegmenter:
    def detect_clauses(self, text):
        clauses = []
        for i, para in enumerate(text.split('. '), start=1):
            para = para.strip()
            if len(para) > 50:
                clauses.append({
                    "clause_id": i,
                    "original_text": para,
                    "clause_type": self.identify_type(para),
                    "word_count": len(para.split()),
                })
        return clauses

    def identify_type(self, text):
        t = text.lower()
        if any(k in t for k in ["confidential", "non-disclosure", "proprietary"]): return "Confidentiality"
        if any(k in t for k in ["indemnify", "liable", "liability", "damages"]): return "Indemnification"
        if any(k in t for k in ["terminate", "termination", "expire"]): return "Termination"
        if any(k in t for k in ["payment", "pay", "fee", "compensation"]): return "Payment"
        if any(k in t for k in ["whereas", "background"]): return "Recital"
        if any(k in t for k in ["force majeure", "act of god"]): return "Force Majeure"
        return "General"


class LegalNER:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_entities(self, text):
        if not self.nlp:
            return {}
        doc = self.nlp(text)
        ents = {"persons": [], "organizations": [], "dates": [], "money": [], "legal_terms": []}
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ents["persons"].append(ent.text)
            elif ent.label_ == "ORG":
                ents["organizations"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                ents["dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                ents["money"].append(ent.text)
        legal_terms = [
            r"\b(indemnity|indemnification|liability|damages|breach|default)\b",
            r"\b(confidential|proprietary|non-disclosure|nda)\b",
            r"\b(force majeure|act of god|natural disaster)\b",
            r"\b(governing law|jurisdiction|arbitration)\b",
        ]
        for patt in legal_terms:
            ents["legal_terms"].extend(re.findall(patt, text, flags=re.I))
        for k in ents:
            ents[k] = list(set(ents[k]))
        return ents


class ClauseSimplifier:
    def __init__(self):
        self.pipeline = simplifier_pipeline

    def simplify_clause(self, text):
        if not self.pipeline:
            return self._rule_based_simplification(text)
        prompt = f"Simplify this legal text into plain English: {text}"
        try:
            res = self.pipeline(prompt, max_new_tokens=150)
            return res[0]["generated_text"]
        except Exception as e:
            logger.warning(f"Simplification error: {e}")
            return self._rule_based_simplification(text)

    def _rule_based_simplification(self, text):
        replacements = {
            r"\bheretofore\b": "before this",
            r"\bhereinafter\b": "from now on",
            r"\bwhereas\b": "considering that",
            r"\btherefore\b": "so",
            r"\bnotwithstanding\b": "despite",
            r"\bpursuant to\b": "according to",
            r"\bin consideration of\b": "in exchange for",
            r"\bshall\b": "will",
            r"\bmay not\b": "cannot",
            r"\bindemnify\b": "protect from financial loss",
        }
        for patt, rep in replacements.items():
            text = re.sub(patt, rep, text, flags=re.I)
        return text


class DocumentClassifier:
    def classify_document(self, text):
        t = text.lower()
        if any(x in t for x in ["non-disclosure", "confidential", "proprietary information"]):
            return "Non-Disclosure Agreement (NDA)"
        if any(x in t for x in ["employment", "employee", "salary", "job duties"]):
            return "Employment Contract"
        if any(x in t for x in ["lease", "rent", "tenant", "landlord"]):
            return "Lease Agreement"
        if any(x in t for x in ["purchase", "buy", "sale", "goods"]):
            return "Purchase Agreement"
        if any(x in t for x in ["service", "perform", "deliverables"]):
            return "Service Agreement"
        if any(x in t for x in ["partnership", "partners", "joint venture"]):
            return "Partnership Agreement"
        if any(x in t for x in ["license", "intellectual property", "copyright"]):
            return "License Agreement"
        return "General Contract"


class DynamicResourceSuggester:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.RESOURCE_MAP = {
            "education": {
                "papers": [
                    "https://doi.org/10.53894/education-related-paper",
                    "https://arxiv.org/abs/education-nlp"
                ],
                "videos": [
                    "https://www.youtube.com/watch?v=education_video1",
                    "https://www.youtube.com/watch?v=education_video2"
                ],
                "advice": [
                    "Ensure compliance with educational privacy laws such as FERPA.",
                    "Avoid using data that may discriminate against protected groups."
                ]
            },
            "crime": {
                "papers": [
                    "https://doi.org/10.53894/crime-related-paper",
                    "https://arxiv.org/abs/crime-nlp"
                ],
                "videos": [
                    "https://www.youtube.com/watch?v=crime_video1",
                    "https://www.youtube.com/watch?v=crime_video2"
                ],
                "advice": [
                    "Be careful when using crime data due to ethical and privacy concerns.",
                    "Always anonymize sensitive personal details related to crime cases."
                ]
            },
        }

    def extract_keywords(self, text, max_keywords=5):
        if not self.nlp:
            return []
        doc = self.nlp(text)
        keywords = set()
        for chunk in doc.noun_chunks:
            keywords.add(chunk.text.lower())
        for ent in doc.ents:
            keywords.add(ent.text.lower())
        return list(keywords)[:max_keywords]

    def suggest_resources(self, text):
        keywords = self.extract_keywords(text)
        papers = set()
        videos = set()
        advice = set()
        for kw in keywords:
            if kw in self.RESOURCE_MAP:
                papers.update(self.RESOURCE_MAP[kw]["papers"])
                videos.update(self.RESOURCE_MAP[kw]["videos"])
                advice.update(self.RESOURCE_MAP[kw]["advice"])
        return keywords, list(papers), list(videos), list(advice)

    def display_suggestions(self, text):
        keywords, papers, videos, advice = self.suggest_resources(text)
        st.markdown("### 🔎 Key Topics Detected in Document")
        st.write(", ".join(keywords) if keywords else "No relevant keywords found.")

        if papers:
            st.markdown("### 📚 Recommended Research Papers")
            for p in papers:
                st.markdown(f"- [{p}]({p})")

        if videos:
            st.markdown("### 🎥 Recommended YouTube Videos")
            for v in videos:
                st.markdown(f"- [{v}]({v})")

        if advice:
            st.markdown("### ⚖️ Legal Advice and Best Practices")
            for a in advice:
                st.markdown(f"- {a}")


# ---------------- NEW LEGAL ADVISOR FEATURE ----------------
class LegalAdvisor:
    def __init__(self):
        self.ADVICE_MAP = {
            "confidentiality": [
                "Ensure all parties clearly understand what constitutes confidential information.",
                "Check if there are exceptions where disclosure is permitted (like court orders)."
            ],
            "indemnification": [
                "Review the scope of indemnity — who covers which losses.",
                "Negotiate caps or limits to indemnity obligations."
            ],
            "termination": [
                "Verify if termination notice periods are reasonable.",
                "Ensure both parties have fair exit clauses."
            ],
            "payment": [
                "Confirm due dates for payments and penalties for delays.",
                "Clarify the currency, taxation terms, and late fee policies."
            ],
            "jurisdiction": [
                "Check if the governing law jurisdiction is favorable for you.",
                "Consider arbitration clauses for faster dispute resolution."
            ]
        }

    def generate_advice(self, clauses):
        advice_list = []
        for clause in clauses:
            ctype = clause["clause_type"].lower()
            if ctype in self.ADVICE_MAP:
                advice_list.extend(self.ADVICE_MAP[ctype])
        return list(set(advice_list))

    def ai_based_advice(self, text, pipeline_ref):
        if not pipeline_ref:
            return []
        prompt = f"Read the following legal document text and provide 5 key practical legal advice points:\n\n{text}"
        try:
            res = pipeline_ref(prompt, max_new_tokens=200)
            return res[0]["generated_text"].split("\n")
        except Exception as e:
            logger.warning(f"AI Advice Error: {e}")
            return []


class ClauseWiseApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.clause_segmenter = ClauseSegmenter()
        self.ner_extractor = LegalNER()
        self.clause_simplifier = ClauseSimplifier()
        self.doc_classifier = DocumentClassifier()
        self.resource_suggester = DynamicResourceSuggester(self.ner_extractor.nlp)
        self.advisor = LegalAdvisor()

    def run(self):
        st.set_page_config(page_title="ClauseWise - Legal Document Analyzer", page_icon="⚖️", layout="wide")
        st.title("⚖️ ClauseWise - AI-Powered Legal Document Analyzer")
        st.markdown("**Simplify, analyze, and understand legal documents with AI**")

        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", ["Document Analysis", "About", "Help"])

        if page == "Document Analysis":
            self.document_analysis_page()
        elif page == "About":
            self.about_page()
        elif page == "Help":
            self.help_page()

    def document_analysis_page(self):
        st.header("📄 Upload Legal Document")
        uploaded_file = st.file_uploader("Choose a legal document", type=["pdf", "docx", "txt"])
        if uploaded_file:
            with st.spinner("Processing..."):
                file_type = uploaded_file.name.split(".")[-1]
                text = self.doc_processor.process_document(uploaded_file, file_type)
            if text:
                doc_type = self.doc_classifier.classify_document(text)
                st.success("Document processed successfully!")
                st.info(f"Document type: {doc_type}")

                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["Clause Analysis", "Entity Extraction", "Summary", "Advice", "Export"]
                )
                with tab1:
                    self.clause_analysis_tab(text)
                with tab2:
                    self.entity_extraction_tab(text)
                with tab3:
                    self.summary_tab(text, doc_type)
                with tab4:
                    self.advice_tab(text)
                with tab5:
                    self.export_tab(text)
            else:
                st.error("Failed to extract text from document.")

    def clause_analysis_tab(self, text):
        st.subheader("Clause Analysis")
        clauses = self.clause_segmenter.detect_clauses(text)
        if not clauses:
            st.warning("No clauses found.")
            return
        clause_options = [f"Clause {c['clause_id']}: {c['clause_type']}" for c in clauses]
        idx = st.selectbox("Select clause:", range(len(clauses)), format_func=lambda x: clause_options[x])
        clause = clauses[idx]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Original Clause:")
            st.text_area(" ", clause["original_text"], height=200, disabled=True, label_visibility="collapsed", key=f"orig_{clause['clause_id']}")
            st.caption(f"Word count: {clause['word_count']}")
        with col2:
            st.markdown("Simplified Clause:")
            with st.spinner("Simplifying clause..."):
                simplified = self.clause_simplifier.simplify_clause(clause["original_text"])
            st.text_area(" ", simplified, height=200, disabled=True, label_visibility="collapsed", key=f"simp_{clause['clause_id']}")
            st.caption(f"Clause type: {clause['clause_type']}")

    def entity_extraction_tab(self, text):
        st.subheader("Named Entity Extraction")
        entities = self.ner_extractor.extract_entities(text)
        if not entities:
            st.warning("No entities extracted.")
            return
        col1, col2 = st.columns(2)
        with col1:
            if entities["persons"]:
                st.write("Persons:", ", ".join(entities["persons"]))
            if entities["organizations"]:
                st.write("Organizations:", ", ".join(entities["organizations"]))
        with col2:
            if entities["dates"]:
                st.write("Dates:", ", ".join(entities["dates"]))
            if entities["money"]:
                st.write("Monetary Amounts:", ", ".join(entities["money"]))
            if entities["legal_terms"]:
                st.write("Legal Terms:", ", ".join(entities["legal_terms"]))

    def summary_tab(self, text, doc_type):
        st.subheader("Document Summary")
        wc = len(text.split())
        cc = len(text)
        st.metric("Word Count", wc)
        st.metric("Character Count", cc)
        st.metric("Document Type", doc_type)
        clauses = self.clause_segmenter.detect_clauses(text)
        clause_types = [c["clause_type"] for c in clauses]
        counts = pd.Series(clause_types).value_counts()
        st.bar_chart(counts)
        risk = self.calculate_risk_score(text)
        st.markdown(f"Risk Assessment: {risk}")

        self.resource_suggester.display_suggestions(text)

    def advice_tab(self, text):
        st.subheader("📌 Personalized Legal Advice")
        clauses = self.clause_segmenter.detect_clauses(text)

        # Rule-based advice
        advice = self.advisor.generate_advice(clauses)
        if advice:
            st.markdown("### 🧾 Rule-Based Recommendations")
            for item in advice:
                st.markdown(f"- {item}")

        # AI-based advice
        ai_advice = self.advisor.ai_based_advice(text, self.clause_simplifier.pipeline)
        if ai_advice:
            st.markdown("### 🤖 AI-Powered Recommendations")
            for item in ai_advice:
                if item.strip():
                    st.markdown(f"- {item}")

        if not advice and not ai_advice:
            st.info("No specific advice found. Document seems straightforward.")

    def calculate_risk_score(self, text):
        keywords = ["penalty", "liquidated damages", "termination", "breach", "indemnify", "liable", "warranty", "guarantee"]
        count = sum(1 for w in keywords if w in text.lower())
        if count >= 5:
            return "🔴 High Risk - Review carefully"
        elif count >= 3:
            return "🟡 Medium Risk - Standard review recommended"
        else:
            return "🟢 Low Risk - Standard contract terms"

    def export_tab(self, text):
        st.subheader("Export Analysis Report")
        clauses = self.clause_segmenter.detect_clauses(text)
        entities = self.ner_extractor.extract_entities(text)
        doc_type = self.doc_classifier.classify_document(text)
        report = self.generate_report(text, clauses, entities, doc_type)
        st.text_area("Report", report, height=300)
        st.download_button("Download Report", data=report, file_name=f"clausewise_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def generate_report(self, text, clauses, entities, doc_type):
        rpt = f"CLAUSEWISE LEGAL DOCUMENT ANALYSIS REPORT\n\nDocument Type: {doc_type}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        rpt += f"Word Count: {len(text.split())}\nCharacter Count: {len(text)}\nNumber of Clauses: {len(clauses)}\n\nCLAUSE BREAKDOWN:\n"
        for i, c in enumerate(clauses,1):
            rpt+=f"Clause {i} ({c['clause_type']}): Word Count: {c['word_count']}\nPreview: {c['original_text'][:100]}...\n\n"
        rpt+="Entity Extraction:\n"
        for k,v in entities.items():
            if v:
                rpt += f"{k.title()}: {', '.join(v)}\n"
        return rpt

    def about_page(self):
        st.header("About ClauseWise")
        st.markdown("""
        ClauseWise is an AI-powered legal document analyzer leveraging IBM Granite foundation models. It assists users to read, simplify, classify, analyze, and query legal documents efficiently.

        Developed for innovation-driven hackathons and professional use cases.
        """)

    def help_page(self):
        st.header("How to Use ClauseWise")
        st.markdown("""
        1. Upload PDF, DOCX, or TXT legal documents.
        2. Review document clause analysis, named entities, and summary.
        3. Get dynamic research papers, videos, and legal advice suggestions.
        4. Export thorough analysis reports to plain text files.
        5. Use as a compliance aid, not a substitute for legal advice.
        """)


if __name__ == "__main__":
    app = ClauseWiseApp()
    app.run()
