import io
import json
import os
import uuid
from datetime import datetime, timezone

import faiss
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from groq import Groq
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Place the trained model and retrieval files in ./models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "oral_cancer_efficientnetb0.keras"))
FAISS_PATH = os.getenv("FAISS_PATH", os.path.join(BASE_DIR, "models", "oscc_literature.faiss"))
METADATA_PATH = os.getenv("METADATA_PATH", os.path.join(BASE_DIR, "models", "chunk_metadata.json"))

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

for required_path in (MODEL_PATH, FAISS_PATH, METADATA_PATH):
    if not os.path.isfile(required_path):
        raise FileNotFoundError(
            f"Required artifact not found: {required_path}. "
            "Add the model/index/metadata files under ./models or set the corresponding environment variable."
        )

model = tf.keras.models.load_model(MODEL_PATH)
faiss_index = faiss.read_index(FAISS_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)
with neo4j_driver.session(database=NEO4J_DATABASE) as session:
    session.run("RETURN 1 AS ok").single()

groq_client = Groq(api_key=GROQ_API_KEY)
IMAGE_SIZE = (224, 224)


# --- Model prediction, GraphRAG retrieval, and graph persistence helpers ---
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)
    score = float(np.squeeze(pred))

    label = "OSCC" if score >= 0.5 else "Normal"
    confidence = score if label == "OSCC" else (1.0 - score)

    return label, confidence


def normalize_case(
    age=None, sex=None, tobacco=None, alcohol=None,
    lesion_site=None, magnification=None
):
    return {
        "age": age or "Not provided",
        "sex": sex or "Not provided",
        "tobacco": tobacco or "Not provided",
        "alcohol": alcohol or "Not provided",
        "lesion_site": lesion_site or "Not provided",
        "magnification": magnification or "Not provided",
    }


def get_oscc_graph_context():
    query = """
    MATCH (d:Disease {name:"Oral Squamous Cell Carcinoma"})
    OPTIONAL MATCH (d)-[r]-(x)
    RETURN d.name AS disease,
           type(r) AS relationship,
           labels(x) AS node_labels,
           x.name AS node_name,
           x.title AS node_title,
           x.paper_id AS paper_id,
           x.year AS year
    ORDER BY relationship, node_name, node_title
    """
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        return [dict(r) for r in session.run(query)]


def build_case_query(prediction_label, confidence, patient_context):
    parts = [
        "oral squamous cell carcinoma histopathology",
        f"model prediction {prediction_label}",
        f"model confidence {confidence:.4f}",
    ]

    for key, value in patient_context.items():
        if value and value != "Not provided":
            parts.append(f"{key}: {value}")

    parts.append(
        "diagnosis histological features biomarkers research evidence"
    )
    return " ".join(parts)


def retrieve_literature(query, top_k=5):
    q = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = faiss_index.search(q, top_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunk_metadata):
            continue
        item = dict(chunk_metadata[idx])
        item["similarity"] = float(score)
        results.append(item)

    return results


def save_case_to_graph(
    case_id, filename, patient_context, prediction_label, confidence
):
    query = """
    MERGE (c:Case {case_id:$case_id})
    SET c.filename=$filename,
        c.age=$age,
        c.sex=$sex,
        c.tobacco=$tobacco,
        c.alcohol=$alcohol,
        c.lesion_site=$lesion_site,
        c.magnification=$magnification,
        c.prediction=$prediction,
        c.confidence=$confidence,
        c.updated_at=$updated_at

    MERGE (p:Prediction {case_id:$case_id})
    SET p.label=$prediction,
        p.confidence=$confidence,
        p.updated_at=$updated_at

    MERGE (c)-[:HAS_MODEL_PREDICTION]->(p)

    WITH p
    OPTIONAL MATCH (d:Disease {
        name:"Oral Squamous Cell Carcinoma"
    })

    FOREACH (
        _ IN CASE
            WHEN $prediction = "OSCC" AND d IS NOT NULL
            THEN [1]
            ELSE []
        END |
        MERGE (p)-[:PREDICTS_CLASS]->(d)
    )
    """

    params = {
        "case_id": case_id,
        "filename": filename,
        **patient_context,
        "prediction": prediction_label,
        "confidence": float(confidence),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        session.run(query, **params).consume()


# --- Research report generation ---
REPORT_SYSTEM_INSTRUCTION = """
You are a research-support assistant for an oral cancer histopathology project.

Generate a careful, structured, research-oriented report from:
1. CNN prediction and confidence
2. Supplied patient/case metadata
3. Neo4j knowledge-graph context
4. Retrieved literature evidence

STRICT RULES:

1. Never invent patient information, clinical findings, or laboratory results.

2. Treat the CNN prediction strictly as a MODEL OUTPUT.
   Example:
   "The CNN classifier predicted the image as Normal with 89.83%
   model confidence."

3. Never convert CNN confidence into:
   - clinical probability
   - cancer probability
   - diagnostic certainty
   - probability that the patient is disease-free.

4. Do not claim that a histopathological feature is present or absent
   unless that feature is explicitly provided as verified evidence.

5. Do not claim that biomarkers such as Cyclin D1, EGFR, Ki-67,
   PD-L1, or p16 were detected in the submitted image unless
   biomarker-specific evidence is explicitly provided.

6. Knowledge-graph relationships provide RESEARCH CONTEXT.
   They do not establish that a risk factor, biomarker, or
   histological feature is present in this specific specimen.

7. Retrieved literature provides population-level or research evidence.
   Do not present literature findings as patient-specific observations.

8. Dataset filenames or labels such as "normal.jpg" may be reported
   as dataset metadata, but must not be presented as independent
   pathological confirmation.

9. Clearly separate:
   A. Case information
   B. CNN model result
   C. Knowledge-graph context
   D. Literature evidence
   E. Integrated interpretation
   F. Limitations

10. Do not provide a definitive clinical diagnosis.

11. Do not recommend treatment.

12. Use cautious research language such as:
   "the model predicts",
   "the knowledge graph associates",
   "the literature reports",
   "the retrieved evidence provides context",
   "this does not establish presence in the submitted image."

13. The final report must explicitly state that the output is
    intended for research/academic use and is not a clinical diagnosis.
"""


def generate_research_report(
    case_id, filename, patient_context, prediction_label, confidence,
    graph_context, retrieval_query, literature_evidence
):
    payload = {
        "case_id": case_id,
        "filename": filename,
        "patient_context": patient_context,
        "cnn_prediction": prediction_label,
        "cnn_confidence": confidence,
        "knowledge_graph_context": graph_context,
        "retrieval_query": retrieval_query,
        "literature_evidence": literature_evidence,
    }

    prompt = f"""
Prepare a concise but detailed research report for this OSCC GraphRAG case.

Use exactly these sections:

1. Case Summary
2. CNN Model Result
3. Knowledge Graph Context
4. Literature Evidence
5. Integrated Research Interpretation
6. Limitations

Source data:

{json.dumps(payload, indent=2, ensure_ascii=False)}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2500
    )

    return response.choices[0].message.content.strip()



# --- FastAPI endpoints and embedded website ---
app = FastAPI(
    title="OSCC GraphRAG Backend",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cnn": model is not None,
        "faiss": faiss_index is not None,
        "neo4j": neo4j_driver is not None,
        "groq": groq_client is not None,
        "groq_model": GROQ_MODEL,
        "model": "EfficientNetB0"
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    age: str = Form(""),
    sex: str = Form(""),
    tobacco: str = Form(""),
    alcohol: str = Form(""),
    lesion_site: str = Form(""),
    magnification: str = Form("")
):
    image_bytes = await image.read()

    prediction_label, confidence = predict_image(image_bytes)

    patient_context = normalize_case(
        age=age,
        sex=sex,
        tobacco=tobacco,
        alcohol=alcohol,
        lesion_site=lesion_site,
        magnification=magnification
    )

    case_id = "CASE-" + uuid.uuid4().hex[:10].upper()

    graph_context = get_oscc_graph_context()

    retrieval_query = build_case_query(
        prediction_label, confidence, patient_context
    )

    literature_evidence = retrieve_literature(
        retrieval_query, top_k=5
    )

    save_case_to_graph(
        case_id,
        image.filename,
        patient_context,
        prediction_label,
        confidence
    )

    report_text = generate_research_report(
        case_id=case_id,
        filename=image.filename,
        patient_context=patient_context,
        prediction_label=prediction_label,
        confidence=confidence,
        graph_context=graph_context,
        retrieval_query=retrieval_query,
        literature_evidence=literature_evidence
    )

    return {
        "case_id": case_id,
        "filename": image.filename,
        "patient_context": patient_context,
        "prediction": prediction_label,
        "confidence": float(confidence),
        "confidence_percent": round(float(confidence) * 100, 2),
        "graph_context": graph_context,
        "retrieval_query": retrieval_query,
        "literature_evidence": literature_evidence,
        "scope_note": (
            "Literature and knowledge-graph evidence provide research context. "
            "They do not establish that a specific histological feature or "
            "biomarker is present in the uploaded image."
        ),
        "report": {
            "provider": "Groq",
            "model": GROQ_MODEL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "text": report_text
        }
    }





# ------------------------------------------------------------
# Built-in frontend
# Polished single-page frontend served by FastAPI.
# No separate frontend project is required.
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

HTML_PAGE = '\\\n<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>OSCC GraphRAG | Research Console</title>\n<style>\n:root{--bg:#07111f;--panel:#0d1b2e;--panel2:#10243b;--line:#1e3652;--text:#eaf2ff;--muted:#8fa6bf;--accent:#36d1dc;--accent2:#6c7cff;--good:#35d07f;--warn:#ffbd59;--danger:#ff6b7a;--shadow:0 24px 70px rgba(0,0,0,.28)}\n*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 15% 0%,#123254 0,transparent 34%),radial-gradient(circle at 90% 10%,#24235b 0,transparent 30%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}.wrap{max-width:1180px;margin:auto;padding:28px 20px 60px}.top{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:24px}.brand{display:flex;align-items:center;gap:13px}.logo{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;font-size:23px;font-weight:900;box-shadow:0 10px 30px rgba(54,209,220,.22)}.brand h1{font-size:18px;margin:0}.brand p{margin:3px 0 0;color:var(--muted);font-size:12px}.status-pill{border:1px solid #27506c;background:#0c2034;padding:9px 12px;border-radius:999px;font-size:12px;color:#b9d3e9}.dot{display:inline-block;width:7px;height:7px;background:var(--good);border-radius:50%;margin-right:7px;box-shadow:0 0 12px var(--good)}\n.hero{position:relative;overflow:hidden;padding:38px;border:1px solid var(--line);border-radius:26px;background:linear-gradient(135deg,rgba(16,36,59,.95),rgba(10,24,43,.82));box-shadow:var(--shadow);margin-bottom:22px}.hero:after{content:"";position:absolute;width:280px;height:280px;border-radius:50%;right:-90px;top:-120px;background:rgba(54,209,220,.12);filter:blur(8px)}.eyebrow{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase}.hero h2{font-size:clamp(28px,5vw,48px);line-height:1.05;margin:10px 0 12px;max-width:760px}.hero p{color:var(--muted);max-width:760px;line-height:1.65;margin:0}.tech{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}.tech span{padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:#0a1829;color:#b8cae0;font-size:11px}\n.layout{display:grid;grid-template-columns:1fr 360px;gap:20px}.card{background:rgba(13,27,46,.94);border:1px solid var(--line);border-radius:22px;box-shadow:var(--shadow);padding:22px}.card h3{margin:0 0 5px;font-size:18px}.sub{color:var(--muted);font-size:12px;margin:0 0 18px}.drop{border:1.5px dashed #376080;border-radius:18px;padding:30px 18px;text-align:center;background:linear-gradient(180deg,#0c1c2f,#0a1727);cursor:pointer;transition:.2s}.drop:hover,.drop.drag{border-color:var(--accent);background:#0d2238}.upload-icon{font-size:34px;margin-bottom:10px}.drop strong{display:block;font-size:15px}.drop small{display:block;color:var(--muted);margin-top:7px}.drop input{display:none}.preview{display:none;margin-top:14px;grid-template-columns:130px 1fr;gap:14px;padding:12px;border:1px solid var(--line);border-radius:15px;background:#091728}.preview img{width:130px;height:110px;object-fit:cover;border-radius:10px}.filemeta{display:flex;flex-direction:column;justify-content:center;gap:5px}.filemeta b{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.filemeta span{font-size:11px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:13px;margin-top:16px}.field label{display:block;font-size:11px;color:#9db2c8;margin:0 0 6px;font-weight:700}.field input,.field select{width:100%;padding:12px 13px;border:1px solid var(--line);border-radius:11px;background:#091728;color:var(--text);outline:none}.field input:focus,.field select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(54,209,220,.08)}.full{grid-column:1/-1}.actions{display:flex;gap:10px;margin-top:16px}.primary,.secondary{border:0;border-radius:12px;padding:13px 16px;font-weight:800;cursor:pointer}.primary{flex:1;color:#04131d;background:linear-gradient(135deg,var(--accent),#7ce7df);box-shadow:0 12px 25px rgba(54,209,220,.14)}.primary:disabled{opacity:.55;cursor:wait}.secondary{background:#10243b;color:#bcd0e4;border:1px solid var(--line)}.notice{margin-top:13px;padding:11px 13px;border-radius:11px;background:#0b1b2d;border:1px solid var(--line);color:#91a8bf;font-size:11px;line-height:1.5}.side-stat{display:grid;grid-template-columns:1fr 1fr;gap:10px}.stat{padding:14px;border:1px solid var(--line);border-radius:15px;background:#0a1829}.stat b{display:block;font-size:17px;margin-top:4px}.stat span{font-size:10px;color:var(--muted)}.steps{display:flex;flex-direction:column;gap:11px;margin-top:18px}.step{display:flex;gap:11px;align-items:flex-start;color:#8ea4bb;font-size:12px}.step .n{width:25px;height:25px;border-radius:8px;border:1px solid var(--line);display:grid;place-items:center;flex:none}.step.active{color:#d9e8f8}.step.active .n{border-color:var(--accent);color:var(--accent)}\n#status{min-height:20px;margin-top:13px;font-size:12px;color:var(--muted)}.loader{display:none;height:5px;background:#10243b;border-radius:999px;overflow:hidden;margin-top:8px}.loader i{display:block;width:35%;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));animation:load 1.1s infinite}.results{display:none;margin-top:20px}.result-head{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap}.case{font-size:11px;color:var(--muted)}.badges{display:flex;flex-wrap:wrap;gap:8px}.badge{padding:8px 11px;border-radius:999px;border:1px solid var(--line);background:#0a1829;font-size:11px;color:#c6d7e8}.badge strong{color:white}.result-grid{display:grid;grid-template-columns:270px 1fr;gap:18px;margin-top:18px}.confidence{display:grid;place-items:center;min-height:245px;border:1px solid var(--line);border-radius:18px;background:#091728}.ring{--p:0deg;width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--accent) var(--p),#19314a 0deg);position:relative}.ring:after{content:"";position:absolute;width:116px;height:116px;background:#091728;border-radius:50%}.ring div{position:relative;z-index:1;text-align:center}.ring b{display:block;font-size:25px}.ring span{font-size:10px;color:var(--muted)}.prediction{margin-top:12px;font-size:13px;font-weight:800}.tabs{display:flex;gap:7px;border-bottom:1px solid var(--line);padding-bottom:8px}.tab{border:0;background:transparent;color:#8198b0;padding:8px 10px;border-radius:8px;cursor:pointer;font-size:11px;font-weight:700}.tab.active{background:#102b43;color:#cfe9f3}.panel{display:none;padding-top:14px}.panel.active{display:block}.report{white-space:pre-wrap;line-height:1.7;color:#c8d7e7;font-size:13px}.evidence{display:grid;gap:9px}.ev{border:1px solid var(--line);background:#091728;border-radius:12px;padding:12px}.ev b{font-size:12px}.ev p{margin:5px 0 0;color:#8fa7bd;font-size:11px;line-height:1.5}.jsonbox{white-space:pre-wrap;background:#07111f;border:1px solid var(--line);padding:13px;border-radius:12px;color:#9db5cb;font-size:11px;line-height:1.55;max-height:360px;overflow:auto}.footer{margin-top:24px;color:#617991;text-align:center;font-size:10px}.toast{position:fixed;right:18px;bottom:18px;background:#102b43;border:1px solid #31536e;color:#d9ebf7;padding:12px 15px;border-radius:12px;box-shadow:var(--shadow);display:none;font-size:12px;z-index:10}@keyframes load{0%{transform:translateX(-120%)}100%{transform:translateX(330%)}}@media(max-width:900px){.layout,.result-grid{grid-template-columns:1fr}.side{order:-1}.side-stat{grid-template-columns:repeat(4,1fr)}}@media(max-width:620px){.wrap{padding:16px 12px 40px}.hero{padding:25px}.grid{grid-template-columns:1fr}.full{grid-column:auto}.top{align-items:flex-start}.status-pill{display:none}.side-stat{grid-template-columns:1fr 1fr}.preview{grid-template-columns:90px 1fr}.preview img{width:90px;height:85px}.actions{flex-direction:column}}\n</style>\n</head>\n<body>\n<div class="wrap">\n  <header class="top"><div class="brand"><div class="logo">◈</div><div><h1>OSCC GraphRAG</h1><p>Research &amp; knowledge intelligence console</p></div></div><div class="status-pill"><span class="dot"></span>Backend ready</div></header>\n  <section class="hero"><div class="eyebrow">AI-assisted histopathology research</div><h2>Explore an OSCC case with vision + knowledge.</h2><p>Upload a histopathology image and combine EfficientNetB0 prediction with retrieved literature, a Neo4j knowledge graph and a Groq-generated research report.</p><div class="tech"><span>EfficientNetB0</span><span>FAISS Retrieval</span><span>Neo4j Graph</span><span>Groq LLM</span></div></section>\n  <div class="layout">\n    <main>\n      <section class="card"><h3>New analysis</h3><p class="sub">Provide the image and optional case context.</p>\n        <form id="analysisForm">\n          <label class="drop" id="dropZone"><input id="image" name="image" type="file" accept="image/*" required><div class="upload-icon">⇧</div><strong>Drop histopathology image here</strong><small>or click to browse · PNG, JPG, JPEG</small></label>\n          <div class="preview" id="preview"><img id="previewImg"><div class="filemeta"><b id="fileName">—</b><span id="fileSize">—</span><span>Image selected and ready for analysis</span></div></div>\n          <div class="grid">\n            <div class="field"><label>AGE</label><input id="age" name="age" placeholder="e.g. 55"></div>\n            <div class="field"><label>SEX</label><select id="sex" name="sex"><option value="">Select</option><option>Male</option><option>Female</option><option>Other</option></select></div>\n            <div class="field"><label>TOBACCO</label><input id="tobacco" name="tobacco" placeholder="Yes / No"></div>\n            <div class="field"><label>ALCOHOL</label><input id="alcohol" name="alcohol" placeholder="Yes / No"></div>\n            <div class="field"><label>LESION SITE</label><input id="lesion_site" name="lesion_site" placeholder="e.g. Buccal mucosa"></div>\n            <div class="field"><label>MAGNIFICATION</label><input id="magnification" name="magnification" placeholder="e.g. 40x"></div>\n          </div>\n          <div class="actions"><button class="primary" id="submitBtn" type="submit">Analyze case →</button><button class="secondary" id="resetBtn" type="button">Reset</button></div>\n          <div class="loader" id="loader"><i></i></div><div id="status"></div>\n          <div class="notice">Research support only. Model confidence is not a clinical probability, and the retrieved literature does not prove that a specific histological feature is present in the uploaded image.</div>\n        </form>\n      </section>\n      <section class="card results" id="resultCard">\n        <div class="result-head"><div><h3>Case analysis</h3><div class="case" id="caseLine">—</div></div><div class="badges" id="badges"></div></div>\n        <div class="result-grid"><div class="confidence"><div><div class="ring" id="ring"><div><b id="confidenceValue">0%</b><span>model confidence</span></div></div><div class="prediction" id="prediction">Prediction: —</div></div></div>\n          <div><div class="tabs"><button class="tab active" data-tab="reportPanel">Research report</button><button class="tab" data-tab="litPanel">Literature</button><button class="tab" data-tab="graphPanel">Knowledge graph</button></div>\n            <div class="panel active" id="reportPanel"><div class="report" id="report"></div><button class="secondary" id="downloadBtn" style="margin-top:12px">Download report</button></div>\n            <div class="panel" id="litPanel"><div class="evidence" id="literature"></div></div>\n            <div class="panel" id="graphPanel"><div class="jsonbox" id="graph"></div></div>\n          </div>\n        </div>\n      </section>\n    </main>\n    <aside class="side"><section class="card"><h3>Pipeline</h3><p class="sub">What happens after you submit</p><div class="side-stat"><div class="stat"><span>VISION</span><b>CNN</b></div><div class="stat"><span>SEARCH</span><b>FAISS</b></div><div class="stat"><span>GRAPH</span><b>Neo4j</b></div><div class="stat"><span>REPORT</span><b>Groq</b></div></div><div class="steps"><div class="step active"><div class="n">1</div><div><b>Image inference</b><br>EfficientNetB0 produces the model output.</div></div><div class="step active"><div class="n">2</div><div><b>Evidence retrieval</b><br>Relevant research chunks are retrieved from FAISS.</div></div><div class="step active"><div class="n">3</div><div><b>Graph context</b><br>Neo4j contributes structured case knowledge.</div></div><div class="step active"><div class="n">4</div><div><b>Research synthesis</b><br>Groq turns the supplied evidence into a report.</div></div></div></section></aside>\n  </div>\n  <div class="footer">OSCC GraphRAG · Academic/research prototype · Not a diagnostic device</div>\n</div><div class="toast" id="toast"></div>\n<script>\nconst $=id=>document.getElementById(id); const form=$(\'analysisForm\'), file=$(\'image\'), drop=$(\'dropZone\'), preview=$(\'preview\'); let latest=null;\nfunction toast(msg){const t=$(\'toast\');t.textContent=msg;t.style.display=\'block\';setTimeout(()=>t.style.display=\'none\',2600)}\nfunction formatBytes(n){if(!n)return \'0 B\';const u=[\'B\',\'KB\',\'MB\',\'GB\'];let i=0;while(n>=1024&&i<3){n/=1024;i++}return n.toFixed(i?1:0)+\' \'+u[i]}\nfunction showFile(f){if(!f)return; $(\'fileName\').textContent=f.name;$(\'fileSize\').textContent=formatBytes(f.size);$(\'previewImg\').src=URL.createObjectURL(f);preview.style.display=\'grid\'}\nfile.addEventListener(\'change\',()=>showFile(file.files[0]));[\'dragenter\',\'dragover\'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.add(\'drag\')}));[\'dragleave\',\'drop\'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.remove(\'drag\')}));drop.addEventListener(\'drop\',e=>{const f=e.dataTransfer.files[0];if(f&&f.type.startsWith(\'image/\')){file.files=e.dataTransfer.files;showFile(f)}else toast(\'Please drop an image file.\')});\n$(\'resetBtn\').onclick=()=>{form.reset();preview.style.display=\'none\';$(\'resultCard\').style.display=\'none\';$(\'status\').textContent=\'\';latest=null};\ndocument.querySelectorAll(\'.tab\').forEach(b=>b.onclick=()=>{document.querySelectorAll(\'.tab\').forEach(x=>x.classList.remove(\'active\'));document.querySelectorAll(\'.panel\').forEach(x=>x.classList.remove(\'active\'));b.classList.add(\'active\');$(b.dataset.tab).classList.add(\'active\')});\nfunction esc(v){return String(v??\'\').replace(/[&<>"\']/g,c=>({\'&\':\'&amp;\',\'<\':\'&lt;\',\'>\':\'&gt;\',\'"\':\'&quot;\',"\'":\'&#39;\'}[c]))}\nfunction literatureHTML(items){if(!Array.isArray(items)||!items.length)return \'<div class="ev"><b>No literature evidence returned</b><p>The retrieval step returned no displayable records.</p></div>\';return items.map((x,i)=>{if(typeof x===\'string\')return `<div class="ev"><b>Evidence ${i+1}</b><p>${esc(x)}</p></div>`;const title=x.title||x.paper_title||x.name||`Evidence ${i+1}`;const body=x.text||x.content||x.chunk||x.snippet||x.abstract||JSON.stringify(x);return `<div class="ev"><b>${esc(title)}</b><p>${esc(body)}</p></div>`}).join(\'\')}\nform.addEventListener(\'submit\',async e=>{e.preventDefault();if(!file.files[0])return toast(\'Select a histopathology image first.\');const fd=new FormData();fd.append(\'image\',file.files[0]);[\'age\',\'sex\',\'tobacco\',\'alcohol\',\'lesion_site\',\'magnification\'].forEach(id=>fd.append(id,$(id).value));$(\'submitBtn\').disabled=true;$(\'loader\').style.display=\'block\';$(\'status\').textContent=\'Running image inference → retrieval → graph → report…\';$(\'resultCard\').style.display=\'none\';try{const r=await fetch(\'/analyze\',{method:\'POST\',body:fd});let d;try{d=await r.json()}catch(_){throw new Error(\'Server returned an invalid response.\')}if(!r.ok)throw new Error(d.detail||\'Analysis failed.\');latest=d;const pct=Number(d.confidence_percent??(Number(d.confidence||0)*100));$(\'confidenceValue\').textContent=pct.toFixed(1)+\'%\';$(\'ring\').style.setProperty(\'--p\',Math.max(0,Math.min(100,pct))*3.6+\'deg\');$(\'prediction\').textContent=\'Prediction: \'+(d.prediction||\'N/A\');$(\'caseLine\').textContent=(d.case_id||\'CASE\')+\' · \'+(d.filename||\'image\');$(\'badges\').innerHTML=`<span class="badge">Model <strong>EfficientNetB0</strong></span><span class="badge">LLM <strong>Groq</strong></span>`;$(\'report\').textContent=d.report?.text||\'No report returned.\';$(\'literature\').innerHTML=literatureHTML(d.literature_evidence);$(\'graph\').textContent=JSON.stringify(d.graph_context??{},null,2);$(\'resultCard\').style.display=\'block\';$(\'status\').textContent=\'Analysis completed successfully.\';toast(\'Case analysis complete.\');setTimeout(()=>$(\'resultCard\').scrollIntoView({behavior:\'smooth\',block:\'start\'}),80)}catch(err){$(\'status\').textContent=\'Error: \'+err.message;toast(\'Analysis failed.\')}finally{$(\'submitBtn\').disabled=false;$(\'loader\').style.display=\'none\'}});\n$(\'downloadBtn\').onclick=()=>{if(!latest)return;const text=`OSCC GraphRAG Research Report\\n\\nCase: ${latest.case_id||\'\'}\\nPrediction: ${latest.prediction||\'\'}\\nConfidence: ${latest.confidence_percent??\'\'}%\\n\\n${latest.report?.text||\'\'}\\n\\nScope note:\\n${latest.scope_note||\'\'}`;const a=document.createElement(\'a\');a.href=URL.createObjectURL(new Blob([text],{type:\'text/plain\'}));a.download=(latest.case_id||\'OSCC_case\')+\'_report.txt\';a.click();URL.revokeObjectURL(a.href)};\n</script></body></html>\n'

print("Enhanced frontend route created successfully.")
print("Website URL: PUBLIC_BACKEND_URL + '/'")


@app.on_event("shutdown")
def shutdown_event():
    try:
        neo4j_driver.close()
    except Exception:
        pass
