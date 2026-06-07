import asyncio
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    mp = None
    logging.warning("MediaPipe is not available: %s", exc)

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    ort = None
    logging.warning("onnxruntime is not available: %s", exc)

try:
    from insightface.app import FaceAnalysis
except Exception as exc:  # pragma: no cover
    FaceAnalysis = None
    logging.warning("InsightFace is not available: %s", exc)

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception as exc:  # pragma: no cover
    hf_hub_download = None
    list_repo_files = None
    logging.warning("huggingface_hub is not available: %s", exc)


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ai-worker")

HF_REPO_ID = os.getenv("HF_REPO_ID", "garciafido/minifasnet-v2-anti-spoofing-onnx")
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "")
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BLINK_EAR_THRESHOLD = float(os.getenv("BLINK_EAR_THRESHOLD", "0.20"))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
AI_WORKER_CONCURRENCY = max(1, int(os.getenv("AI_WORKER_CONCURRENCY", "4")))
AI_WORKER_QUEUE_LIMIT = max(AI_WORKER_CONCURRENCY, int(os.getenv("AI_WORKER_QUEUE_LIMIT", "16")))
AI_WORKER_QUEUE_WAIT_MS = max(0, int(os.getenv("AI_WORKER_QUEUE_WAIT_MS", "25")))
MODEL_INIT_RETRIES = max(1, int(os.getenv("MODEL_INIT_RETRIES", "3")))
MODEL_INIT_BACKOFF_SECONDS = max(0.1, float(os.getenv("MODEL_INIT_BACKOFF_SECONDS", "1.0")))
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
FACE_DET_SIZE = (
    int(os.getenv("FACE_DET_WIDTH", "640")),
    int(os.getenv("FACE_DET_HEIGHT", "640")),
)

app = FastAPI(title="Attendance Liveness AI Worker", version="1.1.0")

_executor: Optional[ThreadPoolExecutor] = None
_work_slots: Optional[asyncio.Semaphore] = None
_inflight = 0
_contexts: list["WorkerContext"] = []
_context_index = 0
_context_lock = threading.Lock()
_face_store: dict[str, np.ndarray] = {}
_face_store_lock = threading.Lock()
_model_message = "model not loaded"


class AnalyzeResponse(BaseModel):
    faceFound: bool
    faceCount: int
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    blink: bool = False
    eyeAspectRatio: float = 0.0
    livenessScore: float = 0.0
    spoofScore: float = 1.0
    qualityScore: float = 0.0
    modelLoaded: bool = False
    faceEnrolled: bool = False
    faceMatched: bool = False
    faceMatchScore: float = 0.0
    message: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class FaceEnrollResponse(BaseModel):
    ok: bool
    employeeId: str
    enrolled: bool
    faceCount: int
    embeddingSize: int = 0
    message: str = ""


class FaceVerifyResponse(BaseModel):
    ok: bool
    employeeId: str
    enrolled: bool
    faceFound: bool
    faceCount: int
    matched: bool
    score: float
    threshold: float
    message: str = ""


class WorkerContext:
    def __init__(self, index: int, model_path: Optional[Path]):
        self.index = index
        self.face_mesh = self._create_face_mesh()
        self.onnx_session = self._create_onnx_session(model_path)
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name if self.onnx_session is not None else None
        self.face_app = self._create_face_app()

    @property
    def onnx_loaded(self) -> bool:
        return self.onnx_session is not None and self.onnx_input_name is not None

    @property
    def face_model_loaded(self) -> bool:
        return self.face_app is not None

    def _create_face_mesh(self):
        if mp is None:
            logger.warning("MediaPipe not loaded in worker context %s", self.index)
            return None
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.55,
        )

    def _create_onnx_session(self, model_path: Optional[Path]):
        if model_path is None or ort is None:
            return None
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def _create_face_app(self):
        if FaceAnalysis is None:
            return None
        face_app = FaceAnalysis(name=FACE_MODEL_NAME, providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=FACE_DET_SIZE)
        return face_app


@app.on_event("startup")
def startup() -> None:
    global _executor, _work_slots, _contexts, _model_message

    _executor = ThreadPoolExecutor(max_workers=AI_WORKER_CONCURRENCY, thread_name_prefix="ai-worker")
    _work_slots = asyncio.Semaphore(AI_WORKER_QUEUE_LIMIT)

    model_path = retry("find/download ONNX model", find_or_download_onnx_model)
    contexts: list[WorkerContext] = []
    for index in range(AI_WORKER_CONCURRENCY):
        context = retry(f"initialize worker context {index}", lambda i=index: WorkerContext(i, model_path))
        contexts.append(context)

    _contexts = contexts
    loaded = sum(1 for context in _contexts if context.onnx_loaded)
    face_loaded = sum(1 for context in _contexts if context.face_model_loaded)
    _model_message = f"contexts={len(_contexts)}, onnxLoaded={loaded}, faceModelLoaded={face_loaded}"
    logger.info("AI worker ready: %s", _model_message)


@app.on_event("shutdown")
def shutdown() -> None:
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
def health():
    return {"ok": True, "service": "Attendance Liveness AI Worker"}


@app.get("/ready")
def ready():
    onnx_loaded = sum(1 for context in _contexts if context.onnx_loaded)
    face_loaded = sum(1 for context in _contexts if context.face_model_loaded)
    return {
        "ok": bool(_contexts),
        "modelMessage": _model_message,
        "contexts": len(_contexts),
        "onnxLoaded": onnx_loaded,
        "faceModelLoaded": face_loaded,
        "queueLimit": AI_WORKER_QUEUE_LIMIT,
        "inflight": _inflight,
        "enrolledFaces": enrolled_face_count(),
        "hfRepoId": HF_REPO_ID,
        "modelDir": str(MODEL_DIR),
        "faceMatchThreshold": FACE_MATCH_THRESHOLD,
    }


@app.post("/face/enroll", response_model=FaceEnrollResponse)
async def face_enroll(employeeId: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    return await run_bounded(lambda: enroll_face(raw, employeeId))


@app.post("/face/verify", response_model=FaceVerifyResponse)
async def face_verify(employeeId: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    return await run_bounded(lambda: verify_face(raw, employeeId))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    sessionId: str = Form(""),
    seq: str = Form(""),
    employeeId: str = Form(""),
):
    raw = await file.read()
    return await run_bounded(lambda: analyze_frame(raw, employeeId))


async def run_bounded(fn):
    global _inflight

    if _executor is None or _work_slots is None:
        raise HTTPException(status_code=503, detail="AI worker is not ready")

    try:
        await asyncio.wait_for(_work_slots.acquire(), timeout=AI_WORKER_QUEUE_WAIT_MS / 1000)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="AI worker queue is full")

    _inflight += 1
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, fn)
    finally:
        _inflight -= 1
        _work_slots.release()


def retry(name: str, fn):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MODEL_INIT_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning("%s failed on attempt %s/%s: %s", name, attempt, MODEL_INIT_RETRIES, exc)
            if attempt < MODEL_INIT_RETRIES:
                time.sleep(MODEL_INIT_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"{name} failed after {MODEL_INIT_RETRIES} attempts") from last_exc


def next_context() -> WorkerContext:
    global _context_index
    if not _contexts:
        raise HTTPException(status_code=503, detail="AI worker is not ready")
    with _context_lock:
        context = _contexts[_context_index % len(_contexts)]
        _context_index += 1
        return context


def find_or_download_onnx_model() -> Optional[Path]:
    local_files = list(MODEL_DIR.glob("*.onnx"))
    if local_files:
        return local_files[0]

    if hf_hub_download is None:
        return None

    filename = HF_MODEL_FILE.strip()
    try:
        if not filename and list_repo_files is not None:
            files = list_repo_files(HF_REPO_ID)
            onnx_files = [x for x in files if x.lower().endswith(".onnx")]
            if onnx_files:
                filename = onnx_files[0]

        if not filename:
            filename = "model.onnx"

        logger.info("Downloading model from Hugging Face: %s/%s", HF_REPO_ID, filename)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)
    except Exception as exc:
        logger.warning("Could not download Hugging Face model: %s", exc)
        return None


def enroll_face(raw: bytes, employee_id: str) -> FaceEnrollResponse:
    employee_id = require_employee_id(employee_id)
    context = next_context()
    embedding_result = extract_face_embedding(raw, context)
    if not embedding_result["ok"]:
        return FaceEnrollResponse(
            ok=False,
            employeeId=employee_id,
            enrolled=False,
            faceCount=int(embedding_result["face_count"]),
            message=str(embedding_result["message"]),
        )

    embedding = embedding_result["embedding"]
    with _face_store_lock:
        _face_store[employee_id] = embedding

    return FaceEnrollResponse(
        ok=True,
        employeeId=employee_id,
        enrolled=True,
        faceCount=1,
        embeddingSize=int(embedding.shape[0]),
        message="face enrolled",
    )


def verify_face(raw: bytes, employee_id: str) -> FaceVerifyResponse:
    employee_id = require_employee_id(employee_id)
    enrolled, stored_embedding = get_stored_embedding(employee_id)
    if not enrolled:
        return FaceVerifyResponse(
            ok=False,
            employeeId=employee_id,
            enrolled=False,
            faceFound=False,
            faceCount=0,
            matched=False,
            score=0.0,
            threshold=FACE_MATCH_THRESHOLD,
            message="face not enrolled",
        )

    context = next_context()
    embedding_result = extract_face_embedding(raw, context)
    if not embedding_result["ok"]:
        return FaceVerifyResponse(
            ok=False,
            employeeId=employee_id,
            enrolled=True,
            faceFound=bool(embedding_result["face_count"]),
            faceCount=int(embedding_result["face_count"]),
            matched=False,
            score=0.0,
            threshold=FACE_MATCH_THRESHOLD,
            message=str(embedding_result["message"]),
        )

    score = cosine_similarity(stored_embedding, embedding_result["embedding"])
    matched = score >= FACE_MATCH_THRESHOLD
    return FaceVerifyResponse(
        ok=True,
        employeeId=employee_id,
        enrolled=True,
        faceFound=True,
        faceCount=1,
        matched=matched,
        score=score,
        threshold=FACE_MATCH_THRESHOLD,
        message="face matched" if matched else "face mismatch",
    )


def analyze_frame(raw: bytes, employee_id: str) -> AnalyzeResponse:
    start = time.perf_counter()
    context = next_context()
    image = decode_jpeg(raw)
    if image is None:
        return AnalyzeResponse(
            faceFound=False,
            faceCount=0,
            qualityScore=0,
            message="Cannot decode JPEG",
            modelLoaded=context.onnx_loaded,
        )

    brightness, blur_score, quality = image_quality(image)
    face = detect_face_mediapipe(image, context)
    if face is None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return AnalyzeResponse(
            faceFound=False,
            faceCount=0,
            qualityScore=quality,
            livenessScore=0.0,
            spoofScore=1.0,
            modelLoaded=context.onnx_loaded,
            message="No face detected",
            metrics={"brightness": brightness, "blur": blur_score, "elapsedMs": elapsed_ms},
        )

    bbox, landmarks, face_count = face
    yaw, pitch, roll = estimate_head_pose(image, landmarks)
    ear = compute_eye_aspect_ratio(image, landmarks)
    blink = ear > 0 and ear < BLINK_EAR_THRESHOLD

    liveness_score, spoof_score, model_msg, spoof_metrics = run_antispoof(image, bbox, quality, context)
    face_enrolled, face_match_score, face_matched, face_message = analyze_face_match(raw, employee_id, context)
    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics = {
        "brightness": float(brightness),
        "blur": float(blur_score),
        "elapsedMs": float(elapsed_ms),
        "bboxX": float(bbox[0]),
        "bboxY": float(bbox[1]),
        "bboxW": float(bbox[2]),
        "bboxH": float(bbox[3]),
        "faceMatchThreshold": float(FACE_MATCH_THRESHOLD),
    }
    metrics.update(spoof_metrics)

    message = model_msg
    if employee_id.strip():
        message = f"{model_msg}; {face_message}"

    return AnalyzeResponse(
        faceFound=True,
        faceCount=face_count,
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        blink=bool(blink),
        eyeAspectRatio=float(ear),
        livenessScore=float(liveness_score),
        spoofScore=float(spoof_score),
        qualityScore=float(quality),
        modelLoaded=context.onnx_loaded,
        faceEnrolled=face_enrolled,
        faceMatched=face_matched,
        faceMatchScore=float(face_match_score),
        message=message,
        metrics=metrics,
    )


def analyze_face_match(raw: bytes, employee_id: str, context: WorkerContext) -> tuple[bool, float, bool, str]:
    employee_id = employee_id.strip()
    if not employee_id:
        return False, 0.0, False, "employeeId missing"

    enrolled, stored_embedding = get_stored_embedding(employee_id)
    if not enrolled:
        return False, 0.0, False, "face not enrolled"

    embedding_result = extract_face_embedding(raw, context)
    if not embedding_result["ok"]:
        return True, 0.0, False, str(embedding_result["message"])

    score = cosine_similarity(stored_embedding, embedding_result["embedding"])
    return True, score, score >= FACE_MATCH_THRESHOLD, "face matched" if score >= FACE_MATCH_THRESHOLD else "face mismatch"


def extract_face_embedding(raw: bytes, context: WorkerContext) -> dict:
    image = decode_jpeg(raw)
    if image is None:
        return {"ok": False, "face_count": 0, "message": "Cannot decode JPEG"}
    if context.face_app is None:
        return {"ok": False, "face_count": 0, "message": "face model not loaded"}

    faces = context.face_app.get(image)
    face_count = len(faces)
    if face_count != 1:
        return {"ok": False, "face_count": face_count, "message": "Expected exactly one face"}

    embedding = np.asarray(faces[0].embedding, dtype=np.float32)
    embedding = normalize_embedding(embedding)
    return {"ok": True, "face_count": 1, "embedding": embedding, "message": "ok"}


def get_stored_embedding(employee_id: str) -> tuple[bool, np.ndarray]:
    with _face_store_lock:
        embedding = _face_store.get(employee_id)
        if embedding is None:
            return False, np.zeros((0,), dtype=np.float32)
        return True, embedding.copy()


def enrolled_face_count() -> int:
    with _face_store_lock:
        return len(_face_store)


def require_employee_id(employee_id: str) -> str:
    employee_id = employee_id.strip()
    if not employee_id:
        raise HTTPException(status_code=400, detail="employeeId is required")
    return employee_id


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-9:
        return embedding
    return embedding / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))


def decode_jpeg(raw: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def image_quality(image: np.ndarray) -> Tuple[float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(lap_var / 220.0, 1.0)
    light_score = 1.0 - min(abs(brightness - 0.52) / 0.52, 1.0)
    quality = max(0.0, min(1.0, 0.45 * light_score + 0.55 * blur_score))
    return brightness, blur_score, quality


def detect_face_mediapipe(image_bgr: np.ndarray, context: WorkerContext):
    if context.face_mesh is None:
        return None

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = context.face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    face_count = len(result.multi_face_landmarks)
    face_landmarks = result.multi_face_landmarks[0].landmark

    xs = np.array([lm.x for lm in face_landmarks], dtype=np.float32)
    ys = np.array([lm.y for lm in face_landmarks], dtype=np.float32)
    x1 = int(max(xs.min() * w, 0))
    y1 = int(max(ys.min() * h, 0))
    x2 = int(min(xs.max() * w, w - 1))
    y2 = int(min(ys.max() * h, h - 1))
    bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    return bbox, face_landmarks, face_count


def lm_point(image: np.ndarray, landmarks, idx: int) -> Tuple[float, float]:
    h, w = image.shape[:2]
    lm = landmarks[idx]
    return float(lm.x * w), float(lm.y * h)


def estimate_head_pose(image: np.ndarray, landmarks) -> Tuple[float, float, float]:
    h, w = image.shape[:2]

    image_points = np.array([
        lm_point(image, landmarks, 1),
        lm_point(image, landmarks, 152),
        lm_point(image, landmarks, 33),
        lm_point(image, landmarks, 263),
        lm_point(image, landmarks, 61),
        lm_point(image, landmarks, 291),
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rot_mat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)

    pitch = normalize_pitch_angle(float(euler_angles[0][0]))
    yaw = float(euler_angles[1][0])
    roll = float(euler_angles[2][0])
    return yaw, pitch, roll


def normalize_pitch_angle(pitch: float) -> float:
    if pitch > 90.0:
        return pitch - 180.0
    if pitch < -90.0:
        return pitch + 180.0
    return pitch


def compute_eye_aspect_ratio(image: np.ndarray, landmarks) -> float:
    def dist(a, b):
        ax, ay = lm_point(image, landmarks, a)
        bx, by = lm_point(image, landmarks, b)
        return math.hypot(ax - bx, ay - by)

    left = (33, 160, 158, 133, 153, 144)
    right = (362, 385, 387, 263, 373, 380)

    def ear(points):
        p1, p2, p3, p4, p5, p6 = points
        return (dist(p2, p6) + dist(p3, p5)) / max(2.0 * dist(p1, p4), 1e-6)

    return float((ear(left) + ear(right)) / 2.0)


def run_antispoof(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    quality: float,
    context: WorkerContext,
) -> Tuple[float, float, str, dict[str, float]]:
    if not context.onnx_loaded:
        heuristic_live = float(max(0.05, min(0.75, 0.35 + 0.55 * quality)))
        heuristic_spoof = 1.0 - heuristic_live
        return (
            heuristic_live,
            heuristic_spoof,
            "heuristic fallback; ONNX model not loaded",
            anti_spoof_metrics(heuristic_live, heuristic_spoof, 0.0),
        )

    try:
        crop = crop_with_scale(image_bgr, bbox, scale=2.7, size=(80, 80))
        x = crop.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]
        outputs = context.onnx_session.run(None, {context.onnx_input_name: x})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        probs = softmax(logits)

        if probs.shape[0] >= 3:
            live = float(probs[0])
            print_attack = float(probs[1])
            replay_attack = float(probs[2])
            spoof = float(print_attack + replay_attack)
        elif probs.shape[0] == 2:
            live = float(probs[0])
            print_attack = float(probs[1])
            replay_attack = 0.0
            spoof = print_attack
        else:
            live = float(probs[0])
            print_attack = 1.0 - live
            replay_attack = 0.0
            spoof = print_attack

        return live, spoof, "onnx minifasnet", anti_spoof_metrics(live, print_attack, replay_attack)
    except Exception as exc:
        logger.warning("Anti-spoof inference failed: %s", exc)
        heuristic_live = float(max(0.05, min(0.65, 0.30 + 0.50 * quality)))
        heuristic_spoof = 1.0 - heuristic_live
        return (
            heuristic_live,
            heuristic_spoof,
            f"onnx failed; heuristic fallback: {exc}",
            anti_spoof_metrics(heuristic_live, heuristic_spoof, 0.0),
        )


def anti_spoof_metrics(live: float, print_attack: float, replay_attack: float) -> dict[str, float]:
    return {
        "antiSpoofLiveProbability": float(live),
        "antiSpoofPrintProbability": float(print_attack),
        "antiSpoofReplayProbability": float(replay_attack),
    }


def crop_with_scale(image: np.ndarray, bbox: Tuple[int, int, int, int], scale: float, size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    cx = x + bw / 2
    cy = y + bh / 2
    side = max(bw, bh) * scale
    x1 = int(max(cx - side / 2, 0))
    y1 = int(max(cy - side / 2, 0))
    x2 = int(min(cx + side / 2, w - 1))
    y2 = int(min(cy + side / 2, h - 1))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image
    return cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False, workers=1)
