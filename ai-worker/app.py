import os
import time
import math
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

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

# Threshold thấp một chút cho demo; production cần calibrate theo camera thật.
BLINK_EAR_THRESHOLD = float(os.getenv("BLINK_EAR_THRESHOLD", "0.20"))

app = FastAPI(title="Attendance Liveness AI Worker", version="1.0.0")

_face_mesh = None
_onnx_session = None
_onnx_input_name = None
_model_loaded = False
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
    message: str = ""
    metrics: dict[str, float] = {}


@app.on_event("startup")
def startup() -> None:
    global _face_mesh, _onnx_session, _onnx_input_name, _model_loaded, _model_message

    if mp is not None:
        # static_image_mode=True: request độc lập, không giữ tracking state; ổn cho WebSocket frames lẻ.
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.55,
        )
        logger.info("MediaPipe FaceMesh loaded")
    else:
        logger.warning("MediaPipe not loaded; fallback will only provide basic quality metrics")

    model_path = find_or_download_onnx_model()
    if model_path and ort is not None:
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _onnx_session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
            _onnx_input_name = _onnx_session.get_inputs()[0].name
            _model_loaded = True
            _model_message = f"loaded {model_path}"
            logger.info("ONNX model loaded: %s", model_path)
        except Exception as exc:
            _model_loaded = False
            _model_message = f"failed to load model: {exc}"
            logger.exception("Failed to load ONNX model")
    else:
        _model_loaded = False
        _model_message = "ONNX model not found; using heuristic fallback"
        logger.warning(_model_message)


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
            # Common fallback. If this fails, set HF_MODEL_FILE in .env/run script.
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


@app.get("/health")
def health():
    return {
        "ok": True,
        "modelLoaded": _model_loaded,
        "modelMessage": _model_message,
        "hfRepoId": HF_REPO_ID,
        "modelDir": str(MODEL_DIR),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    sessionId: str = Form(""),
    seq: str = Form(""),
):
    start = time.perf_counter()
    raw = await file.read()
    image = decode_jpeg(raw)
    if image is None:
        return AnalyzeResponse(
            faceFound=False,
            faceCount=0,
            qualityScore=0,
            message="Cannot decode JPEG",
            modelLoaded=_model_loaded,
        )

    brightness, blur_score, quality = image_quality(image)

    face = detect_face_mediapipe(image)
    if face is None:
        # Fallback no-face response, still reports image quality.
        elapsed_ms = (time.perf_counter() - start) * 1000
        return AnalyzeResponse(
            faceFound=False,
            faceCount=0,
            qualityScore=quality,
            livenessScore=0.0,
            spoofScore=1.0,
            modelLoaded=_model_loaded,
            message="No face detected",
            metrics={"brightness": brightness, "blur": blur_score, "elapsedMs": elapsed_ms},
        )

    bbox, landmarks, face_count = face
    yaw, pitch, roll = estimate_head_pose(image, landmarks)
    ear = compute_eye_aspect_ratio(image, landmarks)
    blink = ear > 0 and ear < BLINK_EAR_THRESHOLD

    liveness_score, spoof_score, model_msg, spoof_metrics = run_antispoof(image, bbox, quality)
    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics = {
        "brightness": float(brightness),
        "blur": float(blur_score),
        "elapsedMs": float(elapsed_ms),
        "bboxX": float(bbox[0]),
        "bboxY": float(bbox[1]),
        "bboxW": float(bbox[2]),
        "bboxH": float(bbox[3]),
    }
    metrics.update(spoof_metrics)

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
        modelLoaded=_model_loaded,
        message=model_msg,
        metrics=metrics,
    )


def decode_jpeg(raw: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def image_quality(image: np.ndarray) -> Tuple[float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(lap_var / 220.0, 1.0)

    # Penalize too dark/too bright and too blurry.
    light_score = 1.0 - min(abs(brightness - 0.52) / 0.52, 1.0)
    quality = max(0.0, min(1.0, 0.45 * light_score + 0.55 * blur_score))
    return brightness, blur_score, quality


def detect_face_mediapipe(image_bgr: np.ndarray):
    if _face_mesh is None:
        return None

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)

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

    # 2D image points from MediaPipe face mesh.
    image_points = np.array([
        lm_point(image, landmarks, 1),    # nose tip
        lm_point(image, landmarks, 152),  # chin
        lm_point(image, landmarks, 33),   # left eye outer corner
        lm_point(image, landmarks, 263),  # right eye outer corner
        lm_point(image, landmarks, 61),   # mouth left
        lm_point(image, landmarks, 291),  # mouth right
    ], dtype=np.float64)

    # Generic 3D face model points, enough for approximate yaw/pitch demo.
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

    # Because browser preview is mirrored, client UX may feel reversed.
    # Backend uses the raw frame as drawn on canvas; keep sign consistent with UI tests.
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

    # MediaPipe face mesh standard EAR points.
    left = (33, 160, 158, 133, 153, 144)
    right = (362, 385, 387, 263, 373, 380)

    def ear(points):
        p1, p2, p3, p4, p5, p6 = points
        return (dist(p2, p6) + dist(p3, p5)) / max(2.0 * dist(p1, p4), 1e-6)

    return float((ear(left) + ear(right)) / 2.0)


def run_antispoof(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], quality: float) -> Tuple[float, float, str, dict[str, float]]:
    if not _model_loaded or _onnx_session is None:
        # Heuristic fallback keeps active challenge demo usable without internet/model.
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
        x = crop.astype(np.float32) / 255.0  # BGR range [0, 1]
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]
        outputs = _onnx_session.run(None, {_onnx_input_name: x})
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
