"""Общие вещи: словарь, извлечение признаков, подготовка последовательности, отрисовка."""
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

PROJECT = Path(__file__).resolve().parents[2]      # .../vibehack
SLOVO_DIR = PROJECT / "slovo"
KP_DIR = PROJECT / "data" / "slovo_kp"              # сюда extract_slovo.py кладёт точки
MODEL_PATH = PROJECT / "models" / "sign_gru.pt"

# id класса (латиница — для папок и надписей OpenCV, он не рисует кириллицу)
#   -> (метка в slovo/annotations.csv, русский текст для вывода)
# Метки должны совпадать с колонкой text символ в символ (регистр, «!», «?»).
# В Slovo нет «спасибо», «нет», «пожалуйста», «как» — поэтому их тут нет.
WORDS = {
    "_none": ("no_event", ""),
    "privet": ("Привет!", "привет"),
    "poka": ("Пока", "пока"),
    "dobro_pozhalovat": ("Добро пожаловать!", "добро пожаловать"),
    "kto": ("кто?", "кто"),
    "chto": ("что", "что"),
    "da": ("да", "да"),
    "ya": ("я", "я"),
    "ty": ("ты", "ты"),
    "my": ("мы", "мы"),
    "vy": ("вы", "вы"),
    "menya": ("меня", "меня"),
    "tebya": ("тебя", "тебя"),
    "imya": ("имя", "имя"),
    "horosho": ("хорошо", "хорошо"),
    "ploho": ("Плохо", "плохо"),
    "drug": ("друг", "друг"),
    "hotet": ("хотеть", "хотеть"),
    "znat": ("знать", "знать"),
    "ponyat": ("понять", "понять"),
    "pomoch": ("помочь", "помочь"),
    "rad": ("рад", "рад"),
    "ochen": ("очень", "очень"),
    "tozhe": ("тоже", "тоже"),
}
NONE = "_none"

WINDOW_SEC = 1.6   # окно в реальном времени ≈ средняя длина клипа в Slovo (~50 кадров)
SEQ_LEN = 32       # до скольких кадров пережимаем клип перед подачей в модель

# --- какие точки берём ---
POSE_IDX = list(range(0, 25))  # верх тела без ног
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
FACE_IDX = LIPS + BROWS

# раскладка вектора признаков одного кадра (всего 425 чисел)
SL_POSE = slice(0, 75)
SL_LH = slice(75, 138)
SL_RH = slice(138, 201)
SL_LH_LOCAL = slice(201, 264)
SL_RH_LOCAL = slice(264, 327)
SL_FACE = slice(327, 423)
N_XYZ = 423            # первые 423 числа — это 141 точка по (x, y, z)
# те же блоки в номерах точек (для отражения)
PT_LH, PT_RH, PT_LH_LOCAL, PT_RH_LOCAL = slice(25, 46), slice(46, 67), slice(67, 88), slice(88, 109)
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
              (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]  # левая/правая точка позы
FLAG_LH, FLAG_RH = 423, 424
FEAT_DIM = 425


def lm_to_np(lms, idx=None, n=21):
    if lms is None:
        k = len(idx) if idx else n
        return np.zeros((k, 3), np.float32), 0.0
    pts = np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32)
    return (pts[idx] if idx else pts), 1.0


def frame_features(res):
    """Результат MediaPipe Holistic -> вектор (425,)."""
    if res.pose_landmarks is None:          # человека нет — пустой кадр
        return np.zeros(FEAT_DIM, np.float32)

    pose, _ = lm_to_np(res.pose_landmarks, POSE_IDX)
    lh, lh_ok = lm_to_np(res.left_hand_landmarks)
    rh, rh_ok = lm_to_np(res.right_hand_landmarks)
    face, face_ok = lm_to_np(res.face_landmarks, FACE_IDX)

    ls, rs = pose[11], pose[12]
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls[:2] - rs[:2]) + 1e-6

    def norm(a, ok=1.0):
        return ((a - center) / scale) * ok

    def local(h, ok):
        return (h - h[0]) / (np.linalg.norm(h[9] - h[0]) + 1e-6) * ok

    return np.concatenate([
        norm(pose).ravel(),
        norm(lh, lh_ok).ravel(), norm(rh, rh_ok).ravel(),
        local(lh, lh_ok).ravel(), local(rh, rh_ok).ravel(),
        norm(face, face_ok).ravel(),
        [lh_ok, rh_ok],
    ]).astype(np.float32)


def resample(seq, T=SEQ_LEN):
    """(N, D) -> (T, D) линейной интерполяцией по времени."""
    n = len(seq)
    if n == 1:
        return np.repeat(seq, T, axis=0)
    src = np.linspace(0, n - 1, T)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    return seq[lo] * (1 - w) + seq[hi] * w


def make_input(seq):
    """Сырой клип (N, 425) -> вход модели (SEQ_LEN, 850): координаты + скорости."""
    x = resample(np.asarray(seq, np.float32))
    vel = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x, vel], axis=1).astype(np.float32)


# --- камера и отрисовка ---
def create_holistic():
    return mp_holistic.Holistic(model_complexity=1, smooth_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)


def open_camera(index=0, width=1280, height=720):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть камеру {index}. Проверь: ls /dev/video*")
    return cap


def process(holistic, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return holistic.process(rgb)


def draw_landmarks(frame, res):
    if res.face_landmarks:
        mp_draw.draw_landmarks(frame, res.face_landmarks, mp_holistic.FACEMESH_CONTOURS, None,
                               mp_styles.get_default_face_mesh_contours_style())
    if res.pose_landmarks:
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_styles.get_default_pose_landmarks_style())
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            mp_draw.draw_landmarks(frame, hand, mp_holistic.HAND_CONNECTIONS,
                                   mp_styles.get_default_hand_landmarks_style(),
                                   mp_styles.get_default_hand_connections_style())


def put_lines(frame, lines, color=(0, 255, 0), x=10, y0=30, scale=0.7):
    for i, text in enumerate(lines):
        y = y0 + i * int(32 * scale / 0.7)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
