"""Проверка камеры и точек Holistic с периодическим статусом."""
import argparse
import time

from console import LiveStatus, add_log_arguments, configure_console

if __name__ == "__main__":
    configure_console()
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CAM_INDEX = 0          # если камера не открылась, попробуй 1, 2...
WIDTH, HEIGHT = 1280, 720


def draw_landmarks(frame, res):
    # лицо: только контуры (468 точек целиком — каша)
    if res.face_landmarks:
        mp_draw.draw_landmarks(
            frame, res.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )
    # поза
    if res.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )
    # руки
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            mp_draw.draw_landmarks(
                frame, hand, mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


def put_status(frame, fps, res):
    def mark(ok): return "OK" if ok else "--"
    lines = [
        f"FPS: {fps:.1f}",
        f"Pose: {mark(res.pose_landmarks)}  Face: {mark(res.face_landmarks)}",
        f"Left hand: {mark(res.left_hand_landmarks)}  Right hand: {mark(res.right_hand_landmarks)}",
        "q / Esc - exit",
    ]
    for i, text in enumerate(lines):
        y = 30 + i * 30
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_log_arguments(ap)
    args = ap.parse_args()
    print(f"Открытие камеры {CAM_INDEX}…", flush=True)
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Не удалось открыть камеру {CAM_INDEX}. Проверь: ls /dev/video*")

    print(f"Камера открыта: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. Инициализация Holistic…", flush=True)
    print("Выход: q / Esc или Ctrl+C.", flush=True)
    live = LiveStatus(args.log_interval)
    reason = "выход пользователя"
    prev = time.time()
    fps = 0.0

    try:
        with mp_holistic.Holistic(
            model_complexity=1,           # 0 — быстрее, 2 — точнее
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            print("Holistic готов. Обработка кадров…", flush=True)
            while True:
                ok, frame = cap.read()
                if not ok:
                    reason = "камера перестала выдавать кадры"
                    print("ОШИБКА: кадр с камеры не получен.", flush=True)
                    break

                # MediaPipe ждёт RGB; writeable=False ускоряет обработку
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = holistic.process(rgb)

                draw_landmarks(frame, res)

                # зеркалим ПОСЛЕ обработки, чтобы MediaPipe не путал левую и правую руку
                frame = cv2.flip(frame, 1)

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))  # сглаженный FPS
                prev = now
                put_status(frame, fps, res)
                def marks(value):
                    return "найдена" if value else "нет"
                live.tick(f"поза: {marks(res.pose_landmarks)} | "
                          f"левая рука: {marks(res.left_hand_landmarks)} | "
                          f"правая рука: {marks(res.right_hand_landmarks)}")

                cv2.imshow("RSL Translator - landmarks", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        reason = "остановлено пользователем"
    except BaseException:
        reason = "ошибка обработки"
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        live.finish(reason)



if __name__ == "__main__":
    main()