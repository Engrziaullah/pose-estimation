"""
pose_utils.py
-----------------------------------------------------------------------------
Wraps MediaPipe Pose for three use cases used by the Flask app:
  1. Single image  -> annotated JPEG + landmark table
  2. Video file     -> annotated MP4 (frame-by-frame tracking)
  3. Live webcam    -> MJPEG generator for <img> streaming

Kept separate from app.py so the Flask layer stays focused on routing.
-----------------------------------------------------------------------------
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Colors for the skeleton overlay (BGR order, since OpenCV draws in BGR).
LINE_COLOR = (40, 40, 230)    # thin bone lines — red
DOT_COLOR = (255, 255, 255)   # joints — small white points

MIN_LANDMARK_VISIBILITY = 0.5  # skip low-confidence points instead of drawing noisy guesses


def build_specs(frame_width, thin=False):
    """
    Scale dot radius / line thickness to the actual frame width so the
    skeleton always reads as clean, thin marks — never a blob, never so
    thin that JPEG compression eats it. Returns (radius, line_thickness).

    thin=True gives a slimmer line weight for the live camera feed, where
    a lighter line reads better against a moving, often-cluttered scene.
    """
    radius = max(2, min(4, round(frame_width / 220)))
    if thin:
        line_thickness = max(1, min(2, round(frame_width / 480)))
    else:
        line_thickness = max(2, min(3, round(frame_width / 300)))
    return radius, line_thickness


def draw_skeleton(frame_bgr, pose_landmarks, radius, line_thickness,
                   line_color=LINE_COLOR, dot_color=DOT_COLOR):
    """
    Draw a thin, anti-aliased skeleton directly with OpenCV instead of
    mediapipe's default draw_landmarks (which doesn't anti-alias and drew
    much thicker markers) — this is what actually removes the jagged/blurry
    look, on top of sizing everything relative to the frame.

    Joints render as small solid-white points with a thin ring in the line
    color, so they stay visible against light backgrounds/clothing instead
    of disappearing.
    """
    h, w = frame_bgr.shape[:2]
    points = {}
    for idx, lm in enumerate(pose_landmarks.landmark):
        if lm.visibility is not None and lm.visibility < MIN_LANDMARK_VISIBILITY:
            continue
        points[idx] = (int(lm.x * w), int(lm.y * h))

    for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(frame_bgr, points[start_idx], points[end_idx], line_color,
                      line_thickness, lineType=cv2.LINE_AA)

    for pt in points.values():
        cv2.circle(frame_bgr, pt, radius, dot_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame_bgr, pt, radius, line_color, thickness=1, lineType=cv2.LINE_AA)

    return frame_bgr


def calculate_angle(a, b, c):
    """Angle (degrees) at point b, formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def landmarks_to_dataframe(pose_landmarks):
    """Turn a MediaPipe pose_landmarks object into a tidy DataFrame."""
    rows = []
    for idx, lm in enumerate(pose_landmarks.landmark):
        rows.append({
            "id": idx,
            "landmark": mp_pose.PoseLandmark(idx).name,
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4),
            "visibility": round(lm.visibility, 4),
        })
    return pd.DataFrame(rows)


def key_joint_angles(pose_landmarks):
    """Compute a handful of readable joint angles for the summary panel."""
    lm = pose_landmarks.landmark

    def pt(landmark):
        p = lm[landmark.value]
        return [p.x, p.y]

    angles = {}
    try:
        angles["Right Elbow"] = calculate_angle(
            pt(mp_pose.PoseLandmark.RIGHT_SHOULDER), pt(mp_pose.PoseLandmark.RIGHT_ELBOW), pt(mp_pose.PoseLandmark.RIGHT_WRIST)
        )
        angles["Left Elbow"] = calculate_angle(
            pt(mp_pose.PoseLandmark.LEFT_SHOULDER), pt(mp_pose.PoseLandmark.LEFT_ELBOW), pt(mp_pose.PoseLandmark.LEFT_WRIST)
        )
        angles["Right Knee"] = calculate_angle(
            pt(mp_pose.PoseLandmark.RIGHT_HIP), pt(mp_pose.PoseLandmark.RIGHT_KNEE), pt(mp_pose.PoseLandmark.RIGHT_ANKLE)
        )
        angles["Left Knee"] = calculate_angle(
            pt(mp_pose.PoseLandmark.LEFT_HIP), pt(mp_pose.PoseLandmark.LEFT_KNEE), pt(mp_pose.PoseLandmark.LEFT_ANKLE)
        )
        angles["Right Hip"] = calculate_angle(
            pt(mp_pose.PoseLandmark.RIGHT_SHOULDER), pt(mp_pose.PoseLandmark.RIGHT_HIP), pt(mp_pose.PoseLandmark.RIGHT_KNEE)
        )
        angles["Left Hip"] = calculate_angle(
            pt(mp_pose.PoseLandmark.LEFT_SHOULDER), pt(mp_pose.PoseLandmark.LEFT_HIP), pt(mp_pose.PoseLandmark.LEFT_KNEE)
        )
    except Exception:
        pass
    return {k: round(v, 1) for k, v in angles.items()}


def process_image(input_path, output_path, model_complexity=1, min_detection_confidence=0.5):
    """
    Run pose estimation on a single image.
    Returns (success, num_people_detected, angles_dict, csv_path or None)
    """
    image = cv2.imread(input_path)
    if image is None:
        return False, 0, {}, None

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
    ) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            cv2.imwrite(output_path, image)
            return True, 0, {}, None

        radius, line_thickness = build_specs(image.shape[1])
        draw_skeleton(image, results.pose_landmarks, radius, line_thickness)
        cv2.imwrite(output_path, image)

        df = landmarks_to_dataframe(results.pose_landmarks)
        csv_path = output_path.rsplit(".", 1)[0] + "_landmarks.csv"
        df.to_csv(csv_path, index=False)

        angles = key_joint_angles(results.pose_landmarks)
        return True, 1, angles, csv_path


def process_video(input_path, output_path, model_complexity=1,
                   min_detection_confidence=0.5, min_tracking_confidence=0.5,
                   progress_callback=None, max_width=None):
    """
    Run pose estimation on every frame of a video file and write an annotated copy.
    progress_callback(current_frame, total_frames) is called periodically if provided.
    max_width: if set and the source is wider than this, frames are downscaled
    before inference/writing — the single biggest lever for speeding up long clips.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = 1.0
    if max_width and src_width > max_width:
        scale = max_width / float(src_width)
    width = int(src_width * scale)
    height = int(src_height * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    radius, line_thickness = build_specs(width)

    frames_with_detection = 0
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                frames_with_detection += 1
                draw_skeleton(frame, results.pose_landmarks, radius, line_thickness)

            writer.write(frame)
            frame_idx += 1
            if progress_callback and total_frames:
                progress_callback(frame_idx, total_frames)

    cap.release()
    writer.release()
    return True, frames_with_detection, total_frames


class LiveFeedGenerator:
    """
    Wraps a webcam capture + MediaPipe Pose instance for MJPEG streaming.
    Keeps the pose object alive across frames for temporal tracking.
    """

    def __init__(self, camera_index=0, model_complexity=1):
        self.camera_index = camera_index
        self.cap = None
        self._specs = None  # (radius, line_thickness), sized once we know the real frame width
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _ensure_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            # Ask for a real resolution — most webcams default to something
            # low (often 640x480 or smaller) unless told otherwise, which is
            # exactly what made fixed-size landmark dots look like a blob.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def frames(self):
        self._ensure_camera()
        if not self.cap.isOpened():
            # Yield a single placeholder frame explaining the failure
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not available", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            ok, buffer = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            return

        while True:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # mirror for a natural "selfie" view

            if self._specs is None:
                self._specs = build_specs(frame.shape[1], thin=True)
            radius, line_thickness = self._specs

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                draw_skeleton(frame, results.pose_landmarks, radius, line_thickness)

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.pose.close()
