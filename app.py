"""
app.py - Flask front-end for the MediaPipe Pose Estimation project.

Three modes, reachable from the home page:
  /image  -> upload a photo, get it back with the skeleton drawn on top
  /video  -> upload a video, tracked in a background thread with a live
             progress bar (no more staring at a frozen page)
  /live   -> open the server's webcam and stream annotated frames live

Run with:
    python app.py
Then open http://127.0.0.1:5000 in a browser.

NOTE on /live: the webcam is opened on the machine running this Flask
server (via OpenCV), not the visitor's browser camera. That means it only
works as "live camera" when you run this app locally on your own laptop/PC.
"""

import os
import uuid
import time
import threading
from flask import (
    Flask, render_template, request, url_for,
    send_from_directory, Response, jsonify
)
from werkzeug.utils import secure_filename

import pose_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

# speed presets -> (model_complexity, max_width). Lower complexity + a capped
# frame width are what actually make long clips finish quickly.
SPEED_PRESETS = {
    "fast": (0, 960),
    "balanced": (1, 1280),
    "accurate": (2, None),
}

app = Flask(__name__)
app.secret_key = "pose-estimation-dev-key"
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB upload cap

# Single shared live-feed generator instance (created lazily)
_live_feed = None

# In-memory job registry for async video processing.
# job_id -> {status, progress, frames_processed, total_frames, result, error, started}
_jobs = {}
_jobs_lock = threading.Lock()


def allowed_file(filename, extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def unique_name(filename):
    ext = filename.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _run_video_job(job_id, input_path, output_path, model_complexity, max_width):
    _set_job(job_id, status="processing")

    def on_progress(current_frame, total_frames):
        _set_job(job_id, progress=int(current_frame / total_frames * 100),
                  frames_processed=current_frame, total_frames=total_frames)

    try:
        success, frames_with_detection, total_frames = pose_utils.process_video(
            input_path, output_path,
            model_complexity=model_complexity,
            max_width=max_width,
            progress_callback=on_progress,
        )
        if not success:
            _set_job(job_id, status="error", error="Could not read that video file.")
            return
        _set_job(
            job_id,
            status="done",
            progress=100,
            result={
                "output_video": os.path.basename(output_path),
                "frames_with_detection": frames_with_detection,
                "total_frames": total_frames,
            },
        )
    except Exception as exc:  # keep the UI informed instead of hanging forever
        message = str(exc)
        lowered = message.lower()
        if "http error" in lowered or "urlopen" in lowered or "urlerror" in lowered:
            message = (
                "MediaPipe needed to download a model file for this speed setting "
                "and couldn't reach the internet. Check your connection and try "
                "again — after the first successful run for a given speed, it's "
                "cached locally and works offline."
            )
        _set_job(job_id, status="error", error=message)
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------------------------------
# Image mode (AJAX: instant, no page reload)
# --------------------------------------------------------------------------
@app.route("/image")
def image_mode():
    return render_template("image.html")


@app.route("/api/image/upload", methods=["POST"])
def api_image_upload():
    file = request.files.get("image_file")
    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image file first."}), 400
    if not allowed_file(file.filename, IMAGE_EXTENSIONS):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, JPEG, WEBP, or BMP."}), 400

    original_name = secure_filename(file.filename)
    saved_name = unique_name(original_name)
    input_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(input_path)

    output_name = f"annotated_{saved_name.rsplit('.', 1)[0]}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    try:
        success, people_found, angles, csv_path = pose_utils.process_image(input_path, output_path)
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass

    if not success:
        return jsonify({"error": "Could not read that image. Try a different file."}), 400

    return jsonify({
        "detected": people_found > 0,
        "output_image_url": url_for("serve_output", filename=output_name),
        "angles": angles,
        "csv_url": url_for("serve_output", filename=os.path.basename(csv_path)) if csv_path else None,
    })


# --------------------------------------------------------------------------
# Video mode (async background job + progress polling)
# --------------------------------------------------------------------------
@app.route("/video")
def video_mode():
    return render_template("video.html")


@app.route("/api/video/upload", methods=["POST"])
def api_video_upload():
    file = request.files.get("video_file")
    if not file or file.filename == "":
        return jsonify({"error": "Please choose a video file first."}), 400
    if not allowed_file(file.filename, VIDEO_EXTENSIONS):
        return jsonify({"error": "Unsupported file type. Use MP4, MOV, AVI, MKV, or WEBM."}), 400

    speed = request.form.get("speed", "fast")
    model_complexity, max_width = SPEED_PRESETS.get(speed, SPEED_PRESETS["fast"])

    original_name = secure_filename(file.filename)
    saved_name = unique_name(original_name)
    input_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(input_path)

    output_name = f"annotated_{saved_name.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "frames_processed": 0,
            "total_frames": 0,
            "result": None,
            "error": None,
            "started": time.time(),
        }

    thread = threading.Thread(
        target=_run_video_job,
        args=(job_id, input_path, output_path, model_complexity, max_width),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/video/status/<job_id>")
def api_video_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job id."}), 404

    payload = dict(job)
    payload["elapsed"] = round(time.time() - job["started"], 1)
    if payload.get("result"):
        payload["result"] = dict(payload["result"])
        payload["result"]["output_video_url"] = url_for(
            "serve_output", filename=payload["result"]["output_video"]
        )
    return jsonify(payload)


# --------------------------------------------------------------------------
# Live camera mode
# --------------------------------------------------------------------------
@app.route("/live")
def live_mode():
    return render_template("live.html")


@app.route("/video_feed")
def video_feed():
    global _live_feed
    if _live_feed is None:
        _live_feed = pose_utils.LiveFeedGenerator(camera_index=0)
    return Response(_live_feed.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_live", methods=["POST"])
def stop_live():
    global _live_feed
    if _live_feed is not None:
        _live_feed.release()
        _live_feed = None
    return jsonify({"stopped": True})


# --------------------------------------------------------------------------
# Serving generated files
# --------------------------------------------------------------------------
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
