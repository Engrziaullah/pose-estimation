# Kinetra — Pose Estimation Web App

A Flask front-end for your MediaPipe pose-estimation notebook, with three modes:

1. **Image** — upload a photo, get it back instantly (no page reload) annotated with the skeleton, radial joint-angle gauges, and a downloadable landmark CSV.
2. **Video** — upload a video, watch a **live progress bar** (% complete, frame counter, elapsed time) while it's tracked in a background thread, then get an annotated copy back. Pick a **speed preset** (Fast / Balanced / Accurate) to trade accuracy for turnaround time.
3. **Live** — opens the webcam on the machine running the server and streams the tracked skeleton live in the browser, inside a camera-style viewfinder with a session timer.

> **Why video used to feel stuck:** the original version processed the whole
> clip synchronously before sending any response, so the browser just sat
> there with no feedback. It's now processed in a background thread with a
> polling endpoint (`/api/video/status/<job_id>`), so the UI can show real
> progress the whole way through — and the "Fast" preset also downscales
> frames and uses MediaPipe's lightest model, so it's genuinely quicker too.
>
> Note: the first time you use a given speed preset, MediaPipe downloads
> that preset's model file (a few MB, one-time, needs internet). After that
> it's cached locally and every run — including offline — is fast.

## Setup

```bash
cd pose_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `mediapipe` is pinned to `0.10.14` in `requirements.txt`. Newer
> MediaPipe releases removed the legacy `mp.solutions.pose` API this app
> (and your original notebook) relies on — installing an unpinned/newer
> version will break at import time.

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## About the "Live" mode

The live camera stream is powered by OpenCV opening the webcam **on the
same machine that runs `app.py`** — not the visitor's browser camera. This
is the standard approach for a local Flask app: run it on your own laptop
and it will use your laptop's webcam. If you deploy this to a remote
server, "Live" would need to be rebuilt using the browser's own
`getUserMedia` API to send frames to the server instead.

## Project structure

```
pose_app/
├── app.py              # Flask routes
├── pose_utils.py        # MediaPipe logic (image, video, live generator)
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html       # home page, mode picker
│   ├── image.html
│   ├── video.html
│   └── live.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── uploads/              # user-submitted files land here (gitignored)
└── outputs/               # annotated results served back to the user
```
