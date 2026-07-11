# Place the target person's photo here

- Drop one or more images (`.jpg` / `.jpeg` / `.png`) of the face of the person you want the robot to track.
- Best results: clear photo, good lighting, face directly or nearly directly facing the camera (similar to a passport photo).
- Avoid sunglasses or anything covering a large part of the face if possible.
- After placing the image, run:

```bash
uv run python -m target_track_robot.face_enroll
```

This will train the face model and save it to `data/models/`.
