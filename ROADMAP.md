# Roadmap

- [x] extract-audio: pull the audio track out of a video (stream copy when the container allows).
- [x] remove-audio: write a copy of the video with the audio track stripped.
- [ ] trim: cut a video to a start/end timestamp range without re-encoding when possible.
- [ ] compress: re-encode a video to a target size or CRF quality.
- [ ] resize: scale a video to a given width/height or preset (720p, 1080p, ...).
- [ ] crop: cut a video to a given rectangle.
- [ ] speed: speed up or slow down a video (and pitch-correct the audio).
- [ ] to-gif: convert a clip to an optimized GIF (palette pass, capped fps/width).
- [ ] extract-frames: dump frames as images at a given interval or fps.
- [ ] thumbnail: pick a representative frame and save it as an image.
- [ ] watermark: overlay an image or text at a corner with opacity.
- [ ] burn-subtitles: hard-render a subtitle file onto the video.
- [ ] transcribe: speech-to-text a video's audio into an .srt (Whisper-class model, auto device).
- [x] upscale: super-resolution upscale via an ML model (Replicate — Real-ESRGAN or Anime4K).
- [ ] interpolate-frames: raise fps with ML frame interpolation (auto device).
