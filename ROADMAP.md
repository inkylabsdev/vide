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
- [ ] upscale: super-resolution upscale via an ML model (auto device).
- [ ] interpolate-frames: raise fps with ML frame interpolation (auto device).
* [ ] `vide inspect-media <file>` — inspect codec, resolution, FPS, duration, audio, metadata, and media type.
* [ ] `vide probe-hardware` — detect GPU, VRAM, RAM, CUDA, available accelerators, and supported runtimes.
* [ ] `vide doctor` — diagnose missing dependencies, broken models, provider credentials, and runtime compatibility.
* [ ] `vide explain <operation>` — explain which model, recipe, and runner Vide would select and why.
* [ ] `vide estimate <operation>` — estimate VRAM, runtime, disk usage, output properties, and API cost.
* [ ] `vide capabilities` — list the operations currently available on the user’s machine.
* [ ] `vide generate-image --prompt <text>` — generate an image from a natural-language description.
* [ ] `vide edit-image <image> --prompt <text>` — modify an existing image using text instructions.
* [ ] `vide inpaint-image <image> --mask <mask>` — regenerate a masked region of an image.
* [ ] `vide outpaint-image <image> --aspect <ratio>` — extend an image beyond its original boundaries.
* [ ] `vide upscale-image <image>` — increase image resolution using an AI upscaler.
* [ ] `vide restore-image <image>` — repair blur, noise, compression, damage, or low-quality details.
* [ ] `vide restyle-image <image> --style <text>` — transform an image into another visual style.
* [ ] `vide match-image-style <image> --reference <image>` — apply the visual style of a reference image.
* [ ] `vide remove-image-background <image>` — extract the foreground and produce a transparent image.
* [ ] `vide replace-image-background <image> --background <image>` — place a subject onto a new background.
* [ ] `vide extract-character <image> --target <text>` — isolate a described character from an image.
* [ ] `vide create-character-asset <image>` — create a clean reusable character asset with transparency.
* [ ] `vide create-character-sheet --prompt <text>` — generate consistent front, side, back, and perspective views.
* [ ] `vide generate-keyframes --storyboard <file>` — generate consistent keyframes for planned scenes.
* [ ] `vide generate-video --prompt <text>` — generate a video from a natural-language description.
* [ ] `vide animate-image <image> --prompt <text>` — turn a still image into a moving video.
* [ ] `vide interpolate-frames <start> <end>` — generate a coherent video between two keyframes.
* [ ] `vide generate-transition <start> <end>` — create a generative transition between two images or clips.
* [ ] `vide continue-video <video>` — extend a video forward while preserving motion and visual continuity.
* [ ] `vide prepend-video <video>` — generate footage that leads naturally into an existing video.
* [ ] `vide create-video-loop <video>` — turn a clip into a seamless repeating loop.
* [ ] `vide generate-long-video --prompt-file <file>` — create a longer video using multi-window generation.
* [ ] `vide transfer-motion <video> --character <image>` — transfer a recorded performance onto another character.
* [ ] `vide extract-motion <video>` — extract pose, depth, optical flow, or other reusable motion controls.
* [ ] `vide apply-motion <character> <motion>` — animate a character using previously extracted motion data.
* [ ] `vide recast-person <video> --target <text> --character <image>` — replace a person while preserving the original scene.
* [ ] `vide replace-character <video> --from <text> --to <image>` — replace a described character with a new reference.
* [ ] `vide transfer-face-performance <video> --character <image>` — transfer facial expressions and head movement.
* [ ] `vide lip-sync-video <video> --audio <audio>` — synchronize a character’s mouth to supplied speech.
* [ ] `vide create-talking-avatar <image> --audio <audio>` — create a speaking character video from an image.
* [ ] `vide retake-video <video> --from <time> --to <time>` — regenerate a selected section of an existing video.
* [ ] `vide edit-video <video> --prompt <text>` — modify an existing video with natural-language instructions.
* [ ] `vide add-object <video> --object <image>` — insert a new object into a video scene.
* [ ] `vide remove-object <video> --target <text>` — remove a described object throughout a video.
* [ ] `vide replace-object <video> --target <text> --with <text>` — replace one object with another.
* [ ] `vide inpaint-video <video> --mask <mask>` — regenerate masked areas consistently across video frames.
* [ ] `vide outpaint-video <video> --aspect <ratio>` — expand a video canvas to a different aspect ratio.
* [ ] `vide restyle-video <video> --style <text>` — transform the visual style of an entire video.
* [ ] `vide relight-video <video> --prompt <text>` — change lighting direction, mood, color, or time of day.
* [ ] `vide recamera-video <video> --camera <text>` — reinterpret footage with a different virtual camera movement.
* [ ] `vide remove-video-background <video>` — extract a moving subject and produce a video with transparency.
* [ ] `vide replace-video-background <video> --background <file>` — place a moving subject into a new scene.
* [ ] `vide stabilize-video <video>` — reduce unwanted camera shake or generated motion instability.
* [ ] `vide deflicker-video <video>` — reduce brightness, texture, and color flickering across frames.
* [ ] `vide repair-video <video> --issue <type>` — repair identity, face, hand, geometry, or background inconsistencies.
* [ ] `vide upscale-video <video>` — increase video resolution using a temporally consistent AI upscaler.
* [ ] `vide restore-video <video>` — repair blur, noise, scratches, compression, or degraded footage.
* [ ] `vide denoise-video <video>` — remove visual noise while preserving temporal detail.
* [ ] `vide deblur-video <video>` — improve blurry or soft footage.
* [ ] `vide interpolate-video <video> --fps <number>` — generate intermediate frames to increase frame rate.
* [ ] `vide slow-motion <video> --factor <number>` — create smooth slow motion using frame interpolation.
* [ ] `vide add-film-grain <video>` — add controlled film grain to reduce synthetic-looking output.
* [ ] `vide color-grade <video> --prompt <text>` — apply a natural-language color grade.
* [ ] `vide color-match <video> --reference <video>` — match the color treatment of reference footage.
* [ ] `vide reframe-video <video> --aspect <ratio>` — intelligently crop and track subjects for another format.
* [ ] `vide convert-video <video>` — convert codecs, containers, resolutions, frame rates, or bitrates.
* [ ] `vide optimize-video <video> --for <platform>` — prepare a video for YouTube, Shorts, TikTok, web, or archive.
* [ ] `vide trim-video <video> --from <time> --to <time>` — extract a selected time range.
* [ ] `vide join-videos <videos...>` — concatenate multiple clips into one video.
* [ ] `vide blend-videos <video1> <video2>` — create an AI-generated transition between overlapping clips.
* [ ] `vide extract-frame <video> --at <time>` — export a frame at a specific timestamp.
* [ ] `vide extract-frames <video>` — export frames at a chosen interval or frame rate.
* [ ] `vide frames-to-video <directory>` — assemble an image sequence into a video.
* [ ] `vide design-voice --prompt <text>` — create a reusable synthetic voice from a description.
* [ ] `vide clone-voice <audio>` — create a reusable voice identity from a reference recording.
* [ ] `vide generate-speech --voice <name> --text <text>` — synthesize speech using a selected voice.
* [ ] `vide speak --voice <name> --text-file <file>` — generate narration from a text file.
* [ ] `vide transfer-voice <audio> --voice <name>` — convert existing speech into another voice.
* [ ] `vide change-voice-emotion <audio> --emotion <name>` — alter the emotional delivery of recorded speech.
* [ ] `vide dub-audio <audio> --language <code>` — translate and redub an audio recording.
* [ ] `vide dub-video <video> --language <code>` — translate and redub a video while preserving speakers.
* [ ] `vide transcribe-audio <audio>` — transcribe speech with optional timestamps and subtitles.
* [ ] `vide transcribe-video <video>` — extract and transcribe speech from a video.
* [ ] `vide diarize-audio <audio>` — identify and separate different speakers.
* [ ] `vide separate-vocals <audio>` — separate vocals from instrumental audio.
* [ ] `vide separate-stems <audio>` — separate vocals, drums, bass, and other musical stems.
* [ ] `vide clean-voice <audio>` — remove noise, echo, hum, and recording artifacts from speech.
* [ ] `vide remove-silence <audio>` — remove or shorten silent sections.
* [ ] `vide normalize-audio <audio>` — normalize audio to a target loudness level.
* [ ] `vide generate-music --prompt <text>` — generate background music from a description.
* [ ] `vide generate-song --lyrics <file>` — generate a complete song from lyrics and a style description.
* [ ] `vide extend-music <audio>` — continue an existing piece of music.
* [ ] `vide remix-music <audio> --prompt <text>` — reinterpret an existing track using new musical direction.
* [ ] `vide generate-sfx --prompt <text>` — generate a sound effect from a description.
* [ ] `vide sound-design <video>` — automatically generate and align sound effects for a video.
* [ ] `vide replace-audio <video> <audio>` — replace or remix the audio track of a video.
* [ ] `vide mix-audio` — combine speech, music, ambience, and sound effects into a final mix.
* [ ] `vide sync-music <video> <audio>` — align video cuts or timing with a music track.
* [ ] `vide detect-beats <audio>` — detect beats, tempo, energy, and musical sections.
* [ ] `vide create-storyboard --prompt <text>` — convert an idea into a structured sequence of planned shots.
* [ ] `vide plan-shots --script <file>` — turn a script into camera shots, timings, prompts, and transitions.
* [ ] `vide generate-shot <storyboard> --shot <number>` — generate one selected storyboard shot.
* [ ] `vide regenerate-shot <project> --shot <number>` — rerun one shot without rebuilding the whole project.
* [ ] `vide create-music-video <audio>` — plan and generate a complete music video around a song.
* [ ] `vide create-short-film <script>` — generate a multi-shot short film from a script or story.
* [ ] `vide create-short --script <file>` — produce a vertical short-form video from a script.
* [ ] `vide render-project <project>` — combine project clips, audio, transitions, and effects into a final export.
* [ ] `vide init <name>` — create a new reproducible media-generation project.
* [ ] `vide run <workflow>` — execute a declarative multi-step Vide workflow.
* [ ] `vide resume <project>` — continue an interrupted or partially completed workflow.
* [ ] `vide history` — show previous operations, models, parameters, inputs, and outputs.
* [ ] `vide reproduce <run>` — reproduce a previous generation using its recorded manifest.
* [ ] `vide export-manifest <run>` — export all settings needed to reproduce a generation.
* [ ] `vide compare <files...>` — compare media quality, metadata, resolution, duration, and file size.
* [ ] `vide models` — list available, compatible, installed, and recommended models.
* [ ] `vide model-info <model>` — show a model’s capabilities, requirements, license, and supported runners.
* [ ] `vide install-model <model>` — download and prepare a model for local execution.
* [ ] `vide remove-model <model>` — delete a locally installed model without removing shared dependencies.
* [ ] `vide load-model <model>` — preload a model into memory.
* [ ] `vide unload-models` — unload resident models and release GPU memory.
* [ ] `vide scan-models <directory>` — discover compatible models in an existing installation.
* [ ] `vide link-model-directory <directory>` — reuse models from WanGP, ComfyUI, or another local tool.
* [ ] `vide recipes` — list available operation recipes and their implementations.
* [ ] `vide recipe-info <recipe>` — inspect a recipe’s models, defaults, inputs, and runtime requirements.
* [ ] `vide install-recipe <source>` — install an operation recipe from a registry or Git repository.
* [ ] `vide validate-recipes` — verify that installed recipes are valid and executable.
* [ ] `vide autotune` — select quantization, offloading, tiling, and memory settings for the current hardware.
* [ ] `vide benchmark <model>` — measure runtime, VRAM use, and output characteristics on local hardware.
* [ ] `vide runners` — list available local and hosted execution backends.
* [ ] `vide configure <runner>` — configure a local runtime or hosted provider.
* [ ] `vide test-runner <runner>` — verify that a runner is installed, reachable, and authenticated.
* [ ] `vide compare-runners <operation>` — compare compatible runners by quality, speed, privacy, and cost.
* [ ] `vide estimate-cost <operation>` — estimate hosted generation costs before submission.
* [ ] `vide auth login <provider>` — securely configure credentials for a hosted provider.
* [ ] `vide auth status` — display configured provider authentication status.
* [ ] `vide auth logout <provider>` — remove stored credentials for a provider.
* [ ] `vide jobs` — list queued, running, completed, failed, and cancelled jobs.
* [ ] `vide job <id>` — show detailed status and outputs for one job.
* [ ] `vide watch <id>` — stream progress and logs for a running job.
* [ ] `vide cancel <id>` — cancel a queued or running operation.
* [ ] `vide retry <id>` — rerun a failed or cancelled job.
* [ ] `vide clean-jobs` — remove old job records and temporary files.
* [ ] `vide cache-info` — show model, asset, workflow, and temporary cache usage.
* [ ] `vide clean-cache` — remove unused cached assets and model files.
* [ ] `vide config show` — display the effective Vide configuration.
* [ ] `vide config edit` — open the XDG configuration file in an editor.
* [ ] `vide config get <key>` — read one configuration value.
* [ ] `vide config set <key> <value>` — update a personal default or preference.
* [ ] `vide config unset <key>` — remove a configuration override.
* [ ] `vide mcp serve` — expose Vide operations through a local MCP server.
* [ ] `vide mcp tools` — list the semantic tools exposed through MCP.
* [ ] `vide mcp inspect <tool>` — show the schema and capabilities of one MCP tool.

