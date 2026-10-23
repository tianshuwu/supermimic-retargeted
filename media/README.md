# Demo assets

`demo.mp4` is a 20-second selection from continuous physical rollouts of LAFAN1 `dance1_subject1`. The edit advances through source time 80.16–100.16 seconds, switching G1 → X2 Ultra → A3 three times with varying shot lengths. Playback is normal speed.

Solid geometry is policy execution in MuJoCo; cyan translucent geometry is the reference trajectory. The reference is visual only and is not repositioned each frame to conceal tracking error. All three recordings use a 3.2 m camera distance and 45° vertical field of view. No robot model was resized.

G1 and X2 use archived LAFAN policies; A3 uses an existing BONES continuation policy. This is a selected demo, not evidence of identical training conditions or full-dataset performance. The GIF is a reduced-resolution preview. Clip timing and settings are in [demo.json](demo.json).

Motion source attribution and terms remain those of LAFAN1; see [UPSTREAM.md](../UPSTREAM.md). No raw motion data or checkpoint weights are included in this directory.
