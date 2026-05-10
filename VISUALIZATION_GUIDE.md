# Live Training Visualization Guide

## Overview

The RL Framework now includes **real-time environment visualization with playback speed control** for monitoring training as it happens. Frames are captured during training and displayed in the GUI dashboard with interactive playback controls.

## Features Added

### 1. Frame Capture During Training
- **Module**: `rl_framework/training/frame_capture_callback.py`
- Automatically captures environment frames every 50 timesteps during training
- Stores up to 200 frames in memory (ring buffer to prevent memory bloat)
- Frames are JPEG-compressed base64 strings for efficient transmission
- Works with any environment that supports rendering

### 2. Training Manager Frame Storage
- **Updated**: `rl_framework/gui/training_manager.py`
- Instantiates `FrameCaptureCallback` for every training run
- Stores frame buffer accessible via `get_frames()` method
- Automatically enables `render_mode="rgb_array"` during training

### 3. Live Visualization API Endpoint
- **Route**: `GET /api/train/frames/<run_id>`
- Returns JSON with list of captured frames
- Each frame includes:
  - `frame_index`: Sequential frame number
  - `timestep`: Environment timestep when captured
  - `episode_num`: Episode number
  - `image_base64`: JPEG image as base64 string

### 4. Interactive Dashboard Visualization
- **Updated**: `templates/index.html` and `static/app.js`
- New "Live Environment Visualization" panel on dashboard
- Canvas-based frame display
- Interactive controls:
  - **Play/Pause** button to start/stop playback
  - **Speed Selector**: 0.5x, 1x, 2x, 4x playback speeds
  - **Timeline Scrubber**: Seek to any frame
  - **Frame Counter**: Shows current frame / total frames

### 5. CSS Styling
- **Updated**: `static/style.css`
- Responsive visualization panel with dark theme
- Smooth canvas rendering with pixel-perfect display

## How to Use

### Start Training with Visualization

1. **Open the GUI**:
   ```bash
   python -m rl_framework.cli.main gui --port 5001
   ```

2. **Navigate to http://127.0.0.1:5001**

3. **Create experiment** using the 4-step wizard (same as before)

4. **Switch to Dashboard** - training starts automatically and visualization appears

### Monitor Training

**Live Metrics** (top section):
- Timesteps, Mean Reward, Episode Length
- Training losses and entropy
- Learning rate (useful for live tuning)

**Environment Visualization** (new):
- Watch the agent learn in real-time
- Play/pause the frame sequence
- Speed up visualization to see progress faster (doesn't affect training speed)
- Scrub to specific frames to review past states

**Playback Controls**:
- **Play/Pause**: Click to start/stop animation
- **Speed**: Select 0.5x (slow-mo), 1x (real-time), 2x (fast), or 4x (very fast)
- **Timeline**: Drag scrubber or click to seek to specific frame
- **Frame Counter**: Shows progress through captured frames

## API Examples

### Get captured frames for a training run:
```bash
curl http://127.0.0.1:5001/api/train/frames/run_abc123
```

Response:
```json
{
  "frames": [
    {
      "frame_index": 0,
      "timestep": 50,
      "episode_num": 0,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."
    },
    ...
  ]
}
```

### Combine with live tuning:
```bash
# Monitor frames and adjust parameters in real-time
curl http://127.0.0.1:5001/api/train/frames/run_abc123
curl -X POST http://127.0.0.1:5001/api/train/tune/run_abc123 \
  -d '{"learning_rate": 0.0001}'
```

## Performance Notes

- **Frame Capture**: ~1 frame every 50 timesteps = minimal overhead
- **Memory Usage**: Max 200 frames × ~50KB each = ~10MB
- **Network**: JPEG compression reduces transfer size by ~90%
- **Browser**: Canvas rendering is optimized for 30 FPS display

## Configuration

### Adjust frame capture rate:
Edit `training_manager.py` line where `FrameCaptureCallback` is created:
```python
frame_cb = FrameCaptureCallback(
    capture_interval=50,  # Capture every N timesteps
    max_frames=200,       # Maximum frames to keep
    verbose=1
)
```

## Supported Environments

- **Walker (Locomotion)**: Full visualization support
- **Organism Arena (Multi-Agent)**: Full visualization support
- Any environment with PyBullet rendering

## Technical Details

### Frame Capture Flow
1. Training loop runs with frame capture callback
2. Every 50 timesteps, callback calls `env.render()`
3. RGB array is compressed to JPEG and base64-encoded
4. Frame added to ring buffer (oldest frame drops if full)

### Frontend Flow
1. Dashboard polls `/api/train/frames/<run_id>` every 1 second
2. New frames update the canvas
3. Playback timer advances frame index based on selected speed
4. User controls seek, play/pause, and speed without affecting training

## Future Enhancements

Potential additions (not yet implemented):
- Video download (MP4/GIF export)
- Frame-by-frame step controls
- Overlay metrics on visualization
- Multiple environment views (for multi-agent)
- Custom camera angles
- Recording entire training runs

## Troubleshooting

**No frames appearing?**
- Check that training is actually running (look at metrics)
- Wait 6+ seconds for first frame (captured at 50 timesteps)
- Refresh dashboard to trigger fresh poll

**Frames loading slowly?**
- Increase capture interval to reduce frequency
- Reduce max_frames buffer size
- Check network latency between client and server

**Canvas blank?**
- Ensure browser supports HTML5 Canvas (modern browsers)
- Check browser console for JavaScript errors
