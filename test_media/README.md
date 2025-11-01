# Test Media Files

This directory contains sample media files for testing strands-mlx vision/audio functionality.

## Files

- **sample_image.jpg**: Simple image with status code labels (800x600)
- **sample_audio.wav**: 1-second 440Hz tone (16kHz mono WAV)
- **sample_video.mp4**: 5-frame color sequence video (optional, requires imageio)

## Regeneration

If you need to regenerate these files, run:

```python
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import soundfile as sf

# Create directory
test_media_dir = Path("./test_media")
test_media_dir.mkdir(exist_ok=True)

# 1. Sample image
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)
status_codes = ['200 OK', '404 Not Found', '500 Error', '301 Redirect']
x, y = 50, 50
for code in status_codes:
    draw.rectangle([x, y, x+150, y+100], outline='black', width=2)
    draw.text((x+20, y+40), code, fill='black')
    x += 200
    if x > 650:
        x = 50
        y += 150
img.save(test_media_dir / "sample_image.jpg")

# 2. Sample audio (440Hz tone)
sample_rate = 16000
t = np.linspace(0, 1.0, sample_rate)
audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)
sf.write(test_media_dir / "sample_audio.wav", audio_data, sample_rate)

# 3. Sample video (optional - requires imageio)
# pip install imageio[ffmpeg]
```

## Usage in Tests

All test files use relative paths:

```python
# test_vision.py
agent("Describe <image>./test_media/sample_image.jpg</image>")

# test_audio.py
agent("Transcribe <audio>./test_media/sample_audio.wav</audio>")

# test_video_single.py
agent("Analyze <video>./test_media/sample_video.mp4</video>")
```

This ensures tests are portable and work on any machine without hardcoded paths.
