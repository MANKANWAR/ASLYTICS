# Contributing to ASLytics

Thank you for helping build open-source, on-device speech and sign accessibility for the Deaf and Hard-of-Hearing (D/HoH) community!

## Core Design Principles

1. **Nothing About Us Without Us**: Accessibility features (watchword alerts, 360° spatial speaker radar, overlap indicators, ASL Gloss, and visual contrast) must prioritize real-world cognitive ergonomics for D/HoH participants in group settings.
2. **100% On-Device Privacy**: Group conversations in homes, classrooms, medical appointments, and workplaces are deeply private. All speech recognition (`Parakeet-TDT`), speaker diarization (`Sortformer`), and translation pipelines must be runnable locally without mandatory cloud APIs.
3. **Zero-Friction Developer Experience**: Every new feature must include a deterministic test in `tests/` and work seamlessly with `python3 -m unittest discover -s tests -v`.

## Development Workflow

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/MANKANWAR/ASLYTICS.git
   cd ASLYTICS
   ```
2. Run the test suite:
   ```bash
   python3 -m unittest discover -s tests -v
   ```
3. Launch the local server to test UI changes:
   ```bash
   python3 -m aslytics serve --port 8765
   ```
4. Submit a Pull Request describing the accessibility impact and hardware backends tested.
