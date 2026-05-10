# TODO: FracAtlas Data Pipeline Builder

## Task
Create a PyTorch dataset pipeline in `src/dataset.py` for binary classification (fracture vs no fracture).

## Steps
1. [x] Explore project structure and research context
2. [x] Plan the dataset pipeline architecture
3. [ ] Implement `src/dataset.py` with all required components:
   - Custom Dataset class reading from CSV
   - Aspect-ratio preserving resize (pad + resize to 224x224)
   - Train transforms (rotation, affine, colorjitter, h-flip, normalize)
   - Val/Test transforms (deterministic pad/resize, normalize)
   - DataLoader creation functions
   - Sanity-check function
4. [ ] Add educational comments throughout
5. [ ] Test that the file is syntactically valid
6. [ ] Create report and deliver assets


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
