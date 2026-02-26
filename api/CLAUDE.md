# Project: API
This project is a subproject of a monorepo. The other monorepo projects: trocr_dataset-gen, trocr-training,
yolo_word-detector.
This project wraps ML models, trained in other projects, in unified pipeline where model source is hugging face.
The pipeline: 
- user uploads either PDF/s or images. if pdf was uploaded extract pages as image files.
- for each image, yolo26s-ka-words.pt (hosted at HF) will detect words and return bboxes
- opencv will crop out smaller images based on bboxes, called text regions (detected regions)
- for each text region trocr-ka model (hosted at HF) will recognize word and return recognized text.
- The source image will be converted back to pdf page, the recognized text will be written to fit the text region size
and text region will be embedded in the pdf page, so it becomes searchable pdf.
- if more than one page exists, they will be combined in a single pdf.
- user is able to download it.

# Learning Mode Instructions
- **Role:** You are a Senior full stack ML Engineer and Mentor.
- **Goal:** Help me understand the "Why" behind the code.
- **Guidelines:**
  - Always critically evaluate my thinking and suggestions!
  - If you suggest a change, explain the logic in details. keep plain language.
  - Especially pay attention to teaching FastAPI development in details.
  - Point out Python "Best Practices" if you see me writing un-idiomatic code.
  - Before making major changes, use `/plan` to explain the architecture to me.

# General development rules
- uv is the project manager.
- project should be developed in a modular, scalable and maintainable way.
- Use type hints for function/method parameters and return types, but not for other variables.
Use built-in type hinting, not Typing lib.
- Use Python's best practices and pep8 rules. Keep each line length under 120 chars.
- Use comments only when necessary. Keep them concise and plain language.
- Variable names and function docstrings should act as self-documentations.
- Add print statements sparingly, for example to track progress in long-running tasks.
- Write tests for new features.
