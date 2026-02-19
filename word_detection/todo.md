- Do not use typing library, but always annotate function parameters
- in dataset_generation: 
1. documents from source_docs should be parsed
2. we should generate synthetic dataset of 2000 images (640 px min dimensions),
each image should have placed parsed text on it and rendered word's exact bounding box coordinate 
must be saved for that image in YOLO26 compatible format.
3. placed text must be at most 3 different font. Fonts should be applied from fonts/ dir.
4. for document like look augraphy lib should be used
5. 