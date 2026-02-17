how pipeline should work:
1. if pdf was uploaded extract pages as image files

2. pipeline will loop through images

3. on a single image yolo will detect bboxes of words on each image and return bboxes

4. opencv will cut the image into smaller images using returned bboxes

5. in another loop on each cropped image trocr-ka will do text recognition and return text string

6. the strings will be embedded on the pdf pages over the detected text as overlay

7. uploaded images will be combined into searchable pdf and become available for download.

for easy data management we can create a dataclass OCRItem that will hold 
image as np.ndarray, bboxes list and probably a dict of bbox and corresponding recognized text. 
this is preliminary idea, and will be refined.