 import cv2
  import numpy as np
  import tesserocr as tr
  from PIL import Image

  cv_img = cv2.imread('12.jpg', cv2.IMREAD_UNCHANGED)

  # since tesserocr accepts PIL images, converting opencv image to pil
  pil_img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))

  #initialize api
  api = tr.PyTessBaseAPI()
  try:
    # set pil image for ocr
    api.SetImage(pil_img)
    # Google tesseract-ocr has a page segmentation methos(psm) option for specifying ocr types
    # psm values can be: block of text, single text line, single word, single character etc.
    # api.GetComponentImages method exposes this functionality
    # function returns:
    # image (:class:`PIL.Image`): Image object.
    # bounding box (dict): dict with x, y, w, h keys.
    # block id (int): textline block id (if blockids is ``True``). ``None`` otherwise.
    # paragraph id (int): textline paragraph id within its block (if paraids is True).
    # ``None`` otherwise.
    boxes = api.GetComponentImages(tr.RIL.TEXTLINE,True)
    # get text
    text = api.GetUTF8Text()
    # iterate over returned list, draw rectangles
    for (im,box,_,_) in boxes:
      x,y,w,h = box['x'],box['y'],box['w'],box['h']
      cv2.rectangle(cv_img, (x,y), (x+w,y+h), color=(0,0,255))
  finally:
    api.End()

  cv2.imshow('output', cv_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
