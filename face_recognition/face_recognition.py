import numpy as np
import cv2, os, codecs, json, argparse, logging, time

pwd = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='face recognition')
parser.add_argument('--log', dest='loglevel', help='log level', default='INFO')
parser.add_argument('--known', dest='known', help='known images saved as json', default=os.path.join(pwd, "known.json"))
parser.add_argument('--test', dest='test', help='test image path', default=os.path.join(pwd, "test.jpeg"))
args = parser.parse_args()

logging.basicConfig(level=args.loglevel.upper())
start = time.time()
logging.debug("start-->")

import face_recognition
current = time.time()
logging.debug(f'lib imported, takes {current - start} seconds')
start = current

known_encoding_path = args.known
if os.path.exists(known_encoding_path):
  obj_text = codecs.open(known_encoding_path, 'r', encoding='utf-8').read()
  features = json.loads(obj_text)
  known_face_encodings, known_names = [], []
  for name, feature in features.items():
   known_face_encodings.append(np.array(feature))
   known_names.append(name)
else :
  encoding_dict = {}
  known_image_path = os.path.join(pwd, "known")
  known_face_encodings = []
  for image_name in os.listdir(known_image_path):
    name = image_name[:-5] # remove extention
    imagePath = os.path.join(known_image_path, image_name)
    img = face_recognition.load_image_file(imagePath)
    face_locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, face_locations)
    encoding_dict[name] = encodings[0].tolist()
    known_face_encodings.append(encodings[0])

    if args.loglevel.upper() == logging.DEBUG:
      showImage = cv2.imread(imagePath)
      top, right, bottom, left = face_locations[0]
      cv2.rectangle(showImage, (left, top), (right, bottom), (0, 255, 0), 2)
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(showImage, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

      cv2.imshow("found", showImage)

      cv2.waitKey(0)
      cv2.destroyAllWindows()

  json.dump(encoding_dict, codecs.open(known_encoding_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

current = time.time()
logging.debug(f'preset images loaded, takes {current - start} seconds')
start = current
logging.debug("ok, now testing......")

testImagePath = args.test
try:
  img = face_recognition.load_image_file(testImagePath)
except FileNotFoundError:
  logging.error("test image file not found")
  exit(1)
face_locations = face_recognition.face_locations(img)
test_encodings = face_recognition.face_encodings(img, face_locations)
current = time.time()
logging.debug(f'image encoding created, takes {current-start} seconds')
start = current

for face_encoding_to_check in test_encodings:
  distances = face_recognition.api.face_distance(known_face_encodings, face_encoding_to_check)
  m = min(distances)
  result = [f'{known_names[i]} {num} {num<0.6}' for i, num in enumerate(distances) if num == m]
  print(" - ".join(result))

current = time.time()
logging.debug(f'face recognition done, takes {current-start} seconds')
