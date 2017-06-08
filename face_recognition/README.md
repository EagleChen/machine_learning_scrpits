# face_recognition

### How to use
1. put known images to 'known' directory(during first run). For example, use `jay.jpeg` for jay's photo.
2. run the script. `known.json` will be created which contains features for persons under `known` folder. This json file path can be changed by specifying `--known` argument. The file can be reused(`known` folder can be removed).
  ```
  python face_recognition.py --test /tmp/test.jpeg
  ```

Have fun!

### Reference
Blog Reference in Chinese: [机器学习之简单的人脸识别](http://www.jianshu.com/p/a3fdb0cec862)
