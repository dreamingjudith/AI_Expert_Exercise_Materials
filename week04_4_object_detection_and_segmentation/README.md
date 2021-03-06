# Object Detection and Segmentation

4주 4일차 **Object Detection and Segmentation** 실습 파일입니다. [Original Repository](https://github.com/alinlab/0730_detection_segmentation)

## Object Detection

`detection.ipynb`을 아래와 같이 복사한 뒤 그 폴더에서 사용해주세요.

```
$ cp detection.ipynb object_detection/tools
```

이미지에 Bounding box를 그리기 위해 Cython을 이용해 라이브러리 몇 개를 빌드해야 합니다.

```
$ cd object_detection/libs/box_utils/cython_utils
$ make clean
$ make
```

실습을 위해서는 `data` 폴더에서 resnet 모델을 미리 다운로드 받아야 합니다.

## Semantic Segmentation

첨부된 `segmentation.ipynb` 파일을 확인하세요. 답안은 `segmentation_solution.ipynb`에 있습니다.
