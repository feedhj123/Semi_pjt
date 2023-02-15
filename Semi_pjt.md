# Local 환경에서 딥러닝 개발 
- 참고 : 컴퓨터 사양
    - RAM: 32GB
    - CPU: intel(R) Core(TM) i7-8700 CPU @ 3.20GHz 3.19 GHz
    - GPU: Geforce RTX 2070

## Conda 설치 및 개발환경 셋팅
- [한양남자](https://www.youtube.com/watch?v=MbHYl7iVoUY&t=527s)<-설치 및 셋팅 참고
- Tensorflow의 경우 영상대로만 따라하면 문제없이 GPU사용가능함을 확인했지만, Pytorch 설치 및 import ,gpu 실행 하려는 경우 cuda의 버전 문제 탓인지 오류가 발생한다 -> cuda version에 맞는 pytorch 및 torchvision 설치가 필요하다.
    - 해결방법은 이 블로그[pytorch 설치 오류 해결](https://sanglee325.github.io/environment/pytorch-cuda112/)참고
    - 또한 conda 명령어로 실행 혹은 설치가 안되는 경우 명령어를 pip으로 바꿔서 한번 해보기를 권장한다.
    - cuda 11.2 기준으로는 torch 1.7.1이 맞는 짝임.


## GPU사용하여 딥러닝 학습
- 발생했던 오류
    1. OSError: [WinError 1455] 이 작업을 완료하기 위한 페이징 파일이 너무 작습니다. -> GPU에 메모리 할당이 계속 되면서 작업중이던 페이지등이 꺼지던가 하는등 문제 발생함 
   ----> batch size 16>8수정해서 해결 이로 안될시, 사이즈 더 줄여보던가 해야할듯함.
   -----> batch size 4로 모델 학습 진행(GPU6.5~7GB정도 사용중인 것으로 확인)

   2. cannot create new folder: cuda.exe를 실행하면 계속해서 뜨던 오류인데 여러가지 해결방법을 찾아봤지만 전부 먹히는 것은 없었다. 그러다가 그냥 사용자 계정을 새로만든 계정이 아닌 원래 Local 계정(stuednt)에서 실행했더니 정상적으로 설치가 됐다.



# 모델 학습 현황


## 1. 1차시도 (2/6일) 실험용
- 사용 데이터셋: Roboflow 공용데이터셋 - Mofu(raw_ingredient)
    - 크기: Train:207 i Valid: 20  Test: 11
    - 클래스 : 85개
    - PREPROCESSING:
        - Auto-Orient: Applied
        - Resize: Stretch to 640x640
    - AUGMENTATIONS:
        - Outputs per training example: 3
        - Brightness: Between -25% and +25%
        - Blur: Up to 4px

- 파라미터 설정
    - 학습시 설정 사항: img:200 , batch:16 , epochs: 30 , 모델:yolov5x


- 학습후 Detection 결과:
    - epochs도 매우 작고 학습 표본이된 데이터셋 자체가 너무 크기가 작아서 성능이 굉장히떨어질거라고 생각햇으나, 의외로 그렇지는 않았다 인터넷에서 그냥 긁어온 사진으로 detection해도 구분 정도는 할 수 있었음(conf0.5로 설정함)


+++++++map f1score loss값
+++++++데이터셋까봤떠니 ~~~~이런거같다추론 다음에는 이부분개섢


## 2. 2차시도 (2/9일)

- 사용 데이터셋: Roboflow(4조 커스텀데이터셋)
    - 크기: Train:4.2k Valid: 216 Test: 88
    - 클래스: 14개
     - PREPROCESSING:
        - Auto-Orient: Applied
        - Resize: Stretch to 640x640
        - Modify Classes: 0 remapped, 2 droppe
    - AUGMENTATIONS:
        - Outputs per training example: 3
        - Flip: Horizontal, Vertical
        - 90° Rotate: Clockwise, Counter-Clockwise, Upside Down
        - Rotation: Between -15° and +15°




- 파라미터 설정
    - 학습시 설정 사항: img 150 , batch:4 , epochs:150 , 모델:yolov5x



- 성능평가지표
    - Matric/map_0.5:0.95: 0.37978(best)
    - Matric/map_0.5: 0.71437(best)
    - F1-score: 
    - box_loss:
    - cls_loss:




- 학습후 Detection 결과:
    - 발견된 문제: 
   
    오인식이 많음 치킨을 돼지고기로 인식한다던가 달걀을 감자로 인식하는 등 , 인식 정확도가 상당히 떨어짐 conf를 0.3~0.5까지 조절해 본 결과, 0.5로 설정한 경우 Bounding box 자체가 사라지는 경우가 꽤 있음
    - 문제 추론:

    데이터셋으로 사용한 이미지 묶음이 그렇게 잘 정제되지 않아서 생긴 문제인 것같음 박스처리가 모호한 것이 많고 객체를 묶음으로 박스처리한다던가 배경이 너무 많이 들어간 것들이 많으며 특징을 뽑아내기에 사진을 찍은 거리나 화질같은 부분이 영향을 많이 준것으로 보여짐.

    - 개선방안:

    우선, 좋은 성능을 위해 몇가지 클래스만 선별하여 좀 더 신중히 박스처리하고 이미지 자체도 명확하게 특징을 뽑아낼 수 있는 것들로 선별해야할 것 같음.
    
    추가적으로, background images(no object)를 train dataset의 1~10%정도 집어 넣어주는 것이 모델 성능 개선에 도움이 될 수 있다는 roboflow training tips에 따라 데이터셋에 추가해줄 생각임.

    roboflow 사용을 위한 첫 테스트 시도를 제외하고 실제로 우리가 모은 이미지들로 학습시킨것은 처음이라서 다른 파라미터는 전혀 건들지 않았는데,
    Docs를 참조하여 다음 시도에는 이부분들을 좀 건드려볼 생각임.

## 3. 3차시도 (2/13일)

****변경사항***
- class 축소 14 -> 6개 

- bounding box 재설정

------------------
- 사용 데이터셋 Detecting food ingredients for yolov5 Image Dataset(decrease class)
 - 크기: Train:989 Valid: 201 Test: 61
    - 클래스: 6개[chicken,onion,potato,beef,radish,eggs]
     - PREPROCESSING:
        - Auto-Orient: Applied
        - Resize: Stretch to 640x640

    - AUGMENTATIONS:
        - Outputs per training example: 3
        - Flip: Horizontal, Vertical
        - Blur: Up to 3px
        - Bounding Box: Flip: Horizontal
        - Bounding Box: Rotation: Between -15° and +15°

- 파라미터 설정
    - 학습시 설정 사항: img:320 , batch:4 , epochs: 150 , 모델:yolov5x


- 성능평가지표
    - Matric/map_0.5:0.95: 0.28964(best)
    - Matric/map_0.5: 0.52199(best)
    - F1-score: 
    - box_loss:
    - cls_loss:


- 학습후 Detection 결과:
    - 문제 사항

    - 문제 추론

    - 개선 방안



## 4. 4차시도(2/13일)

****변경사항***
- background 제거하고 instance만 남긴 이미지 추가 (trainset의 10%정도 약 40장)
- 데이터셋 변경 --> backgroundimage_Add
------------

- 사용 데이터셋 Detecting food ingredients for yolov5 Image Dataset(backgroundimage_Add)
 - 크기: Train:1074 Valid: 214 Test: 64
    - 클래스: 6개[chicken,onion,potato,beef,radish,eggs]
     - PREPROCESSING:
        - Auto-Orient: Applied
        - Resize: Stretch to 640x640

    - AUGMENTATIONS:
        - Outputs per training example: 3
        - Flip: Horizontal, Vertical
        - Blur: Up to 3px
        - Bounding Box: Flip: Horizontal
        - Bounding Box: Rotation: Between -15° and +15°

- 파라미터 설정
    - 학습시 설정 사항: img:320 , batch:4 , epochs: 150 , 모델:yolov5x


- 성능평가지표
    - Matric/map_0.5:0.95: 0.33224(best)
    - Matric/map_0.5: 0.58226(best)
    - F1-score: 
    - box_loss:
    - cls_loss:


- 학습후 Detection 결과:
    - 문제 사항

    - 문제 추론

    - 개선 방안


## 5.5차시도(2/15일)

**변경사항**
- 각 클래스의 이미지 추가적으로 더해줌
- train/valid/split의 비율 조정해줌 (80/10/10 맞출 수 있도록)

---------

- 시용 데이터셋: Detecting food ingredients for YOLOv5 Image Dataset(adjust split ratio-add more image)
- 크기: Train : 2k   Valid:252 Test:253
    - 클래스: 6개[chicken,onion,potato,beef,radish,eggs]
     - PREPROCESSING:
        - Auto-Orient: Applied
        - Resize: Stretch to 640x640
        
    - AUGMENTATIONS:
        - Outputs per training example: 3
        - Flip: Horizontal, Vertical
        - Blur: Up to 3px
        - Bounding Box: Flip: Horizontal
        - Bounding Box: Rotation: Between -15° and +15°

- 파라미터 설정
    - 학습시 설정 사항: img:320 , batch:4 , epochs: 150 , 모델:yolov5x



    
- 성능평가지표
    - Matric/map_0.5:0.95: 0.44637(best)
    - Matric/map_0.5: 0.71552(best)
    - F1-score: 
    - box_loss:
    - cls_loss:



- 학습후 Detection 결과:
    - 문제 사항

    - 문제 추론

    - 개선 방안

#평가지표 설명
https://help.roboflow.com/en_US/roboflow-train-understanding-training-graphs
1. box_loss : 특정 손실 함수를 기반으로 예측된 경계 상자가 실제 객체에 얼마나 가까운지 측정하는 손실 matric입니다.

    값이 낮으면 모델이 일반화를 위해 개선되고 있고, 데이터세트에 식별할 레이블이 지정된 개체 주위에 더 나은 경계 상자를 만들고 있음을 나타냅니다.


2. cls_loss: 분류 손실로 잘알려져 있습니다. 특정 손실 함수를 기반으로 모든 예측 경계 상자 분류의 정확성을 측정하는 손실 메트릭입니다. 각 개별 경계 상자에는 객체 클래스 또는 배경 레이블이 포함될 수 있습니다.

3. mAP_0.5 : "IoU가 0.50 또는 50%인 평균 정밀도"로 더 잘 알려져 있습니다 0.5 또는 50%보다 큰 IoU(Intersection over Union)에서 "검출된 개체"로 평가된 예측의 평균 정밀도(mAP)입니다.

4. mAP_0.5:0.95: "IoU 간격이 0.50 ~ 0.95 또는 50% ~ 95%인 평균 정밀도"로 더 잘 알려져 있습니다 0.50보다 크고 0.95(50%-95%) 이하인 IoU(Intersection over Union)에서 "검출된 개체"로 평가된 예측이 포함된 평균 정밀도(mAP)입니다.

**matric/map_0.5:0.95는 한스텝마다 0.05씩 IOU thred hold를 올린 MAP로평가하는것을 의미합니다. **
