# object_detection_training
## ch1  
이미지 데이터(coco-128-2)를 indexing 한 뒤, SAM을 이용하여 입력 이미지와 비슷한 이미지를 검출
## ch2  
### Segmentation  
SAM을 이용하여 Segmentation한 object에 mask를 구함
mmengindml 코드중 에러코드위치.png에 해당하는 부분이 예전 코드에 해당하는 부분이 설치 됨 - 우선 직접  weights_only=False 를 추가 하여 해결함  

### OCR
https://github.com/yeungchenwa/OCR-SAM 의 클론 코딩  
위 repo의 image, config, dict 사용



