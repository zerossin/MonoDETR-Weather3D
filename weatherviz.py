import os
import cv2
import numpy as np   #numpy 라이브러리 임포트
import matplotlib.pyplot as plt   #matplotlib 라이브러리 임포트

# -----------------------------------------
# category
# KITTI 클래스 객체 분류
category = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
    'Cyclist', 'Tram', 'Misc', 'DontCare'
]


# ---------------------------------
# load_calibration
# 카메라 정보 파일을 읽어 파싱처리 
def load_calibration(calibration_datafile):
    # 파일 오픈
    with open(calibration_datafile, 'r') as file:
        # 한 줄씩 읽기
        for row in file:
            # p2 행렬 정보 찾기 후 segment에 저장
            if row.startswith('P2:'):
                segment = [float(i) for i in row.strip().split()[1:]]
                return np.array(segment).reshape(3, 4) # numpy 행렬로 변환
    raise ValueError("이상")   # 파일 자체에 이상이 있는 경우 에러 처리


# ---------------------------------
# load
# 예측 파일을 읽어 정보를 파싱
def load(prediction_datafile):
    output = []  # 결과를 담을 리스트
    # 읽기 모드로 파일 열기
    with open(prediction_datafile, 'r') as file:
        # 한 줄 씩 읽기
        for row in file:
            # 정보가 없는 빈 줄 + 주석 처리된 줄 제외
            if not row.strip() or row.startswith("//"):
                continue
            segments = row.strip().split()
            name = segments[0]

            # 높이 h, 너비 w, 길이 l
            h, w, l = map(float, segments[8:11])
            x, y, z = map(float, segments[11:14]) # 객체 위치 좌표
            r_yaw = float(segments[14])

            # 카테고리 아이디
            categoryId = category.index(name) if name in category else len(category)-1
            
            #output 어레이에 추가
            output.append({'class_id': categoryId, 'h': h, 'w': w, 'l': l, 'x': x, 'y': y, 'z': z, 'ry': r_yaw})
    return output




# ---------------------------------
# position_2dimaging 함수
# 모서리의 좌표를 2D 이미지로 투영하는 함수
def position_2dimaging(position, P2_Matrix):
    k = position.shape[1]
    homo_coordinates = np.vstack((position, np.ones((1, k))))
    
    # 동차 좌표로 변환
    projection_Matrix = P2_Matrix @ homo_coordinates
    
    # 투영 처리
    projection_Matrix[:2] /= projection_Matrix[2]
    return projection_Matrix[:2].T.astype(np.int32)


# ---------------------------------
# xyz_position 함수
# 객체의 좌표 계산
def xyz_position(item):
    # 높이 h, 너비 w, 길이 l
    l, h, w = item['l'], item['h'], item['w']

    # 총 8개 모서리의 좌표 정리
    xPosit = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    yPosit = [0, -h, -h, 0, 0, -h, -h, 0]
    zPosit = [w/2, w/2, w/2, w/2, -w/2, -w/2, -w/2, -w/2]
    
    position = np.array([xPosit, yPosit, zPosit])
    
    # 회전 행렬 생성 및 처리
    rotationMatrix = np.array([
        [np.cos(item['ry']), 0, np.sin(item['ry'])],
        [0, 1, 0],
        [-np.sin(item['ry']), 0, np.cos(item['ry'])]
    ])

    # 모서리 좌표에 회전 행렬 결과 적용 
    output = rotationMatrix @ position
    output[0, :] += item['x']
    output[1, :] += item['y']
    output[2, :] += item['z'] # 실제 좌표로 이동
    
    return output


# ---------------------------------
# boxing 함수
# 이미지에 직접 박스를 그리는 함수
def boxing(image, matrix, name=None, score=None, color=(0,255,0), strength=2):
    # 12개의 엣지를 정의
    # 현재 선 두께 strength 2로 박스를 그림
    edge_list = [
        (0,1), (1,2), (2,3), (3,0), 
        (4,5), (5,6), (6,7), (7,4), 
        (0,4), (1,5), (2,6), (3,7) 
    ]

    # 실제 박스 그리기
    for first, last in edge_list:
        cv2.line(image, tuple(matrix[first]), tuple(matrix[last]), color, strength)
    

    # 클래스명을 텍스트로 박스 위에 표기
    if name is not None:
        box = [4, 5, 6, 7]

        # 박스 정 중앙에 이름을 표기
        x = int(np.mean(matrix[box, 0]))
        y = int(np.min(matrix[box, 1]))
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    # 이미지 반환
    return image


# -----------------------------
# render
# 최종 이미지화

# 이미지 경로 입력
def render(imagefilePath, calibfilePath, predfilePath):
    # 이미지가 존재하지 않는 경우 에러 처리
    if not (os.path.exists(imagefilePath) and os.path.exists(calibfilePath) and os.path.exists(predfilePath)):
        print("오류")
        return
    
    # 이미지 정보 읽기
    image = cv2.imread(imagefilePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    P2 = load_calibration(calibfilePath) # calibration 정보
    predictionFile = load(predfilePath) # 예측 파일 정보

    # 정보를 바탕으로 박스 그리기
    for item in predictionFile:
        xyzPosition = xyz_position(item)
        points = position_2dimaging(xyzPosition, P2)
        categoryname = category[item['class_id']]
        score = 1.0
        # draw 함수 호출
        image = boxing(image, points, name=categoryname, score=score, color=(0, 255, 0))
    
    # matplotlib으로 이미지 시각화
    plt.figure(figsize=(12,8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()



# -----------------------------
# save
# 파일을 로컬에 최종 저장
# file들의 path를 파라미터로 받음
def save(predictionfilePath, imagefilePath, calibrationfilePath, savePath):
    
    #savePath가 없으면 새로 만들기
    os.makedirs(savePath, exist_ok=True)
    predictionfiles = [file for file in os.listdir(predictionfilePath) if file.endswith('.txt')]
    
    # 예측 파일들 가져오기
    for prediction in predictionfiles:
        start = os.path.splitext(prediction)[0]
        image = os.path.join(imagefilePath, f"{start}.png")
        calib = os.path.join(calibrationfilePath, f"{start}.txt")
        pred = os.path.join(predictionfilePath, prediction)
        # 파일 존재 여부 확인 후 오류처리
        if not (os.path.exists(image) and os.path.exists(calib) and os.path.exists(pred)):
            print(f"오류: {start}")
            continue
        
        # 이미지 읽기
        finalimg = cv2.imread(image)
        finalimg = cv2.cvtColor(finalimg, cv2.COLOR_BGR2RGB)
        P2 = load_calibration(calib) #calibration 및 P2
        pre = load(pred)

        # 이미지에 박스 그리기
        for item in pre:
            position = xyz_position(item) # 좌표 계산
            points = position_2dimaging(position, P2) #이미지에 투영
            name = category[item['class_id']]
            finalimg = boxing(finalimg, points, name=name, color=(0, 255, 0))
        
        # 이미지 결과를 로컬에 저장
        save = os.path.join(savePath, f"{start}_vis.png")
        cv2.imwrite(save, cv2.cvtColor(finalimg, cv2.COLOR_RGB2BGR))
        print(f"저장됨: {save}")


#----------------------------------------
# 사용 방법
# os.path.join의 파일명은 이름에 맞게 변경
if __name__ == "__main__":
    # 1) clean 데이터 시각화하기
    # 데이터 입력
    prediction = os.path.join("clean", "pred(multi_mse_all)")
    image = os.path.join("clean", "image_2")
    calibration = os.path.join("clean", "calib")
    save = "./result/multi_mse_all(clean)"
    save(prediction, image, calibration, save)

    # 2) foggy data 시각화하기
    # 데이터 입력
    prediction = os.path.join("foggy", "pred(multi_mse_all)")
    image = os.path.join("foggy", "image_2")
    calibration = os.path.join("foggy", "calib")
    save = "./result/multi_mse_all(foggy)"
    save(prediction, image, calibration, save)