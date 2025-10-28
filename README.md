# MonoDETR-Weather3D

**[KCC 2025 학부생논문] 악천후 환경에서 단안 3D 객체 검출 성능 향상을 위한 MonoDETR 기반 기법**

[![Paper](https://img.shields.io/badge/Paper-Google_Drive-blue)](https://drive.google.com/your-paper-link)
[![License](https://img.shields.io/badge/License-Academic-green)]()

## 프로젝트 개요

본 프로젝트는 악천후(특히 안개) 환경에서 단안 카메라 기반 3D 객체 검출의 성능을 향상시키기 위한 연구입니다. 기존 MonoDETR 모델을 기반으로 Fine-Tuning, Multi-Domain Learning, Teacher-Student Learning 등 다양한 학습 전략을 구현하고 비교 분석했습니다.

## 주요 특징

### 1. Fine-Tuning
- Clean 환경에서 사전 학습 후 Foggy 환경에서 추가 학습
- 특정 환경에 대한 높은 성능 달성
- Catastrophic Forgetting 현상 관찰

### 2. Multi-Domain Learning
- 하나의 모델이 Clean과 Foggy 이미지를 동시에 학습
- 두 도메인 간 균형잡힌 성능 달성
- 망각 문제 완화

### 3. Teacher-Student Learning
- Clean 환경 전문가(Teacher)의 지식을 Foggy 환경 학습에 전달
- Multi-Layer Knowledge Distillation 구현
- Cross-Domain Knowledge Transfer를 통한 성능 향상

## 프로젝트 구조

```
MonoDETR-Weather3D/
├── configs/                    # 설정 파일들
│   └── monodetr.yaml          # 기본 학습 설정
├── data/                      # 데이터셋 (KITTI, Foggy KITTI)
├── lib/
│   ├── datasets/              # 데이터 로더 및 전처리
│   │   ├── kitti/
│   │   │   ├── kitti_dataset.py
│   │   │   └── kitti_utils.py
│   │   └── utils.py
│   ├── models/                # MonoDETR 모델 구현
│   │   ├── monodetr/
│   │   └── backbone/
│   ├── helpers/               # 학습 및 평가 헬퍼
│   │   ├── trainer_helper.py  # 핵심 학습 로직
│   │   ├── tester_helper.py   # 평가 로직
│   │   ├── save_helper.py
│   │   └── decode_helper.py
│   └── losses/                # Loss 함수들
├── tools/
│   ├── train_val.py           # 학습 및 검증 메인 스크립트
│   └── test.py                # 테스트 스크립트
├── experiments/               # 실험 결과 저장 디렉토리
└── README.md
```

## 환경 설정

### 요구사항
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.1+

### 설치
```bash
# 저장소 클론
git clone https://github.com/yourusername/MonoDETR-Weather3D.git
cd MonoDETR-Weather3D

# 필요한 패키지 설치
pip install -r requirements.txt

# KITTI 데이터셋 다운로드 및 설정
# Foggy KITTI 데이터셋 준비
```

## 데이터셋 준비

### KITTI Dataset
```
data/KITTI/
├── ImageSets/
├── training/
│   ├── image_2/
│   ├── calib/
│   └── label_2/
└── testing/
    ├── image_2/
    └── calib/
```

### Foggy KITTI Dataset
```
data/KITTI/
├── training/
│   └── image_2_foggy/
└── testing/
    └── image_2_foggy/
```

## 사용 방법

### 1. Fine-Tuning 학습

#### Step 1: Clean 환경 학습 (50 epochs)
```bash
python tools/train_val.py \
    --config configs/monodetr.yaml \
    --max_epoch 50 \
    --output_dir experiments/clean_only
```

#### Step 2: Foggy 환경 Fine-tuning (50 epochs)
```bash
python tools/train_val.py \
    --config configs/monodetr.yaml \
    --max_epoch 50 \
    --pretrain_model experiments/clean_only/checkpoint_epoch_50.pth \
    --output_dir experiments/fine_tuning \
    --use_foggy True
```

### 2. Multi-Domain Learning
```bash
python tools/train_val.py \
    --config configs/monodetr.yaml \
    --max_epoch 100 \
    --output_dir experiments/multi_domain \
    --use_foggy True \
    --clean_weight 0.5 \
    --foggy_weight 0.5
```

### 3. Teacher-Student Learning
```bash
python tools/train_val.py \
    --config configs/monodetr.yaml \
    --max_epoch 100 \
    --output_dir experiments/teacher_student \
    --use_foggy True \
    --teacher_ckpt experiments/clean_only/checkpoint_epoch_50.pth \
    --lambda_distill 1.0 \
    --clean_weight 0.5 \
    --foggy_weight 0.5
```

### 4. 모델 평가
```bash
python tools/test.py \
    --config configs/monodetr.yaml \
    --checkpoint_path experiments/multi_domain/checkpoint_epoch_100.pth \
    --test_foggy True
```

## 핵심 구현 사항

### Multi-Domain Learning
```python
# trainer_helper.py의 핵심 로직
student_outputs_clean = self.model(imgs_clean, calibs, targets_list, img_sizes)
student_outputs_foggy = self.model(imgs_foggy, calibs, targets_list, img_sizes)

clean_losses = self.detr_loss(student_outputs_clean, targets_list)
foggy_losses = self.detr_loss(student_outputs_foggy, targets_list)

total_loss = alpha * clean_losses + beta * foggy_losses
```

### Teacher-Student Learning
```python
# Teacher 모델 실행 (가중치 고정)
with torch.no_grad():
    teacher_outputs = self.teacher_model(imgs_clean, calibs, targets_list, img_sizes)

# Student 모델 실행
student_outputs_foggy = self.model(imgs_foggy, calibs, targets_list, img_sizes)

# Multi-Layer Distillation Loss
distill_loss = 0
for i in range(num_layers):
    distill_loss += MSE(student_aux[i]['pred_logits'], teacher_aux[i]['pred_logits'])
    distill_loss += MSE(student_aux[i]['pred_boxes'], teacher_aux[i]['pred_boxes'])

total_loss = alpha * clean_losses + beta * foggy_losses + lambda_distill * distill_loss
```

## 실험 결과

### KITTI Validation Set (Car, Moderate Difficulty, IoU=0.7)

| Method | Clean (Easy/Mod/Hard) | Foggy (Easy/Mod/Hard) |
|--------|----------------------|----------------------|
| Clean Only | 21.15/15.71/12.78 | 10.29/6.39/4.94 |
| Foggy Only | 4.80/3.58/2.89 | 19.95/13.89/11.15 |
| Fine-Tuning | 19.34/12.61/10.42 | 24.35/17.15/13.87 |
| Multi-Domain | 19.24/14.67/11.99 | 20.50/15.35/12.43 |

### 주요 발견사항
- **Fine-Tuning**: Foggy 환경에서 최고 성능, 하지만 Clean 성능 저하 (Catastrophic Forgetting)
- **Multi-Domain**: 두 도메인 간 균형잡힌 성능, 망각 문제 완화
- **실용적 관점**: Multi-Domain이 다양한 날씨 조건에 robust한 자율주행에 더 적합

## 설정 파일 예시

### monodetr.yaml
```yaml
dataset:
  type: 'KITTI'
  root_dir: './data/KITTI'
  writelist: ['Car']
  random_flip: 0.5
  random_crop: 1.0
  scale: 0.4
  shift: 0.1

model:
  type: 'MonoDETR'
  backbone: 'ResNet50'
  num_queries: 50
  
trainer:
  max_epoch: 100
  batch_size: 8
  learning_rate: 0.0002
  lr_drop: 90
  
  # Multi-Domain 설정
  use_foggy: True
  clean_weight: 0.5
  foggy_weight: 0.5
  
  # Teacher-Student 설정
  teacher_ckpt: null  # Teacher 체크포인트 경로
  lambda_distill: 1.0
```

## 논문

전체 논문은 다음 링크에서 확인하실 수 있습니다:
- [논문 PDF (Google Drive)](https://drive.google.com/file/d/your-file-id/view?usp=sharing)
- [발표 자료 (Google Drive)](https://drive.google.com/file/d/your-presentation-id/view?usp=sharing)

## 참고 문헌

### 베이스 모델
- Zhang, R., Qiu, H., Wang, T., Xu, X., Guo, Z., Qiao, Y., ... & Gao, P. (2022). MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection. ICCV 2023.

### 핵심 기법
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

### 데이터셋
- Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite. CVPR 2012.
- Sakaridis, C., Dai, D., & Van Gool, L. (2018). Semantic foggy scene understanding with synthetic data. IJCV.

## Citation

본 연구를 인용하실 경우 다음과 같이 표기해주시기 바랍니다:

```bibtex
@inproceedings{yourname2025monodetr,
  title={악천후 환경에서 단안 3D 객체 검출 성능 향상을 위한 MonoDETR 기반 기법},
  author={Your Name and Co-authors},
  booktitle={한국정보과학회 학술발표논문집 (KCC 2025)},
  year={2025}
}
```

## 라이센스

본 프로젝트는 학술 연구 목적으로 개발되었습니다.

## 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주시기 바랍니다.

## Acknowledgements

본 연구는 [경희대학교] 에서 수행되었습니다.
