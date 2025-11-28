# MonoDETR-Weather3D

**[KCC 2025 학부생논문] 악천후 환경에서 단안 3D 객체 검출 성능 향상을 위한 MonoDETR 기반 기법**

[![Paper](https://img.shields.io/badge/Paper-Google_Drive-blue)](https://drive.google.com/your-paper-link)
[![License](https://img.shields.io/badge/License-Academic-green)]()

## 프로젝트 개요

본 프로젝트는 안개와 같은 악천후 환경에서 **MonoDETR** 모델의 3D 객체 검출 성능을 개선하기 위해 다양한 학습 전략(Fine-Tuning, Multi-Domain, Teacher-Student)을 적용하고 분석한 연구입니다.

## 주요 특징

*   **Fine-Tuning**: Clean 데이터로 사전 학습 후 Foggy 데이터로 미세 조정하여 특정 환경 성능 극대화.
*   **Multi-Domain Learning**: Clean과 Foggy 데이터를 동시에 학습하여 도메인 간 균형 잡힌 성능 달성.
*   **Teacher-Student Learning**: Clean 환경의 Teacher 모델 지식을 Foggy 환경의 Student 모델로 전이 (Knowledge Distillation).

## 설치 및 환경 설정

```bash
# Clone repository
git clone https://github.com/pulqum/MonoDETR-Weather3D.git
cd MonoDETR-Weather3D

# Install dependencies (Python 3.8+, PyTorch 1.9+, CUDA 11.1+)
pip install -r requirements.txt
```

## 데이터셋 구조

KITTI 및 Foggy KITTI 데이터셋을 아래와 같이 구성해야 합니다.

```
data/KITTI/
├── training/
│   ├── image_2/       # Clean Images
│   ├── image_2_foggy/ # Foggy Images
│   ├── calib/
│   └── label_2/
└── testing/
    ├── image_2/
    ├── image_2_foggy/
    └── calib/
```

## 실행 방법

### 학습 (Training)

```bash
# 1. Fine-Tuning (Clean -> Foggy)
python tools/train_val.py --config configs/monodetr.yaml --output_dir experiments/fine_tuning --use_foggy True --pretrain_model <path_to_clean_ckpt>

# 2. Multi-Domain Learning
python tools/train_val.py --config configs/monodetr.yaml --output_dir experiments/multi_domain --use_foggy True --clean_weight 0.5 --foggy_weight 0.5

# 3. Teacher-Student Learning
python tools/train_val.py --config configs/monodetr.yaml --output_dir experiments/teacher_student --use_foggy True --teacher_ckpt <path_to_teacher_ckpt>
```

### 평가 (Testing)

```bash
python tools/test.py --config configs/monodetr.yaml --checkpoint_path <path_to_ckpt> --test_foggy True
```

## 실험 결과 (KITTI Val / Car / Moderate)

| Method | Clean AP3D | Foggy AP3D | 비고 |
|:---:|:---:|:---:|:---|
| **Clean Only** | 15.71 | 6.39 | 기준 모델 |
| **Fine-Tuning** | 12.61 | **17.15** | Foggy 성능 최고, Clean 성능 하락 |
| **Multi-Domain** | 14.67 | 15.35 | 균형 잡힌 성능 (Robust) |

## Citation

```bibtex
@inproceedings{yourname2025monodetr,
  title={악천후 환경에서 단안 3D 객체 검출 성능 향상을 위한 MonoDETR 기반 기법},
  booktitle={KCC 2025},
  year={2025}
}
```

## Acknowledgements

본 연구는 [경희대학교] 에서 수행되었습니다.
