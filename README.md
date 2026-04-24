# MAI-standard

> 국립부경대학교 의료인공지능(MAI) 연구실 표준 개발 스택 및 컨벤션

---

## 공식 스택

### 딥러닝용
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-000000?style=flat-square&logo=github&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

### 의료 이미징용
![MONAI](https://img.shields.io/badge/MONAI-40A8D0?style=flat-square&logoColor=white)
![SimpleITK](https://img.shields.io/badge/SimpleITK-5C5C5C?style=flat-square&logoColor=white)
![nilearn](https://img.shields.io/badge/nilearn-4B8BBE?style=flat-square&logoColor=white)

### 실험 관리용
![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)
![Hydra](https://img.shields.io/badge/Hydra-89B4FA?style=flat-square&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)
![tmux](https://img.shields.io/badge/tmux-1BB91F?style=flat-square&logo=tmux&logoColor=white)

---

## 목차

- [환경 관리](#환경-관리)
- [프로젝트 관리](#프로젝트-관리)
- [Best Practice](#best-practice)
- [코드 컨벤션](#코드-컨벤션)
- [실험 관리](#실험-관리)
- [LLM 도구](#llm-도구)

---

## 환경 관리

| 항목 | 도구 | 비고 |
|------|------|------|
| OS | `Ubuntu` | 24.04 |
| 패키지 매니저 | `uv` | 기본. 빠른 설치 및 lockfile 관리 |
| 대체 환경 | `conda` | uv 오류 시 (e.g. CUDA 버전 관리 필요) |
| Python | `3.10+` | 3.10 권장|
| CUDA | `12.x` | cu121 기준 |

```bash
uv init
uv venv --python 3.10
source .venv/bin/activate
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv add wandb hydra-core omegaconf
```

---
## 의료 이미징

### 모달리티별 스택

| 모달리티 | 패키지 | 주요 용도 |
|----------|--------|-----------|
| CT / MRI (sMRI) | `SimpleITK`, `nibabel` | nii.gz/DICOM 로드, 전처리|
| fMRI | `nilearn`, `nibabel` | ROI 추출, FC matrix, connectome 시각화 |

---

## 프로젝트 관리

| 역할 | 도구 |
|------|------|
| 실험 추적 | WandB |
| 코드 버전 관리 | GitHub |
| 문서 / 회의록 | Notion |
| 논문 타깃 | MICCAI, TMI, MedIA ... |

### WandB 초기화 표준

```python
wandb.init(
    entity="<entity-name>",
    project="<project-name>",
    name=f"{cfg.model}_{cfg.exp_id}",
    tags=[cfg.model, cfg.data.dataset, cfg.exp_id, ... etc],
    config=OmegaConf.to_container(cfg, resolve=True),
)
```

---

## best practice

```
<project-name>/
├── configs/                    # 실험 설정 파일 모음
│   ├── default.yaml            # 기본 하이퍼파라미터 설정
│   └── exp/
│       └── ablation_01.yaml    # Ablation 실험별 오버라이드 설정
├── data/                       # 데이터 로딩 및 전처리
│   ├── __init__.py
│   ├── dataset0.py             # 데이터셋 이름에 맞게 설정
│   ├── dataset1.py             # 데이터셋 이름에 맞게 설정 (opt.)
│   └── transforms.py           # 데이터 증강 / 변환 정의
├── models/                     # 모델 아키텍처
│   ├── __init__.py
│   ├── backbone.py             # Feature extractor (encoder)
│   └── head.py                 # Task-specific output head
├── utils/                      # 공용 유틸리티
│   ├── __init__.py
│   ├── logger.py               # WandB 래퍼
│   ├── metrics.py              # 평가 지표 계산
│   └── visualization.py        # 결과 시각화
├── scripts/
│   ├── run_ablation.sh         # Ablation 실험 일괄 실행 스크립트
│   └── run_{train, eval}.sh    # train, eval 실행 스크립트
├── outputs/                    # 실험 결과 저장 — .gitignore에 추가
│   ├── checkpoints/
│   │   └── <exp_name>/
│   │       ├── best.ckpt
│   │       └── last.ckpt
│   └── logs/
│       └── <exp_name>.log
├── .gitignore                  # Git 추적 제외 파일 목록
├── .claudeignore               # Claude Code 컨텍스트 제외 파일 목록 (opt.)
├── __init__.py                 # 루트 패키지 초기화
├── main.py                     # 진입점 — config 로드, mode dispatch, seed 설정
├── train.py                    # Trainer 클래스 (학습 루프)
├── eval.py                     # Evaluator 클래스 (평가 루프)
├── AGENT.md                    # Claude Code용 프로젝트 가이드 (opt.)
├── pyproject.toml              # 패키지 의존성 및 빌드 설정 (opt.)
└── README.md                   # 프로젝트 개요 및 사용법

data/                           # 프로젝트와 같은 SSD/HDD에 위치 — IO 최소화
    ├── dataset0/               # 데이터셋 이름에 맞게 설정
    │   └── path/to/data/
    └── dataset1/               # (opt.)
        └── path/to/data/
```

---

## 코드 컨벤션

### 설정 관리

Hydra 기반 config 관리를 표준으로 사용.

```yaml
# configs/default.yaml
model:
  name: vit_small_patch16_224
  pretrained: true

train:
  epochs: 100
  batch_size: 32
  lr: 1e-4
  optimizer: adamw

data:
  dataset: chestmnist
  label_ratio: 0.1
  img_size: 224
```

```python
@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    ...
```

```bash
# CLI 오버라이드
python train.py train.lr=1e-3 +exp=ablation_01
```

### 네이밍 컨벤션

| 항목 | 규칙 | 예시 |
|------|------|------|
| 파일명 | `snake_case` | `vq_vae_encoder.py` |
| 클래스 | `PascalCase` | `VQVAEEncoder` |
| 함수/변수 | `snake_case` | `forward_pass` |
| 상수 | `UPPER_SNAKE` | `MAX_EPOCH` |
| WandB run name | `{model}_{exp_id}` | `uvit_cpr4_n3` |

---

## LLM 도구

### AGENT.md

각 프로젝트 루트에 `AGENT.md`를 작성하여 LLM 에이전트가 프로젝트 맥락을 파악하도록 한다.

```markdown
# AGENT.md — <project-name>
업데이트 예정
```

> AGENT.md는 LLM이 MAI 표준에 맞는 코드를 생성하도록 유도하는 안전장치 역할을 한다.

---

## 참고 링크

- [WandB](https://wandb.ai/)
- [MONAI 공식 문서](https://docs.monai.io/)
- [Hydra 공식 문서](https://hydra.cc/docs/intro/)
- [nilearn 공식 문서](https://nilearn.github.io/)
- [Hugging Face](https://huggingface.co/)
- [박연수 개인 블로그 ㅎㅎ](https://kitewatermelon.github.io)

---

*MAI-standard v0.1 — 국립부경대학교 의료인공지능 연구실*