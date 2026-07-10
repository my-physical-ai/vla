#!/usr/bin/env python3
# LeKiwi ACT 학습 상태 종합 평가 스크립트 v1 (merged_40 체크포인트 전용)
# 작성일: 2026-04-23
# 작성자: 빅맥 / ZETA Satellite Robotics
# 기반: eval_train_state_smolvla_v15_E.py (SmolVLA용 v2-E)
#
# ═══════════════════════════════════════════════════════════════════════
# 🎯 이 스크립트의 목적
# ═══════════════════════════════════════════════════════════════════════
#
# LeKiwi ACT 정책이 학습된 체크포인트를 실전 배포 전에 진단합니다.
# 정책이 Vision/State/Action을 올바르게 학습했는지, Mode averaging이나
# Mode collapse가 없는지를 정량적으로 측정합니다.
#
# [SmolVLA vs ACT 차이 — 이 스크립트의 조정점]
#   1. SmolVLA는 언어(task) 조건 → ACT는 언어 없음 (삭제)
#   2. SmolVLA는 6DOF arm → ACT(LeKiwi)는 3DOF mobile (x, y, θ)
#   3. 평가 7 "5문장 일관성" → 평가 7 "Observation 일관성"으로 대체
#   4. Joint 이름/범위 모두 로봇 주행 단위(m/s, deg/s)로 변경
#   5. ACT 원본 설정 검증 추가 (dim_model=512, kl_weight=10)
#
# ═══════════════════════════════════════════════════════════════════════
# 📚 글로벌 레퍼런스 (2023~2026 최신 연구)
# ═══════════════════════════════════════════════════════════════════════
#
# ACT 논문 (Zhao et al. 2023, arXiv 2304.13705):
#   "Chunking amortizes compound errors. Typical batch 8, lr 1e-5,
#    chunk_size 100, dim_model 512, kl_weight 10"
#
# SVRC "ACT Policy Explained" (2026):
#   "Action chunking breaks the compounding error cycle by predicting
#    a sequence of k future actions. Because the plan was generated
#    from a single consistent observation, trajectory is smooth."
#
# Emergent Mind "ACT" (2026):
#   "Typical training: 50 demos, batch 32-64, Adam(W), hours on GPU"
#
# arXiv 2603.11642 "Chunk-Boundary Artifact" (2026):
#   "At the boundary where one chunk ends and the next begins,
#    the policy must replan from a new observation context,
#    often producing visible discontinuities"
#
# LeRobot GitHub Issue #2213:
#   "ACT should learn smooth approach or grasp-like motion in ~30k-100k
#    steps. Recommended minimum batch size ≥ 16"
#
# Medium (Karkada 2025):
#   "ACT evaluated on SO101 with 50 episodes, 100k steps,
#    achieved 70% task success rate. Data quality matters more
#    than quantity"
#
# ═══════════════════════════════════════════════════════════════════════
# 📊 평가 구성 (10개)
# ═══════════════════════════════════════════════════════════════════════
#
# 평가 1: 체크포인트 메타 정보 (스텝, 스케줄러, 체크포인트 수)
# 평가 2: ACT 핵심 설정 검증 (chunk_size, n_action_steps, dim_model 등)
# 평가 3: Vision 인코더 반응 (2개 카메라 각각)
# 평가 4: State 인코더 반응 (주행 상태 10가지)
# 평가 5: Chunk 품질 (boundary smoothness, 안전 범위)
# 평가 6: Vision 다양성 → Mode averaging 진단
# 평가 7: Observation 일관성 (언어 없는 ACT 특화)
# 평가 8: State 다양성 → Mode collapse 진단 (⭐ 핵심)
# 평가 9: Action range 안전성 (속도 한계)
# 평가 10: 체크포인트별 Sweet spot 탐색
#
# ═══════════════════════════════════════════════════════════════════════

import os
import sys
import json
import glob
import warnings
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
warnings.filterwarnings("ignore")

LOG_DIR = os.path.expanduser("~/lerobot_outputs/eval_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR,
    f"act_eval_lekiwi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


class Tee:
    """터미널 출력 + 파일 동시 저장 (원본 스크립트와 동일한 로깅 패턴)"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Tee(LOG_FILE)

import torch
import numpy as np

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# ═══════════════════════════════════════════════════════════════════════
# ⚙️  설정 — 체크포인트 경로와 카메라 키
# ═══════════════════════════════════════════════════════════════════════
# 체크포인트 경로는 학습 스크립트의 --output_dir와 일치해야 함
CKPT_DIR   = "/home/zetabank/lerobot_outputs/act_lekiwi_merged_40_v7_batch32"
CKPT       = os.path.join(CKPT_DIR, "checkpoints/last/pretrained_model")
DATASET_DIR = "/home/zetabank/lerobot_datasets/lekiwi_act_merged_40"

# LeKiwi 카메라 키 (record 시 사용한 이름과 일치)
# lerobot.robots.lekiwi의 LeKiwiClient에서 사용하는 기본 이름
CAM_FRONT = "observation.images.front"  # Pi5 /dev/video0 (lekiwi_host)
CAM_WRIST = "observation.images.wrist"  # 필요시 (보통 LeKiwi는 front만)

# [주의] 실제 데이터셋의 카메라 키를 자동으로 탐지하는 로직이 아래 있음

# LeKiwi 조인트(실제로는 속도 채널)
# x: 전후 속도 (m/s), y: 측면 속도 (m/s), theta: 회전 속도 (deg/s)
ACTION_NAMES = ["vx (m/s)", "vy (m/s)", "vtheta (deg/s)"]
ACTION_SAFE_RANGE = [(-0.5, 0.5), (-0.5, 0.5), (-120, 120)]  # 안전 속도 한계

VISION_THRESHOLD = 0.01      # 속도 단위라 SmolVLA(0.1°)보다 엄격
STATE_THRESHOLD = 0.02       # 10배 엄격 (속도는 작은 값)
MODE_AVG_THRESHOLD = 0.05    # chunk 최종 분산 임계값 (m/s 기준)
EXPECTED_CHUNK_SIZE = 50     # LeKiwi 15fps × 50 = 3.3초

print("=" * 70)
print("  🤖 LeKiwi ACT 학습 상태 종합 평가 v1 (merged_40 전용)")
print(f"  체크포인트: {CKPT_DIR.split('/')[-1]}")
print(f"  데이터셋:   {DATASET_DIR.split('/')[-1]}")
print(f"  목적:       Mode averaging / collapse / Vision 반응 정량화")
print(f"  로그 저장:  {LOG_FILE}")
print("=" * 70)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸리티 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_camera_keys(ckpt_path: str) -> list:
    """체크포인트 config.json에서 실제 카메라 키를 추출한다.

    [근거] LeRobot ACTConfig는 input_features에 카메라 shape를 저장한다.
           따라서 카메라 이름 오타를 사전 방지할 수 있다.

    Returns:
        카메라 키 목록 (예: ["observation.images.front"])
    """
    config_file = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_file):
        print(f"  ⚠️  config.json 없음 → 기본 카메라 키 사용")
        return [CAM_FRONT]

    with open(config_file) as f:
        cfg = json.load(f)

    input_features = cfg.get("input_features", {})
    cam_keys = [k for k in input_features.keys() if "images" in k]
    if not cam_keys:
        cam_keys = [CAM_FRONT]
    print(f"  ℹ️  감지된 카메라 키: {cam_keys}")
    return cam_keys


def rand_img(seed: int) -> torch.Tensor:
    """재현 가능한 랜덤 이미지 생성."""
    torch.manual_seed(seed)
    return torch.rand(1, 3, 480, 640).cuda()


def make_meaningful_test_imgs() -> list:
    """다양한 이미지 생성 (실제 데이터셋 + 극단 케이스).

    [근거] SmolVLA 원본 스크립트와 동일한 전략. 실제 데이터셋 프레임 5장 +
           단색/체커보드 등 극단 케이스로 Vision 반응성 측정.
    """
    imgs = []
    real_imgs = sorted(
        glob.glob(f"{DATASET_DIR}/**/*.jpg", recursive=True)
    )[:5]
    if not real_imgs:
        real_imgs = sorted(
            glob.glob(f"{DATASET_DIR}/**/*.png", recursive=True)
        )[:5]
    if real_imgs:
        from torchvision import transforms
        from PIL import Image
        to_tensor = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
        ])
        for path in real_imgs:
            try:
                img = Image.open(path).convert("RGB")
                t = to_tensor(img).unsqueeze(0).cuda()
                imgs.append(t)
                if len(imgs) >= 5:
                    break
            except Exception:
                pass
        if imgs:
            print(f"  ℹ️  실제 데이터셋 이미지 {len(imgs)}장 로드 완료")

    # 극단 케이스 추가 (Vision 인코더의 민감도 테스트)
    imgs.append(torch.ones(1, 3, 480, 640).cuda())   # 완전 흰색
    imgs.append(torch.zeros(1, 3, 480, 640).cuda())  # 완전 검정
    r = torch.zeros(1, 3, 480, 640).cuda(); r[0, 0] = 1.0; imgs.append(r)
    g = torch.zeros(1, 3, 480, 640).cuda(); g[0, 1] = 1.0; imgs.append(g)
    b = torch.zeros(1, 3, 480, 640).cuda(); b[0, 2] = 1.0; imgs.append(b)

    return imgs[:10]


def reset_policy_state(p):
    """ACT 내부 action queue 초기화.

    [근거] ACT는 temporal ensembling을 위해 내부 queue를 유지한다.
           평가 간 간섭 방지를 위해 매번 초기화 필요.
           (LeRobot ACTPolicy.reset() 또는 _action_queue.clear())
    """
    if hasattr(p, "reset") and callable(getattr(p, "reset", None)):
        try:
            p.reset()
            return
        except Exception:
            pass
    if hasattr(p, "_action_queue"):
        p._action_queue.clear()


def get_action(policy, preprocess, obs_imgs: dict, state=None):
    """ACT select_action — 한 스텝 action 예측.

    Args:
        obs_imgs: {cam_key: tensor} 딕셔너리 (카메라 여러 개 지원)
        state: [1, state_dim] 또는 None (None이면 0벡터)

    Returns:
        action numpy array [action_dim]
    """
    if state is None:
        state = torch.zeros(1, 3).cuda()  # LeKiwi는 3DOF
    reset_policy_state(policy)

    batch = {**obs_imgs, "observation.state": state}
    batch = preprocess(batch)
    with torch.no_grad():
        return policy.select_action(batch).cpu().numpy()[0]


def get_chunk(policy, preprocess, obs_imgs: dict, state=None):
    """ACT predict_action_chunk — chunk_size 만큼의 action 시퀀스.

    [근거] ACT의 핵심은 chunk 예측. 이를 통해 궤적의 다양성, 내부 분산,
           최종점 분산 등을 측정하여 Mode collapse/averaging을 진단한다.

    Returns:
        chunk numpy array [chunk_size, action_dim]
    """
    if state is None:
        state = torch.zeros(1, 3).cuda()
    reset_policy_state(policy)

    batch = {**obs_imgs, "observation.state": state}
    batch = preprocess(batch)
    with torch.no_grad():
        return policy.predict_action_chunk(batch)[0].cpu().numpy()


# ── 등급 함수 (SmolVLA 원본에서 이어받아 ACT 단위로 조정) ──────────────

def grade_vision(responding: int, total: int) -> tuple:
    """Vision 반응 수 기반 등급."""
    if responding == 0:
        return "F", "❌ Vision 완전 미반응 (인코더 학습 실패)"
    elif responding <= int(total * 0.3):
        return "D", "⚠️  Vision 극초기 반응 (불충분)"
    elif responding <= int(total * 0.6):
        return "C", "⚠️  Vision 부분 반응"
    elif responding <= int(total * 0.8):
        return "B", "✅ Vision 상당 부분 반응"
    else:
        return "A", "✅ Vision 충분히 반응"


def grade_mode_avg(final_std: float) -> tuple:
    """Mode averaging 등급 (LeKiwi 속도 단위 m/s·deg/s 기준).

    [근거] SmolVLA는 조인트 각도(°) 단위 → LeKiwi는 속도 단위.
           ACT 원본 chunk_size=50에서 보통 chunk 최종점 분산은
           0.01~0.1 m/s 수준이 정상.
    """
    if final_std < 0.005:
        return "F", "❌ Mode averaging 심각 (모든 이미지에 같은 궤적)"
    elif final_std < 0.01:
        return "D", "⚠️  Mode averaging 강함 (Vision 반응 약)"
    elif final_std < 0.02:
        return "C", "⚠️  모드 분리 약함"
    elif final_std < 0.05:
        return "B", "✅ 모드 분리 양호"
    else:
        return "A", "✅ 모드 분리 뚜렷 (다양한 궤적)"


def grade_mode_collapse(state_std: float, chunk_range: float) -> tuple:
    """Mode collapse 등급 (⭐ 핵심 지표).

    [근거] state가 변해도 chunk 궤적이 비슷하면 collapse.
           LeKiwi 주행 속도 범위 고려 (±0.5 m/s).

    Args:
        state_std: state 변이에 따른 chunk 최종점 분산
        chunk_range: chunk 내부 움직임 범위 (시작~끝 이동)
    """
    if state_std < 0.005 and chunk_range < 0.05:
        return "F", "❌ Mode collapse 심각 (자세 무관 단일 경직 궤적)"
    elif state_std < 0.01 and chunk_range < 0.1:
        return "D", "⚠️  Mode collapse 일부 (공간 적응력 부족)"
    elif state_std < 0.02:
        return "C", "⚠️  공간 적응력 약함"
    elif state_std < 0.05:
        return "B", "✅ 공간 적응력 양호"
    else:
        return "A", "✅ 공간 적응력 뛰어남"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 1: 체크포인트 메타 정보
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 1: 체크포인트 메타 정보")
print("─" * 70)

scores = {}

step_file = os.path.join(
    CKPT_DIR, "checkpoints/last/training_state/training_step.json"
)
if os.path.exists(step_file):
    with open(step_file) as f:
        step_data = json.load(f)
    actual_step = step_data.get("training_step", 0)
    print(f"  실제 학습 완료 스텝: {actual_step:,}")

    # ACT 권장: 최소 30k, 권장 80k~100k (Issue #2213)
    if actual_step < 20000:
        print(f"    └─ ⚠️  20,000 미만 (권장: 80,000 이상)")
    elif actual_step < 50000:
        print(f"    └─ 🔶 중간 단계 — 수렴 미달 가능")
    elif actual_step < 80000:
        print(f"    └─ ✅ 적정 범위 (일반적 수렴)")
    elif actual_step <= 100000:
        print(f"    └─ ✅ 충분히 학습됨")
    else:
        print(f"    └─ ℹ️  100k 이상 — overfitting 주의")
    scores["actual_step"] = actual_step
else:
    print("  ⚠️  training_step.json 없음")
    scores["actual_step"] = 0

sched_file = os.path.join(
    CKPT_DIR, "checkpoints/last/training_state/scheduler_state.json"
)
if os.path.exists(sched_file):
    with open(sched_file) as f:
        sched_data = json.load(f)
    last_epoch = sched_data.get("last_epoch", "?")
    last_lr = sched_data.get("_last_lr", ["?"])[0]
    print(f"  스케줄러 last_epoch: {last_epoch:,}  |  마지막 lr: {last_lr}")
    scores["scheduler_ok"] = True
else:
    print("  ⚠️  scheduler_state.json 없음")
    scores["scheduler_ok"] = False

ckpt_root = os.path.join(CKPT_DIR, "checkpoints")
if os.path.exists(ckpt_root):
    ckpts = sorted(
        [d for d in os.listdir(ckpt_root) if d.isdigit()],
        key=lambda x: int(x)
    )
    print(f"\n  저장된 체크포인트 ({len(ckpts)}개): {ckpts[:5]}..."
          if len(ckpts) > 5 else
          f"\n  저장된 체크포인트 ({len(ckpts)}개): {ckpts}")
    scores["num_checkpoints"] = len(ckpts)
else:
    print("  ⚠️  체크포인트 디렉토리 없음")
    ckpts = []
    scores["num_checkpoints"] = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 로딩 (카메라 키 자동 감지 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 모델 로딩 중...")
if not os.path.exists(CKPT):
    print(f"  ❌ 체크포인트 경로 없음: {CKPT}")
    sys.exit(1)

cam_keys = detect_camera_keys(CKPT)
policy = ACTPolicy.from_pretrained(CKPT).to("cuda").eval()
preprocess, _ = make_pre_post_processors(
    policy.config, CKPT,
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
print("  ✅ 완료\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 2: ACT 핵심 설정 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 평가 2: ACT 핵심 설정 검증")
print("  근거: ACT 원본 논문 (Zhao et al. 2023) 권장값")
print("─" * 70)

cfg = policy.config
chunk_size = getattr(cfg, "chunk_size", None)
n_action_steps = getattr(cfg, "n_action_steps", None)
n_obs_steps = getattr(cfg, "n_obs_steps", None)
dim_model = getattr(cfg, "dim_model", None)
dim_feedforward = getattr(cfg, "dim_feedforward", None)
n_encoder_layers = getattr(cfg, "n_encoder_layers", None)
n_decoder_layers = getattr(cfg, "n_decoder_layers", None)
n_heads = getattr(cfg, "n_heads", None)
use_vae = getattr(cfg, "use_vae", None)
kl_weight = getattr(cfg, "kl_weight", None)

print(f"  chunk_size:       {chunk_size}  (ACT 원본: 100, LeKiwi: 50)")
print(f"  n_action_steps:   {n_action_steps}   (기대: chunk_size와 동일 = {chunk_size})")
print(f"  n_obs_steps:      {n_obs_steps}   (기대: 1)")
print(f"  dim_model:        {dim_model}  (ACT 원본: 512)")
print(f"  dim_feedforward:  {dim_feedforward}  (ACT 원본: 3200)")
print(f"  n_encoder_layers: {n_encoder_layers}  (ACT 원본: 4)")
print(f"  n_decoder_layers: {n_decoder_layers}  (ACT 원본: 1)")
print(f"  n_heads:          {n_heads}  (ACT 원본: 8)")
print(f"  use_vae:          {use_vae}  (ACT 원본: True)")
print(f"  kl_weight:        {kl_weight}  (ACT 원본: 10)")

# 설정 등급
config_issues = []
if n_action_steps != chunk_size:
    config_issues.append(f"n_action_steps({n_action_steps}) ≠ chunk_size({chunk_size})")
if dim_model != 512:
    config_issues.append(f"dim_model({dim_model}) ≠ 512 (원본)")
if use_vae is False:
    config_issues.append("use_vae=False (VAE 비활성화, 학습 안정성 저하 가능)")

if len(config_issues) == 0:
    config_grade = "A"
    print(f"\n  ✅ 모든 설정이 ACT 원본과 일치  등급: A")
elif len(config_issues) <= 1:
    config_grade = "B"
    print(f"\n  🔶 1개 설정 비표준:")
    for issue in config_issues:
        print(f"     - {issue}")
elif len(config_issues) <= 2:
    config_grade = "C"
    print(f"\n  ⚠️  2개 설정 비표준:")
    for issue in config_issues:
        print(f"     - {issue}")
else:
    config_grade = "D"
    print(f"\n  ❌ 다수 설정 비표준 — 성능 저하 가능:")
    for issue in config_issues:
        print(f"     - {issue}")

scores["config_grade"] = config_grade
scores["n_action_steps"] = n_action_steps
scores["chunk_size"] = chunk_size

# 기본 관측 설정 (이후 평가에 공통 사용)
BASE_IMG = rand_img(0)
BASE_STATE = torch.zeros(1, 3).cuda()  # LeKiwi는 3DOF 속도

# 카메라별 관측 딕셔너리 생성 유틸
def make_obs(img):
    """감지된 모든 카메라 키에 같은 이미지를 할당 (테스트용)."""
    return {k: img for k in cam_keys}

base_obs = make_obs(BASE_IMG)
base_act = get_action(policy, preprocess, base_obs, BASE_STATE)
print(f"\n  기준 action (BASE_IMG + 0 state): {base_act}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 3: Vision 인코더 반응
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 3: Vision 인코더 반응 (다양한 이미지)")
print("  근거: ACT는 ResNet-18 백본으로 이미지를 인코딩. 다양한 이미지에")
print("        충분히 반응하지 않으면 Vision 무시 학습 (degenerate policy)")
print("─" * 70)

test_imgs = make_meaningful_test_imgs()
vision_diffs = []
for i, img in enumerate(test_imgs):
    obs = make_obs(img)
    a = get_action(policy, preprocess, obs)
    diff = np.abs(a - base_act).max()
    vision_diffs.append(diff)
    flag = "✅" if diff > VISION_THRESHOLD else "❌"
    print(f"  img_{i+1:02d}: Δmax={diff:8.5f}  {flag}")

vision_responding = sum(1 for d in vision_diffs if d > VISION_THRESHOLD)
vision_mean = np.mean(vision_diffs)
vision_max = np.max(vision_diffs)
vision_grade, vision_msg = grade_vision(vision_responding, len(test_imgs))
print(f"\n  Vision 반응: {vision_responding}/{len(test_imgs)}  "
      f"평균: {vision_mean:.5f}  최대: {vision_max:.5f}")
print(f"  등급: {vision_grade}  {vision_msg}")

scores.update({
    "vision_grade": vision_grade,
    "vision_responding": vision_responding,
    "vision_mean": float(vision_mean),
    "vision_max": float(vision_max),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 4: State 인코더 반응 (LeKiwi 주행 상태)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 4: State 인코더 반응 (LeKiwi 주행 상태 10가지)")
print("  근거: LeKiwi는 현재 속도(vx, vy, vtheta)를 state로 받음.")
print("        다양한 속도 상태에서 action이 달라져야 정상 (공간 적응력)")
print("─" * 70)

# LeKiwi 주행 상태 케이스 (m/s, m/s, deg/s)
state_cases = [
    ("정지",          [0.0,    0.0,   0.0]),
    ("전진 저속",     [0.1,    0.0,   0.0]),
    ("전진 고속",     [0.3,    0.0,   0.0]),
    ("후진",          [-0.1,   0.0,   0.0]),
    ("좌측 이동",     [0.0,   -0.1,   0.0]),
    ("우측 이동",     [0.0,    0.1,   0.0]),
    ("좌회전",        [0.0,    0.0, -30.0]),
    ("우회전",        [0.0,    0.0,  30.0]),
    ("전진 + 좌회전", [0.1,    0.0, -30.0]),
    ("전진 + 우회전", [0.1,    0.0,  30.0]),
]

state_diffs = []
for name, vals in state_cases:
    st = torch.tensor([vals]).float().cuda()
    a = get_action(policy, preprocess, base_obs, st)
    d = np.abs(a - base_act).max()
    state_diffs.append(d)
    flag = "✅" if d > STATE_THRESHOLD else "❌"
    print(f"  {name:14s}: Δmax={d:8.5f}  {flag}")

state_responding = sum(1 for d in state_diffs if d > STATE_THRESHOLD)
state_mean = np.mean(state_diffs)
print(f"\n  반응: {state_responding}/10  평균: {state_mean:.5f}")

if state_responding >= 8:
    state_grade = "A"
    print("  ✅ State 인코더 정상 (공간 적응력 우수)")
elif state_responding >= 5:
    state_grade = "B"
    print("  ✅ State 인코더 양호")
elif state_responding >= 3:
    state_grade = "C"
    print("  ⚠️  State 반응 약함")
else:
    state_grade = "D"
    print("  ❌ State 무시 학습 — 데이터 다양성 부족 의심")

scores.update({
    "state_grade": state_grade,
    "state_responding": state_responding,
    "state_mean": float(state_mean),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 5: Chunk 품질 (boundary smoothness)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 5: Chunk 품질 — Boundary Smoothness")
print("  근거: arXiv 2603.11642 'Chunk-Boundary Artifact' (2026)")
print("        chunk 내부는 부드러운 궤적이어야 하며, step간 변화가 급격하면")
print("        실전에서 jitter 발생 위험")
print("─" * 70)

chunk = get_chunk(policy, preprocess, base_obs, BASE_STATE)

step_diffs = [np.abs(chunk[t] - chunk[t-1]).max() for t in range(1, len(chunk))]
max_step = np.max(step_diffs)
mean_step = np.mean(step_diffs)

# Action range 안전성 점검
oor = 0
for t in range(len(chunk)):
    for j in range(len(ACTION_NAMES)):
        lo, hi = ACTION_SAFE_RANGE[j]
        if not (lo <= chunk[t, j] <= hi):
            oor += 1

print(f"  청크 길이: {len(chunk)}  (기대: {EXPECTED_CHUNK_SIZE})")
print(f"  Step간 최대 변화: {max_step:.5f}  (15fps 기준)")
print(f"  Step간 평균 변화: {mean_step:.5f}")
print(f"  안전 범위 초과: {oor}건")

# LeKiwi 기준: 15fps에서 한 step(0.067초)당 변화가 0.05 이상이면 급격
if max_step < 0.02 and oor == 0:
    chunk_grade = "A"
    print("  ✅ 부드러운 궤적, 안전 범위 내")
elif max_step < 0.05 and oor == 0:
    chunk_grade = "B"
    print("  ✅ 허용 가능 범위")
elif max_step < 0.1:
    chunk_grade = "C"
    print("  ⚠️  다소 급격한 변화")
else:
    chunk_grade = "D"
    print("  ❌ 급격한 변화 — 실전 jitter 위험")

scores["chunk_grade"] = chunk_grade
scores["max_step_diff"] = float(max_step)
scores["chunk_len"] = len(chunk)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 6: Vision 다양성 → Mode averaging 진단
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 6: ⭐ Mode Averaging 진단 (Vision 다양성)")
print("  근거: arXiv 2410.18647 'Data Scaling Laws' — 과다양성은 평균 궤적 유발")
print("  측정: 다른 이미지 5장에 대한 chunk 최종점 분산")
print("        낮으면 → 모든 입력에 같은 궤적 (Vision 무시)")
print("★" * 70)

chunks_per_img = []
for img_idx, img in enumerate(test_imgs[:5]):
    obs = make_obs(img)
    c = get_chunk(policy, preprocess, obs, BASE_STATE)
    chunks_per_img.append(c)
chunks_img_arr = np.array(chunks_per_img)  # [5, chunk_size, 3]

final_points = chunks_img_arr[:, -1, :]  # 마지막 step의 action
mid_points = chunks_img_arr[:, len(chunk)//2, :]
final_std = np.std(final_points, axis=0).mean()
mid_std = np.std(mid_points, axis=0).mean()
traj_std = np.std(chunks_img_arr, axis=0).mean()

print(f"\n  📊 측정 결과:")
print(f"     최종점 분산 (5장): {final_std:.5f}")
print(f"     중간점 분산 (5장): {mid_std:.5f}")
print(f"     전체 궤적 분산:    {traj_std:.5f}")

mode_grade, mode_msg = grade_mode_avg(final_std)
print(f"\n  Mode averaging 등급: {mode_grade}  {mode_msg}")

scores.update({
    "mode_avg_grade": mode_grade,
    "final_std": float(final_std),
    "mid_std": float(mid_std),
    "traj_std": float(traj_std),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 7: Observation 일관성 (ACT는 언어 없으므로 SmolVLA 평가 7 대체)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 7: Observation 일관성 (재현성)")
print("  근거: ACT는 결정론적 추론. 같은 입력에 같은 출력이 나와야 함")
print("        (단, VAE latent 샘플링이 있어 미세한 분산은 정상)")
print("  측정: 같은 이미지 + 같은 state 10회 반복 → action 분산")
print("★" * 70)

repeat_actions = []
obs_for_repeat = make_obs(BASE_IMG)
for i in range(10):
    a = get_action(policy, preprocess, obs_for_repeat, BASE_STATE)
    repeat_actions.append(a)
repeat_arr = np.array(repeat_actions)
repeat_std = np.std(repeat_arr, axis=0).mean()

print(f"\n  10회 반복 action 표준편차: {repeat_std:.6f}")

if repeat_std < 0.001:
    rep_grade = "A"
    print("  ✅ 완전 결정론적 (VAE 없거나 매우 안정)")
elif repeat_std < 0.005:
    rep_grade = "A"
    print("  ✅ 거의 결정론적 (VAE latent 노이즈 미미)")
elif repeat_std < 0.02:
    rep_grade = "B"
    print("  ✅ 허용 가능한 VAE 샘플링 분산")
elif repeat_std < 0.05:
    rep_grade = "C"
    print("  🔶 VAE 분산 다소 큼 (Mode 여러 개 가능성)")
else:
    rep_grade = "D"
    print("  ❌ 같은 입력에도 다른 출력 → 학습 불안정")

scores.update({
    "consistency_grade": rep_grade,
    "consistency_mean": float(repeat_std),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 8: State 다양성 → Mode COLLAPSE 진단 (⭐⭐ 핵심)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 8: ⭐⭐ Mode COLLAPSE 진단 (State 다양성)")
print("  근거: SmolVLA v14 실패 사례 — '수집 때 단조로움이 단조로운 행동으로'")
print("        state가 변해도 비슷한 chunk만 생성되면 공간 적응력 부족")
print("  측정 1: 다른 state 5가지 → chunk 최종점 분산")
print("  측정 2: chunk 내부 움직임 범위 (시작~끝 이동량)")
print("★" * 70)

# 여러 주행 상태에서 chunk 추출
test_states_for_collapse = [
    torch.tensor([[0.0,    0.0,   0.0]]).float().cuda(),   # 정지
    torch.tensor([[0.1,    0.0,   0.0]]).float().cuda(),   # 전진 저속
    torch.tensor([[0.2,    0.0,  20.0]]).float().cuda(),   # 전진 + 우회전
    torch.tensor([[-0.1,   0.0,  -20.0]]).float().cuda(),  # 후진 + 좌회전
    torch.tensor([[0.0,    0.1,   0.0]]).float().cuda(),   # 측면 이동
]
chunks_per_state = []
for st in test_states_for_collapse:
    c = get_chunk(policy, preprocess, make_obs(test_imgs[0]), st)
    chunks_per_state.append(c)
chunks_state_arr = np.array(chunks_per_state)  # [5, chunk_size, 3]

state_final_std = np.std(chunks_state_arr[:, -1, :], axis=0).mean()
state_traj_std = np.std(chunks_state_arr, axis=0).mean()

# chunk 내부 움직임 범위
chunk_ranges = []
for c in chunks_per_img:
    joint_range = np.max(c, axis=0) - np.min(c, axis=0)
    chunk_ranges.append(joint_range.mean())
avg_chunk_range = np.mean(chunk_ranges)
internal_std = np.std(chunks_img_arr, axis=1).mean()

print(f"\n  📊 측정 1: State 변이에 따른 chunk 반응")
print(f"     최종점 분산:   {state_final_std:.5f}")
print(f"     궤적 분산:     {state_traj_std:.5f}")
print(f"     → 낮으면 Mode collapse (자세 달라도 같은 궤적)")

print(f"\n  📊 측정 2: Chunk 내부 움직임 범위")
print(f"     평균 움직임:   {avg_chunk_range:.5f}")
print(f"     내부 분산:     {internal_std:.5f}")
print(f"     → 낮으면 경직 궤적 (Mode collapse 증거)")

collapse_grade, collapse_msg = grade_mode_collapse(
    state_final_std, avg_chunk_range
)
print(f"\n  Mode collapse 등급: {collapse_grade}  {collapse_msg}")

scores.update({
    "collapse_grade": collapse_grade,
    "state_final_std": float(state_final_std),
    "state_traj_std": float(state_traj_std),
    "avg_chunk_range": float(avg_chunk_range),
    "internal_std": float(internal_std),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 9: Action range 안전성 (LeKiwi 속도 한계 검증)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 9: Action Range 안전성 (실전 배포 전 필수 점검)")
print("  근거: LeKiwi 안전 범위: vx,vy ∈ ±0.5 m/s, vtheta ∈ ±120 deg/s")
print("        이 범위 초과 시 모터 제한 또는 안전 이슈 발생")
print("★" * 70)

# 다양한 state에서의 모든 chunk를 flatten
all_actions = chunks_state_arr.reshape(-1, 3)  # [5*chunk_size, 3]
print(f"\n  측정 샘플 수: {len(all_actions)}")

print(f"\n  채널별 통계:")
print(f"  {'채널':<15} | {'최소':>10} | {'최대':>10} | {'평균':>10} | {'초과':>6}")
print(f"  {'-' * 65}")

total_oor = 0
for j, name in enumerate(ACTION_NAMES):
    vals = all_actions[:, j]
    lo, hi = ACTION_SAFE_RANGE[j]
    mn, mx = vals.min(), vals.max()
    mean = vals.mean()
    oor_cnt = np.sum((vals < lo) | (vals > hi))
    total_oor += oor_cnt
    flag = "✅" if oor_cnt == 0 else f"❌ {oor_cnt}"
    print(f"  {name:<15} | {mn:>10.4f} | {mx:>10.4f} | {mean:>10.4f} | {flag:>6}")

if total_oor == 0:
    range_grade = "A"
    print(f"\n  ✅ 모든 action이 안전 범위 내")
elif total_oor < len(all_actions) * 0.01:
    range_grade = "B"
    print(f"\n  🔶 일부 action이 범위 초과 ({total_oor}/{len(all_actions)})")
elif total_oor < len(all_actions) * 0.05:
    range_grade = "C"
    print(f"\n  ⚠️  5% 미만의 action이 범위 초과")
else:
    range_grade = "D"
    print(f"\n  ❌ 다수 action이 범위 초과 — 재학습 필요")

scores.update({
    "range_grade": range_grade,
    "total_oor": int(total_oor),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 10: 체크포인트별 Sweet Spot 탐색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 10: 체크포인트별 Sweet Spot 탐색")
print("  근거: 최종 체크포인트가 반드시 최선은 아님 (overfitting 가능)")
print("        각 체크포인트의 Mode averaging 지표로 최적 선택")
print("─" * 70)

if len(ckpts) >= 2:
    ckpts_to_eval = ckpts[-10:] if len(ckpts) > 10 else ckpts
    print(f"\n  평가 대상: 마지막 {len(ckpts_to_eval)}개 체크포인트\n")
    print(f"  {'체크포인트':>12} | {'최종점 분산':>14} | {'등급':>6} | {'비고':>10}")
    print(f"  {'-' * 60}")

    best_ckpt = None
    best_std = 0
    ckpt_trends = []

    for ckpt_name in ckpts_to_eval:
        ckpt_path = os.path.join(
            CKPT_DIR, "checkpoints", ckpt_name, "pretrained_model"
        )
        if not os.path.exists(ckpt_path):
            continue
        try:
            p = ACTPolicy.from_pretrained(ckpt_path).to("cuda").eval()
            pp, _ = make_pre_post_processors(
                p.config, ckpt_path,
                preprocessor_overrides={"device_processor": {"device": "cuda"}},
            )

            local_finals = []
            for img in test_imgs[:3]:
                reset_policy_state(p)
                batch = {**make_obs(img), "observation.state": BASE_STATE}
                batch = pp(batch)
                with torch.no_grad():
                    c = p.predict_action_chunk(batch)[0].cpu().numpy()
                local_finals.append(c[-1])
            local_finals = np.array(local_finals)
            local_std = np.std(local_finals, axis=0).mean()

            g, _ = grade_mode_avg(local_std)
            recommend = ""
            if local_std > best_std:
                best_std = local_std
                best_ckpt = ckpt_name
                recommend = "⭐ 최고"

            print(f"  step {ckpt_name:>8} | {local_std:>13.5f} | "
                  f"{g:>6} | {recommend:>10}")
            ckpt_trends.append((ckpt_name, local_std, g))

            del p, pp
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  step {ckpt_name:>8} | 로딩 실패: {str(e)[:30]}")

    if best_ckpt:
        print(f"\n  ⭐ 추천 Sweet Spot: step {best_ckpt} "
              f"(최종점 분산: {best_std:.5f})")
        print(f"     실전 배포 시 경로:")
        print(f"     {CKPT_DIR}/checkpoints/{best_ckpt}/pretrained_model")
        scores["best_ckpt"] = best_ckpt
        scores["best_ckpt_std"] = float(best_std)
else:
    print("  ⚠️  체크포인트 부족 — sweet spot 탐색 생략")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최종 종합 평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  📊 최종 종합 평가 (LeKiwi ACT merged_40)")
print("=" * 70)

grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

# 가중치: Mode collapse를 가장 중요하게 (공간 적응력)
total = (
    grade_map.get(scores.get("collapse_grade", "F"), 0) * 5 +  # ⭐⭐ 공간 적응
    grade_map.get(scores.get("mode_avg_grade", "F"), 0) * 4 +  # ⭐ Vision 반응
    grade_map.get(scores.get("config_grade",   "F"), 0) * 3 +
    grade_map.get(scores.get("vision_grade",   "F"), 0) * 3 +
    grade_map.get(scores.get("state_grade",    "F"), 0) * 3 +
    grade_map.get(scores.get("range_grade",    "F"), 0) * 3 +  # 안전성
    grade_map.get(scores.get("chunk_grade",    "F"), 0) * 2 +
    grade_map.get(scores.get("consistency_grade", "F"), 0) * 1
)
max_total = 4 * (5 + 4 + 3 + 3 + 3 + 3 + 2 + 1)
pct = total / max_total * 100

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  ⭐⭐ Mode Collapse (공간 적응):    {scores.get('collapse_grade','?'):>2}   가중치 x5  │
  │  ⭐  Mode Averaging (Vision 반응):  {scores.get('mode_avg_grade','?'):>2}   가중치 x4  │
  │  ACT 설정 (chunk/dim/vae):          {scores.get('config_grade','?'):>2}   가중치 x3  │
  │  Vision 인코더:                      {scores.get('vision_grade','?'):>2}   가중치 x3  │
  │  State 인코더:                       {scores.get('state_grade','?'):>2}   가중치 x3  │
  │  Action Range 안전성:                {scores.get('range_grade','?'):>2}   가중치 x3  │
  │  Chunk 품질 (boundary):              {scores.get('chunk_grade','?'):>2}   가중치 x2  │
  │  Observation 일관성:                 {scores.get('consistency_grade','?'):>2}   가중치 x1  │
  ├──────────────────────────────────────────────────────────┤
  │  종합 점수: {total}/{max_total}  ({pct:.0f}%)                                  │
  └──────────────────────────────────────────────────────────┘""")

# ── 진단 결론 ──────────────────────────────────────────────────────
print("\n  ── 🔬 LeKiwi ACT 진단 결론 ────────────────────────────")

collapse_g = scores.get("collapse_grade", "F")
mode_g = scores.get("mode_avg_grade", "F")
vision_g = scores.get("vision_grade", "F")
range_g = scores.get("range_grade", "F")

if (collapse_g in ("A", "B") and mode_g in ("A", "B") and
        vision_g in ("A", "B") and range_g in ("A", "B")):
    print(f"""
  🎉 학습 성공 — 실전 배포 준비 완료!

     ✅ 공간 적응력 (Mode collapse): {collapse_g}
     ✅ Vision 반응 (Mode averaging): {mode_g}
     ✅ Vision 인코더: {vision_g}
     ✅ Action 안전성: {range_g}

  💡 다음 단계 (우선순위):

  1) [실전 테스트] LeKiwi 실제 주행 검증
     - 추론 서버 실행 (PolicyServer at NUC)
     - RobotClient로 비동기 추론 (Pi5)
     - 10회 이상 노란선 추종 테스트

  2) [Sweet Spot 적용] 평가 10의 추천 체크포인트 사용
     - {scores.get('best_ckpt', 'last')}번 step 추천

  3) [모니터링] 실전 중 chunk_boundary 부드러움 확인
""")
elif collapse_g in ("F", "D"):
    print(f"""
  🚨 Mode COLLAPSE 의심 — 공간 적응력 부족

     ❌ collapse 등급: {collapse_g}
     - state 변이 시 chunk 분산: {scores.get('state_final_std', 0):.5f}
     - chunk 내부 움직임: {scores.get('avg_chunk_range', 0):.5f}

  💡 원인 분석:
     1) 학습 데이터가 너무 단조 (같은 경로만 반복)
     2) 학습 부족 (actual_step={scores.get('actual_step', 0):,})
     3) LR이 너무 작아 수렴 미달

  💡 해결책 (우선순위):

  1) [데이터] 더 다양한 주행 경로 수집
     - 5~9개 위치 × 10~20회 = 50~180ep
     - 직선, 곡선, 급회전 섞어서

  2) [학습] 더 긴 학습 또는 LR 조정
     - steps 100k 이상
     - LR 1e-5 → 3e-5 시도 (batch 32에 맞게)

  3) [체크포인트] Sweet spot 사용 (평가 10)
     - 추천: step {scores.get('best_ckpt', '?')}
""")
elif vision_g in ("F", "D"):
    print(f"""
  🚨 Vision 무시 학습 — 이미지 인코더 학습 실패

     ❌ Vision 등급: {vision_g} ({scores.get('vision_responding', 0)}/10 반응)

  💡 원인:
     1) freeze_vision_encoder=True 로 학습됨
     2) 데이터셋의 이미지 다양성 부족
     3) 학습 부족

  💡 해결책:
     1) freeze_vision_encoder=False 확인
     2) 데이터에 시각적 다양성 추가 (조명, 배경)
     3) 더 긴 학습 (100k steps+)
""")
else:
    print(f"""
  🔶 부분적 성공 — 추가 분석 필요

     collapse: {collapse_g}, averaging: {mode_g}
     vision: {vision_g}, range: {range_g}

  💡 위 상세 등급을 참조하여 약점을 중심으로 개선
""")

print(f"""
  ── 📊 핵심 숫자 정리 ────────────────────────────────────
     학습 스텝:             {scores.get('actual_step', 0):,}
     체크포인트 수:          {scores.get('num_checkpoints', 0)}
     Vision 평균 Δ:          {scores.get('vision_mean', 0):.5f}
     State 평균 Δ:           {scores.get('state_mean', 0):.5f}
     Chunk 최종점 분산:       {scores.get('final_std', 0):.5f} ({mode_g})
     공간 적응력 분산:        {scores.get('state_final_std', 0):.5f} ({collapse_g}) ⭐
     Chunk 내부 움직임:       {scores.get('avg_chunk_range', 0):.5f}
     Repeat 분산:            {scores.get('consistency_mean', 0):.5f}
     Action 범위 초과:        {scores.get('total_oor', 0)}건
     Sweet spot:             step {scores.get('best_ckpt', '?')}

  ── 📚 참고 레퍼런스 (이 스크립트의 근거) ─────────────
     • ACT 원본: Zhao et al. 2023 (arXiv 2304.13705)
     • SVRC Explanation: roboticscenter.ai (2026)
     • Chunk-Boundary Artifact: arXiv 2603.11642 (2026)
     • Data Scaling Laws: arXiv 2410.18647 (2024)
     • LeRobot Issue #2213 (ACT 최소 batch)
     • HuggingFace LeRobot ACT Docs (2026)

  ── 🎓 교육적 가치 (ZETA Satellite Robotics) ──────────
     이 스크립트는 피지컬 AI 학습 디버깅의 실전 예시입니다.
     • Vision/State/Action 각각 독립 검증
     • Mode averaging vs collapse 정량 진단
     • 체크포인트 선택의 과학적 근거 제공
""")

print("=" * 70)
print(f"\n  📄 로그 파일: {LOG_FILE}")
print("=" * 70)
