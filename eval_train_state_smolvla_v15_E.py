#!/usr/bin/env python3
# SmolVLA 학습 상태 종합 평가 스크립트 v2-E (v15 new_merged_100 체크포인트 전용)
# 작성일: 2026-04-21
# 기반: eval_train_state_smolvla_v14_D.py (v2-D)
# 작성자: 빅맨 / ZETA Satellite Robotics
#
# ═══════════════════════════════════════════════════════════════════════
# v2-D → v2-E 주요 변경점 (v15 new_merged_100 검증용)
# ═══════════════════════════════════════════════════════════════════════
#
# [v13 실전 결과 — 2026-04-19]
#   ✅ 색상 구분 성공 (Layer 1 해결)
#   ❌ 10번 중 2번만 '귀', 8번 몸통 → Mode AVERAGING (시연 흩어짐)
#
# [v14 실전 관찰 — 2026-04-20]
#   ✅ 색상 구분됨 (pink/yellow)
#   ✅ 팔(arm)을 잡으려고 함 ← v13과 결정적 차이
#   ⚠️  "수집 때 단조로움이 단조로운 행동으로 옮겨진다" ← 새 발견
#       → Mode COLLAPSE (시연 과단조로움으로 적응력 부족)
#   📊 v14_D 평가 결과:
#       - 공간 적응력: 0.252° (F)
#       - Chunk 움직임 범위: 0.06° (서보 분해능 미만)
#       - Mode Collapse 등급: F (종합 55%)
#
# [v15 기대 — 2026-04-21]
#   🎯 new_merged_100 = v7(30) + v12(20) + v13_yellow(30) + v14_yellow(20)
#   🎯 공간 다양성 확보: v13_yellow 3위치 + v14_yellow 2위치 = 5위치 커버리지
#   🎯 HuggingFace 공식 권장(5위치×10회) 수준 달성
#   🎯 학습: batch 96, 16k steps, LR 1.2e-4, warmup 500 (v15 스크립트)
#
#   ⭐ v2-E 핵심 가설:
#     "100ep(v14보다 20ep 적음)인데도 v14 120ep보다 좋다면
#      = 데이터 품질 > 수량 실증"
#
# [v13 vs v14 vs v15 — 3단 비교]
#   v13: 데이터 과다양성 → Mode averaging (평균 몸통 수렴)
#   v14: 데이터 과단조로움 → Mode collapse (단일 궤적 고착)
#   v15: 계획된 다양성 → 양쪽 모두 해결 기대 ⭐
#
# [글로벌 사이트 근거 (2025~2026 최신 연구)]
#   arXiv 2410.18647 "Data Scaling Laws":
#     "Diversity of environments/objects is far more important than
#      absolute number of demonstrations"
#   arXiv 2512.04813 MOVE (2025):
#     "Single static spatial configuration restricts spatial diversity"
#   arXiv 2411.10203 "Generalizable 3D Manipulation":
#     "Methods tend to overfit specific training trajectories"
#   arXiv 2410.06151 "Quality Diversity IL":
#     "Narrow demonstrations → struggle with unseen situations"
#   HuggingFace SmolVLA 공식:
#     "50 episodes across 5 distinct positions, 25 episodes not enough"
#
# [v2-D → v2-E 변경사항]
#   [1] 경로: CKPT_DIR, DATASET_DIR → v15 / new_merged_100으로 업데이트
#   [2] 해석 관점: "Mode collapse 진단" → "Mode collapse 해결 검증"
#   [3] 평가 지표 동일 유지 (v14와 공정 비교 위해)
#       - 5문장 일관성, Vision 다양성, State 적응력, Chunk 움직임
#   [4] 최종 진단: v15 성공 기준
#       - 공간 적응력: B 이상 (v14 F → 큰 개선)
#       - Chunk 움직임 범위: 50° 이상 (v14 0.06° → 극적 회복)
#       - 5문장 일관성: A 유지
#
# ═══════════════════════════════════════════════════════════════════════

import os
import sys
import json
import glob
import warnings
from datetime import datetime
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
warnings.filterwarnings("ignore")

LOG_DIR = os.path.expanduser("~/lerobot_outputs/eval_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"smolvla_eval_v2e_yellow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


class Tee:
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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# ── 설정 ──────────────────────────────────────────────────────────────
CKPT_DIR  = "/home/zetabank/lerobot_outputs/so101_doll_box_smolvla_v15_yellow_merged_100"
CKPT      = os.path.join(CKPT_DIR, "checkpoints/last/pretrained_model")
DATASET_DIR = "/home/zetabank/lerobot_datasets/doll_box_yellow_new_merged_100"

# ★ v13 학습된 5개 Yellow 태스크
TASKS_YELLOW = [
    "Grasp the yellow doll by its arm and place it in the box",
    "Pick up the yellow doll by its arm and put it in the box",
    "Lift the yellow doll by its arm and drop it into the box",
    "Hold the yellow doll's arm and move it into the box",
    "Take the yellow doll by its arm and carry it to the box",
]
TASK_DEFAULT = TASKS_YELLOW[0]

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_f", "wrist_r", "grip"]
VISION_THRESHOLD = 0.1
CONSISTENCY_THRESHOLD = 0.3  # ★ v2-D 신규: 5문장 간 action 차이가 이보다 작아야 강건
MODE_AVG_THRESHOLD = 2.0     # ★ v2-D 신규: 같은 상태에서 chunk 분산이 이보다 크면 다중모드
EXPECTED_N_ACTION_STEPS = 50

CAM_WRIST = "observation.images.camera2"
CAM_TOP   = "observation.images.camera1"

print("=" * 70)
print("  SmolVLA 학습 상태 종합 평가  v2-E  (v15 new_merged_100 전용)")
print(f"  체크포인트: {CKPT_DIR.split('/')[-1]}")
print(f"  실전 관측: ✅ 색상 구분 성공 / ❌ 10번 중 2번만 'arm(팔)' grasp")
print(f"  주 진단 목표: Mode Averaging (arm vs 다른 부위) 정량화")
print(f"  로그 저장: {LOG_FILE}")
print("=" * 70)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rand_img(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(1, 3, 480, 640).cuda()


def make_meaningful_test_imgs() -> list:
    imgs = []
    real_imgs = sorted(glob.glob(f"{DATASET_DIR}/**/*.jpg", recursive=True))[:5]
    if not real_imgs:
        real_imgs = sorted(glob.glob(f"{DATASET_DIR}/**/*.png", recursive=True))[:5]
    if real_imgs:
        from torchvision import transforms
        to_tensor = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
        ])
        from PIL import Image
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

    imgs.append(torch.ones(1, 3, 480, 640).cuda())
    imgs.append(torch.zeros(1, 3, 480, 640).cuda())
    r = torch.zeros(1, 3, 480, 640).cuda(); r[0, 0] = 1.0; imgs.append(r)
    g = torch.zeros(1, 3, 480, 640).cuda(); g[0, 1] = 1.0; imgs.append(g)
    b = torch.zeros(1, 3, 480, 640).cuda(); b[0, 2] = 1.0; imgs.append(b)
    base = rand_img(0)
    imgs.append(1.0 - base)
    imgs.append(base * 0.05)
    checker = torch.zeros(1, 3, 480, 640).cuda()
    checker[0, :, ::2, ::2] = 1.0
    checker[0, :, 1::2, 1::2] = 1.0
    imgs.append(checker)
    return imgs[:10]


def reset_policy_state(p):
    if hasattr(p, "reset") and callable(getattr(p, "reset", None)):
        try:
            p.reset()
            return
        except Exception:
            pass
    if hasattr(p, "_queues"):
        for q in p._queues.values():
            if hasattr(q, "clear"):
                q.clear()
        return
    if hasattr(p, "_action_queue"):
        p._action_queue.clear()


def get_action(policy, preprocess, wrist, top, state=None, task=None):
    if state is None:
        state = torch.zeros(1, 6).cuda()
    if task is None:
        task = TASK_DEFAULT
    reset_policy_state(policy)
    batch = preprocess({
        CAM_WRIST:           wrist,
        CAM_TOP:             top,
        "observation.state": state,
        "task":              [task],
    })
    with torch.no_grad():
        return policy.select_action(batch).cpu().numpy()[0]


def get_chunk(policy, preprocess, wrist, top, state=None, task=None):
    """★ v2-D 신규: chunk 전체 예측 (평가 9 Mode averaging용)"""
    if state is None:
        state = torch.zeros(1, 6).cuda()
    if task is None:
        task = TASK_DEFAULT
    reset_policy_state(policy)
    batch = preprocess({
        CAM_WRIST:           wrist,
        CAM_TOP:             top,
        "observation.state": state,
        "task":              [task],
    })
    with torch.no_grad():
        return policy.predict_action_chunk(batch)[0].cpu().numpy()


def grade_vision(responding: int, total: int) -> tuple:
    if responding == 0:
        return "F", "❌ Vision 완전 미반응"
    elif responding <= int(total * 0.3):
        return "D", "⚠️  Vision 극초기 반응 (불충분)"
    elif responding <= int(total * 0.6):
        return "C", "⚠️  Vision 부분 반응"
    elif responding <= int(total * 0.8):
        return "B", "✅ Vision 상당 부분 반응"
    else:
        return "A", "✅ Vision 충분히 반응"


def grade_consistency(mean_diff: float) -> tuple:
    """★ v2-D 신규: 5문장 간 일관성 (낮을수록 좋음)
    같은 이미지에 5개 Yellow 문장 넣었을 때 action 차이가 작으면 언어 강건.
    너무 크면 언어 접지가 과적합 (문장 구조에 민감).
    """
    if mean_diff < 0.3:
        return "A", "✅ 5문장 간 action 매우 일관 (언어 강건)"
    elif mean_diff < 0.5:
        return "B", "✅ 5문장 간 action 일관"
    elif mean_diff < 1.0:
        return "C", "⚠️  5문장 간 일부 차이 (문장 구조 민감)"
    elif mean_diff < 2.0:
        return "D", "⚠️  5문장 간 큰 차이 (학습 편향 의심)"
    else:
        return "F", "❌ 5문장이 완전 다른 행동 생성 (학습 실패)"


def grade_mode_avg(chunk_divergence: float) -> tuple:
    """★ v2-D 신규: Mode averaging 심각도.
    같은 상태에서 chunk가 얼마나 '평균적' 궤적인지 vs '뚜렷한 모드' 궤적인지.
    v13 실전: 10번 중 2번만 귀 → Mode averaging 있음 (몸통으로 수렴)
    """
    if chunk_divergence < 1.0:
        return "F", "❌ Mode averaging 심각 (단일 평균 궤적만 생성 - 몸통 grasp)"
    elif chunk_divergence < 2.0:
        return "D", "⚠️  Mode averaging 일부 (대부분 평균 궤적)"
    elif chunk_divergence < 4.0:
        return "C", "⚠️  모드 분리 약함"
    elif chunk_divergence < 6.0:
        return "B", "✅ 모드 분리 양호"
    else:
        return "A", "✅ 모드 분리 뚜렷 (다양한 궤적 생성 가능)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 1: 체크포인트 메타 정보
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 1: 체크포인트 메타 정보")
print("─" * 70)

scores = {}

step_file = os.path.join(CKPT_DIR, "checkpoints/last/training_state/training_step.json")
if os.path.exists(step_file):
    with open(step_file) as f:
        step_data = json.load(f)
    actual_step = step_data.get("training_step", 0)
    print(f"  실제 학습 완료 스텝: {actual_step:,}")
    if actual_step < 10000:
        print(f"    └─ ⚠️  10,000 미만 (목표: 20,000)")
    elif actual_step < 15000:
        print(f"    └─ 🔶 중간 단계 학습")
    elif actual_step <= 20000:
        print(f"    └─ ✅ 권장 범위 내 (v13 v2 목표)")
    elif actual_step <= 30000:
        print(f"    └─ ℹ️  충분히 학습")
    else:
        print(f"    └─ ⚠️  과학습 가능성")
    scores["actual_step"] = actual_step
else:
    print("  ⚠️  training_step.json 없음")
    scores["actual_step"] = 0

sched_file = os.path.join(CKPT_DIR, "checkpoints/last/training_state/scheduler_state.json")
if os.path.exists(sched_file):
    with open(sched_file) as f:
        sched_data = json.load(f)
    last_epoch = sched_data.get("last_epoch", "?")
    last_lr    = sched_data.get("_last_lr", ["?"])[0]
    print(f"  스케줄러 last_epoch: {last_epoch:,}  |  마지막 lr: {last_lr}")
    scores["scheduler_ok"] = True
else:
    print("  ⚠️  scheduler_state.json 없음")
    scores["scheduler_ok"] = False

ckpt_root = os.path.join(CKPT_DIR, "checkpoints")
ckpts = sorted([d for d in os.listdir(ckpt_root) if d.isdigit()], key=lambda x: int(x))
print(f"\n  저장된 체크포인트 ({len(ckpts)}개): {ckpts}")
scores["num_checkpoints"] = len(ckpts)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 로딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 모델 로딩 중...")
policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda").eval()
preprocess, _ = make_pre_post_processors(
    policy.config, CKPT,
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
print("  ✅ 완료\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 2: SmolVLA 핵심 설정 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("▶ 평가 2: SmolVLA 핵심 설정 검증")
print("─" * 70)

cfg = policy.config
n_action_steps = getattr(cfg, "n_action_steps", None)
chunk_size     = getattr(cfg, "chunk_size", None)
n_obs_steps    = getattr(cfg, "n_obs_steps", None)
freeze_vision  = getattr(cfg, "freeze_vision_encoder", None)

print(f"  chunk_size:            {chunk_size}")
print(f"  n_action_steps:        {n_action_steps}   (기대값: {EXPECTED_N_ACTION_STEPS})")
print(f"  n_obs_steps:           {n_obs_steps}")
print(f"  freeze_vision_encoder: {freeze_vision}")

if n_action_steps is None:
    config_grade = "D"
elif n_action_steps == 1:
    config_grade = "F"
elif n_action_steps < EXPECTED_N_ACTION_STEPS * 0.5:
    config_grade = "C"
elif n_action_steps < EXPECTED_N_ACTION_STEPS:
    config_grade = "B"
else:
    config_grade = "A"

print(f"\n  ✅ n_action_steps={n_action_steps}  등급: {config_grade}")
scores["config_grade"] = config_grade
scores["n_action_steps"] = n_action_steps

BASE_IMG   = rand_img(0)
BASE_STATE = torch.zeros(1, 6).cuda()
base_act   = get_action(policy, preprocess, BASE_IMG, BASE_IMG, BASE_STATE, task=TASK_DEFAULT)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 3: Vision 인코더 반응
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 3: Vision 인코더 반응 (Pink task [0] 기준)")
print("─" * 70)

test_imgs = make_meaningful_test_imgs()
vision_diffs = []
for i, img in enumerate(test_imgs):
    a = get_action(policy, preprocess, img, img, task=TASK_DEFAULT)
    diff = np.abs(a - base_act).max()
    vision_diffs.append(diff)
    flag = "✅" if diff > VISION_THRESHOLD else "❌"
    print(f"  img_{i+1:02d}: Δmax={diff:7.4f}°  {flag}")

vision_responding = sum(1 for d in vision_diffs if d > VISION_THRESHOLD)
vision_mean = np.mean(vision_diffs)
vision_max  = np.max(vision_diffs)
vision_grade, vision_msg = grade_vision(vision_responding, len(test_imgs))
print(f"\n  Vision 반응: {vision_responding}/{len(test_imgs)}  평균: {vision_mean:.4f}°  등급: {vision_grade}")

scores.update({
    "vision_grade": vision_grade,
    "vision_responding": vision_responding,
    "vision_mean": float(vision_mean),
    "vision_max": float(vision_max),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 4: State 인코더 학습 상태
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 4: State 인코더 학습 상태")
print("─" * 70)

state_cases = [
    ("홈",          [0.,   -90.,  90.,   0.,  0., 0.]),
    ("뻗기",        [45.,  -45.,  45.,   0.,  0., 0.]),
    ("접기",        [0.,  -160., 160.,   0.,  0., 0.]),
    ("좌회전",      [-90., -90.,  90.,   0.,  0., 0.]),
    ("우회전",      [90.,  -90.,  90.,   0.,  0., 0.]),
    ("그리퍼열림",  [0.,   -90.,  90.,   0.,  0., 25.]),
    ("그리퍼닫힘",  [0.,   -90.,  90.,   0.,  0., -25.]),
    ("낮은자세",    [0.,  -170., 160.,  90.,  0., 0.]),
    ("높은자세",    [0.,   -30.,  30., -30.,  0., 0.]),
    ("수집자세",    [2.5,  -88.,  82.,  63., -2., 0.7]),
]

state_diffs = []
for name, vals in state_cases:
    st = torch.tensor([vals]).float().cuda()
    a  = get_action(policy, preprocess, BASE_IMG, BASE_IMG, st, task=TASK_DEFAULT)
    d  = np.abs(a - base_act).max()
    state_diffs.append(d)
    flag = "✅" if d > 0.5 else "❌"
    print(f"  {name:10s}: Δmax={d:6.3f}°  {flag}")

state_responding = sum(1 for d in state_diffs if d > 0.5)
state_mean = np.mean(state_diffs)
print(f"\n  반응: {state_responding}/10  평균: {state_mean:.3f}°")

if state_responding >= 8:
    state_grade = "A"
elif state_responding >= 5:
    state_grade = "B"
elif state_responding >= 3:
    state_grade = "C"
else:
    state_grade = "D"
print(f"  State 등급: {state_grade}")
scores.update({"state_grade": state_grade, "state_responding": state_responding})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 5: 청크 품질
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 5: 청크 품질")
print("─" * 70)

reset_policy_state(policy)
batch = preprocess({
    CAM_WRIST:           BASE_IMG,
    CAM_TOP:             BASE_IMG,
    "observation.state": BASE_STATE,
    "task":              [TASK_DEFAULT],
})
with torch.no_grad():
    chunk = policy.predict_action_chunk(batch)[0].cpu().numpy()

step_diffs = [np.abs(chunk[t] - chunk[t-1]).max() for t in range(1, len(chunk))]
max_step = np.max(step_diffs)
SAFE = [(-180, 180), (-200, 200), (-180, 180), (-180, 180), (-180, 180), (-30, 30)]
oor = sum(1 for t in range(len(chunk)) for j in range(6)
          if not (SAFE[j][0] <= chunk[t, j] <= SAFE[j][1]))
print(f"  청크 길이: {len(chunk)} / 최대변화: {max_step:.2f}° / 범위초과: {oor}건")
chunk_grade = ("A" if max_step < 10 and oor == 0 else
               "B" if max_step < 20 and oor == 0 else
               "C" if max_step < 30 else "D")
print(f"  청크 등급: {chunk_grade}")
scores["chunk_grade"] = chunk_grade
scores["chunk_len"] = len(chunk)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★★★ 평가 7 (v2-D 재설계): Yellow 5문장 간 action 일관성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 7: ⭐ Yellow 5문장 간 action 일관성 (v2-D 재설계)")
print("  목적: 5문장 모두 '팔(arm)을 잡아 박스에' 의미 → 같은 행동이어야 정상")
print("  Δ 작음 = 언어 강건 (문장 구조 무관)")
print("  Δ 큼 = 언어 과적합 (특정 문장에 과도 민감)")
print("★" * 70)

lang_cases = [
    ("자세 1 (홈)",      torch.zeros(1, 6).cuda()),
    ("자세 2 (뻗기)",    torch.tensor([[45., -45., 45., 0., 0., 0.]]).float().cuda()),
    ("자세 3 (수집자세)", torch.tensor([[2.5, -88., 82., 63., -2., 0.7]]).float().cuda()),
]

test_img_subset = test_imgs[:3]
pairwise_diffs = []
per_case_stats = []

print(f"\n  {'케이스':<20} | {'5문장 평균 Δ':>12} | {'최대 Δ':>8} | {'판정':>10}")
print(f"  {'-' * 62}")

for st_name, st in lang_cases:
    for img_idx, img in enumerate(test_img_subset):
        # 5개 문장 각각 action 계산
        actions = []
        for task in TASKS_YELLOW:
            a = get_action(policy, preprocess, img, img, st, task=task)
            actions.append(a)
        actions = np.array(actions)  # [5, 6]

        # 5개 문장 간 pairwise max 차이 (10개 쌍)
        pair_diffs = []
        for i in range(5):
            for j in range(i+1, 5):
                pair_diffs.append(np.abs(actions[i] - actions[j]).max())
        pair_mean = np.mean(pair_diffs)
        pair_max  = np.max(pair_diffs)
        pairwise_diffs.extend(pair_diffs)

        # 판정: 낮을수록 좋음 (일관성)
        if pair_mean < CONSISTENCY_THRESHOLD:
            judge = "✅ 강건"
        elif pair_mean < 1.0:
            judge = "🔶 보통"
        else:
            judge = "❌ 불일치"

        per_case_stats.append((st_name, img_idx, pair_mean, pair_max))
        case_label = f"{st_name} / img{img_idx+1}"
        print(f"  {case_label:<20} | {pair_mean:>11.4f}° | {pair_max:>7.4f}° | {judge:>10}")

cons_mean = np.mean(pairwise_diffs)
cons_max  = np.max(pairwise_diffs)
cons_grade, cons_msg = grade_consistency(cons_mean)

print(f"\n  ─────────────────────────────────────────────────────────────")
print(f"  5문장 간 pairwise 평균 Δ: {cons_mean:.4f}°  최대: {cons_max:.4f}°")
print(f"  일관성 등급: {cons_grade}  {cons_msg}")
print()
if cons_mean < 0.3:
    print(f"  ✅ 언어 접지 성공 확인 — 5개 문장이 같은 행동으로 수렴")
    print(f"     빅맨님 실전 관찰 (색상 구분 성공)과 일치")
elif cons_mean < 1.0:
    print(f"  🔶 언어 강건성 부분 달성 — 대부분 일관되나 일부 문장에서 편차")
else:
    print(f"  ⚠️  5문장이 서로 다른 행동을 생성 → 특정 문장에 과적합")

scores.update({
    "consistency_grade": cons_grade,
    "consistency_mean": float(cons_mean),
    "consistency_max": float(cons_max),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★★★ 평가 9 (v2-D 신규 핵심): Mode Averaging 진단
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "★" * 70)
print("▶ 평가 9: ⭐⭐ Mode Averaging + Mode COLLAPSE 동시 진단 (v2-D 핵심)")
print("  v13 실전: 10번 중 2번만 '귀' → Mode averaging (과다양성 원인)")
print("  v14 실전: arm 시도하나 단조로움 → Mode collapse (과단조 원인)")
print("  ─────────────────────────────────────────────────────────")
print("  측정 1: Vision 다양성에 따른 chunk 반응 (이미지 변화)")
print("  측정 2: State 다양성에 따른 chunk 반응 (자세 변화)")
print("  측정 3: Chunk 내부 움직임 범위 (궤적의 역동성)")
print("★" * 70)

# 현실적인 수집 자세에서 chunk 추출
collection_state = torch.tensor([[2.5, -88., 82., 63., -2., 0.7]]).float().cuda()

# ── 측정 1: 같은 state, 다른 이미지로 chunk 추출 (Vision 반응 다양성) ──
chunks_per_img = []
for img_idx, img in enumerate(test_imgs[:5]):
    chunk = get_chunk(policy, preprocess, img, img, collection_state, TASK_DEFAULT)  # [50, 6]
    chunks_per_img.append(chunk)
chunks_img_arr = np.array(chunks_per_img)  # [5, 50, 6]

# chunk 최종점, 중간점 분산
final_points = chunks_img_arr[:, -1, :]
final_std = np.std(final_points, axis=0).mean()
mid_points = chunks_img_arr[:, 25, :]
mid_std = np.std(mid_points, axis=0).mean()
traj_std = np.std(chunks_img_arr, axis=0).mean()

# ── 측정 2: ★ v2-D 신규: 같은 이미지, 다른 state로 chunk 추출 (공간 적응력) ──
# Mode collapse가 심하면 state가 달라져도 비슷한 움직임만 생성
test_states_for_collapse = [
    torch.tensor([[2.5, -88., 82., 63., -2., 0.7]]).float().cuda(),     # 수집자세
    torch.tensor([[-20., -85., 85., 65., -5., 0.7]]).float().cuda(),    # 살짝 왼쪽
    torch.tensor([[20., -85., 85., 65., 5., 0.7]]).float().cuda(),      # 살짝 오른쪽
    torch.tensor([[0., -70., 70., 50., 0., 0.7]]).float().cuda(),       # 높은 자세
    torch.tensor([[0., -100., 100., 75., 0., 0.7]]).float().cuda(),     # 낮은 자세
]
chunks_per_state = []
for st in test_states_for_collapse:
    chunk = get_chunk(policy, preprocess, test_imgs[0], test_imgs[0], st, TASK_DEFAULT)
    chunks_per_state.append(chunk)
chunks_state_arr = np.array(chunks_per_state)  # [5, 50, 6]

# state 변이에 따른 chunk 최종점 분산 (높을수록 공간 적응력 좋음)
state_final_std = np.std(chunks_state_arr[:, -1, :], axis=0).mean()
state_traj_std = np.std(chunks_state_arr, axis=0).mean()

# ── 측정 3: ★ v2-D 신규: Chunk 내부 움직임 범위 (collapse 진단) ──
# 각 chunk의 시작-끝 이동량 (joint별 최대 변화)
chunk_ranges = []
for chunk in chunks_per_img:
    joint_range = np.max(chunk, axis=0) - np.min(chunk, axis=0)  # [6]
    chunk_ranges.append(joint_range.mean())
avg_chunk_range = np.mean(chunk_ranges)

# 지표 4: chunk 내부 궤적 다양성 (움직임 역동성)
internal_std = np.std(chunks_img_arr, axis=1).mean()

print(f"\n  📊 측정 1: Vision 다양성 (다른 이미지 5장, 같은 state)")
print(f"     최종 grasp 지점 분산 : {final_std:.4f}°  (Vision 반응)")
print(f"     중간 approach 분산   : {mid_std:.4f}°")
print(f"     전체 trajectory 분산 : {traj_std:.4f}°")

print(f"\n  📊 측정 2: State 다양성 (다른 자세 5가지, 같은 이미지) ★ v2-D 신규")
print(f"     최종 grasp 지점 분산 : {state_final_std:.4f}°  (공간 적응력)")
print(f"     전체 trajectory 분산 : {state_traj_std:.4f}°")
print(f"     → 낮으면 Mode collapse (자세 달라도 같은 궤적만 생성)")

print(f"\n  📊 측정 3: Chunk 내부 움직임 범위 ★ v2-D 신규")
print(f"     평균 움직임 범위     : {avg_chunk_range:.2f}°  (궤적 역동성)")
print(f"     Chunk 내부 분산      : {internal_std:.4f}°")
print(f"     → 낮으면 경직된 단일 모션 (Mode collapse 증거)")

# ── Mode averaging 등급 (Vision 다양성 기반) ──
mode_grade, mode_msg = grade_mode_avg(final_std)

# ── ★ v2-D 신규: Mode COLLAPSE 등급 (State 적응력 기반) ──
def grade_mode_collapse(state_std: float, chunk_range: float) -> tuple:
    """
    Mode collapse 진단: state 변이에 따른 chunk 반응 + 움직임 범위
    v14 실전: "단조로움이 단조로운 행동으로" → state 변해도 비슷한 경직 궤적
    """
    # state 반응이 거의 없으면서 움직임 범위도 좁으면 collapse
    if state_std < 1.5 and chunk_range < 30:
        return "F", "❌ Mode collapse 심각 (자세 무관 단일 경직 궤적)"
    elif state_std < 3.0 and chunk_range < 50:
        return "D", "⚠️  Mode collapse 일부 (공간 적응력 부족)"
    elif state_std < 5.0:
        return "C", "⚠️  공간 적응력 약함"
    elif state_std < 8.0:
        return "B", "✅ 공간 적응력 양호"
    else:
        return "A", "✅ 공간 적응력 우수"

collapse_grade, collapse_msg = grade_mode_collapse(state_final_std, avg_chunk_range)

print(f"\n  ─────────────────────────────────────────────────────────────")
print(f"  Mode Averaging 등급 (Vision 반응): {mode_grade}  {mode_msg}")
print(f"  Mode Collapse 등급 (공간 적응력) : {collapse_grade}  {collapse_msg}")
print()

# ── v13 vs v14 차이 기반 진단 ──
if collapse_grade in ("F", "D"):
    print(f"  🚨 Mode COLLAPSE 확정 — v14의 핵심 문제")
    print(f"     → 빅맨님 관찰 일치: '단조로움이 단조로운 행동으로 옮겨진다'")
    print(f"     → state 변이에도 비슷한 궤적 → 인형 위치 변화에 적응 못함")
    print(f"     → arm은 잡으려 하나 정확도 떨어짐 (경직된 단일 궤적)")
    print(f"")
    print(f"  💡 해결책 (우선순위):")
    print(f"     1) [핵심] 데이터 다양성 증가 재수집:")
    print(f"        - 인형 위치 5~10가지 (좌/중/우 × 앞/뒤)")
    print(f"        - 각 위치마다 10ep 이상 수집")
    print(f"        - arm grasp은 유지, 접근 궤적은 다양하게")
    print(f"        - 근거: arXiv 2410.18647 'Diversity is all you need'")
    print(f"     2) [데이터] Random crop 이미지 증강:")
    print(f"        - --dataset.image_transforms.enable=true (이미 적용됨)")
    print(f"        - 추가로 색상 지터, 밝기 증강 고려")
    print(f"     3) [모델] ACT 또는 Diffusion Policy 비교:")
    print(f"        - Multi-modal 표현이 더 강건")
elif mode_grade in ("F", "D"):
    print(f"  🚨 Mode AVERAGING 확정 — v13 패턴 (예상 밖, 확인 필요)")
    print(f"     → 이론상 v14는 collapse 문제여야 하는데 averaging이 나옴")
    print(f"     → 실전 테스트 재확인 권장")
else:
    print(f"  ✅ 양호한 학습 상태")
    print(f"     → Mode averaging, collapse 모두 양호")
    print(f"     → 실전에서 실패 시 다른 원인 탐색 (카메라, calibration 등)")

scores.update({
    "mode_avg_grade": mode_grade,
    "collapse_grade": collapse_grade,       # ★ v2-D 신규
    "final_std": float(final_std),
    "mid_std": float(mid_std),
    "traj_std": float(traj_std),
    "state_final_std": float(state_final_std),  # ★ v2-D 신규
    "state_traj_std": float(state_traj_std),    # ★ v2-D 신규
    "avg_chunk_range": float(avg_chunk_range),  # ★ v2-D 신규
    "internal_std": float(internal_std),
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ 평가 10 (v2-D 신규): 실제 데이터셋 grasp point 분포 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 10: 실제 데이터셋 grasp point 공간 분포")
print("  → 시연 데이터의 grasp 순간 좌표 분산 측정")
print("  → 분산 크면 → 시연자 간 귀/몸통 지점 불일치")
print("─" * 70)

try:
    import pandas as pd
    data_files = sorted(glob.glob(f"{DATASET_DIR}/data/chunk-*/file-*.parquet"))
    if data_files:
        print(f"  📂 데이터 파일 {len(data_files)}개 발견")

        # 각 에피소드에서 gripper 닫히는 순간의 state 추출
        grasp_points = []
        all_data = pd.concat([pd.read_parquet(f) for f in data_files])
        print(f"  📦 총 {all_data['episode_index'].nunique()} 에피소드, {len(all_data)} 프레임")

        # action 6번째 컬럼(grip)이 임계치 이하로 떨어지는 순간 포착
        # SO-101의 grip action은 음수일수록 닫힘
        for ep_idx in sorted(all_data['episode_index'].unique()):
            ep = all_data[all_data['episode_index'] == ep_idx]
            # grip action 추출 (action[5] = grip)
            actions = np.stack(ep['action'].values)  # [T, 6]
            grips = actions[:, 5]

            # 처음으로 grip이 크게 닫히는 프레임 찾기
            grip_closing = np.where(grips < grips.min() * 0.5)[0]
            if len(grip_closing) > 0:
                grasp_frame = grip_closing[0]
                # 그 순간의 state (인형에 접근한 로봇 자세)
                states = np.stack(ep['observation.state'].values)
                grasp_points.append(states[grasp_frame])

        if grasp_points:
            grasp_arr = np.array(grasp_points)  # [N_episodes, 6]
            print(f"  🎯 Grasp 순간 포착: {len(grasp_points)} / {all_data['episode_index'].nunique()} 에피소드")

            # 각 joint의 grasp 순간 분산
            grasp_std = np.std(grasp_arr, axis=0)
            grasp_mean = np.mean(grasp_arr, axis=0)

            print(f"\n  📊 Grasp 순간 joint별 분산 (각 에피소드 평균 ± 표준편차):")
            for i, name in enumerate(JOINT_NAMES):
                print(f"     {name:8s}: {grasp_mean[i]:>7.2f}° ± {grasp_std[i]:>5.2f}°")

            total_std = grasp_std.mean()
            print(f"\n  총 grasp 분산 평균: {total_std:.2f}°")

            if total_std < 3.0:
                print(f"  ✅ 시연 grasp 지점 일관됨 ({total_std:.2f}° < 3°)")
                print(f"     → 모델이 제대로 학습했다면 귀를 잡아야 함")
                print(f"     → 실전에서 몸통 잡으면 학습 문제")
            elif total_std < 7.0:
                print(f"  🔶 시연 grasp 지점 분산 중간 ({total_std:.2f}°)")
                print(f"     → 시연자 간 귀 포인트 약간 다름")
                print(f"     → 모델이 평균 = 몸통으로 수렴 가능")
            else:
                print(f"  ❌ 시연 grasp 지점 분산 큼 ({total_std:.2f}°)")
                print(f"     → 시연마다 귀/몸통 섞여 있음 = Mode averaging 유발")
                print(f"     → 빅맨님의 실전 관찰(10번 중 2번만 귀) 원인 확정")
                print(f"     💡 해결: 시연 재수집 시 '귀' 위치 엄격히 고정")

            scores["grasp_std"] = float(total_std)
            scores["grasp_n_episodes"] = len(grasp_points)
        else:
            print(f"  ⚠️  grasp 순간 감지 실패 (grip 패턴 다를 수 있음)")
    else:
        print(f"  ⚠️  데이터 parquet 없음")
except Exception as e:
    print(f"  ⚠️  데이터셋 분석 실패: {str(e)[:100]}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 8 (v2-D 재설계): 5개 Yellow 문장 각각의 Vision 반응
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 8: 5개 Yellow 문장 각각의 Vision 반응")
print("  → 각 문장이 Vision 입력에 비슷하게 반응하는가 (언어 강건성 보조 지표)")
print("─" * 70)

per_task_vision = []
for idx, task in enumerate(TASKS_YELLOW):
    base_t = get_action(policy, preprocess, BASE_IMG, BASE_IMG, BASE_STATE, task=task)
    diffs = []
    for img in test_imgs:
        a = get_action(policy, preprocess, img, img, task=task)
        diffs.append(np.abs(a - base_t).max())
    resp = sum(1 for d in diffs if d > VISION_THRESHOLD)
    mean = np.mean(diffs)
    g, _ = grade_vision(resp, len(test_imgs))
    short_task = task[:40] + "..."
    print(f"  [{idx}] {short_task:<45} 반응 {resp:>2}/{len(test_imgs)}  평균 {mean:.4f}°  등급 {g}")
    per_task_vision.append(mean)

task_vision_std = np.std(per_task_vision)
print(f"\n  5개 문장 Vision 반응 분산: {task_vision_std:.4f}°")
if task_vision_std < 0.05:
    print(f"  ✅ 5개 문장이 Vision에 동일하게 반응 (언어 강건)")
else:
    print(f"  🔶 문장별 Vision 반응 차이 있음 (학습 편향 가능)")

scores["per_task_vision_std"] = float(task_vision_std)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 6 (v2-D 재설계): 체크포인트별 Mode averaging 추이
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ 평가 6: 체크포인트별 Mode averaging 추이 (sweet spot 탐색)")
print("  → 학습 진행하며 chunk 궤적의 다양성이 어떻게 변하는가")
print("  → v13 단일 grasp point 학습 후 sweet spot 체크포인트 찾기")
print("─" * 70)

print(f"\n  {'체크포인트':>12} | {'최종 grasp 분산':>18} | {'등급':>6} | {'추천':>8}")
print(f"  {'-' * 60}")

ckpts_to_eval = ckpts[-10:] if len(ckpts) > 10 else ckpts
best_ckpt = None
best_std = 0
ckpt_trends = []

for ckpt_name in ckpts_to_eval:
    ckpt_path = os.path.join(CKPT_DIR, "checkpoints", ckpt_name, "pretrained_model")
    if not os.path.exists(ckpt_path):
        continue
    try:
        p = SmolVLAPolicy.from_pretrained(ckpt_path).to("cuda").eval()
        pp, _ = make_pre_post_processors(
            p.config, ckpt_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )

        # 여러 이미지로 chunk 추출 후 최종점 분산 측정
        local_finals = []
        for img in test_imgs[:3]:
            reset_policy_state(p)
            b = pp({CAM_WRIST: img, CAM_TOP: img,
                    "observation.state": collection_state,
                    "task": [TASK_DEFAULT]})
            with torch.no_grad():
                c = p.predict_action_chunk(b)[0].cpu().numpy()
            local_finals.append(c[-1])
        local_finals = np.array(local_finals)
        local_std = np.std(local_finals, axis=0).mean()

        g, _ = grade_mode_avg(local_std)
        recommend = ""
        if local_std > best_std:
            best_std = local_std
            best_ckpt = ckpt_name
            recommend = "⭐ 최고"

        print(f"  step {ckpt_name:>8} | {local_std:>15.4f}° | {g:>6} | {recommend:>8}")
        ckpt_trends.append((ckpt_name, local_std, g))

        del p, pp
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  step {ckpt_name:>8} | 로딩 실패: {str(e)[:30]}")

if best_ckpt:
    print(f"\n  ⭐ 추천 sweet spot 체크포인트: step {best_ckpt}")
    print(f"     최종 분산 {best_std:.4f}°")
    print(f"     추론 시 CKPT 경로를 이걸로 변경:")
    print(f"     .../checkpoints/{best_ckpt}/pretrained_model")
    scores["best_ckpt"] = best_ckpt
    scores["best_ckpt_std"] = float(best_std)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최종 종합 평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  최종 종합 평가 (v15 new_merged_100, v2-E 진단)")
print("=" * 70)

grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

# v2-D: v14는 Mode collapse가 주 병목이므로 collapse_grade 가중치 최고
# v13은 Mode averaging이 문제였지만 v14는 정반대 문제 (과단조로움)
total = (grade_map.get(scores.get("collapse_grade",   "F"), 0) * 5 +  # ⭐ v14 최우선 (공간 적응력)
         grade_map.get(scores.get("mode_avg_grade",   "F"), 0) * 2 +  # Vision 반응 (보조)
         grade_map.get(scores.get("consistency_grade", "F"), 0) * 3 +  # 언어 일관성
         grade_map.get(scores.get("config_grade",     "F"), 0) * 3 +
         grade_map.get(scores.get("vision_grade",     "F"), 0) * 3 +
         grade_map.get(scores.get("state_grade",      "F"), 0) * 2 +
         grade_map.get(scores.get("chunk_grade",      "F"), 0) * 1)
max_total = 4 * (5 + 2 + 3 + 3 + 3 + 2 + 1)
pct = total / max_total * 100

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  ⭐ Mode Collapse (공간 적응력): {scores.get('collapse_grade','?'):>2}      가중치 x5  │
  │  Mode Averaging (Vision 반응):  {scores.get('mode_avg_grade','?'):>2}      가중치 x2  │
  │  5문장 일관성:                   {scores.get('consistency_grade','?'):>2}      가중치 x3  │
  │  설정 (n_action_steps):         {scores.get('config_grade','?'):>2}      가중치 x3  │
  │  Vision 인코더:                 {scores.get('vision_grade','?'):>2}      가중치 x3  │
  │  State 인코더:                  {scores.get('state_grade','?'):>2}      가중치 x2  │
  │  청크 품질:                     {scores.get('chunk_grade','?'):>2}      가중치 x1  │
  ├──────────────────────────────────────────────────────────┤
  │  종합 점수: {total}/{max_total}  ({pct:.0f}%)                                  │
  └──────────────────────────────────────────────────────────┘""")

# 진단 결론
print("\n  ── 🔬 v15 진단 결론 ────────────────────────────────")

collapse_g = scores.get("collapse_grade", "F")
mode_g = scores.get("mode_avg_grade", "F")
cons_g = scores.get("consistency_grade", "F")

if collapse_g in ("A", "B") and cons_g in ("A", "B"):
    print(f"""
  🎉 v15 성공 — Mode COLLAPSE 해결!
     - 공간 적응력 등급: {collapse_g} (v14 F에서 개선 ⭐)
     - State 변이 시 chunk 최종점 분산: {scores.get('state_final_std', 0):.4f}°
     - Chunk 내부 움직임 범위: {scores.get('avg_chunk_range', 0):.2f}°
     - 5문장 일관성: {cons_g} ({scores.get('consistency_mean', 0):.4f}°) ← 언어 접지 유지

     🎯 빅맨님 가설 검증:
     "100ep(적음)인데도 v14 120ep보다 좋다면 = 데이터 품질 > 수량 실증"
     → new_merged_100의 계획된 다양성이 효과 입증!

  💡 다음 단계 (우선순위):

  1) [실전 검증] 10번 이상 반복 테스트
     ★ 인형 위치를 다양하게 (중앙/좌/우/앞/뒤)
     ★ 5개 태스크 순환
     ★ arm grasp 성공률 기록 (목표: 7/10 이상)

  2) [교육 콘텐츠] v13 vs v14 vs v15 3단 비교 HTML
     - 각 실패/성공의 원인 분석
     - "데이터 품질의 과학" 교재 완성
     - ZETA Satellite Robotics 수업 자료로 활용

  3) [발전 방향] Sweet spot 체크포인트 탐색
     - 평가 6에서 추천한 {scores.get('best_ckpt', '????')} 체크포인트
     - 소량 성능 차이 최적화
""")
elif collapse_g in ("F", "D"):
    print(f"""
  🚨 Mode COLLAPSE 지속 — v14 문제 미해결
     - 공간 적응력 등급: {collapse_g}
     - State 변이 시 chunk 최종점 분산: {scores.get('state_final_std', 0):.4f}°
     - Chunk 내부 움직임 범위: {scores.get('avg_chunk_range', 0):.2f}°
     - 5문장 일관성: {cons_g} ({scores.get('consistency_mean', 0):.4f}°)

     ⚠️ 예상 밖 결과 — new_merged_100도 collapse 발생
     → v13_yellow(3위치) + v14_yellow(2위치)의 다양성이 부족했거나
     → 병합 시 특정 데이터셋이 과대 비중을 가졌을 가능성

  💡 추가 해결책:

  1) [재수집] 더 많은 위치 (9구역 이상, v15_collection_plan 참고)
     - 3x3 그리드 × 11회 = 99ep
     - 계획된 수집 계획서 적용

  2) [체크포인트] Sweet spot 탐색 (평가 6)
     - 추천: {scores.get('best_ckpt', '????')} 체크포인트
     - 학습 초기가 더 다양한 궤적 가능성

  3) [모델 다양화] ACT 또는 Diffusion Policy 비교 실험
     - SmolVLA flow matching의 구조적 한계 우회
     - arXiv Much Ado About Noising (2025) 참고
""")
else:
    print(f"\n  🔶 혼합 상태 — 추가 분석 필요")
    print(f"     collapse: {collapse_g}, averaging: {mode_g}, consistency: {cons_g}")

print(f"""
  ── 📚 v13 vs v14 vs v15 3단 진단 비교 ──────────────────────
     v13 (pink, 120ep 분산)    : Mode AVERAGING (F) → 2/10 귀
     v14 (yellow, 120ep 단조)  : Mode COLLAPSE  (F) → arm 시도하나 단조
     v15 (yellow, 100ep 계획)  : 현재 → {collapse_g}/{mode_g} (목표: B↑)

     가설: 수량보다 품질 (100ep < 120ep 임에도 개선되면 검증)

  ── 핵심 숫자 ─────────────────────────────────────
     Vision 평균:              {scores.get('vision_mean', 0):.4f}°
     5문장 일관성 평균:        {scores.get('consistency_mean', 0):.4f}°  ({cons_g})
     Vision 다양성 (image)     : {scores.get('final_std', 0):.4f}°  ({mode_g})
     공간 적응력 (state)       : {scores.get('state_final_std', 0):.4f}°  ({collapse_g}) ⭐
     Chunk 움직임 범위         : {scores.get('avg_chunk_range', 0):.2f}°
     Chunk 내부 분산           : {scores.get('internal_std', 0):.4f}°
     학습 steps:               {scores.get('actual_step', 0):,}
     추천 sweet spot:          step {scores.get('best_ckpt', '????')}

  ── 🎓 교육적 가치 (ZETA Satellite Robotics) ──────────
     v13 vs v14 vs v15 3단 비교가 "데이터 설계의 과학"을 증명:
     • v13: 너무 다양 → Mode averaging (F)
     • v14: 너무 단조 → Mode collapse (F)
     • v15: 계획된 다양성 → 해결 검증 ⭐
     • 올바른 답: "grasp 일관성 유지 + 구조적 공간 다양성"
""")

print("=" * 70)
print(f"\n  📄 로그 파일: {LOG_FILE}")
