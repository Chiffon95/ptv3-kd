#!/usr/bin/env bash
set -euo pipefail

# -------- User options --------
CFG="${1:-configs/train_s3dis.yaml}"   # 첫 번째 인자: config 경로 (기본값 위와 같음)
CUDA_DEVICES="${CUDA_DEVICES:-0}"      # 사용할 GPU 목록, 예: "0" 또는 "0,1"
RESUME="${RESUME:-}"                   # 이어서 학습할 ckpt 경로 (없으면 빈 값)
PYTHON_BIN="${PYTHON_BIN:-python}"     # 가상환경 python 지정 가능

# -------- Environment --------
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TORCH_USE_CUDA_DSA=0            # 디버그용 아니면 0 유지
export NVIDIA_TF32_OVERRIDE=1          # Amp + TF32 허용(연산속도↑, 정확도 영향 미미)

# -------- Create dirs if missing --------
LOG_DIR="$(awk '/^[[:space:]]*dir:[[:space:]]*/ && prev ~ /^log:/ {print $2}' "${CFG}" || true)"
CKPT_DIR="$(awk '/^[[:space:]]*dir:[[:space:]]*/ && prev ~ /^ckpt:/ {print $2}' "${CFG}" || true)"
prev=""
while IFS= read -r line; do prev="$line"; done < /dev/null # no-op to satisfy shellcheck

mkdir -p logs checkpoints || true

# -------- Info --------
echo "== PTv3KD Training =="
echo "CFG     : ${CFG}"
echo "GPU     : ${CUDA_VISIBLE_DEVICES}"
echo "RESUME  : ${RESUME:-<none>}"
echo "PYTHON  : ${PYTHON_BIN}"

# -------- Run --------
if [[ -n "${RESUME}" ]]; then
  # NOTE: 현재 train.py는 --cfg만 받음. RESUME은 config 파일에 넣어두는 걸 권장.
  # 빠르게 덮어쓰기 실행이 필요하면, 임시 YAML을 만들어 넣습니다.
  TMP_CFG="$(mktemp "/tmp/ptv3kd_cfg_XXXX.yaml")"
  cp "${CFG}" "${TMP_CFG}"
  # train.resume 필드가 있으면 교체, 없으면 train 블록 아래에 추가
  if grep -qE '^[[:space:]]*resume:' "${TMP_CFG}"; then
    sed -i "s|^[[:space:]]*resume:.*$|  resume: \"${RESUME}\"|g" "${TMP_CFG}"
  else
    awk '{
      print $0
      if($0 ~ /^train:[[:space:]]*$/ && !p){print "  resume: \"__RESUME__\""; p=1}
    }' "${TMP_CFG}" | sed "s|__RESUME__|${RESUME}|g" > "${TMP_CFG}.new"
    mv "${TMP_CFG}.new" "${TMP_CFG}"
  fi
  "${PYTHON_BIN}" -m src.train --cfg "${TMP_CFG}"
  rm -f "${TMP_CFG}"
else
  "${PYTHON_BIN}" -m src.train --cfg "${CFG}"
fi
