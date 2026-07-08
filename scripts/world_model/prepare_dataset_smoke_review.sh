#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-smoke05}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
SUMMARY="${SUMMARY:-${OUT_DIR}/summary.json}"
REVIEW_MD="${REVIEW_MD:-${OUT_DIR}/review_request.md}"
APPROVAL="${APPROVAL:-${OUT_DIR}/human_review_approved.txt}"
REJECTION="${REJECTION:-${OUT_DIR}/human_review_rejected.txt}"

if [[ ! -d "${OUT_DIR}" ]]; then
  echo "review_ready=false"
  echo "reason=output_dir_missing"
  echo "output_dir=${OUT_DIR}"
  exit 40
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "review_ready=false"
  echo "reason=summary_missing"
  echo "summary=${SUMMARY}"
  exit 41
fi

mapfile -t videos < <(find "${OUT_DIR}" -type f -name '*.mp4' | sort)
mapfile -t frames < <(find "${OUT_DIR}" -type f -path '*/review/frames/*.png' | sort | head -80)

if [[ "${#videos[@]}" -eq 0 ]]; then
  echo "review_ready=false"
  echo "reason=videos_missing"
  echo "output_dir=${OUT_DIR}"
  exit 42
fi

if [[ "${#frames[@]}" -eq 0 ]]; then
  echo "review_ready=false"
  echo "reason=review_frames_missing"
  echo "output_dir=${OUT_DIR}"
  exit 43
fi

{
  echo "# Dataset Smoke Review"
  echo
  echo "Run: ${RUN_GROUP}/${RUN_NAME}"
  echo
  echo "Summary: ${SUMMARY}"
  echo
  echo "Status: target blocked on human visual review."
  echo
  echo "Do not start full production or B/C/D/E production before this review"
  echo "is approved."
  echo
  echo "Human approval file to create after review:"
  echo
  echo '```text'
  echo "${APPROVAL}"
  echo '```'
  echo
  echo "Approval content:"
  echo
  echo '```text'
  echo "approved=true"
  echo '```'
  echo
  echo "Structured decision helper:"
  echo
  echo '```bash'
  echo "scripts/world_model/record_dataset_smoke_review_decision.sh --decision approved --reviewer <name> --notes <text>"
  echo "scripts/world_model/record_dataset_smoke_review_decision.sh --decision rejected --reviewer <name> --notes <text>"
  echo '```'
  echo
  echo "Rejection record path:"
  echo
  echo '```text'
  echo "${REJECTION}"
  echo '```'
  echo
  echo "Accept only if:"
  echo
  echo "- the video is non-empty and visually renders the ManiSkill peg insertion scene;"
  echo "- the camera, objects, peg, target/hole, and robot are visible enough for review;"
  echo "- there is no black/blank video, corrupt frame, severe flicker, disappearing object, or obvious render artifact;"
  echo "- this is understood as Stage 1 static RGB render quality approval, not dynamic-method success."
  echo
  echo "Reject or request rerender if:"
  echo
  echo "- the video is blank/corrupt or too small to inspect;"
  echo "- RGB content does not match the expected task;"
  echo "- render artifacts make downstream Cosmos training unreliable."
  echo
  echo "After approval:"
  echo
  echo '- `scripts/world_model/require_dataset_smoke_approved.sh` should pass;'
  echo "- A static RGB full production may be launched through the guarded Slurm tmux launcher;"
  echo "- B/C/D/E still require their own smoke gates and real in-allocation runners."
  echo
  echo "Videos:"
  echo
  for video in "${videos[@]}"; do
    size="$(wc -c < "${video}")"
    echo "- ${video} (${size} bytes)"
  done
  echo
  echo "Review frames:"
  echo
  for frame in "${frames[@]}"; do
    size="$(wc -c < "${frame}")"
    echo "- ${frame} (${size} bytes)"
  done
} > "${REVIEW_MD}"

echo "review_ready=true"
echo "review_request=${REVIEW_MD}"
echo "summary=${SUMMARY}"
echo "approval=${APPROVAL}"
echo "video_count=${#videos[@]}"
echo "review_frame_count=${#frames[@]}"
