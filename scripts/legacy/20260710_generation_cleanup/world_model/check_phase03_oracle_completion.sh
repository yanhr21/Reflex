#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/experiments/maniskill/runs/03_oracle}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/public/home/yanhongru/ICLR2027/archive/Reflex}"

if ! command -v jq >/dev/null 2>&1; then
  echo "phase03_oracle_completion_check_ok=false"
  echo "reason=jq_missing"
  exit 2
fi

accepted_tmp="$(mktemp)"
strict_tmp="$(mktemp)"
trap 'rm -f "${accepted_tmp}" "${strict_tmp}"' EXIT

print_tsv_row() {
  local first=true
  local field
  for field in "$@"; do
    if [[ "${first}" == "true" ]]; then
      printf '%s' "${field}"
      first=false
    else
      printf '\t%s' "${field}"
    fi
  done
  printf '\n'
}

while IFS= read -r -d '' verdict_path; do
  accepted="$(
    jq -r '
      (.validation_key_single_case_success_confirmed == true)
      or (.single_case_oracle_success_confirmed == true)
      or (.verdict == "accepted_single_case_oracle_success")
    ' "${verdict_path}"
  )"
  if [[ "${accepted}" != "true" ]]; then
    continue
  fi
  verdict_dir="$(dirname "${verdict_path}")"
  if [[ "$(basename "${verdict_dir}")" == "review" ]]; then
    run_dir="$(dirname "${verdict_dir}")"
  else
    run_dir="${verdict_dir}"
  fi
  summary_path="${run_dir}/summary.json"
  artifact_audit_path="${run_dir}/artifact_audit.json"
  action_trace_path="${run_dir}/action_trace.json"
  rel="${run_dir#${RUN_ROOT}/}"
  group="${rel%%/*}"
  attempt="${rel#*/}"
  attempt="${attempt%%/*}"
  source_key="$(jq -r '.source_key // empty' "${verdict_path}")"
  if [[ -z "${source_key}" && -f "${summary_path}" ]]; then
    source_key="$(jq -r '.source_key // empty' "${summary_path}")"
  fi
  source_key="${source_key:-unknown_source_key}"
  verdict="$(jq -r '.verdict // "accepted_by_boolean_flag"' "${verdict_path}")"
  visual_full="$(jq -r 'if has("visual_full_insertion_confirmed") then .visual_full_insertion_confirmed else empty end' "${verdict_path}")"
  if [[ -z "${visual_full}" && -f "${summary_path}" ]]; then
    visual_full="$(jq -r 'if has("visual_full_insertion_confirmed") then .visual_full_insertion_confirmed else empty end' "${summary_path}")"
  fi
  visual_full="${visual_full:-false}"
  target_assisted_rejected="$(jq -r '.target_assisted_insertion_rejected // false' "${verdict_path}")"
  active_robot_insertion_confirmed="$(jq -r '.active_robot_insertion_confirmed // false' "${verdict_path}")"
  cosmos_rgb_prediction_confirmed="$(jq -r '.cosmos_rgb_prediction_confirmed // false' "${verdict_path}")"
  cosmos_action_prediction_confirmed="$(jq -r '.cosmos_action_prediction_confirmed // false' "${verdict_path}")"
  full_sequence_video_reviewed="$(jq -r '.full_sequence_video_reviewed // false' "${verdict_path}")"
  video_covers_cosmos_control_and_finisher="$(jq -r '.video_covers_cosmos_control_and_finisher // false' "${verdict_path}")"
  video_covers_final_insertion_or_physical_failure="$(jq -r '.video_covers_final_insertion_or_physical_failure // false' "${verdict_path}")"
  no_snap_or_teleport_observed="$(jq -r '.no_snap_or_teleport_observed // false' "${verdict_path}")"
  no_wall_observed="$(jq -r '.no_wall_insertion_or_wall_penetration_observed // false' "${verdict_path}")"
  no_disappearing_objects_observed="$(jq -r '.no_disappearing_objects_observed // false' "${verdict_path}")"
  simulator_success="$(jq -r 'if has("simulator_success_metric") then .simulator_success_metric else empty end' "${verdict_path}")"
  if [[ -z "${simulator_success}" && -f "${summary_path}" ]]; then
    simulator_success="$(jq -r 'if has("simulator_success_metric") then .simulator_success_metric else empty end' "${summary_path}")"
  fi
  simulator_success="${simulator_success:-false}"
  method_allowed="$(jq -r 'if has("method_evidence_allowed") then .method_evidence_allowed else empty end' "${verdict_path}")"
  if [[ -z "${method_allowed}" && -f "${summary_path}" ]]; then
    method_allowed="$(jq -r 'if has("method_evidence_allowed") then .method_evidence_allowed else empty end' "${summary_path}")"
  fi
  method_allowed="${method_allowed:-missing}"
  overall_complete="$(jq -r '.overall_task_complete // false' "${verdict_path}")"
  summary_cosmos_dynamic_count=0
  min_cosmos_dynamic_before_finisher=4
  no_snap=false
  if [[ -f "${summary_path}" ]]; then
    summary_cosmos_dynamic_count="$(jq -r '.cosmos_dynamic_action_count // 0' "${summary_path}")"
    min_cosmos_dynamic_before_finisher="$(jq -r '.min_cosmos_dynamic_actions_before_finisher // 4' "${summary_path}")"
    no_snap="$(jq -r '(.discontinuity_audit.snap_detected == false)' "${summary_path}")"
  fi
  audit_ok=false
  premotion_reports=0
  required_premotion_reports=2
  postmotion_reports=0
  audit_dynamic_rows=0
  source_h5_ok=false
  diagnostic_exclusion_ok=false
  if [[ -f "${artifact_audit_path}" ]]; then
    audit_ok="$(jq -r 'if has("ok") then .ok else false end' "${artifact_audit_path}")"
    premotion_reports="$(jq -r '.premotion_cosmos_report_count // 0' "${artifact_audit_path}")"
    required_premotion_reports="$(jq -r '.required_premotion_cosmos_report_count // 2' "${artifact_audit_path}")"
    postmotion_reports="$(jq -r '.postmotion_cosmos_report_count // 0' "${artifact_audit_path}")"
    audit_dynamic_rows="$(jq -r '.cosmos_dynamic_rows // 0' "${artifact_audit_path}")"
    source_h5_ok="$(
      jq -r '
        (.source_h5_path | type == "string" and length > 0)
        and (.source_key | type == "string" and length > 0)
        and (.target_motion_source == "fix3_733_source_h5_protocol")
        and (.validation_key_success_allowed == true)
      ' "${artifact_audit_path}"
    )"
  fi
  if [[ -f "${summary_path}" ]]; then
    diagnostic_exclusion_ok="$(
      jq -r '
        (.synthetic_target_motion_diagnostic_only != true)
        and ((.cosmos_action_adapter.action_row_offset_diagnostic // false) != true)
        and ((.dynamic_controller.future_label_teacher_dynamic_diagnostic // false) != true)
        and ((.finisher.future_label_teacher_suffix_diagnostic // false) != true)
        and ((.source_h5_teacher_suffix_enabled // false) != true)
        and ((.source_h5_teacher_dynamic_enabled // false) != true)
      ' "${summary_path}"
    )"
  fi
  trace_dynamic_rows=0
  trace_all_cosmos=false
  trace_finisher_rows=0
  first_target_motion_trace_index=""
  dp_static_after_target_motion_count=0
  first_dynamic_trace_index=""
  first_finisher_trace_index=""
  stage_order_ok=false
  first_success_trace_index=""
  first_success_stage=""
  first_success_stage_is_finisher=false
  success_after_finisher_start=false
  pre_prefix_rgb_videos=0
  pre_vision_videos=0
  post_prefix_rgb_videos=0
  post_vision_videos=0
  rendered_raw_video_present=false
  rendered_annotated_video_present=false
  distance_evidence_ok=false
  initial_peg_head_l2=""
  pre_finisher_peg_head_l2=""
  final_peg_head_l2=""
  if [[ -f "${action_trace_path}" ]]; then
    trace_dynamic_rows="$(jq -r '[.[] | select(.stage == "cosmos_dynamic_control")] | length' "${action_trace_path}")"
    trace_all_cosmos="$(jq -r '[.[] | select(.stage == "cosmos_dynamic_control") | .action_source] | (length > 0 and all(. == "cosmos3_policy_output"))' "${action_trace_path}")"
    trace_finisher_rows="$(
      jq -r '
        [
          .[]
          | select(
              .stage == "oracle_physical_dp_finisher"
              or .stage == "oracle_physical_manual_finisher"
            )
        ]
        | length
      ' "${action_trace_path}"
    )"
    first_target_motion_trace_index="$(
      jq -r '
        to_entries
        | map(
            select(
              ((.value.target_motion_delta_xyz // [0,0,0]) | tostring) != ([0,0,0] | tostring)
            )
          )
        | if length == 0 then "" else .[0].key end
      ' "${action_trace_path}"
    )"
    if [[ -n "${first_target_motion_trace_index}" ]]; then
      dp_static_after_target_motion_count="$(
        jq -r --argjson first "${first_target_motion_trace_index}" '
          [
            to_entries[]
            | select(.key > $first)
            | select(.value.stage == "dp_static_prefix")
          ]
          | length
        ' "${action_trace_path}"
      )"
    fi
    first_dynamic_trace_index="$(
      jq -r '
        to_entries
        | map(select(.value.stage == "cosmos_dynamic_control"))
        | if length == 0 then "" else .[0].key end
      ' "${action_trace_path}"
    )"
    first_finisher_trace_index="$(
      jq -r '
        to_entries
        | map(
            select(
              .value.stage == "oracle_physical_dp_finisher"
              or .value.stage == "oracle_physical_manual_finisher"
            )
          )
        | if length == 0 then "" else .[0].key end
      ' "${action_trace_path}"
    )"
    if [[ -n "${first_target_motion_trace_index}" \
      && -n "${first_dynamic_trace_index}" \
      && -n "${first_finisher_trace_index}" \
      && "${dp_static_after_target_motion_count}" -eq 0 \
      && "${first_dynamic_trace_index}" -gt "${first_target_motion_trace_index}" \
      && "${first_finisher_trace_index}" -gt "${first_dynamic_trace_index}" ]]; then
      stage_order_ok=true
    fi
    first_success_trace_index="$(
      jq -r '
        to_entries
        | map(select((.value.live_eval.success // false) == true))
        | if length == 0 then "" else .[0].key end
      ' "${action_trace_path}"
    )"
    if [[ -n "${first_success_trace_index}" ]]; then
      first_success_stage="$(
        jq -r --argjson first_success "${first_success_trace_index}" '
          .[$first_success].stage // ""
        ' "${action_trace_path}"
      )"
      if [[ "${first_success_stage}" == "oracle_physical_dp_finisher" || "${first_success_stage}" == "oracle_physical_manual_finisher" ]]; then
        first_success_stage_is_finisher=true
      fi
      success_after_finisher_start="$(
        jq -r --argjson first_success "${first_success_trace_index}" '
          (
            to_entries
            | map(
                select(
                  .value.stage == "oracle_physical_dp_finisher"
                  or .value.stage == "oracle_physical_manual_finisher"
                )
              )
            | if length == 0 then null else .[0].key end
          ) as $finisher_start
          | ($finisher_start != null and $first_success >= $finisher_start)
        ' "${action_trace_path}"
      )"
    fi
  fi
  if [[ -d "${run_dir}/cosmos_policy/pre" ]]; then
    pre_prefix_rgb_videos="$(find "${run_dir}/cosmos_policy/pre" -type f -name 'prefix_rgb.mp4' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
    pre_vision_videos="$(find "${run_dir}/cosmos_policy/pre" -type f -path '*/outputs/sample/vision.mp4' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if [[ -d "${run_dir}/cosmos_policy/post" ]]; then
    post_prefix_rgb_videos="$(find "${run_dir}/cosmos_policy/post" -type f -name 'prefix_rgb.mp4' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
    post_vision_videos="$(find "${run_dir}/cosmos_policy/post" -type f -path '*/outputs/sample/vision.mp4' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if [[ -s "${run_dir}/videos/raw.mp4" ]]; then
    rendered_raw_video_present=true
  fi
  if [[ -s "${run_dir}/videos/annotated.mp4" ]]; then
    rendered_annotated_video_present=true
  fi
  near_target_before_finisher=false
  finisher_start_step=""
  if [[ -f "${summary_path}" ]]; then
    near_target_before_finisher="$(jq -r '.near_target_before_finisher // false' "${summary_path}")"
    finisher_start_step="$(jq -r 'if .finisher_start_step == null then "" else .finisher_start_step end' "${summary_path}")"
    distance_evidence_ok="$(
      jq -r '
        (.initial_eval.peg_head_l2 | type == "number")
        and (.target_motion_live_gate.peg_head_l2 | type == "number")
        and (.final_eval.peg_head_l2 | type == "number")
        and (.final_eval.success == true)
        and (.final_success == true)
        and (.final_eval.peg_head_l2 < .target_motion_live_gate.peg_head_l2)
      ' "${summary_path}"
    )"
    initial_peg_head_l2="$(jq -r 'if (.initial_eval.peg_head_l2 | type == "number") then .initial_eval.peg_head_l2 else "" end' "${summary_path}")"
    pre_finisher_peg_head_l2="$(jq -r 'if (.target_motion_live_gate.peg_head_l2 | type == "number") then .target_motion_live_gate.peg_head_l2 else "" end' "${summary_path}")"
    final_peg_head_l2="$(jq -r 'if (.final_eval.peg_head_l2 | type == "number") then .final_eval.peg_head_l2 else "" end' "${summary_path}")"
  fi
  protocol_artifacts_ok=false
  if [[ -f "${summary_path}" && -f "${artifact_audit_path}" && -f "${action_trace_path}" \
    && "${audit_ok}" == "true" \
    && "${source_h5_ok}" == "true" \
    && "${diagnostic_exclusion_ok}" == "true" \
    && "${premotion_reports}" -ge "${required_premotion_reports}" \
    && "${postmotion_reports}" -gt 0 \
    && "${pre_prefix_rgb_videos}" -ge "${required_premotion_reports}" \
    && "${pre_vision_videos}" -ge "${required_premotion_reports}" \
    && "${post_prefix_rgb_videos}" -gt 0 \
    && "${post_vision_videos}" -gt 0 \
    && "${rendered_raw_video_present}" == "true" \
    && "${rendered_annotated_video_present}" == "true" \
    && "${distance_evidence_ok}" == "true" \
    && "${audit_dynamic_rows}" -ge 4 \
    && "${trace_dynamic_rows}" -ge 4 \
    && "${trace_all_cosmos}" == "true" \
    && "${trace_finisher_rows}" -gt 0 \
    && -n "${first_target_motion_trace_index}" \
    && "${dp_static_after_target_motion_count}" -eq 0 \
    && "${stage_order_ok}" == "true" \
    && -n "${first_success_trace_index}" \
    && "${success_after_finisher_start}" == "true" \
    && "${first_success_stage_is_finisher}" == "true" \
    && "${near_target_before_finisher}" == "true" \
    && -n "${finisher_start_step}" \
    && "${summary_cosmos_dynamic_count}" -ge "${min_cosmos_dynamic_before_finisher}" \
    && "${summary_cosmos_dynamic_count}" -ge 4 \
    && "${no_snap}" == "true" ]]; then
    protocol_artifacts_ok=true
  fi
  print_tsv_row \
    "${rel}" \
    "${group}" \
    "${attempt}" \
    "${source_key}" \
    "${verdict}" \
    "${visual_full}" \
    "${target_assisted_rejected}" \
    "${simulator_success}" \
    "${method_allowed}" \
    "${overall_complete}" \
    "${protocol_artifacts_ok}" \
    "${audit_ok}" \
    "${source_h5_ok}" \
    "${diagnostic_exclusion_ok}" \
    "${premotion_reports}" \
    "${required_premotion_reports}" \
    "${postmotion_reports}" \
    "${audit_dynamic_rows}" \
    "${trace_dynamic_rows}" \
    "${trace_all_cosmos}" \
    "${no_snap}" \
    "${trace_finisher_rows}" \
    "${near_target_before_finisher}" \
    "${finisher_start_step}" \
    "${pre_prefix_rgb_videos}" \
    "${pre_vision_videos}" \
    "${post_prefix_rgb_videos}" \
    "${post_vision_videos}" \
    "${active_robot_insertion_confirmed}" \
    "${rendered_raw_video_present}" \
    "${rendered_annotated_video_present}" \
    "${distance_evidence_ok}" \
    "${initial_peg_head_l2}" \
    "${pre_finisher_peg_head_l2}" \
    "${final_peg_head_l2}" \
    "${no_snap_or_teleport_observed}" \
    "${no_wall_observed}" \
    "${no_disappearing_objects_observed}" \
    "${cosmos_rgb_prediction_confirmed}" \
    "${cosmos_action_prediction_confirmed}" \
    "${full_sequence_video_reviewed}" \
    "${video_covers_cosmos_control_and_finisher}" \
    "${video_covers_final_insertion_or_physical_failure}" \
    "${first_target_motion_trace_index}" \
    "${dp_static_after_target_motion_count}" \
    "${first_dynamic_trace_index}" \
    "${first_finisher_trace_index}" \
    "${stage_order_ok}" \
    "${first_success_trace_index}" \
    "${first_success_stage}" \
    "${success_after_finisher_start}" >>"${accepted_tmp}"
  if [[ "${visual_full}" == "true" && "${target_assisted_rejected}" == "true" && "${active_robot_insertion_confirmed}" == "true" && "${cosmos_rgb_prediction_confirmed}" == "true" && "${cosmos_action_prediction_confirmed}" == "true" && "${full_sequence_video_reviewed}" == "true" && "${video_covers_cosmos_control_and_finisher}" == "true" && "${video_covers_final_insertion_or_physical_failure}" == "true" && "${no_snap_or_teleport_observed}" == "true" && "${no_wall_observed}" == "true" && "${no_disappearing_objects_observed}" == "true" && "${simulator_success}" == "true" && "${method_allowed}" == "false" && "${overall_complete}" == "false" && "${protocol_artifacts_ok}" == "true" ]]; then
    print_tsv_row \
      "${rel}" \
      "${group}" \
      "${attempt}" \
      "${source_key}" \
      "${verdict}" \
      "${visual_full}" \
      "${target_assisted_rejected}" \
      "${simulator_success}" \
      "${method_allowed}" \
      "${overall_complete}" \
      "${protocol_artifacts_ok}" \
      "${audit_ok}" \
      "${source_h5_ok}" \
      "${diagnostic_exclusion_ok}" \
      "${premotion_reports}" \
      "${required_premotion_reports}" \
      "${postmotion_reports}" \
      "${audit_dynamic_rows}" \
      "${trace_dynamic_rows}" \
      "${trace_all_cosmos}" \
      "${no_snap}" \
      "${trace_finisher_rows}" \
      "${near_target_before_finisher}" \
      "${finisher_start_step}" \
      "${pre_prefix_rgb_videos}" \
      "${pre_vision_videos}" \
      "${post_prefix_rgb_videos}" \
      "${post_vision_videos}" \
      "${active_robot_insertion_confirmed}" \
      "${rendered_raw_video_present}" \
      "${rendered_annotated_video_present}" \
      "${distance_evidence_ok}" \
      "${initial_peg_head_l2}" \
      "${pre_finisher_peg_head_l2}" \
      "${final_peg_head_l2}" \
      "${no_snap_or_teleport_observed}" \
      "${no_wall_observed}" \
      "${no_disappearing_objects_observed}" \
      "${cosmos_rgb_prediction_confirmed}" \
      "${cosmos_action_prediction_confirmed}" \
      "${full_sequence_video_reviewed}" \
      "${video_covers_cosmos_control_and_finisher}" \
      "${video_covers_final_insertion_or_physical_failure}" \
      "${first_target_motion_trace_index}" \
      "${dp_static_after_target_motion_count}" \
      "${first_dynamic_trace_index}" \
      "${first_finisher_trace_index}" \
      "${stage_order_ok}" \
      "${first_success_trace_index}" \
      "${first_success_stage}" \
      "${success_after_finisher_start}" >>"${strict_tmp}"
  fi
done < <(find "${RUN_ROOT}" -path '*/visual_review_verdict.json' -print0 2>/dev/null | sort -z)

accepted_count="$(wc -l <"${accepted_tmp}" | tr -d ' ')"
strict_count="$(wc -l <"${strict_tmp}" | tr -d ' ')"
unique_source_key_count="$(
  if [[ -s "${accepted_tmp}" ]]; then
    cut -f4 "${accepted_tmp}" | sort -u | wc -l | tr -d ' '
  else
    echo 0
  fi
)"
strict_unique_source_key_count="$(
  if [[ -s "${strict_tmp}" ]]; then
    cut -f4 "${strict_tmp}" | sort -u | wc -l | tr -d ' '
  else
    echo 0
  fi
)"

count_group() {
  local table="$1"
  local group="$2"
  awk -F '\t' -v group="${group}" '$2 == group { count += 1 } END { print count + 0 }' "${table}"
}

count_accepted_group() {
  local group="$1"
  count_group "${accepted_tmp}" "${group}"
}

count_strict_group() {
  local group="$1"
  count_group "${strict_tmp}" "${group}"
}

continuous_count="$(count_accepted_group h5_continuous_insert)"
fastshift_count="$(count_accepted_group h5_fastshift)"
reverse_count="$(count_accepted_group h5_reverse)"
move_stop_count="$(count_accepted_group h5_move_stop)"
peg_disturb_count="$(count_accepted_group peg_disturb)"
strict_continuous_count="$(count_strict_group h5_continuous_insert)"
strict_fastshift_count="$(count_strict_group h5_fastshift)"
strict_reverse_count="$(count_strict_group h5_reverse)"
strict_move_stop_count="$(count_strict_group h5_move_stop)"
strict_peg_disturb_count="$(count_strict_group peg_disturb)"
missing_target_assisted_review_count="$(
  awk -F '\t' '$7 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}"
)"

forward_backward_count="$((reverse_count + move_stop_count))"
left_right_count="${fastshift_count}"
strict_forward_backward_count="$((strict_reverse_count + strict_move_stop_count))"
strict_left_right_count="${strict_fastshift_count}"
required_forward_backward_success="$([[ "${strict_forward_backward_count}" -gt 0 ]] && echo true || echo false)"
required_left_right_success="$([[ "${strict_left_right_count}" -gt 0 ]] && echo true || echo false)"
required_peg_or_stick_disturb_success="$([[ "${strict_peg_disturb_count}" -gt 0 ]] && echo true || echo false)"
required_multiple_approved_keys_success="$([[ "${strict_unique_source_key_count}" -ge 3 ]] && echo true || echo false)"
required_modern_target_assisted_review_success="$([[ "${missing_target_assisted_review_count}" -eq 0 ]] && echo true || echo false)"

missing_coverage_items=()
if [[ "${required_forward_backward_success}" != "true" ]]; then
  missing_coverage_items+=(forward_backward_target_motion)
fi
if [[ "${required_left_right_success}" != "true" ]]; then
  missing_coverage_items+=(left_right_target_motion)
fi
if [[ "${required_peg_or_stick_disturb_success}" != "true" ]]; then
  missing_coverage_items+=(peg_or_wooden_stick_disturbance)
fi
if [[ "${required_multiple_approved_keys_success}" != "true" ]]; then
  missing_coverage_items+=(multiple_approved_fix3_733_keys)
fi
if [[ "${required_modern_target_assisted_review_success}" != "true" ]]; then
  missing_coverage_items+=(modern_target_assisted_review)
fi
missing_coverage_csv="$(
  IFS=,
  echo "${missing_coverage_items[*]}"
)"
next_required_coverage_group="none"
if [[ "${#missing_coverage_items[@]}" -gt 0 ]]; then
  next_required_coverage_group="${missing_coverage_items[0]}"
fi

echo "phase03_oracle_completion_check_ok=true"
echo "run_root=${RUN_ROOT}"
echo "accepted_single_case_count=${accepted_count}"
echo "accepted_unique_source_key_count=${unique_source_key_count}"
echo "modern_strict_single_case_count=${strict_count}"
echo "modern_strict_unique_source_key_count=${strict_unique_source_key_count}"
echo "accepted_h5_continuous_insert_count=${continuous_count}"
echo "accepted_forward_backward_group_count=${forward_backward_count}"
echo "accepted_left_right_group_count=${left_right_count}"
echo "accepted_peg_disturb_count=${peg_disturb_count}"
echo "accepted_rows_missing_protocol_artifact_check_count=$(awk -F '\t' '$11 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_source_h5_check_count=$(awk -F '\t' '$13 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_failing_diagnostic_exclusion_count=$(awk -F '\t' '$14 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_physical_finisher_count=$(awk -F '\t' '$22 <= 0 || $23 != "true" || $24 == "" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_cosmos_rgb_video_count=$(awk -F '\t' '$25 < $16 || $26 < $16 || $27 <= 0 || $28 <= 0 { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_active_robot_insertion_count=$(awk -F '\t' '$29 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_rendered_video_count=$(awk -F '\t' '$30 != "true" || $31 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_distance_evidence_count=$(awk -F '\t' '$32 != "true" || $33 == "" || $34 == "" || $35 == "" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_physical_validity_visual_flags_count=$(awk -F '\t' '$36 != "true" || $37 != "true" || $38 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_cosmos_visual_confirmation_count=$(awk -F '\t' '$39 != "true" || $40 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_missing_full_sequence_video_review_count=$(awk -F '\t' '$41 != "true" || $42 != "true" || $43 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_with_overall_complete_flag_count=$(awk -F '\t' '$10 == "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_with_dp_static_after_target_motion_count=$(awk -F '\t' '$44 == "" || $45 != 0 { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_bad_stage_order_count=$(awk -F '\t' '$46 == "" || $47 == "" || $48 != "true" { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "accepted_rows_success_before_finisher_count=$(awk -F '\t' '$49 == "" || $51 != "true" || ($50 != "oracle_physical_dp_finisher" && $50 != "oracle_physical_manual_finisher") { count += 1 } END { print count + 0 }' "${accepted_tmp}")"
echo "modern_strict_h5_continuous_insert_count=${strict_continuous_count}"
echo "modern_strict_forward_backward_group_count=${strict_forward_backward_count}"
echo "modern_strict_left_right_group_count=${strict_left_right_count}"
echo "modern_strict_peg_disturb_count=${strict_peg_disturb_count}"
echo "accepted_rows_missing_target_assisted_rejection_count=${missing_target_assisted_review_count}"
echo "required_forward_backward_success=${required_forward_backward_success}"
echo "required_left_right_success=${required_left_right_success}"
echo "required_peg_or_stick_disturb_success=${required_peg_or_stick_disturb_success}"
echo "required_multiple_approved_keys_success=${required_multiple_approved_keys_success}"
echo "required_modern_target_assisted_review_success=${required_modern_target_assisted_review_success}"
echo "missing_coverage_items=${missing_coverage_csv}"
echo "next_required_coverage_group=${next_required_coverage_group}"

if [[ -s "${accepted_tmp}" ]]; then
  echo "--- accepted_single_case_rows ---"
  echo "rel_path	group	attempt	source_key	verdict	visual_full	target_assisted_rejected	simulator_success	method_evidence_allowed	overall_task_complete	protocol_artifacts_ok	artifact_audit_ok	source_h5_ok	diagnostic_exclusion_ok	premotion_reports	required_premotion_reports	postmotion_reports	audit_dynamic_rows	trace_dynamic_rows	trace_all_cosmos	no_snap	trace_finisher_rows	near_target_before_finisher	finisher_start_step	pre_prefix_rgb_videos	pre_vision_videos	post_prefix_rgb_videos	post_vision_videos	active_robot_insertion_confirmed	rendered_raw_video_present	rendered_annotated_video_present	distance_evidence_ok	initial_peg_head_l2	pre_finisher_peg_head_l2	final_peg_head_l2	no_snap_or_teleport_observed	no_wall_insertion_or_wall_penetration_observed	no_disappearing_objects_observed	cosmos_rgb_prediction_confirmed	cosmos_action_prediction_confirmed	full_sequence_video_reviewed	video_covers_cosmos_control_and_finisher	video_covers_final_insertion_or_physical_failure	first_target_motion_trace_index	dp_static_after_target_motion_count	first_dynamic_trace_index	first_finisher_trace_index	stage_order_ok	first_success_trace_index	first_success_stage	success_after_finisher_start"
  cat "${accepted_tmp}"
fi

if [[ -s "${strict_tmp}" ]]; then
  echo "--- modern_strict_single_case_rows ---"
  echo "rel_path	group	attempt	source_key	verdict	visual_full	target_assisted_rejected	simulator_success	method_evidence_allowed	overall_task_complete	protocol_artifacts_ok	artifact_audit_ok	source_h5_ok	diagnostic_exclusion_ok	premotion_reports	required_premotion_reports	postmotion_reports	audit_dynamic_rows	trace_dynamic_rows	trace_all_cosmos	no_snap	trace_finisher_rows	near_target_before_finisher	finisher_start_step	pre_prefix_rgb_videos	pre_vision_videos	post_prefix_rgb_videos	post_vision_videos	active_robot_insertion_confirmed	rendered_raw_video_present	rendered_annotated_video_present	distance_evidence_ok	initial_peg_head_l2	pre_finisher_peg_head_l2	final_peg_head_l2	no_snap_or_teleport_observed	no_wall_insertion_or_wall_penetration_observed	no_disappearing_objects_observed	cosmos_rgb_prediction_confirmed	cosmos_action_prediction_confirmed	full_sequence_video_reviewed	video_covers_cosmos_control_and_finisher	video_covers_final_insertion_or_physical_failure	first_target_motion_trace_index	dp_static_after_target_motion_count	first_dynamic_trace_index	first_finisher_trace_index	stage_order_ok	first_success_trace_index	first_success_stage	success_after_finisher_start"
  cat "${strict_tmp}"
fi

legacy_boundary="${ARCHIVE_ROOT}/experiments/maniskill/runs/03_oracle/p03_oracle_no_teleport_trace_20260703_151937_162757_mgmtserver02"
if [[ -d "${legacy_boundary}" ]]; then
  echo "legacy_boundary_only_trace_archived=true"
  echo "legacy_boundary_only_trace_path=${legacy_boundary}"
  echo "legacy_boundary_only_trace_counts_as_success=false"
fi

overall_complete=false
if [[ "${strict_forward_backward_count}" -gt 0 && "${strict_left_right_count}" -gt 0 && "${strict_peg_disturb_count}" -gt 0 && "${strict_unique_source_key_count}" -ge 3 && "${missing_target_assisted_review_count}" -eq 0 ]]; then
  overall_complete=true
fi
echo "phase03_oracle_overall_complete=${overall_complete}"
if [[ "${overall_complete}" != "true" ]]; then
  exit 3
fi
