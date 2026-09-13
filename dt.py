"""
run_pipeline_multimodal.py

Multi-modal (Head + Lip + Hand) active speaker detection + AMI
ground-truth scoring, in one run. Same structure as run_pipeline.py /
run_pipeline_lips.py / run_pipeline_hands.py, adapted to your fused
detector.

Just run:
    python run_pipeline_multimodal.py --meetings-xml "C:\\path\\to\\meetings.xml"

Fixes applied vs. your original multi-modal script:
  1. Two stale-reference bugs, same class as the ones fixed in the
     single-signal scripts:
       a. When no face is detected in a frame, `bbox` is None and the
          entire lip block is skipped -- but `state.prev_lip_shape` was
          never cleared, so if a face reappears N frames later, the lip
          movement is computed against a shape from N frames ago instead
          of being treated as "no baseline yet". Now reset to None
          whenever bbox is None.
       b. Same issue for `state.prev_head_center`: previously only updated
          on a successful detection, never cleared on a miss, so a
          re-detected face after a gap could register a fake big jump.
          Now reset to None on a miss.
     (Note: your hand-movement design already sidesteps the worse version
     of this problem that the hands-only script has, because you aggregate
     to a single mean centroid across all detected hands instead of
     pairing individual landmarks by list position -- that's a solid
     design choice, kept as-is.)
  2. Same fps-consistency check across all 4 videos (was trusting Person1 only).
  3. Same "Silence" decision using --movement-threshold on the combined
     `total` score, instead of always crowning a winner when everyone's
     near 0.
  4. Output CSV now stores WindowStartSec/WindowEndSec (seconds) instead
     of only a raw Frame number, needed to align to the AMI transcript.
  5. Added --headless mode, file-existence checks for all 3 models + 4
     videos, auto-discovery of meetings.xml + segments/words xml (dotted
     or underscored naming).
  6. Runs the AMI evaluation immediately after detection, in the same run.
  7. Config moved from hardcoded module-level constants to CLI args with
     the same defaults, so you can override anything without editing the
     file (weights, thresholds, window, paths all included).

Same Person->Closeup->AMI-channel caveat as before, resolved automatically
from meetings.xml.
"""

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode

# ----------------------------------------------------------------------
# EDIT THESE IF YOUR FOLDERS ARE DIFFERENT
# ----------------------------------------------------------------------
DEFAULT_VIDEO_DIR = r"C:\Users\Ehtiram\Desktop\DS_Project\ES2008a\video"
DEFAULT_AMI_DIR = r"C:\Users\Ehtiram\Desktop\DS_Project\ES2008a\video"
DEFAULT_FACE_DETECTOR_MODEL = r"C:\Users\Ehtiram\Downloads\blaze_face_short_range.tflite"
DEFAULT_FACE_LANDMARKER_MODEL = r"C:\Users\Ehtiram\Downloads\face_landmarker.task"
DEFAULT_HAND_LANDMARKER_MODEL = r"C:\Users\Ehtiram\Downloads\hand_landmarker.task"
DEFAULT_MEETING_ID = "ES2008a"
# ----------------------------------------------------------------------

NITE_NS = "{http://nite.sourceforge.net/}"
HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")

LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 78,
    191, 80, 81, 82, 13, 312, 311,
    310, 415
]

FACE_DETECTOR_MIN_CONF = 0.35
FACE_LM_MIN_FACE_DET_CONF = 0.35
FACE_LM_MIN_FACE_PRES_CONF = 0.35
FACE_LM_MIN_TRACK_CONF = 0.35
HAND_MIN_DET_CONF = 0.35
HAND_MIN_PRES_CONF = 0.35
HAND_MIN_TRACK_CONF = 0.35
MAX_HANDS_PER_PERSON = 2


# ----------------------------------------------------------------------
# File discovery helpers (same as the other pipeline scripts)
# ----------------------------------------------------------------------

def find_file(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_meetings_xml(search_dirs):
    for d in search_dirs:
        if not d:
            continue
        for depth_dir in [d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))]:
            candidate = os.path.join(depth_dir, "meetings.xml")
            if os.path.exists(candidate):
                return candidate
            matches = glob.glob(os.path.join(depth_dir, "*meetings.xml"))
            if matches:
                return matches[0]
    return None


def channel_files(ami_dir, meeting_id, channel):
    segs = find_file(
        os.path.join(ami_dir, f"{meeting_id}.{channel}.segments.xml"),
        os.path.join(ami_dir, f"{meeting_id}_{channel}_segments.xml"),
    )
    words = find_file(
        os.path.join(ami_dir, f"{meeting_id}.{channel}.words.xml"),
        os.path.join(ami_dir, f"{meeting_id}_{channel}_words.xml"),
    )
    return segs, words


def video_file(video_dir, meeting_id, cam_index):
    return find_file(
        os.path.join(video_dir, f"{meeting_id}.Closeup{cam_index}.avi"),
        os.path.join(video_dir, f"{meeting_id}_Closeup{cam_index}.avi"),
    )


# ----------------------------------------------------------------------
# AMI ground truth parsing (identical to the other pipeline scripts)
# ----------------------------------------------------------------------

def get_camera_channel_map(meetings_xml_path, meeting_id):
    tree = ET.parse(meetings_xml_path)
    root = tree.getroot()
    camera_to_channel = {}
    for meeting in root.findall("meeting"):
        if meeting.attrib.get("observation") != meeting_id:
            continue
        for speaker in meeting.findall("speaker"):
            camera = speaker.attrib.get("camera")
            channel = speaker.attrib.get("nxt_agent")
            if camera and channel:
                camera_to_channel[camera] = channel
        break
    if not camera_to_channel:
        raise ValueError(f"Meeting '{meeting_id}' not found in {meetings_xml_path}")
    return camera_to_channel


def load_word_times(words_xml_path):
    tree = ET.parse(words_xml_path)
    root = tree.getroot()
    times = {}
    for el in root:
        eid = el.attrib.get(f"{NITE_NS}id")
        st = el.attrib.get("starttime")
        et = el.attrib.get("endtime")
        if eid is None or st is None or et is None:
            continue
        times[eid] = (float(st), float(et))
    return times


def load_channel_segments(segments_xml_path, word_times):
    tree = ET.parse(segments_xml_path)
    root = tree.getroot()
    segs = []
    for seg in root.findall("segment"):
        child = seg.find(f"{NITE_NS}child")
        if child is None:
            continue
        href = child.attrib.get("href", "")
        m = HREF_RE.search(href)
        if not m:
            continue
        start_id, end_id = m.group(1), m.group(2) or m.group(1)
        if start_id not in word_times or end_id not in word_times:
            continue
        start_t = word_times[start_id][0]
        end_t = word_times[end_id][1]
        if end_t > start_t:
            segs.append((start_t, end_t))
    segs.sort()
    return segs


def build_ground_truth(ami_dir, meeting_id, camera_to_channel):
    person_segments = {}
    for i in range(1, 5):
        person = f"Person{i}"
        camera = f"Closeup{i}"
        channel = camera_to_channel.get(camera)
        if channel is None:
            raise ValueError(f"No AMI channel found for {camera} in meetings.xml")

        segs_path, words_path = channel_files(ami_dir, meeting_id, channel)
        if not segs_path or not words_path:
            raise FileNotFoundError(
                f"Could not find segments/words xml for channel {channel} in {ami_dir}"
            )

        word_times = load_word_times(words_path)
        segs = load_channel_segments(segs_path, word_times)
        person_segments[person] = segs
        print(f"  {person} = Closeup{i} = AMI channel {channel}: {len(segs)} speech segments loaded")

    return person_segments


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def ground_truth_label_for_window(window_start, window_end, person_segments, silence_threshold):
    overlaps = {}
    for person, segs in person_segments.items():
        total = 0.0
        for s, e in segs:
            if e <= window_start:
                continue
            if s >= window_end:
                break
            total += overlap_seconds(window_start, window_end, s, e)
        overlaps[person] = total
    best_person = max(overlaps, key=overlaps.get)
    if overlaps[best_person] < silence_threshold:
        return "Silence", overlaps
    return best_person, overlaps


def compute_metrics(df):
    labels = sorted(set(df["GroundTruth"]) | set(df["Predicted"]))
    confusion = pd.DataFrame(0, index=labels, columns=labels)
    for gt, pred in zip(df["GroundTruth"], df["Predicted"]):
        confusion.loc[gt, pred] += 1
    n = len(df)
    accuracy = (df["GroundTruth"] == df["Predicted"]).sum() / n if n else float("nan")
    per_class = {}
    for label in labels:
        tp = confusion.loc[label, label]
        fp = confusion[label].sum() - tp
        fn = confusion.loc[label].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) > 0 else float("nan"))
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                             "support": int(confusion.loc[label].sum())}
    return accuracy, confusion, per_class


# ----------------------------------------------------------------------
# Evaluation metric plots (confusion matrix, per-class bars, ROC-AUC)
# ----------------------------------------------------------------------

def plot_evaluation_metrics(result_df, preds_df, confusion, per_class, args, score_cols):
    """
    Saves three PNGs into args.output_dir:
      - confusion_matrix.png : heatmap of the confusion matrix
      - per_class_metrics.png : precision/recall/f1 bar chart per label
      - roc_curve.png : one-vs-rest ROC curves (with AUC) per label, using
        each person's raw per-window combined ("total") movement score as
        the decision score. Silence's score is taken as the negative of
        the max movement score across people for that window (higher when
        everyone is still).
    score_cols: dict mapping person label (e.g. "Person1") -> column name
        in preds_df holding that person's raw per-window movement score
        (here, the "<name>_Total" column produced by run_detection).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1) Confusion matrix heatmap ----
    labels = list(confusion.index)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    vmax = confusion.values.max() if confusion.values.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(confusion.values[i, j]), ha="center", va="center",
                    color="white" if confusion.values[i, j] > vmax / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- 2) Per-class precision/recall/f1 bar chart ----
    class_labels = list(per_class.keys())
    precisions = [per_class[l]["precision"] for l in class_labels]
    recalls = [per_class[l]["recall"] for l in class_labels]
    f1s = [per_class[l]["f1"] for l in class_labels]

    x = np.arange(len(class_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "per_class_metrics.png"), dpi=150)
    plt.close(fig)

    # ---- 3) One-vs-rest ROC curves using raw movement scores ----
    y_true = result_df["GroundTruth"].tolist()
    n = len(y_true)
    if n == 0:
        return

    person_scores = {}
    for person, col in score_cols.items():
        if col in preds_df.columns:
            person_scores[person] = preds_df[col].to_numpy()[:n]

    if person_scores:
        stacked = np.vstack(list(person_scores.values()))
        max_person_score = stacked.max(axis=0)
    else:
        max_person_score = np.zeros(n)

    fig, ax = plt.subplots(figsize=(6, 6))
    plotted_any = False
    for label in sorted(set(y_true)):
        y_bin = np.array([1 if g == label else 0 for g in y_true])
        if y_bin.sum() == 0 or y_bin.sum() == n:
            continue  # ROC-AUC undefined with only one class present
        if label in person_scores:
            scores = person_scores[label]
        elif label == "Silence":
            scores = -max_person_score
        else:
            continue
        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")
        plotted_any = True

    if plotted_any:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150)
    else:
        print("  (Skipping ROC curve plot: not enough class variation to compute it.)")
    plt.close(fig)

    print(f"  Saved evaluation plots to: {os.path.abspath(args.output_dir)}")


# ----------------------------------------------------------------------
# Detection helpers (from your script, unchanged)
# ----------------------------------------------------------------------

def euclidean(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def bbox_diag(w, h):
    return max(np.hypot(w, h), 1e-6)


def bbox_relative_points(points, bbox):
    x, y, w, h = bbox
    diag = bbox_diag(w, h)
    return [((px - x) / diag, (py - y) / diag) for (px, py) in points]


def mean_point_movement(curr_points, prev_points):
    if prev_points is None or len(curr_points) != len(prev_points):
        return 0.0
    dists = [euclidean(c, p) for c, p in zip(curr_points, prev_points)]
    return float(np.mean(dists)) if dists else 0.0


def build_detectors(args):
    face_detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=args.face_detector_model),
            running_mode=RunningMode.VIDEO,
            min_detection_confidence=FACE_DETECTOR_MIN_CONF,
        )
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=args.face_landmarker_model),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=FACE_LM_MIN_FACE_DET_CONF,
            min_face_presence_confidence=FACE_LM_MIN_FACE_PRES_CONF,
            min_tracking_confidence=FACE_LM_MIN_TRACK_CONF,
        )
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=args.hand_landmarker_model),
            running_mode=RunningMode.VIDEO,
            num_hands=MAX_HANDS_PER_PERSON,
            min_hand_detection_confidence=HAND_MIN_DET_CONF,
            min_hand_presence_confidence=HAND_MIN_PRES_CONF,
            min_tracking_confidence=HAND_MIN_TRACK_CONF,
        )
    )
    return face_detector, face_landmarker, hand_landmarker


class PersonState:
    def __init__(self, args):
        self.prev_head_center = None
        self.prev_lip_shape = None
        self.prev_hand_centroid = None
        self.face_detector, self.face_landmarker, self.hand_landmarker = build_detectors(args)
        self.cap = None
        self.fps = None

    def close(self):
        self.face_detector.close()
        self.face_landmarker.close()
        self.hand_landmarker.close()
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------
# Detection (multi-modal)
# ----------------------------------------------------------------------

def run_detection(args):
    for label, path in [("face detector", args.face_detector_model),
                         ("face landmarker", args.face_landmarker_model),
                         ("hand landmarker", args.hand_landmarker_model)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} model not found: {path}")

    video_paths = {}
    for i in range(1, 5):
        vp = video_file(args.video_dir, args.meeting_id, i)
        if not vp:
            raise FileNotFoundError(f"Could not find Closeup{i} video in {args.video_dir}")
        video_paths[f"Person{i}"] = vp

    people = {name: PersonState(args) for name in video_paths}

    for name, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            for p in people.values():
                p.close()
            raise RuntimeError(f"Cannot open {path}")
        people[name].cap = cap
        people[name].fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fps_values = {n: p.fps for n, p in people.items()}
    fps_set = {round(v, 2) for v in fps_values.values()}
    if len(fps_set) > 1:
        for p in people.values():
            p.close()
        raise RuntimeError(f"Videos have mismatched FPS, alignment would be wrong: {fps_values}")
    fps = people["Person1"].fps

    start_frame = int(args.start_time * fps)
    end_frame = int(args.end_time * fps)
    window_size = int(args.window_seconds * fps)

    for p in people.values():
        p.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    window_scores = {name: {"head": 0.0, "lip": 0.0, "hand": 0.0, "total": 0.0} for name in people}
    results = []

    predicted_speaker = "None"
    frame_counter = 0
    current_frame = start_frame
    window_start_frame = start_frame

    try:
        while current_frame <= end_frame:
            finished = False
            frames = {}

            for name, state in people.items():
                ret, frame = state.cap.read()
                if not ret:
                    finished = True
                    break

                h_frame, w_frame = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((current_frame / fps) * 1000)

                head_movement = 0.0
                lip_movement = 0.0
                hand_movement = 0.0

                # ---------------- Head ----------------
                face_result = state.face_detector.detect_for_video(mp_image, timestamp_ms)

                bbox = None
                if len(face_result.detections) > 0:
                    bb = face_result.detections[0].bounding_box
                    bbox = (bb.origin_x, bb.origin_y, bb.width, bb.height)
                    cx = bbox[0] + bbox[2] / 2
                    cy = bbox[1] + bbox[3] / 2

                    if state.prev_head_center is not None:
                        diag = bbox_diag(bbox[2], bbox[3])
                        head_movement = euclidean((cx, cy), state.prev_head_center) / diag

                    state.prev_head_center = (cx, cy)

                    if not args.headless:
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 255, 0), 2)
                else:
                    # FIX: no face this frame -> clear stale reference so a
                    # re-detected face doesn't register a fake jump.
                    state.prev_head_center = None

                # ---------------- Lips ----------------
                if bbox is not None:
                    face_lm_result = state.face_landmarker.detect_for_video(mp_image, timestamp_ms)

                    if face_lm_result.face_landmarks:
                        landmarks = face_lm_result.face_landmarks[0]
                        abs_points = [(lm.x * w_frame, lm.y * h_frame) for lm in landmarks]
                        lip_points_abs = [abs_points[i] for i in LIP_INDICES]

                        lip_shape = bbox_relative_points(lip_points_abs, bbox)
                        lip_movement = mean_point_movement(lip_shape, state.prev_lip_shape)
                        state.prev_lip_shape = lip_shape

                        if not args.headless:
                            for (lx, ly) in lip_points_abs:
                                cv2.circle(frame, (int(lx), int(ly)), 2, (0, 200, 255), -1)
                    else:
                        state.prev_lip_shape = None
                else:
                    # FIX: no face -> lip block above was entirely skipped in the
                    # original, leaving prev_lip_shape stale. Clear it here too.
                    state.prev_lip_shape = None

                # ---------------- Hands ----------------
                hand_result = state.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                if hand_result.hand_landmarks:
                    all_centroids = []
                    for hand_landmarks in hand_result.hand_landmarks:
                        pts = [(lm.x * w_frame, lm.y * h_frame) for lm in hand_landmarks]
                        hx = sum(p[0] for p in pts) / len(pts)
                        hy = sum(p[1] for p in pts) / len(pts)
                        all_centroids.append((hx, hy))
                        if not args.headless:
                            for (px, py) in pts:
                                cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 255), -1)

                    agg_x = sum(c[0] for c in all_centroids) / len(all_centroids)
                    agg_y = sum(c[1] for c in all_centroids) / len(all_centroids)

                    if state.prev_hand_centroid is not None and bbox is not None:
                        diag = bbox_diag(bbox[2], bbox[3])
                        hand_movement = euclidean((agg_x, agg_y), state.prev_hand_centroid) / diag

                    state.prev_hand_centroid = (agg_x, agg_y)
                else:
                    state.prev_hand_centroid = None

                # ---------------- Combine ----------------
                window_scores[name]["head"] += head_movement
                window_scores[name]["lip"] += lip_movement
                window_scores[name]["hand"] += hand_movement
                window_scores[name]["total"] += (
                    args.w_head * head_movement + args.w_lip * lip_movement + args.w_hand * hand_movement
                )

                if not args.headless:
                    color = (0, 255, 0) if name == predicted_speaker else (0, 0, 255)
                    cv2.rectangle(frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), color, 3)
                    cv2.putText(frame, name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Total: {window_scores[name]['total']:.2f}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame,
                                f"H:{window_scores[name]['head']:.2f} "
                                f"L:{window_scores[name]['lip']:.2f} "
                                f"Ha:{window_scores[name]['hand']:.2f}",
                                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if name == predicted_speaker:
                        cv2.putText(frame, "SPEAKING", (20, 115),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frames[name] = frame

            if finished:
                break

            frame_counter += 1

            if frame_counter >= window_size:
                top_person = max(window_scores, key=lambda n: window_scores[n]["total"])

                if window_scores[top_person]["total"] < args.movement_threshold:
                    predicted_speaker = "Silence"
                else:
                    predicted_speaker = top_person

                print("\nWindow Result")
                for name in window_scores:
                    s = window_scores[name]
                    print(f"{name}: head={s['head']:.2f} lip={s['lip']:.2f} "
                          f"hand={s['hand']:.2f} total={s['total']:.2f}")
                print("Predicted:", predicted_speaker)

                row = {
                    "WindowStartSec": window_start_frame / fps,
                    "WindowEndSec": (current_frame + 1) / fps,
                    "PredictedSpeaker": predicted_speaker,
                }
                for name in window_scores:
                    row[f"{name}_Head"] = window_scores[name]["head"]
                    row[f"{name}_Lip"] = window_scores[name]["lip"]
                    row[f"{name}_Hand"] = window_scores[name]["hand"]
                    row[f"{name}_Total"] = window_scores[name]["total"]
                results.append(row)

                for name in window_scores:
                    window_scores[name] = {"head": 0.0, "lip": 0.0, "hand": 0.0, "total": 0.0}

                frame_counter = 0
                window_start_frame = current_frame + 1

            if not args.headless:
                for name, frame in frames.items():
                    cv2.imshow(name, frame)

            current_frame += 1

            if not args.headless and (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    finally:
        for p in people.values():
            p.close()
        if not args.headless:
            cv2.destroyAllWindows()

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    return df


# ----------------------------------------------------------------------
# Evaluation (identical logic to the other pipeline scripts)
# ----------------------------------------------------------------------

def run_evaluation(preds_df, args):
    print("\nResolving Closeup->AMI channel mapping from meetings.xml ...")
    camera_to_channel = get_camera_channel_map(args.meetings_xml, args.meeting_id)
    print(f"  {camera_to_channel}")

    print("\nBuilding ground truth from AMI segments/words XML ...")
    person_segments = build_ground_truth(args.ami_dir, args.meeting_id, camera_to_channel)

    print("\nAligning windows to ground truth ...")
    rows = []
    for _, row in preds_df.iterrows():
        ws, we = float(row["WindowStartSec"]), float(row["WindowEndSec"])
        gt_label, overlaps = ground_truth_label_for_window(ws, we, person_segments, args.silence_threshold)
        rows.append({
            "WindowStartSec": ws,
            "WindowEndSec": we,
            "Predicted": row["PredictedSpeaker"],
            "GroundTruth": gt_label,
            **{f"GT_overlap_{p}": round(v, 3) for p, v in overlaps.items()},
        })

    result_df = pd.DataFrame(rows)
    result_df["Correct"] = result_df["Predicted"] == result_df["GroundTruth"]
    accuracy, confusion, per_class = compute_metrics(result_df)

    os.makedirs(args.output_dir, exist_ok=True)
    result_df.to_csv(os.path.join(args.output_dir, "window_scores.csv"), index=False)
    confusion.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))
    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write(f"Meeting: {args.meeting_id}\n")
        f.write(f"Windows evaluated: {len(result_df)}\n")
        f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
        f.write("Per-class metrics:\n")
        f.write(f"{'label':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}\n")
        for label, m in per_class.items():
            f.write(f"{label:<12}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}\n")
        f.write("\nConfusion matrix (rows=ground truth, cols=predicted):\n")
        f.write(confusion.to_string())
        f.write("\n")

    print("\nGenerating evaluation plots (confusion matrix, per-class bars, ROC-AUC) ...")
    plot_evaluation_metrics(
        result_df, preds_df, confusion, per_class, args,
        score_cols={f"Person{i}": f"Person{i}_Total" for i in range(1, 5)},
    )

    print("\n================ RESULTS ================")
    print(f"Windows evaluated: {len(result_df)}")
    print(f"Overall accuracy:  {accuracy:.4f}")
    print("\nPer-class metrics:")
    print(f"{'label':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label, m in per_class.items():
        print(f"{label:<12}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}")
    print("\nConfusion matrix (rows=ground truth, cols=predicted):")
    print(confusion.to_string())
    print(f"\nSaved to: {os.path.abspath(args.output_dir)}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-modal (head+lip+hand) speaker detection + AMI evaluation")
    p.add_argument("--face-detector-model", default=DEFAULT_FACE_DETECTOR_MODEL)
    p.add_argument("--face-landmarker-model", default=DEFAULT_FACE_LANDMARKER_MODEL)
    p.add_argument("--hand-landmarker-model", default=DEFAULT_HAND_LANDMARKER_MODEL)
    p.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--ami-dir", default=DEFAULT_AMI_DIR)
    p.add_argument("--meetings-xml", default=None,
                    help="Path to meetings.xml. If omitted, auto-searches video-dir/ami-dir and their parents.")
    p.add_argument("--meeting-id", default=DEFAULT_MEETING_ID)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--end-time", type=float, default=150.0)
    p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--w-head", type=float, default=1.0)
    p.add_argument("--w-lip", type=float, default=4.0)
    p.add_argument("--w-hand", type=float, default=0.5)
    p.add_argument("--movement-threshold", type=float, default=1.0,
                    help="Combined 'total' score below which a window counts as Silence. "
                         "Tune based on your printed Window Result totals -- these are "
                         "normalized (bbox-relative) signals, much smaller than raw pixel scores.")
    p.add_argument("--silence-threshold", type=float, default=0.1,
                    help="Seconds of overlapping ground-truth speech below which a window "
                         "counts as Silence in the AMI ground truth")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--output", default="MultiModalSpeakerScores.csv")
    p.add_argument("--output-dir", default="eval_out_multimodal")
    return p.parse_args()


def main():
    args = parse_args()

    if args.meetings_xml is None:
        found = find_meetings_xml([args.video_dir, args.ami_dir])
        if not found:
            raise FileNotFoundError(
                "Could not auto-locate meetings.xml near your video/ami folders. "
                "Pass its path explicitly with --meetings-xml \"C:\\path\\to\\meetings.xml\""
            )
        args.meetings_xml = found
        print(f"Found meetings.xml at: {found}")

    print("=== STEP 1: Running multi-modal (head+lip+hand) detection ===")
    preds_df = run_detection(args)

    print("\n=== STEP 2: Scoring against AMI ground truth ===")
    run_evaluation(preds_df, args)


if __name__ == "__main__":
    main()