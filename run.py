"""
run_pipeline.py

Runs head-movement speaker detection AND scores it against AMI ground truth
in one go -- no need to run two scripts / two commands manually.

Just run:
    python run_pipeline.py

All paths below are pre-filled to match your machine's folder layout. If
anything is in a different place, just edit the DEFAULT_* constants below
(or pass the equivalent --flag, see --help).

Your AMI xml files use the real AMI naming convention with DOTS:
    ES2008a.A.segments.xml   ES2008a.A.words.xml   (etc for B/C/D)
This script looks for that pattern first, and also the underscore pattern
(ES2008a_A_segments.xml) as a fallback, so it works either way.
"""

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# EDIT THESE IF YOUR FOLDERS ARE DIFFERENT
# ----------------------------------------------------------------------
DEFAULT_VIDEO_DIR = r"C:\Users\Ehtiram\Desktop\DS_Project\ES2008a\video"
DEFAULT_AMI_DIR = r"C:\Users\Ehtiram\Desktop\DS_Project\ES2008a\video"   # your xml files sit here too
DEFAULT_MODEL_PATH = r"C:\Users\Ehtiram\Downloads\blaze_face_short_range.tflite"
DEFAULT_MEETING_ID = "ES2008a"
# ----------------------------------------------------------------------

NITE_NS = "{http://nite.sourceforge.net/}"
HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


# ----------------------------------------------------------------------
# Helpers to locate files regardless of dot vs underscore naming
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
            # also glob in case of different case
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
# AMI ground truth parsing
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
                f"Could not find segments/words xml for channel {channel} in {ami_dir} "
                f"(tried both '{meeting_id}.{channel}.segments.xml' and "
                f"'{meeting_id}_{channel}_segments.xml' patterns)"
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
        each person's raw per-window movement score as the decision score.
        Silence's score is taken as the negative of the max movement score
        across people for that window (higher when everyone is still).
    score_cols: dict mapping person label (e.g. "Person1") -> column name
        in preds_df holding that person's raw per-window movement score.
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
# Detection
# ----------------------------------------------------------------------

def run_detection(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    video_paths = {}
    for i in range(1, 5):
        vp = video_file(args.video_dir, args.meeting_id, i)
        if not vp:
            raise FileNotFoundError(f"Could not find Closeup{i} video in {args.video_dir}")
        video_paths[f"Person{i}"] = vp

    caps = {}
    for person, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        caps[person] = cap

    fps_values = {p: c.get(cv2.CAP_PROP_FPS) for p, c in caps.items()}
    fps_set = {round(v, 2) for v in fps_values.values()}
    if len(fps_set) > 1:
        raise RuntimeError(f"Videos have mismatched FPS, alignment would be wrong: {fps_values}")
    fps = int(round(fps_values["Person1"]))

    start_frame = int(args.start_time * fps)
    end_frame = int(args.end_time * fps)
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    previous_centers = {person: None for person in video_paths}
    window_scores = {person: 0.0 for person in video_paths}
    results = []
    window_size = fps
    predicted_speaker = "None"
    frame_counter = 0
    current_frame = start_frame
    window_start_frame = start_frame

    try:
        while current_frame <= end_frame:
            finished = False
            frames = {}

            for person, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    finished = True
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                movement = 0
                if len(result.detections) > 0:
                    bbox = result.detections[0].bounding_box
                    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    cx, cy = x + w / 2, y + h / 2
                    if previous_centers[person] is not None:
                        px, py = previous_centers[person]
                        movement = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    previous_centers[person] = (cx, cy)
                    if not args.headless:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                else:
                    previous_centers[person] = None

                window_scores[person] += movement

                if not args.headless:
                    color = (0, 255, 0) if person == predicted_speaker else (0, 0, 255)
                    cv2.rectangle(frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), color, 3)
                    cv2.putText(frame, person, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Score: {window_scores[person]:.2f}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if person == predicted_speaker:
                        cv2.putText(frame, "SPEAKING", (20, 95), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)
                    frames[person] = frame

            if finished:
                break

            frame_counter += 1

            if frame_counter >= window_size:
                top_person = max(window_scores, key=window_scores.get)
                if window_scores[top_person] < args.movement_threshold:
                    predicted_speaker = "Silence"
                else:
                    predicted_speaker = top_person

                print("\nWindow Result")
                for p in window_scores:
                    print(p, round(window_scores[p], 2))
                print("Predicted:", predicted_speaker)

                results.append({
                    "WindowStartSec": window_start_frame / fps,
                    "WindowEndSec": (current_frame + 1) / fps,
                    "Person1": window_scores["Person1"],
                    "Person2": window_scores["Person2"],
                    "Person3": window_scores["Person3"],
                    "Person4": window_scores["Person4"],
                    "PredictedSpeaker": predicted_speaker,
                })

                for p in window_scores:
                    window_scores[p] = 0
                frame_counter = 0
                window_start_frame = current_frame + 1

            if not args.headless:
                for person in video_paths:
                    cv2.imshow(person, frames[person])

            current_frame += 1

            if not args.headless and (cv2.waitKey(1) & 0xFF == ord('q')):
                break
    finally:
        for cap in caps.values():
            cap.release()
        detector.close()
        if not args.headless:
            cv2.destroyAllWindows()

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    return df


# ----------------------------------------------------------------------
# Evaluation
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
        score_cols={"Person1": "Person1", "Person2": "Person2",
                    "Person3": "Person3", "Person4": "Person4"},
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
    p = argparse.ArgumentParser(description="Head-movement speaker detection + AMI evaluation, one shot")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--ami-dir", default=DEFAULT_AMI_DIR)
    p.add_argument("--meetings-xml", default=None,
                    help="Path to meetings.xml. If omitted, auto-searches video-dir/ami-dir and their parents.")
    p.add_argument("--meeting-id", default=DEFAULT_MEETING_ID)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--end-time", type=float, default=150.0)
    p.add_argument("--movement-threshold", type=float, default=1.0)
    p.add_argument("--silence-threshold", type=float, default=0.1)
    p.add_argument("--headless", action="store_true", default=False,
                    help="Pass this flag to skip the live video windows and just run to completion faster")
    p.add_argument("--output", default="HeadSpeakerScores.csv")
    p.add_argument("--output-dir", default="eval_out")
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

    print("=== STEP 1: Running head-movement detection ===")
    preds_df = run_detection(args)

    print("\n=== STEP 2: Scoring against AMI ground truth ===")
    run_evaluation(preds_df, args)


if __name__ == "__main__":
    main()