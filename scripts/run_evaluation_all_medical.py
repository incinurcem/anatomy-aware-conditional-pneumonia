import os
import sys
import shlex
import argparse
import subprocess


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--image_col", type=str, default="image_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mean", type=float, default=0.485)
    parser.add_argument("--std", type=float, default=0.229)
    parser.add_argument("--mc_runs", type=int, default=20)
    parser.add_argument("--do_gradcam", action="store_true")
    parser.add_argument("--gradcam_num_samples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    python_bin = shlex.quote(sys.executable)

    cls_dir = os.path.join(args.output_root, "classification")
    unc_dir = os.path.join(args.output_root, "uncertainty")
    cal_dir = os.path.join(args.output_root, "calibration")
    os.makedirs(cls_dir, exist_ok=True)
    os.makedirs(unc_dir, exist_ok=True)
    os.makedirs(cal_dir, exist_ok=True)

    run_cmd(
        f"{python_bin} {shlex.quote(os.path.join(base_dir, 'eval_classifier_medical.py'))} "
        f"--test_csv {shlex.quote(args.test_csv)} --output_dir {shlex.quote(cls_dir)} "
        f"--model_path {shlex.quote(args.model_path)} --image_col {shlex.quote(args.image_col)} "
        f"--label_col {shlex.quote(args.label_col)} --model_name {shlex.quote(args.model_name)} "
        f"--image_size {args.image_size} --batch_size {args.batch_size} --num_workers {args.num_workers} "
        f"--device {shlex.quote(args.device)} --mean {args.mean} --std {args.std}"
    )

    run_cmd(
        f"{python_bin} {shlex.quote(os.path.join(base_dir, 'eval_calibration_medical.py'))} "
        f"--predictions_csv {shlex.quote(os.path.join(cls_dir, 'predictions_with_labels.csv'))} "
        f"--output_dir {shlex.quote(cal_dir)}"
    )

    run_cmd(
        f"{python_bin} {shlex.quote(os.path.join(base_dir, 'eval_uncertainty_medical.py'))} "
        f"--test_csv {shlex.quote(args.test_csv)} --model_path {shlex.quote(args.model_path)} "
        f"--output_dir {shlex.quote(unc_dir)} --image_col {shlex.quote(args.image_col)} "
        f"--label_col {shlex.quote(args.label_col)} --model_name {shlex.quote(args.model_name)} "
        f"--image_size {args.image_size} --batch_size {min(args.batch_size, 16)} --num_workers {args.num_workers} "
        f"--device {shlex.quote(args.device)} --mean {args.mean} --std {args.std} --mc_runs {args.mc_runs}"
    )

    if args.pred_dir and args.gt_dir:
        seg_dir = os.path.join(args.output_root, "segmentation")
        os.makedirs(seg_dir, exist_ok=True)
        run_cmd(
            f"{python_bin} {shlex.quote(os.path.join(base_dir, 'eval_segmentation_medical.py'))} "
            f"--pred_dir {shlex.quote(args.pred_dir)} --gt_dir {shlex.quote(args.gt_dir)} "
            f"--output_dir {shlex.quote(seg_dir)}"
        )

    if args.do_gradcam:
        gc_dir = os.path.join(args.output_root, "gradcam")
        os.makedirs(gc_dir, exist_ok=True)
        run_cmd(
            f"{python_bin} {shlex.quote(os.path.join(base_dir, 'eval_gradcam_medical.py'))} "
            f"--test_csv {shlex.quote(args.test_csv)} --model_path {shlex.quote(args.model_path)} "
            f"--output_dir {shlex.quote(gc_dir)} --image_col {shlex.quote(args.image_col)} "
            f"--label_col {shlex.quote(args.label_col)} --model_name {shlex.quote(args.model_name)} "
            f"--image_size {args.image_size} --device {shlex.quote(args.device)} --mean {args.mean} --std {args.std} "
            f"--num_samples {args.gradcam_num_samples}"
        )

    print("\n===== ALL MEDICAL EVALUATION COMPLETE =====")


if __name__ == "__main__":
    main()
