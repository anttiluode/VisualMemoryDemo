"""Local camera/video demo of bounded visual memory (no training)."""

import argparse
import time

import numpy as np

from memory import MergeMemory, RecentMemory, pool_patches


SIZE = 224
GRID = 16
DISPLAY = 448


class DinoEncoder:
    def __init__(self):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading frozen DINOv2-small (first run downloads its weights)...")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
        ).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    def encode(self, image, cv2):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()
        tensor = (tensor.to(self.device) / 255 - self.mean) / self.std
        with self.torch.inference_mode():
            patches = self.model.forward_features(tensor)["x_norm_patchtokens"][0]
        return patches.float().cpu().numpy().reshape(GRID, GRID, -1)


class RgbEncoder:
    def encode(self, image, cv2):
        rgb = cv2.cvtColor(cv2.resize(image, (GRID, GRID), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        colors = rgb.astype(np.float32) / 255
        # A fixed component makes black patches valid nonzero vectors.
        return np.concatenate((colors, np.full((GRID, GRID, 1), 0.25, np.float32)), axis=2)


def color(slot_id):
    rng = np.random.default_rng(slot_id)
    return rng.integers(55, 245, 3, dtype=np.uint8)


def caption(cv2, canvas, label, x, y, scale=0.57):
    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)


def draw_match(cv2, canvas, match, title, x, y):
    caption(cv2, canvas, title, x, y)
    if match is None:
        caption(cv2, canvas, "No earlier patch yet", x, y + 36)
        return
    preview = cv2.resize(match.entry.preview, (140, 140), interpolation=cv2.INTER_NEAREST)
    canvas[y + 9:y + 149, x:x + 140] = preview
    caption(cv2, canvas, f"cos {match.score:.2f}  first frame {match.entry.first_frame}", x + 150, y + 43, 0.48)
    caption(cv2, canvas, f"combined patches {match.entry.mass}", x + 150, y + 67, 0.48)
    if match.entry.position is not None:
        caption(cv2, canvas, f"old region {match.entry.position}", x + 150, y + 91, 0.48)


def render(cv2, image, features, merged_memory, recent_memory, past_merge, past_recent,
           selected, frame, milliseconds):
    canvas = np.full((DISPLAY, DISPLAY * 2, 3), 26, np.uint8)
    slot_ids = merged_memory.assign(features.reshape(-1, features.shape[-1]))
    tint = np.array([color(slot_id) for slot_id in slot_ids], np.uint8).reshape(GRID, GRID, 3)
    tint = cv2.resize(tint, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    camera = cv2.addWeighted(cv2.resize(image, (DISPLAY, DISPLAY)), 0.72, tint, 0.28, 0)
    row, col = selected
    cv2.rectangle(camera, (col * 28, row * 28), ((col + 1) * 28 - 1, (row + 1) * 28 - 1), (0, 0, 255), 2)
    canvas[:, :DISPLAY] = camera

    feature = features[row, col]
    merge_hit = merged_memory.search(feature, entries=past_merge)
    recent_hit = recent_memory.search(feature, entries=past_recent)
    caption(cv2, canvas, f"Visual memory | frame {frame} | encoder {milliseconds:.0f} ms", 462, 27)
    caption(cv2, canvas, f"Click image patch | {len(merged_memory.entries)}/{merged_memory.capacity} slots", 462, 55)
    draw_match(cv2, canvas, merge_hit, "Similarity merge: earlier match", 462, 87)
    draw_match(cv2, canvas, recent_hit, "Recent only: earlier match", 462, 262)
    caption(cv2, canvas, "Colors show patch families, not object masks.", 462, 435, 0.48)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or local video file")
    parser.add_argument("--encoder", choices=("dino", "rgb"), default="dino",
                        help="DINOv2 features or an RGB-only control")
    parser.add_argument("--slots", type=int, default=64, help="Slots for EACH memory (default: 64)")
    parser.add_argument("--every", type=int, default=10, help="Process every Nth input frame (default: 10)")
    parser.add_argument("--output", help="Write the visualized video to an MP4 file")
    parser.add_argument("--headless", action="store_true", help="No desktop window (requires --output)")
    parser.add_argument("--query-cell", default="8,8", help="Initial selected patch row,col, each 0..15")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames")
    args = parser.parse_args()
    if args.every < 1 or args.slots < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--every, --slots and --max-frames must be positive")
    if args.headless and not args.output:
        parser.error("--headless requires --output")
    try:
        row, col = (int(part) for part in args.query_cell.split(","))
    except ValueError:
        parser.error("--query-cell needs row,col")
    if not (0 <= row < GRID and 0 <= col < GRID):
        parser.error("--query-cell coordinates must be 0..15")
    return args, (row, col)


def main():
    args, selected = parse_args()
    import cv2

    encoder = DinoEncoder() if args.encoder == "dino" else RgbEncoder()
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")
    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (DISPLAY * 2, DISPLAY))
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Could not write MP4: {args.output}")

    merged_memory, recent_memory = MergeMemory(args.slots), RecentMemory(args.slots)
    selection = [selected]
    if not args.headless:
        cv2.namedWindow("Visual memory", cv2.WINDOW_AUTOSIZE)

        def click(event, x, y, _flags, _userdata):
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < DISPLAY and 0 <= y < DISPLAY:
                selection[0] = (min(y // 28, 15), min(x // 28, 15))

        cv2.setMouseCallback("Visual memory", click)

    read_frames = processed = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            read_frames += 1
            if (read_frames - 1) % args.every:
                continue
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            start = time.perf_counter()
            features = encoder.encode(image, cv2)
            milliseconds = (time.perf_counter() - start) * 1000
            past_merge, past_recent = merged_memory.snapshot(), recent_memory.snapshot()
            pooled = pool_patches(features)
            for row in range(4):
                for col in range(4):
                    preview = image[row * 56:(row + 1) * 56, col * 56:(col + 1) * 56]
                    merged_memory.observe(pooled[row, col], preview, processed, position=(row, col))
                    recent_memory.observe(pooled[row, col], preview, processed, position=(row, col))
            panel = render(cv2, image, features, merged_memory, recent_memory, past_merge, past_recent,
                           selection[0], processed, milliseconds)
            processed += 1
            if writer:
                writer.write(panel)
            if not args.headless:
                cv2.imshow("Visual memory", panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    cv2.imwrite("visual-memory.png", panel)
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Processed {processed} frames; {len(merged_memory.entries)} merged slots retained")


if __name__ == "__main__":
    main()
