import argparse
import os
import sys

# sam_extractor.py 在当前目录下
#try:
    #from sam_extractor import FeatureExtractor
#except ImportError:
    #sys.path.append(os.path.dirname(__file__))
    #try:
        #from sam_extractor import FeatureExtractor
    #except ImportError:
        #print("[Error] Could not import 'sam_extractor'. Make sure sam_extractor.py is in the same directory.")
        #sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from sam_extractor import FeatureExtractor        

def main():
    parser = argparse.ArgumentParser(description="Run SAM & LSD feature extraction for HSR Calibration")
    parser.add_argument("--image", type=str, help="Single image file path")
    parser.add_argument("--img_dir", type=str, required=False, help="Directory containing images (png/jpg)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save masks and line features")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", help="SAM model type (vit_h, vit_l, vit_b)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--frame_start", type=str, default="", help="Start frame ID")
    parser.add_argument("--frame_end", type=str, default="", help="End frame ID")
    parser.add_argument("--max_frames", type=int, default=0, help="Max number of frames to process")

    args = parser.parse_args()

    # 1. 检查输入路径（单个文件或目录）
    if args.image and args.img_dir:
        print("[Error] Cannot specify both --image and --img_dir")
        sys.exit(1)

    if not args.image and not args.img_dir:
        print("[Error] Must specify either --image or --img_dir")
        sys.exit(1)

    # 检查checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        print("Please download it from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sys.exit(1)

    # 2. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 初始化提取器
    print("--- Initializing Feature Extractor ---")
    extractor = FeatureExtractor(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device
    )

   # 4. 处理图像
    if args.image:
        # 单个图像处理模式
        if not os.path.exists(args.image):
            print(f"[Error] Image file does not exist: {args.image}")
            sys.exit(1)
        
        print(f"--- Processing single image: {args.image} ---")
        extractor.process_image(args.image, args.output_dir)
        print(f"\n[Done] Feature extracted and saved to: {args.output_dir}")
    
    else:
        # 批量处理模式
        if not os.path.exists(args.img_dir):
            print(f"[Error] Image directory does not exist: {args.img_dir}")
            sys.exit(1)
        
        image_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        selected_files = image_files

        def parse_frame_id(filename):
            stem = os.path.splitext(filename)[0]
            return int(stem) if stem.isdigit() else None

        if args.frame_start or args.frame_end:
            start_id = int(args.frame_start) if args.frame_start else None
            end_id = int(args.frame_end) if args.frame_end else None
            filtered = []
            for name in image_files:
                frame_id = parse_frame_id(name)
                if frame_id is None:
                    continue
                if start_id is not None and frame_id < start_id:
                    continue
                if end_id is not None and frame_id > end_id:
                    continue
                filtered.append(name)
            selected_files = filtered
        elif args.max_frames > 0:
            selected_files = image_files[:args.max_frames]

        if not selected_files:
            print("[Warning] No images found in directory.")
            return

        print(f"--- Processing {len(selected_files)} images ---")
        for img_file in selected_files:
            full_path = os.path.join(args.img_dir, img_file)
            extractor.process_image(full_path, args.output_dir)

        print("\n[Done] All features extracted.")
        print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()