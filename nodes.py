import os
import cv2
import yaml
import torch
import random
import folder_paths
import numpy as np
from PIL import Image


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# Our any instance wants to be a wildcard string
any = AnyType("*")


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def cv_frame_generator(video):
    try:
        video_cap = cv2.VideoCapture(video)
        if not video_cap.isOpened():
            raise ValueError(f"{video} could not be loaded with cv.")
        # set video_cap to look at start_index frame
        total_frame_count = 0
        total_frames_evaluated = -1
        frames_added = 0
        base_frame_time = 1/video_cap.get(cv2.CAP_PROP_FPS)
        width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        prev_frame = None
        target_frame_time = base_frame_time
        yield (width, height, target_frame_time)
        time_offset=target_frame_time - base_frame_time
        while video_cap.isOpened():
            if time_offset < target_frame_time:
                is_returned = video_cap.grab()
                # if didn't return frame, video has ended
                if not is_returned:
                    break
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue
            time_offset -= target_frame_time
            # if not at start_index, skip doing anything with frame
            total_frame_count += 1
            total_frames_evaluated += 1

            # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
            # follow up: can videos ever have an alpha channel?
            # To my testing: No. opencv has no support for alpha
            unused, frame = video_cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert frame to comfyui's expected format
            # TODO: frame contains no exif information. Check if opencv2 has already applied
            frame = np.array(frame, dtype=np.float32) / 255.0
            if prev_frame is not None:
                inp  = yield prev_frame
                if inp is not None:
                    #ensure the finally block is called
                    return
            prev_frame = frame
            frames_added += 1

        if prev_frame is not None:
            yield prev_frame
    finally:
        video_cap.release()

    
class HalloNode:
    @classmethod
    def INPUT_TYPES(s):
        audio_extensions = ["wav", "mp3", "flac"]
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                    "source_image": ("IMAGE", ),
                    "driving_audio": (sorted(files), ),
                    "pose_weight" :("FLOAT",{"default": 1.0}),
                    "face_weight":("FLOAT",{"default": 1.0}),
                    "lip_weight":("FLOAT",{"default": 1.0}),
                    "face_expand_ratio":("FLOAT",{"default": 1.2}),
                     },}

    CATEGORY = "HalloNode"

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", )
    RETURN_NAMES = ("images", "count", "frame_rate", )
    FUNCTION = "inference"


    def inference(self, source_image, driving_audio, pose_weight, face_weight, lip_weight, face_expand_ratio):
        ckpt_dir = os.path.join(folder_paths.models_dir, "hallo")
        cur_dir = get_ext_dir()
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(ckpt_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="fudan-generative-ai/hallo", local_dir=ckpt_dir, local_dir_use_symlinks=False)

        infer_py = os.path.join(cur_dir, "scripts/inference.py")
        default_yaml_path = os.path.join(cur_dir, "configs/inference/default.yaml")
        with open(default_yaml_path, 'r', encoding="utf-8") as f:
            yaml_data = yaml.load(f.read(),Loader=yaml.SafeLoader)
        yaml_data['save_path'] = output_dir
        yaml_data['audio_ckpt_dir'] = os.path.join(ckpt_dir, "hallo")
        yaml_data['base_model_path'] = os.path.join(ckpt_dir, "stable-diffusion-v1-5")
        yaml_data['motion_module_path'] = os.path.join(ckpt_dir, "motion_module/mm_sd_v15_v2.ckpt")
        yaml_data['face_analysis']['model_path'] = os.path.join(ckpt_dir, "face_analysis")
        yaml_data['wav2vec']['model_path'] = os.path.join(ckpt_dir, "wav2vec/wav2vec2-base-960h")
        yaml_data['audio_separator']['model_path'] = os.path.join(ckpt_dir, "audio_separator/Kim_Vocal_2.onnx")
        yaml_data['vae']['model_path'] = os.path.join(ckpt_dir, "sd-vae-ft-mse")
        yaml_data["face_landmarker"]['model_path'] = os.path.join(ckpt_dir, "face_analysis/models/face_landmarker_v2_with_blendshapes.task")

        tmp_yaml_path = os.path.join(cur_dir, 'tmp.yaml')
        with open(tmp_yaml_path, 'w', encoding="utf-8") as f:
            yaml.dump(data=yaml_data, stream=f, Dumper=yaml.Dumper)

        output_name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        output_video_path = os.path.join(output_dir, f"hallo_{output_name}.mp4")

        # get src image
        for (_, img) in enumerate(source_image):
            img = 255. * img.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            src_img_path = os.path.join(output_dir, f"hallo_{output_name}_src_img.png")
            img.save(src_img_path)
            print(f'saved src image to {src_img_path}')
            break

        # get src audio
        src_audio_path = os.path.join(folder_paths.get_input_directory(), driving_audio)

        env = ':'.join([os.environ.get('PYTHONPATH', ''), cur_dir])
        cmd = f"""PYTHONPATH={env} python {infer_py} --config "{tmp_yaml_path}" --source_image "{src_img_path}" --driving_audio "{src_audio_path}" --output {output_video_path} --pose_weight {pose_weight} --face_weight {face_weight} --lip_weight {lip_weight} --face_expand_ratio {face_expand_ratio}"""
        
        print(cmd)
        os.system(cmd)
        os.remove(tmp_yaml_path)

        gen = cv_frame_generator(output_video_path)
        (width, height, target_frame_time) = next(gen)
        width = int(width)
        height = int(height)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        if len(images) == 0:
            raise RuntimeError("No frames generated")
        return (images, len(images), 25)

NODE_CLASS_MAPPINGS = {
    "D_HalloNode": HalloNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_HalloNode": "Hallo Node",
}