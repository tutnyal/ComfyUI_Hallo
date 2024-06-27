def cv_frame_generator(video):
    try:
        # Debug: Check if the video file exists
        if not os.path.exists(video):
            raise FileNotFoundError(f"Video file {video} does not exist.")
        
        # Debug: Check file permissions
        if not os.access(video, os.R_OK):
            raise PermissionError(f"Video file {video} is not readable.")
        
        video_cap = cv2.VideoCapture(video)
        if not video_cap.isOpened():
            raise ValueError(f"{video} could not be loaded with cv.")
        
        # set video_cap to look at start_index frame
        total_frame_count = 0
        total_frames_evaluated = -1
        frames_added = 0
        base_frame_time = 1 / video_cap.get(cv2.CAP_PROP_FPS)
        width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        prev_frame = None
        target_frame_time = base_frame_time
        yield (width, height, target_frame_time)
        time_offset = target_frame_time - base_frame_time
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
                inp = yield prev_frame
                if inp is not None:
                    # ensure the finally block is called
                    return
            prev_frame = frame
            frames_added += 1

        if prev_frame is not None:
            yield prev_frame
    finally:
        video_cap.release()
