import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

video_path = "/home/user_lerobot/.cache/huggingface/lerobot/yaak-ai/L2D/videos/chunk-001/observation.images.front_left/episode_001002.mp4"
fps = 10


SYSTEM_PROMPT = (
    "You are an expert driver. Analyze the driving video at exactly 3 timestamps: 0s, 10s, and 20s.\n"
    "You MUST output exactly three lines: one for Frame 0, one for Frame 10, one for Frame 20.\n\n"
    "Rules for each line:\n"
    "- Start with 'Frame X:' (X = 0, 5, 10, 15)\n"
    "- After the colon, the first word must be a verb (Drive, Stop, Accelerate, Decelerate, Change, etc.)\n"
    "- Write a short, complete action sentence in present tense (not just a single verb).\n"
    "- Do not begin with 'The car', 'It', or any subject — start directly with the verb.\n"
    "- Do not describe scenery unless it directly affects the car’s action.\n\n"
    "Format strictly as:\n"
    "### OUTPUT START\n"
    "Frame 0: Drive straight in the current lane.\n"
    "Frame 5: Decelerate before the roundabout.\n"
    "Frame 10: Turn right at the roundabout.\n"
    "### OUTPUT END\n"
    "This is the ONLY correct style."
)



# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

# Messages containing a local video path and a text query
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": SYSTEM_PROMPT}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": fps, 
             "max_pixels": 360 * 420},
            {"type": "text", "text": "What to do next?"},
        ],
    },
]


#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    # fps=fps,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)