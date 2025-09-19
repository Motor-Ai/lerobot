import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText, TrainingArguments, Trainer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mai.utils.utils import target_in_ego


SMOL = True
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if SMOL else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEVICE = "cuda:2"
VIDEO_BASE_PATH = "/home/user_lerobot/.cache/huggingface/lerobot/yaak-ai/lerobot-driving-school/videos/chunk-000/observation.images.front_left"


def load_model(use_lora=False, use_qlora=False, smol=True):
    model_id = MODEL_ID
    processor = AutoProcessor.from_pretrained(model_id)

    if use_qlora or use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            use_dora=False if use_qlora else True,
            init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config if use_qlora else None,
            _attn_implementation="flash_attention_2",
            device_map=DEVICE
        )
        # model.add_adapter(lora_config)
        # model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print('Trainable params', model.get_nb_trainable_parameters())
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to(DEVICE)

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False

    peak_mem = torch.cuda.max_memory_allocated()
    print(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

    return model, processor


def train(model, processor, dataset):
    def collate_fn(examples):
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
                            processor.tokenizer.additional_special_tokens.index("<image>")]

        instances = []
        for example in examples:
            prompt = example['task.instructions']
            episode_idx = example["episode_index"]
            example["video_path"] = f"{VIDEO_BASE_PATH}/episode_{episode_idx:06d}.mp4"

            # Extract values
            lat0 = example["observation.state.vehicle"][3]   # hp_loc_latitude
            lon0 = example["observation.state.vehicle"][4]   # hp_loc_longitude
            heading_deg = example["observation.state.vehicle"][1]

            # last waypoint as target (lat, lon) => note your arrays are [X=lon list], [Y=lat list]
            target_lon = example["observation.state.waypoints"][0][-1]
            target_lat = example["observation.state.waypoints"][1][-1]

            x_fwd, y_left, z_up = target_in_ego(lat0, lon0, heading_deg, target_lat, target_lon)

            # Build dynamic text
            text_instruction = (
                f"Current speed: {example['observation.state.vehicle'][0]:.3f}. "
                f"Target (in ego frame): x={x_fwd:.2f} m, y={y_left:.2f} m. "
                "Drive to the target while adhering to traffic rules and regulations."
            )
            
            user_content = [{"type": "text", 
                             "text": f"{text_instruction}"}]
            user_content.append({"type": "video", 
                                 "path": example["video_path"], 
                                 "max_pixels": 360 * 420, 
                                 "fps": 2.0 })

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
            ]

            instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                    tokenize=True, return_dict=True, return_tensors="pt").to(model.device).to(model.dtype)
            instances.append(instance)


        input_ids = pad_sequence(
            [inst["input_ids"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(
            [inst["attention_mask"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=0
        )
        labels = pad_sequence(
            [inst["input_ids"].squeeze(0).clone() for inst in instances],
            batch_first=True,
            padding_value=-100
        )

        labels[labels == image_token_id] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


        # Step 1: figure out maximum frames, height, width across the batch
        pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
        if pvs:  # there is at least one non-None pixel_values
            max_frames = max(pv.shape[0] for pv in pvs)
            max_h = max(pv.shape[-2] for pv in pvs)
            max_w = max(pv.shape[-1] for pv in pvs)
        else:
            max_h = max_w = processor.video_size['longest_edge']
            max_frames = 1

        padded_pixel_values_list = []
        for ex in instances:
            pv = ex.get("pixel_values", None).squeeze(0)

            if pv is None:
                # text-only => fill pixel data + mask with zeros
                shape_pv = (max_frames, 3, max_h, max_w)
                padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
            else:
                f, c, h, w = pv.shape
                # Prepare final storage
                padded_pv = torch.zeros(
                    (max_frames, c, max_h, max_w),
                    dtype=pv.dtype,
                    device=pv.device
                )
                padded_pv[:f, :, :h, :w] = pv
            padded_pixel_values_list.append(padded_pv)

        out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
        return out

    # processed_dataset = preprocess_dataset(dataset, processor)

    model_name = MODEL_ID.split("/")[-1]

    training_args = TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        optim="paged_adamw_8bit", # for 8-bit, keep paged_adamw_8bit, else adamw_hf
        bf16=True,
        output_dir=f"./outputs/bs_8_{model_name}-driving",
        hub_model_id=f"bs_8_{model_name}-driving",
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False, 
        label_names=["labels"]
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=dataset["train"],
        )

    trainer.train()
    trainer.save_model(f"./outputs/bs_8_{model_name}-driving_last")


def main():
    use_lora = True
    use_qlora = True
    smol = True

    model, processor = load_model(use_lora, use_qlora, smol)
    dataset = load_dataset("yaak-ai/lerobot-driving-school")
    train(model, processor, dataset)


if __name__ == "__main__":
    main()
