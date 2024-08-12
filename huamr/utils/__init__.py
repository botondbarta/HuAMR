from transformers import BitsAndBytesConfig


def get_bnb_config(quantize: str):
    if quantize == '4bit':
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == '8bit':
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Invalid quantization type: {quantize}")