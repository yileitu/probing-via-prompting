from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Model, GPT2Tokenizer, TextGenerationPipeline

HF_PATH = "akhooli/gpt2-small-arabic"

tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
model = GPT2LMHeadModel.from_pretrained(HF_PATH)
text_generator = TextGenerationPipeline(model, tokenizer)
outputs = text_generator("Hi. How are you?", max_length=100, do_sample=True)
print(outputs[0]['generated_text'])
