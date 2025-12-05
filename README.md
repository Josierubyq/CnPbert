# CnPbert
# CnPBERT: Facilitating Chinese Online Petition Classification through Domain-Specific Pretraining
This is the pretrained language model CnPBERT, designed for petition categorization in Chinese.

Here is the Huggingface link:
https://huggingface.co/ZhangQianML/CnPBERT

# Model Useage:

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("JosieZhang/CnPBERT")
model = AutoModel.from_pretrained("JosieZhang/CnPBERT")

inputs = tokenizer("Sample text for classification", return_tensors="pt")
outputs = model(**inputs)
