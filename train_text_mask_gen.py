import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig, BertForSequenceClassification, BertTokenizer, BertConfig
from datasets import load_dataset,load_metric
import numpy as np


from accelerate import Accelerator


accelerator = Accelerator()
device = accelerator.device


texts = "I [MASK] this movie!"

# pretrained_name = "distilbert-base-uncased-finetuned-sst-2-english"
pretrained_name = "textattack/bert-base-uncased-SST-2"

# pred_config = DistilBertConfig.from_pretrained(pretrained_name)
# pred_tokenizer = DistilBertTokenizer.from_pretrained(pretrained_name)
# pred_model = DistilBertForSequenceClassification.from_pretrained(pretrained_name).to(device)

pred_config = BertConfig.from_pretrained(pretrained_name)
pred_tokenizer = BertTokenizer.from_pretrained(pretrained_name)
pred_model = BertForSequenceClassification.from_pretrained(pretrained_name).to(device)

inputs = pred_tokenizer(texts, return_tensors="pt")
with torch.no_grad():
    inputs = {key:val.to(device) for key,val in inputs.items()}
    logits = pred_model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(pred_model.config.id2label[predicted_class_id])

print(inputs)


from maskgen.text_models.text_maskgen_model2 import MaskGeneratingModel

pred_hidden_dim = pred_model.config.hidden_size
num_labels = pred_model.config.num_labels

mask_gen_model = MaskGeneratingModel(pred_model, hidden_size=pred_hidden_dim, num_classes=num_labels)
mask_gen_model.to(device)
print()


from datasets import load_dataset
imdb = load_dataset("imdb")
idx = 0
texts = imdb["test"][idx]['text']
print(texts)

inputs = pred_tokenizer(texts, return_tensors="pt")
with torch.no_grad():
    inputs = {key:val.to(device) for key,val in inputs.items()}
    logits = pred_model(**inputs).logits

predicted_class_id = logits.argmax().item()
pred_label = pred_model.config.id2label[predicted_class_id]
print("pred label", pred_label)
print("True label", pred_model.config.id2label[imdb["test"][idx]['label']])


from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import load_dataset

imdb = load_dataset("imdb")
train_ds = imdb['train']

# def preprocess_function(examples):
#     return pred_tokenizer(examples["text"], truncation=True, padding="max_length")

# tokenized_imdb = train_ds.map(preprocess_function, batched=True)

def collate_fn(examples):
    texts = [example['text'] for example in examples]
    labels = [example['label'] for example in examples]
    
    # Tokenize texts
    batch = pred_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    
    # Add labels
    batch['labels'] = torch.tensor(labels, dtype=torch.long)
    return batch
# train_ds.set_transform(preprocess)


batch_size = 256
train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
n_steps = 2
n_samples = 3

params_to_optimize = [name for name, param in mask_gen_model.named_parameters() if param.requires_grad]
print("params to be optimized: ")
print(params_to_optimize)


from tqdm import tqdm

params_to_optimize = [param for param in mask_gen_model.parameters() if param.requires_grad]
# optimizer = torch.optim.Adam(params_to_optimize, lr=1e-3, weight_decay=1e-5)
optimizer = torch.optim.Adam(params_to_optimize, lr=1e-3, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)


print()

for epoch in range(10):
    pbar = tqdm(train_dataloader)
    for idx, data in enumerate(pbar):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        loss_dict = mask_gen_model.train_one_batch(input_ids, attention_mask, optimizer=optimizer, n_steps=n_steps, n_samples=n_samples)
        # scheduler.step()
        pbar.set_description(f"Epoch {epoch+1}, Step {idx+1}: Loss = {loss_dict['loss'].item():.4f}, " 
                             f"Reward Loss = {loss_dict['reward_loss'].item():.4f}, "
                            #  f"Regret Loss = {loss_dict['regret_loss'].item():.4f}, "
                             f"Mask Loss = {loss_dict['mask_loss'].item():.4f} "
                            #  f"alt_mask_loss = {loss_dict['alt_mask_loss'].item():.4f} "
                             f"mask_mean = {loss_dict['mask_mean'].item():.4f} "
                             f"prob_mean = {loss_dict['prob_mean'].item():.4f} "
                             )
        if idx % 10 == 0:
            print()
        if (idx) % 10 == 0:
            
            torch.save(mask_gen_model.state_dict(), f'text_mask_gen_model/mask_gen_model_{epoch}_{idx}.pth') 



torch.save(mask_gen_model.state_dict(), f'text_mask_gen_model/mask_gen_model_final_{epoch}_{idx}.pth') 
