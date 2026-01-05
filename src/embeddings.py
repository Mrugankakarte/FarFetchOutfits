from os.path import exists

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import os

from PIL import Image
import open_clip
import requests
import torch

# Specify model from Hugging Face Hub
model_name = 'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
# model_name = 'hf-hub:Marqo/marqo-ecommerce-embeddings-B'

model, _, preprocessor = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

def text_collate_func(batch):
    pids = [i['prod_id'] for i in batch]
    text = [i['ptext'] for i in batch]

    text_tokens = tokenizer(text)

    return torch.tensor(pids), text_tokens

class Textfeatures_Generator(torch.utils.data.Dataset):
    def __init__(self, products_path):

        self.product_df = pd.read_parquet(products_path)
        self.product_df = self.product_df.fillna(value="N/A")

    def __len__(self):
        return self.product_df.shape[0]

    def __getitem__(self, idx):
        category = self.product_df['product_category'][idx]
        family = self.product_df['product_family'][idx]
        sub_cat_text = self.product_df['product_sub_category'][idx]
        prd_gender = self.product_df['product_gender'][idx]
        main_color = str(self.product_df['product_main_colour'][idx])
        brand = self.product_df['product_brand']
        materials = str(self.product_df['product_materials'])
        highlights = str(self.product_df['product_highlights'])

        text = self.product_df['product_short_description'][idx]
        prod_id = self.product_df['product_node_id'][idx]

        ptext = (f"This is a {prd_gender} {brand} {family} product and belongs to {category} category, {sub_cat_text} sub category. "
                 f"It is a {main_color} color and made of {materials}. Highlights: {highlights}. Description: {text}")

        return {"prod_id": prod_id, "ptext": ptext}

def main():
    textfeatures_gen = Textfeatures_Generator("data/graph/products.parquet")
    prod_highlight_dl = torch.utils.data.DataLoader(textfeatures_gen, collate_fn=text_collate_func, batch_size=64, shuffle=False, num_workers=2)
    os.makedirs("text_embedding", exist_ok=True)
    device = "cuda"
    model.to(device)
    colnames = [f"embd_{i}" for i in range(1024)]
    batch = 0
    with torch.no_grad(), torch.amp.autocast(device):
        for prod_id, text in tqdm(prod_highlight_dl):

            text_features = model.encode_text(text.to(device), normalize=True).to(torch.float16)
            temp = pd.DataFrame(text_features.cpu().numpy())
            temp.columns = colnames
            temp.insert(0, 'product_node_id', value=prod_id)
            temp.to_parquet(f"text_embedding/product_embeddings_batch{batch}.parquet")
            batch+=1


if __name__=="__main__":
    main()