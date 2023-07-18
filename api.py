from flask import Flask, request
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
from diffusers.utils import load_image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
import numpy as np
import os

app = Flask(__name__)

DEVICE = torch.device('cuda:0')

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    subfolder='image_encoder'
).half().to(DEVICE)

unet = UNet2DConditionModel.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    subfolder='unet'
).half().to(DEVICE)

prior = KandinskyV22PriorPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    image_encoder=image_encoder,
    torch_dtype=torch.float16
).to(DEVICE)

decoder = KandinskyV22Pipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    unet=unet,
    torch_dtype=torch.float16
).to(DEVICE)

torch.manual_seed(42)

negative_prior_prompt ='worst quality, low quality'

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json(force=True)
    prompt = data['prompt']
    
    img_emb = prior(
        prompt=prompt,
        num_inference_steps=25,
        num_images_per_prompt=1
    )

    negative_emb = prior(
        prompt=negative_prior_prompt,
        num_inference_steps=25,
        num_images_per_prompt=1
    )

    images = decoder(
        image_embeds=img_emb.image_embeds,
        negative_image_embeds=negative_emb.image_embeds,
        num_inference_steps=75,
        height=1024,
        width=512)
    
    # Return your image data here. You might need to convert it to a suitable format.
    return {"image": images.images[0].tolist()}  

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(threaded=True, host='0.0.0.0', port=port)
