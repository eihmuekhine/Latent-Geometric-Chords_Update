import torchvision.transforms as transforms
import torchvision.models as torch_models
import numpy as np
import torch
import os
from PIL import Image
import argparse
import glob
import time 
import random
import re

import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from models import builder
from attacks.lgc_attack import LGC_Attack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=992):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def optimize_target_latent(encoder, decoder, model_wrapper, x_0_raw, x_rec_0, x_target_raw, target_label, mean_t, std_t, device):
    pseudo_target = torch.clamp(x_target_raw - (x_0_raw - x_rec_0), 0.0, 1.0)
    with torch.no_grad():
        z_target = encoder(pseudo_target)
    
    adv_pix = model_wrapper.get_pixel_image(z_target)
    pred_label = model_wrapper.classifier((adv_pix - mean_t) / std_t).argmax(1).item()
    if pred_label == target_label:
        return z_target
        
    z_opt = z_target.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_opt], lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    target_tensor = torch.tensor([target_label], dtype=torch.long).to(device)

    for step in range(50):
        optimizer.zero_grad()
        out_pix = model_wrapper.get_pixel_image(z_opt)
        logits = model_wrapper.classifier((out_pix - mean_t) / std_t)
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()

        if logits.argmax(1).item() == target_label:
            break
            
    return z_opt.detach()

def load_places365_model(model_name, device):
    if model_name == 'resnet50':
        model = torch_models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 365)
        url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    else:
        raise ValueError(f"Unknown architecture: {model_name}")

    checkpoint = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()

class LatentModelWrapper(torch.nn.Module):
    def __init__(self, decoder, classifier, mean, std):
        super().__init__()
        self.decoder = decoder
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def set_reference_images(self, x_orig, x_rec_0):
        self.x_orig = x_orig
        self.x_rec_0 = x_rec_0

    def get_pixel_image(self, z):
        delta = self.decoder(z) - self.x_rec_0
        return torch.clamp(self.x_orig + delta, 0.0, 1.0)

    def forward(self, z):
        x_final = self.get_pixel_image(z)
        return self.classifier((x_final - self.mean) / self.std)

def main():
    set_seed(992)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['targeted', 'non_targeted'], required=True)
    parser.add_argument('--target_arch', type=str, default='resnet50')
    parser.add_argument('--attack_method', type=str, choices=['LGC', 'LGC_H'], default='LGC_H')
    parser.add_argument('--ae_checkpoint', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='./data/Places365')
    parser.add_argument('--num_samples', type=int, default=30)
    args = parser.parse_args()

    image_dir = os.path.join(args.base_dir, 'images')
    labels_txt = os.path.join(args.base_dir, 'adv_test_labels.txt')
    
    img2label = {}
    if os.path.exists(labels_txt):
        with open(labels_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2: img2label[parts[0]] = int(parts[1])

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean_t, std_t = torch.tensor(mean).view(1,3,1,1).to(device), torch.tensor(std).view(1,3,1,1).to(device)

    # Load Autoencoder properly handling DataParallel
    class MockArgs: arch, parallel, gpu = 'vgg16', 0, 0
    ae_model = builder.BuildAutoEncoder(MockArgs())
    sd = torch.load(args.ae_checkpoint, map_location=device)
    state_dict = sd['state_dict'] if 'state_dict' in sd else sd
    ae_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    ae_model = ae_model.to(device).eval()
    
    if isinstance(ae_model, torch.nn.DataParallel):
        decoder_to_use, encoder_to_use = ae_model.module.decoder, ae_model.module.encoder
    else:
        decoder_to_use, encoder_to_use = ae_model.decoder, ae_model.encoder

    classifier = load_places365_model(args.target_arch, device)
    model_wrapper = LatentModelWrapper(decoder_to_use, classifier, mean, std).to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    all_files = sorted(glob.glob(os.path.join(image_dir, "*.JPEG")))
    random.shuffle(all_files)
    
    output_dir = f'Results/{args.mode}_{args.attack_method}'
    vis_dir = os.path.join(output_dir, 'Visuals')
    os.makedirs(vis_dir, exist_ok=True)

    attacked_count = 0
    metrics = {k: [] for k in ['queries', 'l2_lat', 'l2_pix', 'ssim', 'lpips', 'asr']}
    
    for full_path in all_files:
        if attacked_count >= args.num_samples: break
        
        img_name = os.path.basename(full_path)
        true_label = img2label.get(img_name)
        if true_label is None: continue

        x_0_raw = transforms.ToTensor()(transforms.Resize((224, 224))(Image.open(full_path).convert('RGB'))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            orig_label = classifier((x_0_raw - mean_t) / std_t).argmax(1).item()
            if orig_label != true_label: continue
            
            z_source = encoder_to_use(x_0_raw)
            model_wrapper.set_reference_images(x_0_raw, decoder_to_use(z_source))

        z_target = None
        if args.mode == 'targeted':
            target_path = random.choice([f for f in all_files if f != full_path])
            x_target_raw = transforms.ToTensor()(transforms.Resize((224, 224))(Image.open(target_path).convert('RGB'))).unsqueeze(0).to(device)
            target_label = classifier((x_target_raw - mean_t) / std_t).argmax(1).item()
            
            if target_label == orig_label: continue
            
            z_target = optimize_target_latent(encoder_to_use, decoder_to_use, model_wrapper, x_0_raw, model_wrapper.x_rec_0, x_target_raw, target_label, mean_t, std_t, device)

        print(f"\n[{attacked_count+1}/{args.num_samples}] Attack: {img_name} - run_places365_attack.py:169")
        attack = LGC_Attack(model_wrapper, z_source, mean, std, loss_fn_vgg, ssim_fn, 
                            tar_img=z_target, attack_method=args.attack_method, 
                            iteration=60, save_img_dir=vis_dir, img_name=img_name.split(".")[0])
        
        z_adv, n_query, l2_lat, _, l2_pix, _, ssim_vals, lpips_vals, asr_vals, _ = attack.Attack()
        
        metrics['queries'].append(n_query)
        metrics['l2_lat'].append(l2_lat)
        metrics['l2_pix'].append(l2_pix)
        metrics['ssim'].append(ssim_vals)
        metrics['lpips'].append(lpips_vals)
        metrics['asr'].append(asr_vals)
        attacked_count += 1

    np.savez(os.path.join(output_dir, 'metrics.npz'), **{k: np.array(v, dtype=object) for k, v in metrics.items()})
    print("\n[+] Experiment Completed & Results Saved! - run_places365_attack.py:185")

if __name__ == "__main__":
    main()