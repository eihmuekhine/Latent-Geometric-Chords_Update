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
from torch.hub import load_state_dict_from_url

# --- METRIC LIBRARIES ---
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
# ------------------------

try:
    from models import builer
except ImportError:
    import builer

from IEEE_Places_Image_Tar_proposed_attack import Proposed_attack

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

def load_places365_categories(cat_path):
    idx2label = {}
    try:
        with open(cat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    idx = int(parts[1])
                    idx2label[idx] = name
        print(f"=> Loaded {len(idx2label)} class names from {cat_path}")
    except Exception as e:
        pass
    return idx2label

def load_places365_labels(label_path):
    img2label = {}
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    idx = int(parts[1])
                    img2label[img_name] = idx
        print(f"=> Loaded {len(img2label)} image labels from {label_path}")
    except Exception as e:
        pass
    return img2label

def load_places365_model(model_name, device):
    print(f"\n=> Loading Target Classifier for Places365: {model_name}")
    if model_name == 'resnet50':
        model = torch_models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 365)
        url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    elif model_name == 'densenet161':
        model = torch_models.densenet161(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 365)
        url = 'http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar'
    else:
        raise ValueError(f"Unknown architecture: {model_name}")

    checkpoint = load_state_dict_from_url(url, map_location=device, progress=True)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    
    if 'densenet' in model_name:
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.([12])\.(.*)$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = f"{res.group(1)}{res.group(2)}.{res.group(3)}"
                state_dict[new_key] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    return model.to(device).eval()

def valid_bounds(img, delta=255):
    im = np.asarray(img).astype(np.int32)
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)
    lb = im - delta
    ub = im + delta
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))
    return lb.astype(np.uint8), ub.astype(np.uint8)

class LatentModelWrapper(torch.nn.Module):
    def __init__(self, decoder, classifier, mean, std):
        super(LatentModelWrapper, self).__init__()
        self.decoder = decoder
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer('lb', torch.zeros(1, 3, 1, 1))
        self.register_buffer('ub', torch.ones(1, 3, 1, 1))
        self.register_buffer('x_orig', torch.zeros(1, 3, 224, 224))
        self.register_buffer('x_rec_0', torch.zeros(1, 3, 224, 224))

    def set_bounds(self, lb, ub):
        self.lb, self.ub = lb, ub

    def set_reference_images(self, x_orig, x_rec_0):
        self.x_orig, self.x_rec_0 = x_orig, x_rec_0

    def get_pixel_image(self, z):
        x_rec = self.decoder(z)
        delta = x_rec - self.x_rec_0
        x_final = torch.clamp(self.x_orig + delta, self.lb, self.ub)
        return x_final

    def forward(self, z):
        x_final = self.get_pixel_image(z)
        normalized_img = (x_final - self.mean) / self.std
        return self.classifier(normalized_img)

def run_targeted_latent_attack():
    set_seed(992)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=os.path.expanduser('~/autodl-tmp/Places365'))
    parser.add_argument('--arch', type=str, default='vgg16')
    parser.add_argument('--ae_checkpoint', type=str, required=True)
    args = parser.parse_args()

    image_dir = os.path.join(args.base_dir, 'images')
    labels_txt_path = os.path.join(args.base_dir, 'adv_test_labels.txt')
    categories_txt_path = os.path.join(args.base_dir, 'categories_places365.txt')

    idx2label = load_places365_categories(categories_txt_path)
    img2label = load_places365_labels(labels_txt_path)

    # --- EXPERIMENT CONFIGURATION ---
    TARGET_IMAGE_COUNT = 30 
    iteration = 70          
    initial_query = 30    
    query_checkpoints = [500, 1000, 2000, 5000, 10000] # Added Checkpoints
    max_queries = 10000 # Added Max Queries
    
    target_archs = ['resnet50'] 
    attack_methods = ['LGC', 'LGC_H'] 
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    mean_tensor = torch.tensor(mean).view(1,3,1,1).to(device)
    std_tensor = torch.tensor(std).view(1,3,1,1).to(device)
    # --------------------------------

    class MockArgs:
        arch = args.arch
        parallel = 0
        gpu = 0
        
    ae_model = builer.BuildAutoEncoder(MockArgs())
    if os.path.exists(args.ae_checkpoint):
        sd = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
        if 'state_dict' in sd: sd = sd['state_dict']
        is_parallel = isinstance(ae_model, torch.nn.DataParallel)
        new_sd = {('module.' + k if is_parallel and not k.startswith('module.') else k.replace('module.', '') if not is_parallel and k.startswith('module.') else k): v for k, v in sd.items()}
        ae_model.load_state_dict(new_sd)

    ae_model = ae_model.to(device).eval()
    decoder_to_use = ae_model.module.decoder if isinstance(ae_model, torch.nn.DataParallel) else ae_model.decoder
    encoder_to_use = ae_model.module.encoder if isinstance(ae_model, torch.nn.DataParallel) else ae_model.encoder

    all_files = sorted(glob.glob(os.path.join(image_dir, "*.JPEG")))
    if not all_files:
        print(f"Error: No JPEG images found in {image_dir}")
        return

    output_dir = 'IEEE_Places_Image_Tar'
    os.makedirs(output_dir, exist_ok=True)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    for target_arch in target_archs:
        classifier = load_places365_model(target_arch, device)
        model_wrapper = LatentModelWrapper(decoder_to_use, classifier, mean, std).to(device)
        
        # --- (၁) ပုံ ၃၀ အတွဲ (Source, Target) ရွေးချယ်ခြင်းနှင့် Text File အသုံးပြုခြင်း ---
        pgds_txt_path = f"places365_{target_arch}_selected_pairs_targeted.txt"
        valid_pairs = []
        
        if os.path.exists(pgds_txt_path):
            print(f"\n[*] Found '{pgds_txt_path}'. Loading exactly {TARGET_IMAGE_COUNT} pairs...")
            with open(pgds_txt_path, 'r') as f:
                for line in f.read().splitlines()[:TARGET_IMAGE_COUNT]:
                    s_name, t_name = line.strip().split(',')
                    s_full = os.path.join(image_dir, s_name)
                    t_full = os.path.join(image_dir, t_name)
                    if os.path.exists(s_full) and os.path.exists(t_full):
                        valid_pairs.append((s_full, t_full))
        else:
            print(f"\n[*] '{pgds_txt_path}' not found! Generating {TARGET_IMAGE_COUNT} pairs now...")
            used_source_indices = set()
            attacked_count = 0
            
            while attacked_count < TARGET_IMAGE_COUNT:
                source_idx = random.randint(0, len(all_files) - 1)
                if source_idx in used_source_indices: continue
                    
                full_path = all_files[source_idx]
                img_name = os.path.basename(full_path)
                true_label = img2label.get(img_name, None)

                im_orig = Image.open(full_path).convert('RGB')
                resize = transforms.Resize((224, 224))
                to_tensor = transforms.ToTensor()
                im_resized = resize(im_orig)
                
                lb_np, ub_np = valid_bounds(im_resized, delta=255) 
                model_wrapper.set_bounds(to_tensor(lb_np).unsqueeze(0).to(device), to_tensor(ub_np).unsqueeze(0).to(device))
                x_0_raw = to_tensor(im_resized).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    z_source = encoder_to_use(x_0_raw)
                    x_rec_0 = decoder_to_use(z_source)
                    model_wrapper.set_reference_images(x_0_raw, x_rec_0)
                    orig_label = classifier((x_0_raw - mean_tensor) / std_tensor).argmax(1).item()

                if true_label is None or orig_label != true_label: continue 

                valid_target_found = False
                target_path = None
                target_img_name = None
                
                candidate_indices = list(range(len(all_files)))
                candidate_indices.remove(source_idx)
                random.shuffle(candidate_indices)
                
                for target_idx in candidate_indices:
                    t_path = all_files[target_idx]
                    t_img_name = os.path.basename(t_path)
                    target_true_label = img2label.get(t_img_name, None)
                    x_0_t_raw = to_tensor(resize(Image.open(t_path).convert('RGB'))).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        tar_label = classifier((x_0_t_raw - mean_tensor) / std_tensor).argmax(1).item()
                        if target_true_label is not None and tar_label == target_true_label and orig_label != tar_label:
                            
                            # --- Strict validation check for z_target ---
                            z_target = encoder_to_use(torch.clamp(x_0_t_raw - (x_0_raw - x_rec_0), 0.0, 1.0))
                            if classifier((model_wrapper.get_pixel_image(z_target) - mean_tensor) / std_tensor).argmax(1).item() == tar_label:
                                valid_target_found = True
                                target_path = t_path
                                target_img_name = t_img_name
                                break 
                            # --------------------------------------------
                
                if not valid_target_found: continue
                
                used_source_indices.add(source_idx)
                valid_pairs.append((full_path, target_path))
                attacked_count += 1
                print(f"   Pair {attacked_count} generated: {img_name} -> {target_img_name}")
                
            with open(pgds_txt_path, 'w') as f_out:
                for s_path, t_path in valid_pairs:
                    f_out.write(f"{os.path.basename(s_path)},{os.path.basename(t_path)}\n")
            print(f"[*] Created '{pgds_txt_path}' with {len(valid_pairs)} pairs.")
        # -------------------------------------------------------------

        for attack_method in attack_methods:
            print(f"\n{'='*20} Running TARGETED Attack: {attack_method} on {target_arch} {'='*20}")
            all_source_names, all_target_names, all_original_labels, all_target_labels, all_adversarial_labels = [], [], [], [], []
            all_queries, all_l2_latent, all_linf_latent, all_l2_pixel, all_linf_pixel = [], [], [], [], []
            all_ssim, all_lpips, all_asr = [], [], []# all_ssim, all_lpips, all_asr = [], []
            all_adv_labels_list = [] # Added for tracking labels

            npz_save_path = os.path.join(output_dir, f'Npz/Targeted_attack/Places365/{target_arch}_{attack_method}_TARGETED_PGDS_Places365_{TARGET_IMAGE_COUNT}.npz')
            os.makedirs(os.path.dirname(npz_save_path), exist_ok=True)
            vis_dir = os.path.join(output_dir, f'Visual_Comparison/{target_arch}/Targeted_{attack_method}')
            os.makedirs(vis_dir, exist_ok=True)
            
            attacked_count = 0

            for full_path, target_path in valid_pairs:
                img_name = os.path.basename(full_path)
                target_img_name = os.path.basename(target_path)
                
                im_orig = Image.open(full_path).convert('RGB')
                resize = transforms.Resize((224, 224))
                to_tensor = transforms.ToTensor()
                im_resized = resize(im_orig)
                lb_np, ub_np = valid_bounds(im_resized, delta=255) 
                model_wrapper.set_bounds(to_tensor(lb_np).unsqueeze(0).to(device), to_tensor(ub_np).unsqueeze(0).to(device))

                x_0_raw = to_tensor(im_resized).unsqueeze(0).to(device)
                x_0_t_raw = to_tensor(resize(Image.open(target_path).convert('RGB'))).unsqueeze(0).to(device)

                with torch.no_grad():
                    z_source = encoder_to_use(x_0_raw)
                    x_rec_0 = decoder_to_use(z_source)
                    model_wrapper.set_reference_images(x_0_raw, x_rec_0)
                    orig_label = classifier((x_0_raw - mean_tensor) / std_tensor).argmax(1).item()
                    tar_label = classifier((x_0_t_raw - mean_tensor) / std_tensor).argmax(1).item()
                    
                    z_target = encoder_to_use(torch.clamp(x_0_t_raw - (x_0_raw - x_rec_0), 0.0, 1.0))
                
                attacked_count += 1
                print(f"\n[{attacked_count}/{TARGET_IMAGE_COUNT}] Starting Attack: {img_name} -> {target_img_name}")
                start_time = time.time()
                img_base_name = img_name.split(".")[0]
                
                # --- Added Parameter Passing for Proposed_attack ---
                attack = Proposed_attack(model_wrapper, z_source, mean, std, loss_fn_vgg=loss_fn_vgg, ssim_fn=ssim_fn, tar_img=z_target, 
                                         attack_method=attack_method, iteration=iteration, initial_query=initial_query, idx2label=idx2label,
                                         save_img_dir=vis_dir, img_name=img_base_name, query_checkpoints=query_checkpoints, max_queries=max_queries) 
                
                # --- Retrieve 10 return values including adv_labels_vals ---
                z_adv, n_query, l2_lat, linf_lat, l2_pix, linf_pix, ssim_vals, lpips_vals, asr_vals, adv_labels_vals = attack.Attack()
                # ----------------------------------------------------
                
                with torch.no_grad():
                    adv_label = classifier((model_wrapper.get_pixel_image(z_adv) - mean_tensor) / std_tensor).argmax(1).item()
                    
                elapsed_time = time.time() - start_time
                print(f"   => Result: {'✅ SUCCESS' if adv_label == tar_label else '❌ FAILED'} (Label {adv_label})")
                
                all_source_names.append(img_name)
                all_target_names.append(target_img_name)
                all_original_labels.append(orig_label)
                all_target_labels.append(tar_label)
                all_adversarial_labels.append(adv_label)
                all_queries.append(n_query)
                all_l2_latent.append(l2_lat)
                all_linf_latent.append(linf_lat)
                all_l2_pixel.append(l2_pix)
                all_linf_pixel.append(linf_pix)
                all_ssim.append(ssim_vals)
                all_lpips.append(lpips_vals)
                all_asr.append(asr_vals)
                all_adv_labels_list.append(adv_labels_vals) # Saving updated labels
                
                try:
                    norm_median = np.median(np.array(all_l2_pixel).astype(float), axis=0)
                    query_median = np.median(np.array(all_queries).astype(float), axis=0)
                except ValueError:
                    norm_median, query_median = np.array([]), np.array([])

                # --- Saving with new attributes ---
                np.savez(npz_save_path, 
                         norm=norm_median,
                         query=query_median,
                         image_names=np.array(all_source_names, dtype=object), 
                         source_names=np.array(all_source_names, dtype=object), 
                         target_names=np.array(all_target_names, dtype=object), 
                         original_labels=np.array(all_original_labels, dtype=np.int64), 
                         target_labels=np.array(all_target_labels, dtype=np.int64), 
                         adversarial_labels=np.array(all_adversarial_labels, dtype=np.int64), 
                         all_queries=np.array(all_queries, dtype=object),
                         all_adv_labels=np.array(all_adv_labels_list, dtype=object), 
                         l2_latent=np.array(all_l2_latent, dtype=object),
                         linf_latent=np.array(all_linf_latent, dtype=object),
                         l2_pixel=np.array(all_l2_pixel, dtype=object),
                         linf_pixel=np.array(all_linf_pixel, dtype=object),
                         ssim=np.array(all_ssim, dtype=object), 
                         lpips=np.array(all_lpips, dtype=object),
                         asr=np.array(all_asr, dtype=object))
                         
if __name__ == "__main__":
    run_targeted_latent_attack()