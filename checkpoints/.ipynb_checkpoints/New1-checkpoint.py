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

# --- NEW: Import metric libraries ---
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

try:
    from models import builer
except ImportError:
    import builer

# Make sure this matches your actual filename for the merged attack class
from New_proposed_attacked import Proposed_attack 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Random Seed Setup ---
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"=> Random Seed set to {seed}")

def valid_bounds(img, delta=255):
    im = np.asarray(img).astype(np.int32)
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)
    lb = im - delta
    ub = im + delta
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))
    return lb.astype(np.uint8), ub.astype(np.uint8)

def get_clean_label_name(synset_list, index):
    if 0 <= index < len(synset_list):
        return synset_list[index].split(',')[0].strip()
    return "Unknown"

# --- Classifier Selection ---
def get_classifier(arch):
    print(f"=> Loading Classifier: {arch}")
    if arch == 'vit_basic':
        return torch_models.vit_b_16(weights='DEFAULT').to(device).eval()
    elif arch == 'densenet121':
        return torch_models.densenet121(weights='DEFAULT').to(device).eval()
    elif arch == 'vgg16':
        return torch_models.vgg16(weights='DEFAULT').to(device).eval()
    elif arch == 'resnet50':
        return torch_models.resnet50(weights='DEFAULT').to(device).eval()
    else:
        raise ValueError(f"Model architecture {arch} not supported")

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
        self.lb = lb
        self.ub = ub

    def set_reference_images(self, x_orig, x_rec_0):
        self.x_orig = x_orig
        self.x_rec_0 = x_rec_0

    def get_pixel_image(self, z):
        x_rec = self.decoder(z)
        delta = (x_rec - self.x_rec_0)
        x_final = self.x_orig + delta
        x_final = torch.clamp(x_final, self.lb, self.ub)
        return x_final

    def forward(self, z):
        x_final = self.get_pixel_image(z)
        normalized_img = (x_final - self.mean) / self.std
        return self.classifier(normalized_img)

def run_latent_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/root/autodl-tmp/Images_without95%/images')
    parser.add_argument('--arch', type=str, default='vgg16') 
    parser.add_argument('--ae_checkpoint', type=str, required=True)
    parser.add_argument('--seed', type=int, default=992) 
    parser.add_argument('--val_file', type=str, default='/root/autodl-tmp/Images_without95%/val.txt')
    parser.add_argument('--synset_file', type=str, default='/root/autodl-tmp/Images_without95%/synset_words.txt')
    args = parser.parse_args()

    args.image_path = os.path.abspath(os.path.expanduser(args.image_path))
     
    num_img_to_scan = 1000 
    
    iteration = 100         
    initial_query = 30       
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 

    # Metadata Loading
    try:
        with open(args.synset_file, 'r') as f:
            synset_labels = f.read().strip().split('\n')
        val_map = {}
        with open(args.val_file, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2: val_map[parts[0]] = int(parts[1])
        print("=> Metadata Loaded Successfully.")
    except Exception as e:
        print(f"Error loading metadata: {e}"); return

    # Build AutoEncoder
    class MockArgs:
        arch = args.arch 
        parallel = 0
        gpu = 0
        
    ae_model = builer.BuildAutoEncoder(MockArgs())
    if os.path.exists(args.ae_checkpoint):
        sd = torch.load(args.ae_checkpoint, map_location=device)
        ae_model.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd)
        print(f"=> Loaded AE Checkpoint: {args.ae_checkpoint}")

    ae_model = ae_model.to(device).eval()
    decoder_to_use = ae_model.module.decoder if hasattr(ae_model, 'module') else ae_model.decoder
    encoder_to_use = ae_model.module.encoder if hasattr(ae_model, 'module') else ae_model.encoder

    # Find Images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_files = []
    for ext in extensions: all_files.extend(glob.glob(os.path.join(args.image_path, ext)))
    image_files = sorted(list(set(all_files)))[:num_img_to_scan]
    print(f"Found {len(image_files)} images available to scan.")

    # --- Load LPIPS metric model before the loops ---
    print("=> Loading LPIPS VGG model for metrics...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    #target_archs = ['resnet50', 'vgg16', 'densenet121', 'vit_basic']
    target_archs = ['resnet50']
    attack_methods = ['CGBA_H','CGBA'] 

    for target_arch in target_archs:
        print(f"\n{'#'*30} Loading Model: {target_arch} {'#'*30}")
        
        # Load Classifier
        classifier = get_classifier(target_arch)
        model_wrapper = LatentModelWrapper(decoder_to_use, classifier, mean, std).to(device)

        # Loop through Attack Methods
        for current_method in attack_methods:
            print(f"\n{'='*20} Running Attack: {current_method} on {target_arch} {'='*20}")

            output_dir = f'Without95%Data/Our_Methor_Mar_2/{current_method}'
            if not os.path.exists(output_dir): 
                os.makedirs(output_dir)

            # Setup arrays to track variables for .npz saving
            all_queries = []
            all_norms = []
            all_l2_pixel = []
            all_linf_pixel = []
            all_ssim = []
            all_lpips = []
            all_asr = []
            image_names = []
            original_labels = []
            adversarial_labels = []
            
            correct_count = 0 
            target_limit = 10 # Target set to 10 for testing (Change to 200 later if needed)

            for i, full_path in enumerate(image_files):
                if correct_count >= target_limit:
                    print(f"=> Target reached: {target_limit} correctly classified images attacked. Stopping loop.")
                    break

                img_name = os.path.basename(full_path)
                im_orig = Image.open(full_path).convert('RGB')
                
                to_tensor = transforms.ToTensor()
                im_resized = transforms.Resize((224, 224))(im_orig)
                lb_np, ub_np = valid_bounds(im_resized, delta=255) 
                
                model_wrapper.set_bounds(to_tensor(lb_np).unsqueeze(0).to(device), 
                                         to_tensor(ub_np).unsqueeze(0).to(device))
                x_0_raw = to_tensor(im_resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    z_source = encoder_to_use(x_0_raw)
                    x_rec_0 = decoder_to_use(z_source)
                    model_wrapper.set_reference_images(x_0_raw, x_rec_0)
                    orig_label = classifier((x_0_raw - torch.tensor(mean).view(1,3,1,1).to(device))/torch.tensor(std).view(1,3,1,1).to(device)).argmax(1).item()

                # Get Names
                gt_label = val_map.get(img_name, -1)
                gt_name = get_clean_label_name(synset_labels, gt_label)
                orig_name = get_clean_label_name(synset_labels, orig_label)

                # Check correct classification
                if orig_label != gt_label:
                    print(f"[{i+1}] Skipping {img_name}: Pred {orig_label} != GT {gt_label}")
                    continue
                
                correct_count += 1
                
                print(f"[{i+1}] Processing Valid Image ({correct_count}/{target_limit}): {img_name}")
                print(f"   Ground Truth: {gt_label} ({gt_name})")
                print(f"   Predicted:    {orig_label} ({orig_name})")

                start_time = time.time()
                
                attack = Proposed_attack(
                    model_wrapper, z_source, mean, std, 
                    loss_fn_vgg=loss_fn_vgg, 
                    ssim_fn=ssim_fn,
                    attack_method=current_method, 
                    iteration=iteration, 
                    initial_query=initial_query
                ) 
                
                z_adv, query_log, l2_lat, linf_lat, l2_pix, linf_pix, ssim_vals, lpips_vals, asr_vals = attack.Attack()
                
                norm_log = l2_pix

                with torch.no_grad():
                    x_rec_adv = decoder_to_use(z_adv)
                    x_final = torch.clamp(x_0_raw + (x_rec_adv - x_rec_0), model_wrapper.lb, model_wrapper.ub)
                    adv_label = classifier((x_final - model_wrapper.mean)/model_wrapper.std).argmax(1).item()
                    adv_name = get_clean_label_name(synset_labels, adv_label)

                elapsed_time = time.time() - start_time
                
                print(f"   Result: {orig_name} -> {adv_name}")
                print(f"   Time: {elapsed_time:.4f}s")

                all_queries.append(np.array(query_log))
                all_norms.append(np.array(norm_log))
                all_l2_pixel.append(np.array(l2_pix))
                all_linf_pixel.append(np.array(linf_pix))
                all_ssim.append(np.array(ssim_vals))
                all_lpips.append(np.array(lpips_vals))
                all_asr.append(np.array(asr_vals))
                image_names.append(img_name)
                original_labels.append(orig_label)
                adversarial_labels.append(adv_label)

            # --- Save Results ---
            if len(all_norms) > 0:
                npz_path = os.path.join(output_dir, f'{target_arch}_{current_method}_Top200_ImageNet.npz')
                
                # =================================================================================
                # --- FIXED: Padding logic for arrays to handle Early Stopping array mismatches ---
                # =================================================================================
                def pad_list_of_arrays(list_of_arrays):
                    max_len = max(len(arr) for arr in list_of_arrays)
                    padded_list = []
                    for arr in list_of_arrays:
                        arr = np.array(arr)
                        if len(arr) < max_len:
                            # edge ကိုသုံး၍ နောက်ဆုံးတန်ဖိုးအတိုင်း ကူးယူဖြည့်စွက်ပေးပါမည်
                            pad_width = max_len - len(arr)
                            padded = np.pad(arr, (0, pad_width), mode='edge')
                            padded_list.append(padded)
                        else:
                            padded_list.append(arr)
                    return np.stack(padded_list)

                # np.stack အစား pad_list_of_arrays ကို ပြောင်းသုံးထားပါသည်
                all_queries_arr = pad_list_of_arrays(all_queries).astype(np.float64) 
                all_norms_arr = pad_list_of_arrays(all_norms).astype(np.float32)   
                l2_pix_arr = pad_list_of_arrays(all_l2_pixel).astype(np.float32)
                linf_pix_arr = pad_list_of_arrays(all_linf_pixel).astype(np.float32)
                ssim_arr = pad_list_of_arrays(all_ssim).astype(np.float32)
                lpips_arr = pad_list_of_arrays(all_lpips).astype(np.float32)
                asr_arr = pad_list_of_arrays(all_asr).astype(np.float32)

                # --- Calculate Median/Mean values across all images ---
                query_mean = np.mean(all_queries_arr, axis=0).astype(np.float64)
                norm_median = np.median(all_norms_arr, axis=0).astype(np.float32)
                l2_pix_median = np.median(l2_pix_arr, axis=0).astype(np.float32)
                linf_pix_median = np.median(linf_pix_arr, axis=0).astype(np.float32)
                ssim_median = np.median(ssim_arr, axis=0).astype(np.float32)
                lpips_median = np.median(lpips_arr, axis=0).astype(np.float32)

                np.savez(npz_path, 
                         norm=norm_median,
                         query=query_mean,
                         all_norms=all_norms_arr,
                         all_queries=all_queries_arr,
                         image_names=np.array(image_names, dtype=object),
                         original_labels=np.array(original_labels, dtype=np.int64),
                         adversarial_labels=np.array(adversarial_labels, dtype=np.int64),
                         l2_pixel=l2_pix_arr,
                         linf_pixel=linf_pix_arr,
                         ssim=ssim_arr,
                         lpips=lpips_arr,
                         asr=asr_arr,
                         l2_pixel_median=l2_pix_median,
                         linf_pixel_median=linf_pix_median,
                         ssim_median=ssim_median,
                         lpips_median=lpips_median
                         )
                
                print(f"Success: Saved pure norm/query results and comprehensive pixel metrics to {npz_path}")
                print("-" * 50)
            else:
                print("No correctly classified images found to save.")

if __name__ == "__main__":
    run_latent_attack()