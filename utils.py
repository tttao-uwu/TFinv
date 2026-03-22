import os
import cv2
import torch
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from scipy import stats
from typing import List, Tuple, Union
from torchvision import utils as vutils
from torchvision.transforms import ToPILImage
from scipy.stats import norm, entropy
from sklearn.decomposition import PCA




def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, fontScale=1, thickness=2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img    

def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)

    return pil_img



def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    
    image = (image - image.min()) / (image.max() - image.min()+1e-8)
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis    

def show_cross_attention_blackwhite(prompts,
                         attention_maps, 
                         display_image=True,
                         ):
    # tokens = tokenizer.encode(prompts[select])
    # decoder = tokenizer.decode
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    split_imgs = []
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        split_imgs.append(image)
        image = text_under_image(image, prompts[i])
        images.append(image)
    pil_img=view_images(np.stack(images, axis=0),display_image=display_image)
    return pil_img, split_imgs

def show_cross_attention(prompts,
                         attention_maps, 
                         display_image=True,
                         ):
    # tokens = tokenizer.encode(prompts[select])
    # decoder = tokenizer.decode
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    split_imgs = []
    white_image = Image.new('RGB', (500, 500), (255, 255, 255))
    
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        image = show_image_relevance(image, white_image)
        
        image = image.astype(np.uint8)
        
        # image = 255 * image / image.max()
        # image = image.unsqueeze(-1).expand(*image.shape, 3)
        # image = image.numpy().astype(np.uint8)
        
        image = np.array(Image.fromarray(image).resize((256, 256)))
        split_imgs.append(image)
        # image = text_under_image(image, decoder(int(tokens[i])))
        image = text_under_image(image, prompts[i])
        images.append(image)
    pil_img=view_images(np.stack(images, axis=0),display_image=display_image)
    return pil_img, split_imgs



    
def show_cross_attention_plus_orig_img(
                        prompts,
                        crossattn, 
                        display_image=True,
                        orig_image=None,
                        indices_to_alter=None,
                        res=16,
                        ):
    images = []
    split_imgs = []
    if indices_to_alter is None:
        indices_to_alter=list(range(len(prompts)))
    for i in range(len(prompts)):
        image = crossattn[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            split_imgs.append(image)

            image = text_under_image(image, prompts[i])
            images.append(image)

    pil_img=view_images(np.stack(images, axis=0),display_image=display_image)
    return pil_img, images    

    
def save_cross_attn(args, cross_avg_dict, RES=16):
    org_image = Image.open(args.input_image_path).convert("RGB")
    prompts=["<|startoftext|>",] + args.caption.split(' ') + ["<|endoftext|>",]

    crossattn = cross_avg_dict['attn'][RES]

    attn_img2, _ = show_cross_attention_plus_orig_img(prompts, crossattn, orig_image=org_image)
    attn_img2_wo, _ =show_cross_attention(prompts, crossattn)
    attn_img2_blakcwhite, _ =show_cross_attention_blackwhite(prompts, crossattn) 

    attn_img2.save(os.path.join(args.path_attention_imgs_save,'crossattn.png')) 
    attn_img2_wo.save(os.path.join(args.path_attention_imgs_save,'crossattn_wo.png')) 
    attn_img2_blakcwhite.save(os.path.join(args.path_attention_imgs_save,'crossattn_blackwhite.png')) 
    
    


def show_image_distribution(latents, path_dis_save):
    """
    Visualizes the distribution of latent variables and compares it to a standard Gaussian distribution.
    Calculates the KL divergence between the latent distribution and a standard Gaussian (mean=0, variance=1).
    
    Parameters:
        latents: PyTorch tensor of latent variables.
        path_dis_save: File path where the plot will be saved.
    
    Returns:
        mean (float): Mean of the latent variables.
        var (float): Variance of the latent variables.
        kl_divergence (float): KL divergence with a standard Gaussian.
    """
    # Convert latents to a NumPy array
    data = latents.cpu().numpy().copy()

    # Compute mean and variance of the data
    mean = np.mean(data)
    var = np.var(data)

    # Create histograms for data (latent distribution) and standard Gaussian
    data_hist, bins = np.histogram(data, bins=100, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    standard_gaussian_pdf = norm.pdf(bin_centers, loc=0, scale=1)

    # Ensure no division by zero or log of zero
    data_hist += 1e-10
    standard_gaussian_pdf += 1e-10

    # Calculate KL divergence
    kl_divergence = np.sum(data_hist * np.log(data_hist / standard_gaussian_pdf) * np.diff(bins))

    # Plot the distribution of the data
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.6)

    # Plot the standard Gaussian distribution (mean=0, std=1)
    x = np.linspace(min(data), max(data), 1000)
    standard_gaussian_curve = norm.pdf(x, loc=0, scale=1)
    plt.plot(x, standard_gaussian_curve, color='red', linestyle='--')

    # Add the mean, variance, and KL divergence as text on the figure
    plt.text(
        0.40, 0.95, 
        f"Mean: {mean:.4f}\nVar: {var:.4f}\nKL-D: {kl_divergence:.4f}", 
        transform=plt.gca().transAxes, 
        fontsize=30, 
        verticalalignment='top', 
        horizontalalignment='right'
    )

    # Hide axis ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # # Add legend for clarity
    # plt.legend()

    # Save the figure
    plt.savefig(path_dis_save)
    plt.close()

    # Return the computed statistics
    return mean, var, kl_divergence


    

def show_self_attention(selfattn_maps, 
                        display_image=False):
    # tokens = tokenizer.encode(prompts[select])
    # decoder = tokenizer.decode
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []

    white_image = Image.new('RGB', (256, 256), (255, 255, 255))
    

    image = selfattn_maps
    image = show_image_relevance(image, white_image)
    
    image = image.astype(np.uint8)
    
    image = Image.fromarray(image).resize((256, 256))

    return image


def show_mean_and_var(mean, variance, path_mean_and_var):
    x = []
    
    for i in range(len(mean)):
        x.append(i*2)
    
    plt.plot(x,mean,'-',color = 'b',label="Mean")
    plt.plot(x,variance,'-',color = 'purple',label="Variance")
    plt.legend(fontsize=20)
    plt.ylim(-0.1, 1.1)
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    # plt.title("Mean and Variance across Iteration", fontsize=22)
    plt.savefig(path_mean_and_var)
    plt.close()
    
    
def show_loss(loss_, path_loss):
    y = []
    
    for j in range(len(loss_)):
        y.append(j*2)
    
    plt.plot(y,loss_,'-',color = 'purple')
        # Set the y-axis range from 0 to 5
    plt.ylim(0, 5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig(path_loss)
    
    
    
    
def cluster(self_attention, num_segments=5,):
    np.random.seed(1)
    resolution, feat_dim = self_attention.shape[0], self_attention.shape[-1]
    attn = self_attention.cpu().numpy().reshape(resolution ** 2, feat_dim)
    kmeans = KMeans(n_clusters=num_segments, n_init=10).fit(attn)
    clusters = kmeans.labels_
    clusters = clusters.reshape(resolution, resolution)
    return clusters

def run_clusters(avg_dict, resolution, dict_key, save_path, special_name):
    clusters = cluster(avg_dict[dict_key][resolution], num_segments=5,)
    output_name=f'cluster_{dict_key}_{resolution}_{special_name}.png'
    plt.imshow(clusters)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, output_name), bbox_inches='tight', pad_inches=0)
    
    
def draw_pca(caption, avg_dict, resolution, save_path, spec):
    
    pca = PCA(n_components=3)
    RESOLUTION=resolution

    before_pca = avg_dict.reshape(RESOLUTION*RESOLUTION,-1).cpu().numpy()
    # print(before_pca.shape)

    pca.fit(before_pca)
    after_pca = pca.transform(before_pca)

    after_pca = after_pca.reshape(RESOLUTION,RESOLUTION,-1)
    pca_img_min = after_pca.min(axis=(0, 1))
    pca_img_max = after_pca.max(axis=(0, 1))
    pca_img = (after_pca - pca_img_min) / (pca_img_max - pca_img_min)

    output_name=f'pca_{spec}_{caption}.png'
    pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_img=pca_img.resize((512,512))
    pca_img.save(os.path.join(save_path, output_name))
