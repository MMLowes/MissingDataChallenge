from skimage import io
import numpy as np
import copy

def read_file_list(file_list):
    try:
        file = open(file_list, 'r')
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}: {file_list}")
        return None

    lines = file.readlines()
    if len(lines) < 1:
        print(f"Could not read from {file_list}")
        return

    samples = []
    for line in lines:
        ls = line.strip()
        samples.append(ls)
    return samples

def save_image(save_name, output, mask, image):
    
    output = output.permute(1,2,0).detach().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    image = image.permute(1,2,0).cpu().numpy()
    mask = np.stack((mask,mask,mask),-1)
    combined = copy.deepcopy(image)
    combined[mask.astype(np.bool)] = output[mask.astype(np.bool)]
    arr = np.concatenate((image,mask,output,combined),1)
    arr = np.clip(arr,0,1)
    io.imsave(save_name, arr)

def save_inp_out(save_name, output, mask, masked_image):
    
    output = output.permute(1,2,0).detach().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    masked_image = masked_image.permute(1,2,0).cpu().numpy()[:,:,:3]
    mask = np.stack((mask,mask,mask),-1)
    combined = copy.deepcopy(masked_image)
    combined[mask.astype(np.bool)] = output[mask.astype(np.bool)]
    arr = np.concatenate((masked_image,combined),1)
    arr = np.clip(arr,0,1)
    io.imsave(save_name, arr)

def save_test_image(save_name, output, mask, masked_image):
    output = output.permute(1,2,0).detach().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    masked_image = masked_image.permute(1,2,0).cpu().numpy()[:,:,:3]
    mask = np.stack((mask,mask,mask),-1)
    combined = copy.deepcopy(masked_image)
    combined[mask.astype(np.bool)] = output[mask.astype(np.bool)]
    # arr = np.concatenate((image,mask,output,combined),1)
    # arr = np.clip(arr,0,1)
    io.imsave(save_name, combined)