import math

import numpy as np
import skimage

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import images, processing, devices
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state

# this function is taken from https://github.com/parlance-zz/g-diffuser-bot
def get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05):
    # helper fft routines that keep ortho normalization and auto-shift before and after fft
    def _fft2(data):
        if data.ndim > 2:  # has channels
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
                out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
        else:  # one channel
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
            out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

        return out_fft

    def _ifft2(data):
        if data.ndim > 2:  # has channels
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
                out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
        else:  # one channel
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
            out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

        return out_ifft

    def _get_gaussian_window(width, height, std=3.14, mode=0):
        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        window = np.zeros((width, height))
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            if mode == 0:
                window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
            else:
                window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (std / 3.14)  # hey wait a minute that's not gaussian

        return window

    def _get_masked_window_rgb(np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        for c in range(3):
            np_mask_rgb[:, :, c] = hardened[:]
        return np_mask_rgb

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    # create a generator with a static seed to make outpainting deterministic / only follow global seed
    rng = np.random.default_rng(0)

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = rng.random((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist ** noise_q) * src_phase  # perform the actual shaping

    brightness_variation = 0.  # color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1., contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb

    matched_noise = shaped_noise[:]

    return np.clip(matched_noise, 0., 1.)

class Script(scripts.Script):
    def title(self):
        return "Outpaint Canvas Region"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        canvasButton = gr.Button("Show/Hide Canvas")
        leftcoord = gr.Slider(label="Left start coord", minimum=-400, maximum=2048, step=1, value=0, elem_id="leftCoord")
        topcoord = gr.Slider(label="top start coord", minimum=-400, maximum=2048, step=1, value=0, elem_id="topCoord")
        dummy = gr.Slider(label="unused", minimum=-1, maximum=1, step=1, value=0)
        
        canvasButton.click(None, [], dummy, _js="(x) => {   let grap = document.body.children[0];\
                                                            let tabDiv = grap.shadowRoot.getElementById('tab_img2img');\
                                                            let img2imgDiv = grap.shadowRoot.getElementById('img2img_image');\
                                                            let imgB64 = img2imgDiv.children[2].children[0].children[1].src;\
                                                            let canvDiv = grap.shadowRoot.getElementById('outDrawCanvasDiv');\
                                                            let canv = grap.shadowRoot.getElementById('outDrawCanvas');\
                                                            console.info('run',canvDiv);\
                                                            if (!canvDiv) {\
                                                              canvDiv = document.createElement('div');\
                                                              canvDiv.id = 'outDrawCanvasDiv';\
                                                              canv = document.createElement('canvas');\
                                                              canv.id = 'outDrawCanvas';\
                                                              canvDiv.append(canv);\
                                                              tabDiv.append(canvDiv);\
                                                              canvDiv.style.display = 'none';\
                                                              canvDiv.style.position = 'absolute';\
                                                              canvDiv.style.left = '50px';\
                                                              canvDiv.style.right = '50px';\
                                                              canvDiv.style.top = '50px';\
                                                              canvDiv.style.bottom = '50px';\
                                                              canvDiv.style.zIndex = '1000';\
                                                              canvDiv.style.background = '#d0d0d0';\
                                                              canvDiv.style.overflow = 'auto';\
                                                              canv.onclick = function(event) {\
                                                                event.stopPropagation();\
                                                                let rect = canv.getBoundingClientRect();\
                                                                let x = event.clientX - rect.left;\
                                                                let y = event.clientY - rect.top;\
                                                                if (x>canv.width-512 || y>canv.height-512) return;\
                                                                let ctx = canv.getContext('2d');\
                                                                ctx.fillStyle = 'black';\
                                                                ctx.fillRect(0, 0, canv.width, canv.height);\
                                                                ctx.drawImage(canv.storeImage, 400, 400, canv.width-800, canv.height-800);\
                                                                ctx.beginPath();\
                                                                ctx.lineWidth = '2';\
                                                                ctx.strokeStyle = 'white';\
                                                                ctx.rect(x, y, 512, 512);\
                                                                ctx.stroke();\
                                                                grap.shadowRoot.getElementById('leftCoord').getElementsByTagName('input')[0].value = x - 400;\
                                                                grap.shadowRoot.getElementById('leftCoord').getElementsByTagName('input')[1].value = x - 400;\
                                                                grap.shadowRoot.getElementById('topCoord').getElementsByTagName('input')[0].value = y -400;\
                                                                grap.shadowRoot.getElementById('topCoord').getElementsByTagName('input')[1].value = y - 400;\
                                                                grap.shadowRoot.getElementById('leftCoord').getElementsByTagName('input')[0].dispatchEvent(new Event('input'));\
                                                                grap.shadowRoot.getElementById('topCoord').getElementsByTagName('input')[0].dispatchEvent(new Event('input'));\
                                                              }\
                                                            }\
                                                            console.info(canvDiv.style.display);\
                                                            if (canvDiv.style.display!=='none') {\
                                                              canvDiv.style.display = 'none';\
                                                              return 0;\
                                                            }\
                                                            if (canv && imgB64) {\
                                                              let ctx = canv.getContext('2d');\
                                                              let image = new Image();\
                                                              image.onload = function() {\
                                                                console.info('onLoad');\
                                                                canv.width = this.width;\
                                                                canv.height = this.height;\
                                                                ctx.drawImage(this, 0, 0);\
                                                                let pixelData = ctx.getImageData(0, 0, canv.width, canv.height).data;\
                                                                let firstX = 9999;\
                                                                let firstY = 9999;\
                                                                let lastX = 0;\
                                                                let lastY = 0;\
                                                                for (let y=0;y<this.height;y=y+10) {\
                                                                  for (let x=0;x<this.width;x++) {\
                                                                    if (pixelData[y*this.width*3+x*3] || pixelData[y*this.width*3+x*3+1] || pixelData[y*this.width*3+x*3+2]) {\
                                                                      if (x<firstX) firstX = x;\
                                                                      if (x>lastX) lastX = x;\
                                                                    }\
                                                                  }\
                                                                }\
                                                                for (let x=0;x<this.width;x=x+10) {\
                                                                  for (let y=0;y<this.height;y++) {\
                                                                    if (pixelData[y*this.width*3+x*3] || pixelData[y*this.width*3+x*3+1] || pixelData[y*this.width*3+x*3+2]) {\
                                                                      if (y<firstY) firstY = y;\
                                                                      if (y>lastY) lastY = y;\
                                                                    }\
                                                                  }\
                                                                }\
                                                                if (lastX<firstX || lastY < firstY) return 0;\
                                                                canv.width = (lastX - firstX) + 800;\
                                                                canv.style.width = canv.width + 'px';\
                                                                canv.height = (lastY - firstY) + 800;\
                                                                canv.style.height = canv.height + 'px';\
                                                                ctx.fillStyle = 'black';\
                                                                ctx.fillRect(0, 0, canv.width, canv.height);\
                                                                ctx.drawImage(image, 400, 400, (lastX - firstX), (lastY - firstY));\
                                                                canvDiv.style.display = 'block';\
                                                                canvDiv.style.position = 'fixed';\
                                                                canvDiv.style.left = '400px';\
                                                                canvDiv.style.width = 'calc(100% - 400px)';\
                                                                canvDiv.style.top = '0px';\
                                                                canvDiv.style.height = '100%';\
                                                                canv.storeImage = this; \
                                                              };\
                                                              console.info('loading image');\
                                                              image.src = imgB64;\
                                                            };\
                                                            return 0}")
        return [leftcoord, topcoord,canvasButton,dummy]

    def run(self, p, leftcoord, topcoord,canvasButton,dummy):
        initial_seed = None
        initial_info = None
        p.mask_blur = 0
        p.inpaint_full_res = False
        p.do_not_save_samples = True
        p.do_not_save_grid = True
        origInBaseLeft = 0
        origInBaseTop = 0
        workItemLeft = leftcoord
        workItemTop = topcoord
        newwidth = p.init_images[0].width
        newheight = p.init_images[0].height
        if leftcoord<0: 
          newwidth = newwidth - leftcoord
          origInBaseLeft = -leftcoord
          workItemLeft = 0          
        if topcoord<0:
          newheight = newheight - topcoord
          origInBaseTop = -topcoord
          workItemTop = 0
        if leftcoord + p.width > newwidth:
          newwidth = leftcoord + p.width
        if topcoord + p.height > newheight:
          newheight = topcoord + p.height
        newBase = Image.new("RGB", (newwidth, newheight), "black")
        newBase.paste(p.init_images[0], (origInBaseLeft, origInBaseTop))
        workItem = Image.new("RGB", (p.width, p.height))
        region = newBase.crop((workItemLeft, workItemTop, workItemLeft+p.width, workItemTop + p.height))
        workItem.paste(region, (0,0))
        workData = np.array(workItem).astype(np.float32) / 255.0
        mask = Image.new("L", (p.width, p.height),color=255)
        maskData = np.array(mask)
        for y in range(p.height):
          for x in range(p.width):
            if workData[y][x][0] + workData[y][x][1] + workData[y][x][2] > 0.001:
              maskData[y][x] = 0
        p.image_mask = Image.fromarray(maskData, mode="L")
        np_image = (np.asarray(workItem) / 255.0).astype(np.float64)
        np_mask = (np.asarray(p.image_mask.convert('RGB')) / 255.0).astype(np.float64)
        noised = get_matched_noise(np_image, np_mask)
        workItem = Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")
        workImages = []
        for n in range(p.batch_size):
          workImages.append(workItem)
        p.init_images = workImages
        p.latent_mask = None
        proc = process_images(p)
        results = []
        for n in range(p.batch_size):
          proc_img = proc.images[n]
          final_image = newBase.copy()
          final_image.paste(proc_img,(workItemLeft,workItemTop))
          proc.images[n] = final_image
        return proc
