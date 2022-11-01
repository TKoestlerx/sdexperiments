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
        return "Alpha Canvas"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        canvasButton = gr.Button("Show/Hide AlphaCanvas")
        dummy = gr.Slider(label="", minimum=-1, maximum=1, step=1, value=0)
        
        javaScriptFunction = """(x) => {
            let alphaWindow,alphaPosition,alphaCanvas,alphaFile,alphaSideMenu,alphaItem,alphaTopMenu;
            if (!gradioApp().getElementById('alphaWindow')) {
                console.info('first run. create Canvas.');
                let tabDiv = gradioApp().getElementById('tab_img2img');
                let Outer = document.createElement('div');
                HTML = `<div id='alphaWindow' style='display:none;position:absolute;min-width:500px;min-height:200px;z-index:1000;overflow:hidden;resize:both;background:#f0f0f0;border-radius:5px;border: 1px solid black;'>
                            <div id='alphaPosition' style='left:0px;top:0px;width:30px;height:30px;display:block;position:absolute;background:#ffffff;color:#000000'>A</div>
                            <div id='alphaTitle' style='left:30px;top:0px;right:75px;height:30px;display:block;position:absolute;background:#4444cc;color:#ffffff;padding-left: 5px;'>AlphaCanvas</div>
                            <div id='alphaClose' style='right:0px;top:0px;width:75px;height:30px;display:block;position:absolute;background:#ffeeee;color:#000000;border:1px solid black;padding-left: 5px;'>Close</div>
                            <input id='alphaFile' style='display:none' type='file'></input>
                            <div id='alphaTopMenu' style='left:0px;top:30px;right:0px;height:64px;display:block;position:absolute;background:#eeeeff'>
                                <img id='alphaItem' style='left:5px;top:0px;width:64px;height:64px;display:block;position:absolute'/>
                                <div id='alphaGrab' style='right:0px;top:15px;width:15%;height:30px;display:block;position:absolute;background:#ddddff;border:1px solid black;padding-left: 5px;overflow:hidden;color:#000000'>Grab Results</div>
                                <div id='alphaMerge' style='right:20%;top:15px;width:15%;height:30px;display:block;position:absolute;background:#ddddff;border:1px solid black;padding-left: 5px;overflow:hidden;color:#000000'>Apply Patch</div>
                                <input id='alphaHue' style='left:75px;top:0px;right:70%;height:20px;display:block;position:absolute' type='range' min='-0.1', max='0.1', value='0.0' step='0.01'></input>
                                <div id='alphaHueLabel' style='left:30%;top:0px;width:10%;height:20px;display:block;position:absolute;color:#000000'>Hue:0</div>
                                <input id='alphaSaturation' style='left:75px;top:21px;right:70%;height:20px;display:block;position:absolute' type='range' min='-0.1', max='0.1', value='0.0' step='0.01'></input>
                                <div id='alphaSaturationLabel' style='left:30%;top:21px;width:10%;height:20px;display:block;position:absolute;color:#000000'>S:0</div>
                                <input id='alphaLightness' style='left:75px;top:42px;right:70%;height:20px;display:block;position:absolute' type='range' min='-0.1', max='0.1', value='0.0' step='0.01'></input>
                                <div id='alphaLightnessLabel' style='left:30%;top:42px;width:10%;height:20px;display:block;position:absolute;color:#000000'>L:0</div>
                                <div id='alphaUpload' style='left:42%;top:1px;width:15%;height:30px;display:block;position:absolute;background:#ddddff;border:1px solid black;padding-left: 5px;color:#000000'>Load Canvas</div>
                                <div id='alphaDownload' style='left:42%;top:33px;width:15%;height:30px;display:block;position:absolute;background:#ddddff;border:1px solid black;padding-left: 5px;color:#000000'>Save Canvas</div>
                            </div>
                            <div id='alphaSideMenu' style='right:0px;top:100px;width:64px;bottom:0px;display:block;position:absolute;background:#eeeeff'></div>
                            <div id='alphaCanvasContainer' style='left:0px;top:100px;right:80px;bottom:0px;display:block;position:absolute;overflow:auto;background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAHCAYAAADEUlfTAAAAAXNSR0IArs4c6QAAABtJREFUGFdjTKi+7cuAAzCCJBe0qm7GJj/oJAGwGxoLJP3XYQAAAABJRU5ErkJggg==)'>
                                <canvas id='alphaCanvas' style='display:block;position:absolute;left:0px;top:0px;'></canvas>
                            </div>
                        </div>`;
                Outer.innerHTML = HTML;
                while (Outer.firstChild) {
                     tabDiv.appendChild(Outer.firstChild);
                }

                // Fixed / Absolute
                gradioApp().getElementById('alphaPosition').onclick = function(event) {
                    if (alphaWindow.style.position==='absolute') {
                        alphaWindow.style.position = 'fixed';
                        alphaWindow.style.top = '100px';
                        alphaPosition.innerHTML = 'F';
                    } else {
                        alphaWindow.style.position = 'absolute';
                        alphaPosition.innerHTML = 'A';
                    }
                }

                // Drag
                function dragElement(elmnt,nn) {
                    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0, pos5 = 0, pos6 = 0;
                    if (gradioApp().getElementById(nn)) {
                        gradioApp().getElementById(nn).onmousedown = dragMouseDown;
                    } else {
                        elmnt.onmousedown = dragMouseDown;
                    }
                    function dragMouseDown(e) {
                        e = e || window.event;
                        let totalOffsetX = 0; let totalOffsetY = 0;
                        let currentElement = elmnt;
                        do{
                            totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
                            totalOffsetY += currentElement.offsetTop - currentElement.scrollTop;
                        }
                        while(currentElement = currentElement.offsetParent)
                        let mpos_x = e.pageX - totalOffsetX;
                        let mpos_y = e.pageY - totalOffsetY;
                        pos3 = e.clientX;
                        pos4 = e.clientY;
                        e.preventDefault();
                        pos5 = elmnt.offsetTop;
                        pos6 = elmnt.offsetLeft;
                        document.onmouseup = closeDragElement;
                        document.onmousemove = elementDrag;
                        e.preventDefault();
                    }
                    function elementDrag(e) {
                        e = e || window.event;
                        e.preventDefault();
                        pos1 = pos3 - e.clientX+5;
                        pos2 = pos4 - e.clientY+5;
                        if (pos5 - pos2 > 0) {
                            elmnt.style.top = (pos5 - pos2) + 'px';
                        }
                        elmnt.style.left = (pos6 - pos1) + 'px';
                    }
                    function closeDragElement() {
                        document.onmouseup = null;
                        document.onmousemove = null;
                    }
                }
                dragElement(gradioApp().getElementById('alphaWindow'),'alphaTitle');

                // Load File
                function loadImage(image) {
                    let ctx = alphaCanvas.getContext('2d');
                    alphaCanvas.width = image.width + 800;
                    alphaCanvas.height = image.height + 800;
                    alphaCanvas.style.width = alphaCanvas.width + 'px';
                    alphaCanvas.style.height = alphaCanvas.height + 'px';
                    alphaCanvas.lastX = '';
                    alphaCanvas.lastY = '';
                    alphaCanvas.markedX = '';
                    alphaCanvas.markedY = '';
                    alphaCanvas.patched = '';
                    alphaSideMenu.innerHTML = '';
                    ctx.clearRect(0, 0, alphaCanvas.width, alphaCanvas.height);
                    ctx.drawImage(image, 400, 400);
                    alphaCanvas.storeImage = image;                 
                }
                gradioApp().getElementById('alphaFile').onchange = function(e) {
                    let reader = new FileReader();
                    reader.onload = function(event){
                        let image = new Image();
                        image.onload = function() {
                            loadImage(this); 
                        };
                        image.src = event.target.result;
                    }
                    reader.readAsDataURL(this.files[0]);
                }
                gradioApp().getElementById('alphaUpload').onclick = function(e) {
                    alphaFile.click();
                }
                
                // Save Image
                gradioApp().getElementById('alphaDownload').onclick = function(e) {
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = alphaCanvas.width-800;
                    tempCanvas.height = alphaCanvas.height-800;
                    let ctx2 = tempCanvas.getContext('2d');
                    ctx2.drawImage(alphaCanvas.storeImage, 0, 0);
                    const link = document.createElement('a');
                    link.download = 'canvas.png';
                    link.href = tempCanvas.toDataURL();
                    link.click();
                    link.delete;
                };
                
                // Color Shifted Patch
                function getColorShiftedPatch() {
                    const colorShift = parseFloat(gradioApp().getElementById('alphaHue').value);
                    const saturationShift = parseFloat(gradioApp().getElementById('alphaSaturation').value);
                    const lightnessShift = parseFloat(gradioApp().getElementById('alphaLightness').value);
                    
                    // HSL Functions from: https://gist.github.com/mjackson/5311256
                    function rgbToHsl(r, g, b) {
                      r /= 255, g /= 255, b /= 255;
                      var max = Math.max(r, g, b), min = Math.min(r, g, b);
                      var h, s, l = (max + min) / 2;
                      if (max == min) {
                        h = s = 0; // achromatic
                      } else {
                        var d = max - min;
                        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
                        switch (max) {
                          case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                          case g: h = (b - r) / d + 2; break;
                          case b: h = (r - g) / d + 4; break;
                        }
                        h /= 6;
                      }
                      return [ h, s, l ];
                    }
                    function hslToRgb(h, s, l) {
                      var r, g, b;
                      if (s == 0) {
                        r = g = b = l; // achromatic
                      } else {
                        function hue2rgb(p, q, t) {
                          if (t < 0) t += 1;
                          if (t > 1) t -= 1;
                          if (t < 1/6) return p + (q - p) * 6 * t;
                          if (t < 1/2) return q;
                          if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                          return p;
                        }
                        var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                        var p = 2 * l - q;
                        r = hue2rgb(p, q, h + 1/3);
                        g = hue2rgb(p, q, h);
                        b = hue2rgb(p, q, h - 1/3);
                      }
                      return [ r * 255, g * 255, b * 255 ];
                    }
                    if (alphaCanvas.patched && alphaCanvas.storeImage) {
                        const tempCanvasOrig = document.createElement('canvas');
                        tempCanvasOrig.width = alphaCanvas.patched.width;
                        tempCanvasOrig.height = alphaCanvas.patched.height;
                        let ctx1 = tempCanvasOrig.getContext('2d');
                        const tempCanvasNew = document.createElement('canvas');
                        tempCanvasNew.width = alphaCanvas.patched.width;
                        tempCanvasNew.height = alphaCanvas.patched.height;
                        let ctx2 = tempCanvasNew.getContext('2d');
                        ctx1.drawImage(alphaCanvas.storeImage,alphaCanvas.patchedX-400,alphaCanvas.patchedY-400,512,512,0,0,512,512);
                        ctx2.drawImage(alphaCanvas.patched, 0, 0);
                        let pixelData1 = ctx1.getImageData(0, 0, tempCanvasOrig.width, tempCanvasOrig.height).data;
                        let pixel2 = ctx2.getImageData(0, 0, tempCanvasNew.width, tempCanvasNew.height);
                        let pixelData2 = pixel2.data;
                        for (let y=0;y<tempCanvasOrig.height;y++) {
                          for (let x=0;x<tempCanvasOrig.width;x++) {
                            const index = y*tempCanvasOrig.width*4+x*4;
                            if (pixelData1[index]!==pixelData2[index] || pixelData1[index+1]!==pixelData2[index+1] || pixelData1[index+2]!==pixelData2[index+2]) {
                              /*
                              pixelData2[index] = Math.abs(pixelData2[index] - pixelData1[index]);
                              pixelData2[index+1] = Math.abs(pixelData2[index+1] - pixelData1[index+1]);
                              pixelData2[index+2] = Math.abs(pixelData2[index+2] - pixelData1[index+2]);
                              */
                              const hsl = rgbToHsl(pixelData2[index],pixelData2[index+1],pixelData2[index+2]);
                              hsl[0] += colorShift;
                              hsl[1] += saturationShift;
                              hsl[2] += lightnessShift;
                              if (hsl[0]<0) { hsl[0]+=1;}
                              if (hsl[0]>1) { hsl[0]-=1;}
                              if (hsl[1]<0) { hsl[1]=0;}
                              if (hsl[1]>1) { hsl[1]=1;}
                              if (hsl[2]<0) { hsl[2]=0;}
                              if (hsl[2]>1) { hsl[2]=1;}
                              let rgb = hslToRgb(hsl[0],hsl[1],hsl[2]);
                              pixelData2[index] = rgb[0];
                              pixelData2[index+1] = rgb[1];
                              pixelData2[index+2] = rgb[2];
                            }
                          }
                        }
                        ctx2.putImageData(pixel2, 0, 0);
                        return tempCanvasNew;
                    }
                }
                gradioApp().getElementById('alphaHue').onchange = function(e) {
                    if (alphaCanvas.patched && alphaCanvas.storeImage) {
                        redrawCanvas();
                        gradioApp().getElementById('alphaHueLabel').innerHTML = 'Hue:' + gradioApp().getElementById('alphaHue').value;
                    }
                }
                gradioApp().getElementById('alphaSaturation').onchange = function(e) {
                    if (alphaCanvas.patched && alphaCanvas.storeImage) {
                        redrawCanvas();
                        gradioApp().getElementById('alphaSaturationLabel').innerHTML = 'S:' + gradioApp().getElementById('alphaSaturation').value;
                    }
                }
                gradioApp().getElementById('alphaLightness').onchange = function(e) {
                    if (alphaCanvas.patched && alphaCanvas.storeImage) {
                        redrawCanvas();
                        gradioApp().getElementById('alphaLightnessLabel').innerHTML = 'L:' + gradioApp().getElementById('alphaLightness').value;
                    }
                }
                
                gradioApp().getElementById('alphaMerge').onclick = function(e) {
                    if (alphaCanvas.patched && alphaCanvas.storeImage) {
                        let leftShift = 0;
                        let topShift = 0;
                        let xMove = alphaCanvas.patchedX - 400;
                        let yMove = alphaCanvas.patchedY - 400;
                        let newwidth = alphaCanvas.storeImage.width;
                        let newheight = alphaCanvas.storeImage.height;
                        if (alphaCanvas.patchedX < 400) {
                            newwidth = newwidth + (400 - alphaCanvas.patchedX);
                            leftShift = (400 - alphaCanvas.patchedX);
                            xMove = 0;
                        }
                        if (alphaCanvas.patchedY < 400) {
                            newheight = newheight + (400 - alphaCanvas.patchedY);
                            topShift = (400 - alphaCanvas.patchedY);
                            yMove = 0;
                        }
                        if (alphaCanvas.patchedX + 112 >newwidth) newwidth = alphaCanvas.patchedX + 112;
                        if (alphaCanvas.patchedY + 112 >newheight) newheight = alphaCanvas.patchedY + 112;
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = newwidth;
                        tempCanvas.height = newheight;
                        let ctx2 = tempCanvas.getContext('2d');
                        ctx2.drawImage(alphaCanvas.storeImage, leftShift, topShift);
                        const shiftedImage = getColorShiftedPatch();
                        ctx2.drawImage(shiftedImage, xMove, yMove);
                        alphaCanvas.patched = ''
                        alphaCanvas.markedX = ''
                        alphaCanvas.markedY = ''
                        alphaSideMenu.innerHTML = ''
                        loadImage(tempCanvas);
                    }
                };
                
                
                function importRegion(image) {
                    let ctx = alphaCanvas.getContext('2d');
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = 512;
                    tempCanvas.height = 512;
                    let ctx2 = tempCanvas.getContext('2d');
                    ctx2.drawImage(image, 0,0);
                    alphaCanvas.patched = tempCanvas;
                    alphaCanvas.patchedX = alphaCanvas.markedX;
                    alphaCanvas.patchedY = alphaCanvas.markedY;
                }
                
                function convertToDataUrl(image) {
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = 512;
                    tempCanvas.height = 512;
                    let ctx2 = tempCanvas.getContext('2d');
                    ctx2.drawImage(image, 0,0);
                    return tempCanvas.toDataURL('image/png');
                }

                // Import results
                function getImages() {
                    let imgParent = gradioApp().getElementById('img2img_gallery');
                    const choices = imgParent.getElementsByClassName('gallery-item group');
                    let nr = 0;
                    alphaSideMenu.innerHTML = '';
                    console.info('reading results... checking ' + choices.length + ' elements.');
                    for (choice in choices) {
                        if (choices[choice].children && choices[choice].children[0]) {
                            console.info('creating new img Element');
                            let image2 = new Image();
                            image2.style.right = '20px';
                            image2.style.top = (nr*70)+'px';
                            image2.style.width = '64px';
                            image2.style.height = '64px';
                            image2.style.display = 'block';
                            image2.style.position = 'absolute';
                            image2.style.zIndex = '1';
                            alphaSideMenu.append(image2);
                            image2.onload = function() {
                                console.info('result image load complete.');
                            };
                            image2.onclick = function(e) {
                                if (alphaCanvas.markedX) {
                                    importRegion(this);
                                    redrawCanvas();
                                } else {
                                    console.info('dont know where to put this region.');
                                }
                            }
                            //image2.src = convertToDataUrl(choices[choice].children[0]);
                            image2.src = choices[choice].children[0].src;
                            nr++;
                        }
                    }
                    console.info('importing results complete. ' + nr + ' images');
                }
                gradioApp().getElementById('alphaGrab').onclick = function(e) {
                    getImages();
                }

                function redrawCanvas() {
                    let ctx = alphaCanvas.getContext('2d');
                    ctx.clearRect(0, 0, alphaCanvas.width, alphaCanvas.height);
                    ctx.drawImage(alphaCanvas.storeImage, 400, 400, alphaCanvas.width-800, alphaCanvas.height-800);
                    if (alphaCanvas.patched) {
                        const colorShift = parseFloat(gradioApp().getElementById('alphaHue').value);
                        const saturationShift = parseFloat(gradioApp().getElementById('alphaSaturation').value);
                        const lightnessShift = parseFloat(gradioApp().getElementById('alphaLightness').value);
                        if (Math.abs(colorShift)+Math.abs(saturationShift)+Math.abs(lightnessShift)>0.005) {
                          const shiftedImage = getColorShiftedPatch()
                          ctx.drawImage(shiftedImage, alphaCanvas.markedX, alphaCanvas.markedY);
                        } else {
                          ctx.drawImage(alphaCanvas.patched, alphaCanvas.markedX, alphaCanvas.markedY);
                        }
                    }
                    // current marker
                    if (alphaCanvas.lastX) {
                        ctx.beginPath();
                        ctx.lineWidth = '1';
                        ctx.strokeStyle = 'white';
                        ctx.rect(alphaCanvas.lastX, alphaCanvas.lastY, 512, 512);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.lineWidth = '1';
                        ctx.strokeStyle = 'black';
                        ctx.rect(alphaCanvas.lastX-1, alphaCanvas.lastY-1, 514, 514);
                        ctx.stroke();
                    }
                    // drop marker
                    if (alphaCanvas.markedX) {
                        ctx.beginPath();
                        ctx.lineWidth = '1';
                        ctx.strokeStyle = 'red';
                        ctx.rect(alphaCanvas.markedX, alphaCanvas.markedY, 512, 512);
                        ctx.stroke();
                    }  
                }
                // Select Working Region
                gradioApp().getElementById('alphaCanvas').onclick = function(event) {
                    event.stopPropagation();
                    let rect = alphaCanvas.getBoundingClientRect();
                    let x = Math.floor(event.clientX - rect.left) - 256;
                    let y = Math.floor(event.clientY - rect.top) - 256;
                    if (x < 0 || y<0) return;
                    if (x > alphaCanvas.width - 512 || y > alphaCanvas.height - 512) return;           
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = 512;
                    tempCanvas.height = 512;
                    let ctx2 = tempCanvas.getContext('2d');
                    ctx2.fillStyle = 'rgb(0,0,0)'
                    ctx2.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                    ctx2.drawImage(alphaCanvas.storeImage, x-400, y-400, 512, 512,0,0,512,512);
                    alphaItem.src = tempCanvas.toDataURL('image/png');
                    alphaCanvas.lastX = x;
                    alphaCanvas.lastY = y;
                    redrawCanvas();
                }

                // full window on right click
                function toggleFullScreen(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    if (alphaWindow.fullS) {
                        alphaWindow.style.position = alphaWindow.oldPos;
                        alphaWindow.style.left = alphaWindow.oldPosLeft;
                        alphaWindow.style.width = alphaWindow.oldWidth;
                        alphaWindow.style.top = alphaWindow.oldPosTop;
                        alphaWindow.style.height = alphaWindow.oldHeight;
                        alphaPosition.innerHTML = alphaWindow.oldPosString;
                        alphaPosition.style.display = 'block';
                        alphaWindow.fullS = false;
                    } else {
                        alphaWindow.oldPos = alphaWindow.style.position;
                        alphaWindow.oldPosLeft = alphaWindow.style.left;
                        alphaWindow.oldWidth = alphaWindow.style.width;
                        alphaWindow.oldPosTop = alphaWindow.style.top;
                        alphaWindow.oldHeight = alphaWindow.style.height;
                        alphaWindow.oldPosString = alphaPosition.innerHTML;
                        alphaPosition.style.display = 'none';
                        alphaWindow.fullS = true;
                        alphaWindow.style.position = 'fixed';
                        alphaWindow.style.left = '0px';
                        alphaWindow.style.width = '100%';
                        alphaWindow.style.top = '0px';
                        alphaWindow.style.height = '100%';
                    }
                    return false;
                }                
                gradioApp().getElementById('alphaCanvas').addEventListener('contextmenu', toggleFullScreen, false);
                gradioApp().getElementById('alphaCanvasContainer').addEventListener('contextmenu', toggleFullScreen, false);

                gradioApp().getElementById('alphaClose').onclick = function(e) {
                    alphaWindow.style.display = 'none';
                }
                gradioApp().getElementById('alphaItem').onclick= function(e) {
                    const dt= new DataTransfer()
                    let fileString = gradioApp().getElementById('alphaItem').src;
                    fetch(fileString).then(
                        function ok(o) {
                            o.blob().then(
                                function ok2(o2) {
                                    let file = new File([o2], "transfer.png", { type: 'image/png'})
                                    dt.items.add(file);
                                    let imgParent = gradioApp().getElementById('img2img_image');
                                    if  (get_tab_index('mode_img2img')===1) {
                                        imgParent = gradioApp().getElementById('img2maskimg');
                                    }
                                    const fileInput = imgParent.querySelector('input[type="file"]');
                                    fileInput.files = dt.files;
                                    fileInput.dispatchEvent(new Event('change'));
                                },
                                function failed2(e2) {
                                });
                        },
                        function failed(e) {
                        });
                    alphaCanvas.markedX = alphaCanvas.lastX;
                    alphaCanvas.markedY = alphaCanvas.lastY;
                    alphaCanvas.lastX = '';
                    alphaCanvas.lastY = '';
                    alphaCanvas.patched = '';
                    alphaSideMenu.innerHTML = '';
                    redrawCanvas();
                }
            }
            alphaWindow = gradioApp().getElementById('alphaWindow');
            alphaPosition = gradioApp().getElementById('alphaPosition');
            alphaCanvas = gradioApp().getElementById('alphaCanvas');
            alphaFile = gradioApp().getElementById('alphaFile');
            alphaSideMenu = gradioApp().getElementById('alphaSideMenu');
            alphaTopMenu = gradioApp().getElementById('alphaTopMenu');
            alphaItem = gradioApp().getElementById('alphaItem');
            
            if (alphaWindow.style.display!=='none') {
                alphaWindow.style.display = 'none';
                return 0;
            }

            function resetView() {
                alphaWindow.style.display = 'block';
                alphaWindow.style.position = 'fixed';
                alphaWindow.style.left = '400px';
                alphaWindow.style.width = '800px';
                alphaWindow.style.top = '0px';
                alphaWindow.style.height = '50%';
                alphaPosition.innerHTML = 'F';
                alphaWindow.fullS = false;
            }
            if (alphaCanvas) {
                resetView();
            } else {
                console.info('failed to get Image data');
            }
        return 0}"""

        canvasButton.click(None, [], dummy, _js = javaScriptFunction)
        return [canvasButton,dummy]

    def run(self, p, canvasButton, dummy):
        p.mask_blur = 0
        p.inpaint_full_res = False
        p.do_not_save_samples = True
        p.do_not_save_grid = True
        newBase = Image.new("RGB", (p.width, p.height), "black")
        newBase.paste(p.init_images[0], (0,0))
        workItem = newBase.copy()
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
          final_image.paste(proc_img,(0,0))
          proc.images[n] = final_image
        return proc
