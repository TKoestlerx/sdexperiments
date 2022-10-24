Basic Canvas Outpainting Test.
===============================
Script for AUTOMATIC1111/stable-diffusion-webui to allow for easier outpainting.
I have used the two existing outpainting scripts from the AUTOMATIC1111 repo as a basis.

Installation:
Copy the file to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui
Set the denoising strength to a high value (1.0)

+ supports selection of region to expand with x/y coords in the current src Image.
+ supports selection of the same x/y coord via canvas selection.
+ Image is expanded to the new dimensions before processing. (empty = black)
+ Mask is auto-generated from the image (where color == (0,0,0))
+ batch support

- currently hardcoded to 512x512
- iterations currently not supported.
- images containing the perfect "black" (0,0,0) will be problematic. (because it wants to draw over it)
- "rough" ui. (canvas is created in gradio event handler)
- currently no automatic saving and probably a lot of bugs.
- currently limited to ~2500x2500px max.

But, i have a lot of fun playing around with it :)


The UI

======

Newer Version(branch: windowedcanvas)

+window is draggable

+window is resizeable

+window has update/close buttons

![windowed](https://user-images.githubusercontent.com/86352149/197577511-63bed66e-3cc0-4077-8d01-bd38c6fede19.jpg)
