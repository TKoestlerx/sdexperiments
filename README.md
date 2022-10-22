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


The Ui:

![canv01](https://user-images.githubusercontent.com/86352149/197335086-bd793e1b-58cd-482a-818c-42a34fd1c4ef.jpg)

Click on show/hide will open the canvas:

![canv02](https://user-images.githubusercontent.com/86352149/197335135-cf0f5aff-85fa-4c45-9088-e6f3a938e565.jpg)

If you are good with one of the resulting images just copy it over as the new src (send to img2img):

![canv03](https://user-images.githubusercontent.com/86352149/197335228-e4ea1559-c609-498b-b797-3c98c4be009a.jpg)

And repeat....

![canv04](https://user-images.githubusercontent.com/86352149/197335278-5624a152-82bb-4c51-8e09-647dd14fefb9.jpg)
