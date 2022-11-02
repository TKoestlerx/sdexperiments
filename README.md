Alpha Canvas
===============================
Script for AUTOMATIC1111/stable-diffusion-webui to allow for easier outpainting.
I have used the two existing outpainting scripts from the AUTOMATIC1111 repo as a basis.

Installation:
Copy the file to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui

- currently hardcoded to 512x512
- "rough" ui. (canvas is created in gradio event handler)

Workflow:  
=========
The idea is to take the amount of data out of the gradio interface when outpainting or inpainting larger images. You select on the canvas only the region you want to edit or generate. Then you "drag" this region into the respective interface (img2img or img2img inpaint). When the calculation is done, you get the results and insert them seamlessly (with color correction) into the big image.

UI:  OutPainting
======
Select the "Alpha Canvas" Script in the img2img Tab:

![alpha01](https://user-images.githubusercontent.com/86352149/198781565-d30e8b66-9b6f-49bc-b354-d33a4bf4f5e7.jpg)

You can Open and Close the Canvas Element with the Button below the Script Selection.

![alpha02](https://user-images.githubusercontent.com/86352149/198782179-33008d47-c38d-4fc5-9005-f621d4c88051.jpg)

Grabbing the Title Bar will allow you to place it where you want it to be.
In the lower right corner you can extend or shrink the window.
In the upper Left Corner you can switch between Absolute and Fixed positioning. Fixed is the default and will make the window stay where it is (in Screen space)
If you change it to "A" or Absolute you can place it somewhere on the Page, and it will stay there, scrolling with the page. So you can for Example put it in this neat space of nothingness in the lower right of the img2img Tab.
Load and Save Canvas will save/load the entire Canvas as one Image. Right clicking on the canvas will toggle between std and maximized size.

Lets start with loading an image with the Load Canvas Button.

![alpha2_1](https://user-images.githubusercontent.com/86352149/199514459-fb7e5c6e-f26b-47b5-bb8c-1fae1795cfeb.jpg)

The Image will appear in the center of the canvas Element. clicking inside the canvas Element will define a 512x512 region with a rectangle, and, at the same time create a Thumbnail in the upper Left Corner of the window.

This works both for the img2img Tab (Alphacanvas script must be active, set denoising to 1.0)

![alpha2_2](https://user-images.githubusercontent.com/86352149/199515072-0dfa8d92-9f1e-464b-8388-eb5e8e5eed55.jpg)

And for the inpainting Tab. Which will upload a seperate alpha mask. (chosse latent noise and set denoising at 1.0)

![alpha2_3](https://user-images.githubusercontent.com/86352149/199515609-aa5c8d6e-b386-4219-9fb3-3c22bb473423.jpg)

Then hit Generate. When the Images are complete hit the "Grab" Button to pull them into the Canvas Window.

![alpha04](https://user-images.githubusercontent.com/86352149/198785744-9136571c-88fd-4c8c-8abd-b471b4e90244.jpg)

Clicking on the choices on the right Side will display them in the big-Canvas Element too.

![alpha2_4](https://user-images.githubusercontent.com/86352149/199517887-23a43101-4f1b-4c39-b9e3-761a49d37df1.jpg)

![alpha2_5](https://user-images.githubusercontent.com/86352149/199517938-3430170b-adca-487c-992b-eb89b3b63681.jpg)

When you are fine with one of the resulting image hit "Apply" und this Patch will now become a part of the canvas. 
At this point the process can be repeated again and again.

![alpha2_6](https://user-images.githubusercontent.com/86352149/199518469-abb867b0-13b8-4fdc-9c97-5e8cd95edb3c.jpg)

Both Outpainting variants use different types of noise and both have advantages and disadvantages.
The Outpainting in the img2img Tab uses black (0,0,0) as the marker Color und will fail if the image contains this color.

Inpainting:
======
If the selected region does not contain transparent areas, the image will be transferred to the gradio interface normally. There you can use the marking tools as usual and at the end, just like in the outdraw variants, copy the images back.

![alpha2_7](https://user-images.githubusercontent.com/86352149/199520305-e4805097-a737-431d-8583-1cad997d827b.jpg)


