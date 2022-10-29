Alpha Canvas
===============================
Script for AUTOMATIC1111/stable-diffusion-webui to allow for easier outpainting.
I have used the two existing outpainting scripts from the AUTOMATIC1111 repo as a basis.

Installation:
Copy the file to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui

- currently hardcoded to 512x512
- iterations currently not supported.
- images containing the perfect "black" (0,0,0) will be problematic. (because it wants to draw over it) inpainting will be fine though.
- "rough" ui. (canvas is created in gradio event handler)

Workflow:  
=========
The idea is to take the amount of data out of the gradio interface when outpainting or inpainting larger images. You select on the canvas only the region you want to edit or generate. Then you "drag" this region into the respective interface (img2img or img2img inpaint). When the calculation is done, you get the results and insert them seamlessly (with color correction) into the big image.

UI:  OutPainting.
======
Select the "Alpha Canvas" Script in the img2img Tab:

![alpha01](https://user-images.githubusercontent.com/86352149/198781565-d30e8b66-9b6f-49bc-b354-d33a4bf4f5e7.jpg)

You can Open and Close the Canvas Element with the Button below the Script Selection.

![alpha02](https://user-images.githubusercontent.com/86352149/198782179-33008d47-c38d-4fc5-9005-f621d4c88051.jpg)

Grabbing the Title Bar will allow you to place it where you want it to be.
In the lower right corner you can extend or shrink the window.
In the upper Left Corner you can switch between Absolute and Fixed positioning. Fixed is the default and will make the window stay where it is (in Screen space)
If you change it to "A" or Absolute you can place it somewhere on the Page, and it will stay there, scrolling with the page. So you can for Example put it in this neat space of nothingness in the lower right of the img2img Tab.
Load and Save Canvas will save/load the entire Canvas as one Image.

Lets start with loading an image with the Load Canvas Button.

![alpha03](https://user-images.githubusercontent.com/86352149/198783348-65915d71-1ca7-41d7-8263-12dd0571192a.jpg)

The Image will appear in the center of the canvas Element. clicking inside the canvas Element will define a 512x512 region with a rectangle, and, at the same time create a draggable Thumbnail in the upper Left Corner of the window.

Now dragg this Thumbnail to the srcImage Selection of img2img. Set denoising to 1.0 and use a matching prompt to the scene you want to expand. Then hit Generate. When the Images are complete hit the "Grab" Button to pull them into the Canvas Window.

![alpha04](https://user-images.githubusercontent.com/86352149/198785744-9136571c-88fd-4c8c-8abd-b471b4e90244.jpg)

Clicking on the choices on the right Side will display them in the big-Canvas Element too.

![alpha05](https://user-images.githubusercontent.com/86352149/198786131-24602cef-5e49-4912-a6e5-346cf5e412b1.jpg)

You can also make some basic Color Adjustments with the HSL Sliders. This will only affect newly generated Pixel-Values, and can help sometimes.

![alpha06](https://user-images.githubusercontent.com/86352149/198787115-65f5dc6d-1082-4f57-955e-f758def862d6.jpg)

When you are fine with one of the resulting image hit "Apply" und this Patch will now become a part of the canvas. 
At this point the process can be repeated again and again.

![alpha07](https://user-images.githubusercontent.com/86352149/198788069-692a1722-8389-4bf8-867b-9df73fceabe9.jpg)

Inpainting:
======
The biggest change between Outpainting and Inpainting is that the script for Inpainting must be deselected. Only the selection and the read back are used.

![alpha09](https://user-images.githubusercontent.com/86352149/198789363-049795a8-629a-4fdc-b750-6ffd23a1d030.jpg)

The selection process is the same, dragging the area are you want to edit in the src-Image of the Inpainting Tab.
Marking that tree, telling the ai you want to see a hut in its place and hit generate. Once the Images are complete "Gather" them, and take a closer Look.


![alpha10](https://user-images.githubusercontent.com/86352149/198790658-d931ed94-3f85-48b3-9884-49f9bc0e174f.jpg)

![alpha11](https://user-images.githubusercontent.com/86352149/198790729-f57efe5f-c429-4cc1-acb5-18ea3f6fd8cf.jpg)

And the HSL Control is working too.... Maybe a little bit more Saturation. Only applied to changed Pixels.

![alpha12](https://user-images.githubusercontent.com/86352149/198791127-847f7e7a-536a-4f62-b193-64413fce9439.jpg)

