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


Workflow:  
=========
Select the Script in the img2img section:

![img01](https://user-images.githubusercontent.com/86352149/197693588-51454cdc-8fd8-4822-b943-b08f14f1dcc1.jpg)

LeftCoord is the X-position in the source Image that will be the upperLeft Corner of the new Region.

TopCoord is the Y-position in the source Image that will be the upperLeft Corner of the new Region.

With both set, hit Generate and let the AI do its work. If you are fine with one of the resulting Images use the "send to img2img" Button do define this as your new source Image.

Repeat.

UI:  
======
An easier way to define the Region you want to extend is to use the canvas Interface to select it directly in the image.
Just click the "Show/Hide" Button, and a new window should appear.

![img02](https://user-images.githubusercontent.com/86352149/197694743-7b73e105-8944-4763-907a-06a5856000af.jpg)

Clicking in the canvas Area will set the leftCoord / topCoord in the mainUI (drawing a white rectangle where the new region should be created.)

Grabbing the Title Bar will allow you to place it where you want it to be.
In the lower right corner you can extend or shrink the window.
In the upper Left Corner you can switch between Absolute and Fixed positioning. Fixed is the default and will make the window stay where it is (in Screen space)
If you change it to "A" or Absolute you can place it somewhere on the Page, and it will stay there, scrolling with the page. So you can for Example put in this neat space of nothingness in the lower right of the img2img Tab.

![img03](https://user-images.githubusercontent.com/86352149/197695664-75457a68-2882-48f6-bebf-9894685cd952.jpg)

And if you scroll around the Tab.... it will stay there:

![img04](https://user-images.githubusercontent.com/86352149/197695730-bcc0d984-f9f2-43de-93cd-f7893c944586.jpg)

The update Button should(must) be used to update the Canvas from the src-Image.
The Close Button... well .. yeah. exactly.
