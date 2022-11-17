Alpha Canvas
===============================
Script for AUTOMATIC1111/stable-diffusion-webui to allow for easier outpainting and inpainting (of larger Images).
I have used the two existing outpainting scripts from the AUTOMATIC1111 repo as a basis.

Installation:
Copy the file to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui

Workflow:  
=========
The idea is to take the massive amount of data out of the gradio interface when outpainting or inpainting larger images. You select on the canvas only the region you want to edit or generate. Then you "transfer" this region into the respective interface (img2img or img2img inpaint). 
When the calculation is done, you get the results and insert them seamlessly (with color correction) into the big image.


UI: 
===
Select the "Alpha Canvas" Script in the img2img Tab:

The Script needs to be active if you are outpainting in the img2img Tab. It will replace all perfect black (0,0,0) pixels with newly generated ones.
The Canvas UI will still work for inpainting if the script is de-activated. (if the ui has already been opened.)

![alpha5_1](https://user-images.githubusercontent.com/86352149/202400485-a7490aa2-c24f-40ae-a45d-256a31cdd39f.jpg)

snapGrid defines the possible "step" size when doing the region selection. 

maxOutpainting defines the "border" put around the current Image in which regions can be selected.

You can Open and Close the Canvas Element with the Button below the Script Selection.

![alpha5_2](https://user-images.githubusercontent.com/86352149/202401073-e517b951-97f0-4e75-80f4-288f3e025afb.jpg)

Grabbing the Title Bar will allow you to place it where you want it to be.

In the lower right corner you can extend or shrink the window.

In the upper Left Corner you can switch between Absolute and Fixed positioning. Fixed will make the window stay where it is (in Screen space), while "A" or absolute allows you can place it somewhere on the Page, and it will stay there, scrolling with the page. 

Load and Save Canvas will save/load the entire Canvas as one Image.

Right clicking on the canvas will toggle between std and maximized size.

The HSL Sliders allow you to make manuell color correction when pasting the results back in.

Apply Patch will make the current selected Patch part of the Canvas

Grab Results is used the read the result Images back to the Canvas Element.


Lets start with loading an image with the Load Canvas Button.

The Image will appear in the center of the canvas Element. clicking inside the canvas Element will define a region with a rectangle, and, at the same time create a Thumbnail in the upper Left Corner of the window. Clicking on the Thumbnail will transfer the data to the gradio Interface and draw a red rectangle to mark the active area.

This works both for the img2img Tab (Alphacanvas script must be active, set denoising to 1.0)

![alpha5_3](https://user-images.githubusercontent.com/86352149/202404340-eb470a98-2c18-40cf-a95b-129a24daaf17.jpg)

And for the inpainting Tab. Which will upload a seperate alpha mask. (chosse latent noise or latent nothing and set denoising at 1.0. latent nothing with denoising set to 1 will give equal results for the same seeds as txt2img)

![alpha5_4](https://user-images.githubusercontent.com/86352149/202404665-c79a657b-474e-4f0f-8151-979c43d98552.jpg)

Then hit Generate. When the Images are complete hit the "Grab" Button to pull them into the Canvas Window.

Clicking on the choices on the right Side will show them in the "Big" Image. Color Correction can be applied at this point.

When you are fine with one of the resulting image hit "Apply" und this Patch will now become a part of the canvas. 
At this point the process can be repeated again and again.

![alpha5_6](https://user-images.githubusercontent.com/86352149/202404987-05c8ae08-3218-44a2-98b1-b2629460ad0a.jpg)

Both Outpainting variants use different types of noise and both have advantages and disadvantages.
The Outpainting in the img2img Tab uses black (0,0,0) as the marker Color und will fail if the image contains this color.

Inpainting:
======
If the selected region does not contain transparent areas, the image will be transferred to the gradio interface normally. 

![alpha5_7](https://user-images.githubusercontent.com/86352149/202405080-10bdc270-6684-4cb6-b76d-2be92947dce3.jpg)

Use the masking tool as usual and grab the result back with some color correction.

![alpha5_8](https://user-images.githubusercontent.com/86352149/202405301-97fe33cb-512b-4cd7-a853-ac35124c497a.jpg)




