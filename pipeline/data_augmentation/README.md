This is a README for project Data Augmentation

There are 7 mandatory tasks in this project as follows

## Task 0. Flip
Write a function  `def flip_image(image):`  that flips an image horizontally:

-   `image`  is a 3D  `tf.Tensor`  containing the image to flip
-   Returns the flipped image

## Task 1. Crop
Write a function  `def crop_image(image, size):`  that performs a random crop of an image:

-   `image`  is a 3D  `tf.Tensor`  containing the image to crop
-   `size`  is a tuple containing the size of the crop
-   Returns the cropped image

## Task 2. Rotate
Write a function  `def rotate_image(image):`  that rotates an image by 90 degrees counter-clockwise:

-   `image`  is a 3D  `tf.Tensor`  containing the image to rotate
-   Returns the rotated image

## Task 3. Shear
Write a function  `def shear_image(image, intensity):`  that randomly shears an image:

-   `image`  is a 3D  `tf.Tensor`  containing the image to shear
-   `intensity`  is the intensity with which the image should be sheared
-   Returns the sheared image

## Task 4. Brightness
Write a function  `def change_brightness(image, max_delta):`  that randomly changes the brightness of an image:

-   `image`  is a 3D  `tf.Tensor`  containing the image to change
-   `max_delta`  is the maximum amount the image should be brightened (or darkened)
-   Returns the altered image

## Task 5. Hue
Write a function  `def change_hue(image, delta):`  that changes the hue of an image:

-   `image`  is a 3D  `tf.Tensor`  containing the image to change
-   `delta`  is the amount the hue should change
-   Returns the altered image

## Task 6. Automation
Write a blog post describing step by step how to perform automated data augmentation. Try to explain every step you know of, and give examples. A total beginner should understand what you have written.

-   Have at least one picture, at the top of the blog post
-   Publish your blog post on Medium or LinkedIn
-   Share your blog post at least on LinkedIn
-   Write professionally and intelligibly
-   Please, remember that these blogs must be written in English to further your technical ability in a variety of settings

## Task 7. PCA Color Augmentation (Advanced)
Write a function  `def pca_color(image, alphas):`  that performs PCA color augmentation as described in the  [AlexNet](https://intranet.hbtn.io/rltoken/_NDxb8XjIX-JgmBCCWvYHQ "AlexNet")  paper:

-   `image`  is a 3D  `tf.Tensor`  containing the image to change
-   `alphas`  a tuple of length 3 containing the amount that each channel should change
-   Returns the augmented image
