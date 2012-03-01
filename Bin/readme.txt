Face Detect 
Adding new content:

folder hierarchy
`-- masks
    |-- backgrounds -->  images or videos that will replace background
    |   |-- images
    |   `-- videos
    |-- faces	    -->  mask to cover faces
    `-- faces_alpha -->  alpha channels of masks


Only 8-bit depth images are supported.
When adding faces, you must extract alpha channel:

convert <img_in.png> -channel Alpha -negate -separate <img_out.png>