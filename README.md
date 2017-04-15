# crystal-bath
Video Steganography/Steganalysis for high compressed video files in C++ using DCT and Pixel Value Differencing
Hides any binary data inside of a video file.
Features functionality for basic error checking
Performs statistical detection using K-means

I'm in the process of refactoring this in C++ since there will probably be an upcoming lack of support for C# or Delphi for Linux/BSD, 
(and those are primarily Windows Languages)

TODO:

Write a tutorial with screenshots
I also need to make it more "user friendly".

Requirements:

Requires ffmpeg to be in the same directory as the executable.
The video input format has to be in raw YUV.

Once video steganography is performed, you can re-convert the video to a secondary format MPEG-4, H.264/265, WebM, even upload to YouTube and recover the hidden file with no error.

Email network (dot) succubus (at) gmail (dot) com
