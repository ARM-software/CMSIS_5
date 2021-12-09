#ifndef _VIDEOCONFIG_H_
#define _VIDEOCONFIG_H_ 

// <<< Use Configuration Wizard in Context Menu >>>

// <h>Video Configuration
// <o>Width in pixels <16-640>
#ifndef VIDEO_DRV_WIDTH
#define VIDEO_DRV_WIDTH 32
#endif

// <o>Height in pixels <16-640>
#ifndef VIDEO_DRV_HEIGHT
#define VIDEO_DRV_HEIGHT 32
#endif

// <o>Pixel size in bytes <1=> 1 <2=> 2
#ifndef VIDEO_DRV_PIXEL_SIZE
#define VIDEO_DRV_PIXEL_SIZE 1
#endif

// <o>Frame rate <10=> 10 <25=> 25 <30=> 30 <60=> 60
#ifndef VIDEO_DRV_FRAME_RATE
#define VIDEO_DRV_FRAME_RATE 10
#endif

// </h>



// <<< end of configuration section >>>

#define SDF_VIDEO_CONFIG 

#endif
