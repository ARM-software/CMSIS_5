#ifndef _RINGCONFIG_H_
#define _RINGCONFIG_H_ 

// <<< Use Configuration Wizard in Context Menu >>>


// <h>Ring Buffer Configuration
// <o>Number of buffers <2=> 2 <4=> 4 <8=> 8 <16=> 16 <32=> 32  
#ifndef RING_NBBUFS
#define RING_NBBUFS 4
#endif
// </h>

// <<< end of configuration section >>>

#if defined(SDF_AUDIO_CONFIG)
#define RING_BUFSIZE_RX (AUDIO_DRV_NBSAMPLES_RX * AUDIO_DRV_NBCHANNELS_RX * AUDIO_DRV_CHANNEL_ENCODING_RX)
#define RING_BUFSIZE_TX (AUDIO_DRV_NBSAMPLES_TX * AUDIO_DRV_NBCHANNELS_TX * AUDIO_DRV_CHANNEL_ENCODING_TX)
#endif

#if defined(SDF_VIDEO_CONFIG)
#define RING_BUFSIZE_RX (VIDEO_DRV_WIDTH * VIDEO_DRV_HEIGHT * VIDEO_DRV_PIXEL_SIZE)
#define RING_BUFSIZE_TX 0
#endif

#endif
