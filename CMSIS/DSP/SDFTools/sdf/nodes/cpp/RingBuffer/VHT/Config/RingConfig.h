#ifndef _RINGCONFIG_H_
#define _RINGCONFIG_H_ 

// <<< Use Configuration Wizard in Context Menu >>>

// <h>Audio Configuration
// <o>Sampling Frequency <8000=>   8000 kHz  <16000=>   16000 kHz
//                     <44100=>   44100 kHz  <48000=>   48000 kHz
#ifndef AUDIO_SAMPLINGFREQUENCY
#define AUDIO_SAMPLINGFREQUENCY 16000
#endif

// <o>Number of samples <256=> 256 <512=> 512 <1024=> 1024 <2048=> 2048  
// <i> Must be consistent with the settings of the Audio source
#ifndef AUDIO_NBSAMPLES
#define AUDIO_NBSAMPLES 2048
#endif

// <o>Number of channels <1=>   Mono <2=>   Stereo
#ifndef AUDIO_NBCHANNELS
#define AUDIO_NBCHANNELS 1U
#endif

// <o>Channel encoding <2=>   16 Bits
#ifndef AUDIO_CHANNEL_ENCODING
#define AUDIO_CHANNEL_ENCODING 2U
#endif

// <q> RX_ENABLED: Enable RX 
#define RX_ENABLED 1

// <q> TX_ENABLED: Enable TX 
#define TX_ENABLED 1

// <q> SDF_VHT_TX_RX_ORDERING: Force TX RX ordering
#define SDF_VHT_TX_RX_ORDERING 0

// </h>

// <h>Ring Buffer Configuration
// <o>Number of buffers <2=> 2 <4=> 4 <8=> 8 <16=> 16 <32=> 32  
#ifndef RING_NBBUFS
#define RING_NBBUFS 4
#endif
// </h>

// <<< end of configuration section >>>

#define RING_BUFSIZE (AUDIO_NBSAMPLES * AUDIO_NBCHANNELS * AUDIO_CHANNEL_ENCODING)

#endif
