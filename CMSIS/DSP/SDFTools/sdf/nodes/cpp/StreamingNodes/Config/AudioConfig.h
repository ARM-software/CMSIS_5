#ifndef _AUDIOCONFIG_H_
#define _AUDIOCONFIG_H_ 

// <<< Use Configuration Wizard in Context Menu >>>

// <e>Audio Configuration for RX
#ifndef AUDIO_DRV_RX_ENABLED
#define AUDIO_DRV_RX_ENABLED 1
#endif
// <o>Sampling Frequency <8000=>   8000 kHz  <16000=>   16000 kHz
//                     <44100=>   44100 kHz  <48000=>   48000 kHz
#ifndef AUDIO_DRV_SAMPLINGFREQUENCY_RX
#define AUDIO_DRV_SAMPLINGFREQUENCY_RX 16000
#endif

// <o>Number of samples <256=> 256 <512=> 512 <1024=> 1024 <2048=> 2048  
// <i> Must be consistent with the settings of the Audio source
#ifndef AUDIO_DRV_NBSAMPLES_RX
#define AUDIO_DRV_NBSAMPLES_RX 2048
#endif

// <o>Number of channels <1=>   Mono <2=>   Stereo
#ifndef AUDIO_DRV_NBCHANNELS_RX
#define AUDIO_DRV_NBCHANNELS_RX 1U
#endif

// <o>Channel encoding <2=>   16 Bits
#ifndef AUDIO_DRV_CHANNEL_ENCODING_RX
#define AUDIO_DRV_CHANNEL_ENCODING_RX 2U
#endif

// </e>

// <e>Audio Configuration for TX
#ifndef AUDIO_DRV_TX_ENABLED
#define AUDIO_DRV_TX_ENABLED 1
#endif
// <o>Sampling Frequency <8000=>   8000 kHz  <16000=>   16000 kHz
//                     <44100=>   44100 kHz  <48000=>   48000 kHz
#ifndef AUDIO_DRV_SAMPLINGFREQUENCY_TX
#define AUDIO_DRV_SAMPLINGFREQUENCY_TX 16000
#endif

// <o>Number of samples <256=> 256 <512=> 512 <1024=> 1024 <2048=> 2048  
// <i> Must be consistent with the settings of the Audio source
#ifndef AUDIO_DRV_NBSAMPLES_TX
#define AUDIO_DRV_NBSAMPLES_TX 2048
#endif

// <o>Number of channels <1=>   Mono <2=>   Stereo
#ifndef AUDIO_DRV_NBCHANNELS_TX
#define AUDIO_DRV_NBCHANNELS_TX 1U
#endif

// <o>Channel encoding <2=>   16 Bits
#ifndef AUDIO_DRV_CHANNEL_ENCODING_TX
#define AUDIO_DRV_CHANNEL_ENCODING_TX 2U
#endif

// </e>

// <q> SDF_VHT_TX_RX_ORDERING: Force TX RX ordering
#define SDF_VHT_TX_RX_ORDERING 0

// <<< end of configuration section >>>

#define SDF_AUDIO_CONFIG 

#endif
