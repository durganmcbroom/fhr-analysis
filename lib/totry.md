 - Try foundation model for UNet
   - Use this foundation model and then do a linear layer projection at the end to produce the output (try 1, and then 3)
   - Or try to retrain the whole model from scratch
   - https://huggingface.co/docs/transformers/en/model_doc/timesfm
 - Try UNet downsample with raw waveform
   - Check after downsampling and see if the waveform looks reasonable
 - Resnet
   - https://huggingface.co/docs/transformers/en/model_doc/resnet
 - More layers not better on MLP


TODO:

 - Diagnostics on PT13+14 for FUNet
 - Diagnostics for SSNet on 13+14+12
 - PansNet w/ small MLP