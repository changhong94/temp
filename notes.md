[comment]: <> (Some notes)

1. cuda codes should be compiles based on server architecture. In short, for example, ` -code=sm_xx` , xx should match the server GPU architecture.Per: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

2. when compile your own cuda codes, should always add `-cudart shared` to dynamic link cuda runtime lib. Otherwise, cudaruntime lib is linked statically. Not sure for other libs (cudnn, cublas...), but I guess should follow the same pattern.
