[comment]: <> (Some notes)

1. cuda codes should be compiles based on server architecture. In short, for example, ` -code=sm_xx` , xx should match the server GPU architecture.Per: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

2. when compile your own cuda codes, should always add `-cudart shared` to dynamic link cuda runtime lib. Otherwise, cudaruntime lib is linked statically. Not sure for other libs (cudnn, cublas...), but I guess should follow the same pattern.

3. to test cudnn, e.g. run `REMOTE_GPU_ADDRESS=127.0.0.1 LD_PRELOAD=/home/hwhiaiuser/cchen/cricket/bin/cricket-client.so ~/cchen/experi/cricket/tests/samples/cudnn-samples/mnistCUDNN/mnistCUDNN`

4. if use torch, env LD_LIBRARY_PATH should include path of `libnvfuser_codegen.so` in client side (created by torch, locate in `python3.10/dist-packages/torch/lib`), maybe PYTORCH_JIT_ENABLE_NVFUSER. 

readelf -S /usr/local/lib/python3.10/dist-packages/torch/_C.cpython-310-x86_64-linux-gnu.so 

5. debug compile command: LOG=DEBUG WITH_DEBUG=1 (CKPT=1) make --debug

6. everytime recompile, need to delete ./bin dir

7. export CRICKET_RESTORE=1 to restore.

8. dir `cricket/tests/test_ckp` provide checkpoint test case, SIGUSR1 make server checkpoint, SIGUSR2 make server restore. 
