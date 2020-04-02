C:\Users\jayhy\.conda\envs\AI\python.exe "C:/Users/jayhy/OneDrive/바탕 화면/Home/SSAFY_Project/03_특화프로젝트/sub02/s02p22a405/doc/현동/self_study/imagecaptioning.py"
2020-04-02 19:09:57.212671: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
2020-04-02 19:09:59.514021: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2020-04-02 19:09:59.517460: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-04-02 19:09:59.542895: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1660 SUPER major: 7 minor: 5 memoryClockRate(GHz): 1.845
pciBusID: 0000:07:00.0
2020-04-02 19:09:59.543019: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-04-02 19:09:59.543416: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-04-02 19:10:00.021231: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-02 19:10:00.021320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-04-02 19:10:00.021366: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-04-02 19:10:00.021986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2457 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 SUPER, pci bus id: 0000:07:00.0, compute capability: 7.5)
['../../../datasets/images/4690435409.jpg', '../../../datasets/images/4690435409.jpg', '../../../datasets/images/4690435409.jpg']
['<start> A man in a black shirt walking down a crowded street with a girl in a flower shirt and Capri pants following closely behind him . <end>', '<start> A man and woman are walking in a restaurant that has signs in Chinese . <end>', '<start> A couple walks past lanterns and people eating at a restaurant . <end>']
2020-04-02 19:10:00.251570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1660 SUPER major: 7 minor: 5 memoryClockRate(GHz): 1.845
pciBusID: 0000:07:00.0
2020-04-02 19:10:00.251695: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-04-02 19:10:00.252085: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-04-02 19:10:00.252552: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1660 SUPER major: 7 minor: 5 memoryClockRate(GHz): 1.845
pciBusID: 0000:07:00.0
2020-04-02 19:10:00.252660: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-04-02 19:10:00.253080: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-04-02 19:10:00.253277: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-02 19:10:00.253357: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-04-02 19:10:00.253402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-04-02 19:10:00.253785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2457 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 SUPER, pci bus id: 0000:07:00.0, compute capability: 7.5)
------------image_dataset---------------------
<BatchDataset shapes: ((None, 299, 299, 3), (None,)), types: (tf.float32, tf.string)>
0it [00:00, ?it/s]2020-04-02 19:10:04.185495: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-04-02 19:10:05.171331: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once.
2020-04-02 19:10:05.407449: W tensorflow/core/common_runtime/bfc_allocator.cc:305] Garbage collection: deallocate free memory regions (i.e., allocations) so that we can re-allocate a larger region to avoid OOM due to memory fragmentation. If you see this message frequently, you are running near the threshold of the available device memory and re-allocation may incur great performance overhead. You may try smaller batch sizes to observe the performance impact. Set TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to disable this feature.
2020-04-02 19:10:05.549791: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
2020-04-02 19:10:05.808955: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.08GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:10:05.809128: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.08GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:10:06.393026: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 785.27MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:10:06.393201: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 785.27MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
595it [00:59, 10.64it/s]2020-04-02 19:11:04.088072: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.26GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:11:04.088250: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.26GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:11:04.171773: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.07GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:11:04.172023: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.07GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:11:04.757955: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 781.82MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-04-02 19:11:04.758136: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 781.82MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
596it [01:00,  9.81it/s]
=======================================Create training and validation sets=======================================
38140 38140 9535 9535

2020-04-02 19:12:42.012072: W tensorflow/core/grappler/optimizers/implementation_selector.cc:310] Skipping optimization due to error while loading function libraries: Invalid argument: Functions '__inference___backward_standard_gru_994505_994881' and '__inference___backward_cudnn_gru_with_fallback_929472_929610_specialized_for_StatefulPartitionedCall_60_at___inference_train_step_1001123' both implement 'gru_9b8255c0-9037-4998-b90b-8db689a931aa' but their signatures do not match.
Epoch 8 Batch 0 Loss 1.1248
Epoch 8 Batch 100 Loss 0.9203
Epoch 8 Batch 200 Loss 0.8169
Epoch 8 Batch 300 Loss 0.7849
Epoch 8 Batch 400 Loss 0.8159
Epoch 8 Batch 500 Loss 0.7004
2020-04-02 19:16:18.551167: W tensorflow/core/grappler/optimizers/implementation_selector.cc:310] Skipping optimization due to error while loading function libraries: Invalid argument: Functions '__inference___backward_cudnn_gru_with_fallback_1032781_1032919' and '__inference___backward_cudnn_gru_with_fallback_1032781_1032919_specialized_for_StatefulPartitionedCall_20_at___inference_train_step_1082672' both implement 'gru_4a14058c-9a48-4a12-8200-e94301bcb40f' but their signatures do not match.
Epoch 8 Loss 0.832104
Time taken for 1 epoch 321.9235031604767 sec

Epoch 9 Batch 0 Loss 0.7289
Epoch 9 Batch 100 Loss 0.6875
Epoch 9 Batch 200 Loss 0.6815
Epoch 9 Batch 300 Loss 0.6564
Epoch 9 Batch 400 Loss 0.7198
Epoch 9 Batch 500 Loss 0.7515
Epoch 9 Loss 0.692327
Time taken for 1 epoch 155.79750609397888 sec

Epoch 10 Batch 0 Loss 0.6998
Epoch 10 Batch 100 Loss 0.6222
Epoch 10 Batch 200 Loss 0.6198
Epoch 10 Batch 300 Loss 0.6590
Epoch 10 Batch 400 Loss 0.5700
Epoch 10 Batch 500 Loss 0.6131
Epoch 10 Loss 0.641823
Time taken for 1 epoch 155.97969508171082 sec

Epoch 11 Batch 0 Loss 0.6497
Epoch 11 Batch 100 Loss 0.6072
Epoch 11 Batch 200 Loss 0.5820
Epoch 11 Batch 300 Loss 0.5733
Epoch 11 Batch 400 Loss 0.6441
Epoch 11 Batch 500 Loss 0.6093
Epoch 11 Loss 0.606682
Time taken for 1 epoch 155.76215434074402 sec

Epoch 12 Batch 0 Loss 0.6571
Epoch 12 Batch 100 Loss 0.6328
Epoch 12 Batch 200 Loss 0.5294
Epoch 12 Batch 300 Loss 0.5177
Epoch 12 Batch 400 Loss 0.5362
Epoch 12 Batch 500 Loss 0.6123
Epoch 12 Loss 0.577742
Time taken for 1 epoch 157.80379629135132 sec

Epoch 13 Batch 0 Loss 0.5389
Epoch 13 Batch 100 Loss 0.6243
Epoch 13 Batch 200 Loss 0.5967
Epoch 13 Batch 300 Loss 0.5835
Epoch 13 Batch 400 Loss 0.5182
Epoch 13 Batch 500 Loss 0.5167
Epoch 13 Loss 0.553067
Time taken for 1 epoch 156.53457188606262 sec

Epoch 14 Batch 0 Loss 0.6071
Epoch 14 Batch 100 Loss 0.5384
Epoch 14 Batch 200 Loss 0.5360
Epoch 14 Batch 300 Loss 0.5500
Epoch 14 Batch 400 Loss 0.5296
Epoch 14 Batch 500 Loss 0.4816
Epoch 14 Loss 0.528089
Time taken for 1 epoch 156.15958333015442 sec

Epoch 15 Batch 0 Loss 0.5364
Epoch 15 Batch 100 Loss 0.4916
Epoch 15 Batch 200 Loss 0.4762
Epoch 15 Batch 300 Loss 0.4997
Epoch 15 Batch 400 Loss 0.4244
Epoch 15 Batch 500 Loss 0.4705
Epoch 15 Loss 0.504786
Time taken for 1 epoch 156.12481713294983 sec

Epoch 16 Batch 0 Loss 0.4912
Epoch 16 Batch 100 Loss 0.4753
Epoch 16 Batch 200 Loss 0.4813
Epoch 16 Batch 300 Loss 0.5002
Epoch 16 Batch 400 Loss 0.4113
Epoch 16 Batch 500 Loss 0.4549
Epoch 16 Loss 0.483777
Time taken for 1 epoch 156.24694848060608 sec

Epoch 17 Batch 0 Loss 0.5436
Epoch 17 Batch 100 Loss 0.4596
Epoch 17 Batch 200 Loss 0.4442
Epoch 17 Batch 300 Loss 0.4471
Epoch 17 Batch 400 Loss 0.4021
Epoch 17 Batch 500 Loss 0.4643
Epoch 17 Loss 0.462575
Time taken for 1 epoch 157.44009470939636 sec

Epoch 18 Batch 0 Loss 0.4736
Epoch 18 Batch 100 Loss 0.4462
Epoch 18 Batch 200 Loss 0.4009
Epoch 18 Batch 300 Loss 0.4285
Epoch 18 Batch 400 Loss 0.4261
Epoch 18 Batch 500 Loss 0.4085
Epoch 18 Loss 0.444443
Time taken for 1 epoch 157.38748598098755 sec

Epoch 19 Batch 0 Loss 0.4526
Epoch 19 Batch 100 Loss 0.4180
Epoch 19 Batch 200 Loss 0.4966
Epoch 19 Batch 300 Loss 0.4634
Epoch 19 Batch 400 Loss 0.3865
Epoch 19 Batch 500 Loss 0.3927
Epoch 19 Loss 0.424950
Time taken for 1 epoch 156.18172001838684 sec

Epoch 20 Batch 0 Loss 0.4227
Epoch 20 Batch 100 Loss 0.4512
Epoch 20 Batch 200 Loss 0.3748
Epoch 20 Batch 300 Loss 0.4333
Epoch 20 Batch 400 Loss 0.4159
Epoch 20 Batch 500 Loss 0.3750
Epoch 20 Loss 0.409116
Time taken for 1 epoch 156.49594449996948 sec

Real Caption: <start> sports fans are <unk> <end>
Prediction Caption: four people talking into a street corner is waiting for his booth with his mouth prepares for the camera and blue helmets are awaiting an outdoor fountain on <end>
C:\Program Files\JetBrains\PyCharm 2019.3.3\plugins\python\helpers\pycharm_matplotlib_backend\backend_interagg.py:64: UserWarning: Tight layout not applied. tight_layout cannot make axes width small enough to accommodate all axes decorations
  self.figure.tight_layout()

Process finished with exit code 0

