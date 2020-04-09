ERROR

1. OOM 

	드디어 만난 OOM...

	```
	ResourceExhaustedError: OOM when allocating tensor with shape[10,17,17,192] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:Relu]
	```
	
	[참고](https://datamasters.co.kr/33)
	```python
	# 지금까지 생성된 tensor 그래프를 제거
	tf.compat.v1.reset_default_graph()
	```

2. 

   ```
   WARNING:tensorflow:Entity <function load_image at 0x000001408CF66F78> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
   ```

    [참고](https://github.com/tensorflow/autograph/issues/3)
   ```python
   tf.autograph.experimental.do_not_convert()
   ```


3. 

   ```
   WARNING: Entity <bound method BahdanauAttention.call of <models.decoder.BahdanauAttention object at 0x000001AF31442E88>> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
   ```

   [참고](https://www.gitmemory.com/issue/tensorflow/autograph/2/561403120) [conda install](https://anaconda.org/conda-forge/gast)

   ```
   conda install -c conda-forge gast
   ```

   