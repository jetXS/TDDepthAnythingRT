'''
May 2025 Update:
- Streamlining
- Added support for TDPyEnvManager
- Installation from within TouchDesigner
- Focus on Depth Anything V2
- Removing Depth Anything repo dependency, switched to HuggingFace
- Original code by Oleg Chomp, Keith Lostracco
- Update by JetXS (Michel Didier)

License for https://github.com/IntentDev/TopArray

MIT License

Copyright (c) 2024 Keith Lostracco

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from typing import Optional, Any
import gc
import pathlib
import threading
from typing import Type
import numpy as np
import logging

logger = logging.getLogger('TDAppLogger')
try:
	import tensorrt as trt
	import torch
	import torchvision.transforms as transforms
except ImportError as e:
	logger.error(f'TDDepthAnythingHF - An error occured trying to import some of the required libraries. Make sure that the environment is setup properly.')
	logger.error(f'TDDepthAnythingHF - {e}')
	logger.error(f'TDDepthAnythingHF - If you are using a custom python environment, make sure that the following packages are installed: tensorrt, torch, torchvision')
except Exception as e:
	logger.error(f'TDDepthAnythingHF - An error occured trying to import some of the required libraries. Make sure that the environment is setup properly.')
	logger.error(f'TDDepthAnythingHF - {e}')
	
from TDDepthAnythingHFAccelerate import TDDepthAnythingHFAccelerate

class TDDepthAnythingHFExt:
	"""
	Main class for managing TensorRT-based depth estimation models in TouchDesigner.
	"""

	def __init__(self, ownerComp: Any) -> None:
		"""
		Initialize the extension with the owner component.

		Args:
			ownerComp (Any): The TouchDesigner component that owns this extension.
		"""
		self.ownerComp = ownerComp
		self.ownerComp.par.Modelstatus = 'None'
		self.Logger = self.ownerComp.op('logger')
		self.SafeLogger = self.Logger.Logger
		self.ThreadManager = op.TDResources.ThreadManager

		self.Accelerate = TDDepthAnythingHFAccelerate(
			width=self.ownerComp.par.Resolutionw.eval(),
			height=self.ownerComp.par.Resolutionh.eval(),
			model_type=self.ownerComp.par.Modeltype.eval(),
			checkpoints_dir=self.ownerComp.par.Checkpointsdir.eval(),
		)

		# TensorRT-related attributes
		self.trt_path: str = self.Accelerate.engine_path
		self.device: str = "cuda"
		self.EngineLock: threading.Lock = threading.Lock()
		self.engine: Optional[trt.ICudaEngine] = None
		self.context: Optional[trt.IExecutionContext] = None
		self.stream: Optional[torch.cuda.Stream] = None

		# Input/output and processing attributes
		self.source = op('inputImage')
		self.trt_input: Optional[torch.Tensor] = None
		self.trt_output: Optional[torch.Tensor] = None
		self.to_tensor: Optional[Any] = None
		self.normalize: Optional[transforms.Normalize] = None
		self.scriptBuffer = op('script1')

	def onInitTD(self) -> None:
		"""
		Initialize the TouchDesigner environment by populating the script buffer with random data.
		"""
		self.scriptBuffer.copyNumpyArray(
			np.random.randint(
				0,
				high=255,
				size=(
					int(self.ownerComp.par.Resolutionh.eval()),
					int(self.ownerComp.par.Resolutionw.eval()),
					4
				),
				dtype=np.uint16
			)
		)

	def onDelTD(self) -> None:
		"""
		Clean up resources when the extension is deleted.
		"""
		self.UnloadModelFromGPU()
		self.trt_input = None
		self.trt_output = None
		self.to_tensor = None
		self.normalize = None
		self.Accelerate.free_acc_mem()

	def _load_engine(self) -> Optional[trt.ICudaEngine]:
		"""
		Load the TensorRT engine from the specified path.

		Returns:
			Optional[trt.ICudaEngine]: The loaded TensorRT engine, or None if loading fails.
		"""
		try:
			TRTbin = self.trt_path
			with open(TRTbin, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
				return runtime.deserialize_cuda_engine(f.read())
		except FileNotFoundError:
			self.SafeLogger.error(f"TensorRT engine file not found: {TRTbin}")
		except Exception as e:
			self.SafeLogger.error(f"Failed to load TensorRT engine: {e}\n{traceback.format_exc()}")
		return None

	def setupTensor(self) -> None:
		"""
		Set up input and output tensors for inference.
		"""
		with self.stream:
			self.trt_input = torch.zeros((self.source.height, self.source.width), device=self.device)
			self.trt_output = torch.zeros((self.source.height, self.source.width), device=self.device)
			self.to_tensor = TopArrayInterface(self.source, self.stream.cuda_stream)
			self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

	def run(self) -> None:
		"""
		Run the inference pipeline if the model is active and loaded.
		"""
		with self.EngineLock:
			if self.ownerComp.par.Active.eval() and self.engine and hasattr(self.stream, 'cuda_stream') and self.to_tensor:
				self._prepare_input_tensor()
				self._run_inference(self.trt_input, self.trt_output)
				self._process_output_tensor()

	def _prepare_input_tensor(self) -> None:
		"""
		Prepare the input tensor for inference by normalizing and reshaping it.
		"""
		with self.stream:
			self.to_tensor.update(self.stream.cuda_stream)
			self.trt_input = torch.as_tensor(self.to_tensor, device=self.device)
			self.trt_input = self.normalize(self.trt_input[1:, :, :]).ravel()

	def _run_inference(self, img: torch.Tensor, output: torch.Tensor) -> None:
		"""
		Run inference on the input tensor.

		Args:
			img (torch.Tensor): The input tensor.
			output (torch.Tensor): The output tensor to store inference results.
		"""
		self.bindings = [img.data_ptr()] + [output.data_ptr()]
		for i in range(self.engine.num_io_tensors):
			self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

		# Run inference
		with self.stream:
			self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

	def _process_output_tensor(self) -> None:
		"""
		Process the output tensor and convert it to RGBA format.
		"""
		with self.stream:
			if self.ownerComp.par.Normalize == 'normal':
				tensor_min = self.trt_output.min()
				tensor_max = self.trt_output.max()
				self.trt_output = (self.trt_output - tensor_min) / (tensor_max - tensor_min)

			rgba_tensor = torch.zeros((4, self.trt_output.shape[0], self.trt_output.shape[1]), device=self.device, dtype=self.trt_output.dtype)
			rgba_tensor[0, :, :] = self.trt_output


			output = TopCUDAInterface(
				self.source.width,
				self.source.height,
				4,
				np.float32
			)
			self.scriptBuffer.copyCUDAMemory(rgba_tensor.permute(1, 2, 0).contiguous().data_ptr(), output.size, output.mem_shape, stream=self.stream.cuda_stream)

	
	"""
	Setup methods
	"""
	def LoadModelThreaded(self) -> None:
		"""
		Start a threaded task to load the model from HuggingFace.

		This method enqueues a task to download and load the model in a separate thread.
		"""
		myThread = self.ThreadManager.TDTask(
			target=self.Accelerate.load_model,
			SuccessHook=self.LoadModelSuccess,
			RefreshHook=self.LoadModelRefresh
		)
		self.ThreadManager.EnqueueTask(myThread)
		self.Logger.Info('Loading model from HuggingFace.')

	def LoadModelSuccess(self) -> None:
		"""
		Callback executed when the model is successfully loaded.

		Updates the logger to indicate success.
		"""
		self.Logger.Info('Model loaded successfully.')
	
	def LoadModelRefresh(self) -> None:
		"""
		Refresh callback executed during the model loading process.

		Updates the logger to indicate progress.
		"""
		self.Logger.Info('Loading model from HuggingFace.')
	
	def AccelerateModelThreaded(self) -> None:
		"""
		Start a threaded task to accelerate the model.

		This method enqueues a task to accelerate the model using TensorRT in a separate thread.
		"""
		if self.Accelerate.model:
			myThread = self.ThreadManager.TDTask(
				target=self.Accelerate.accelerate,
				SuccessHook=self.AccelerateModelSuccess,
				RefreshHook=self.AccelerateModelRefresh
			)
			self.ThreadManager.EnqueueTask(myThread)
			self.Logger.Info('Accelerating model.')
		else:
			self.Logger.Error('The model was not downloaded or loaded. Click "Download Model" first.')


	def AccelerateModelSuccess(self) -> None:
		"""
		Callback executed when the model is successfully accelerated.

		Updates the logger to indicate success.
		"""
		self.Logger.Info('Model accelerated successfully.')

	def AccelerateModelRefresh(self) -> None:
		"""
		Refresh callback executed during the model acceleration process.

		Updates the logger to indicate progress.
		"""
		self.Logger.Debug('Accelerating model.')
	
	def UploadModelToGPUThreaded(self) -> None:
		"""
		Start a threaded task to upload the accelerated model to the GPU.

		This method enqueues a task to load the TensorRT engine and upload it to the GPU in a separate thread.
		"""
		myThread = self.ThreadManager.TDTask(
			target=self.UploadModelToGPU,
			SuccessHook=self.UploadModelToGPUSuccess,
			RefreshHook=self.UploadModelToGPURefresh
		)
		self.ThreadManager.EnqueueTask(myThread)
		self.Logger.Info('Uploading model to GPU.')

	def UploadModelToGPU(self) -> None:
		"""
		Upload the accelerated model to the GPU.

		Loads the TensorRT engine from the specified path and creates the execution context and CUDA stream.
		"""
		with self.EngineLock:
			if pathlib.Path(self.trt_path).exists():
				self.SafeLogger.info(f"Loading TensorRT engine from: {self.trt_path}")
				self.engine = self._load_engine()
				if self.engine:
					self.context = self.engine.create_execution_context()
					self.stream = torch.cuda.Stream(device=self.device)
					self.SafeLogger.info("TensorRT engine loaded successfully.")
				else:
					self.SafeLogger.error("Failed to load TensorRT engine.")
			else:
				self.SafeLogger.error(f"TensorRT engine file does not exist: {self.trt_path}")
		
	def UploadModelToGPUSuccess(self) -> None:
		"""
		Callback executed when the model is successfully uploaded to the GPU.

		Updates the logger and sets up tensors if the model is active.
		"""
		self.Logger.Info('Model uploaded to GPU successfully.')
		if self.ownerComp.par.Active.eval():
			self.setupTensor()
			self.Logger.Info('Active was on, setting up tensors.')
		self.ownerComp.par.Modelstatus = f'{pathlib.Path(self.trt_path).name} loaded.'

	def UploadModelToGPURefresh(self) -> None:
		"""
		Refresh callback executed during the model upload process.

		Updates the logger to indicate progress.
		"""
		self.Logger.Info('Uploading model to GPU.')

	def UnloadModelFromGPUThreaded(self) -> None:
		"""
		Start a threaded task to unload the model from the GPU.

		This method enqueues a task to release the TensorRT engine and associated resources in a separate thread.
		"""
		if self.ownerComp.par.Active.eval():
			self.ownerComp.par.Active = False
			self.Logger.Info('Inference was turned off because the user requested to unload the model.')

		myThread = self.ThreadManager.TDTask(
			target=self.UnloadModelFromGPU,
			SuccessHook=self.UnloadModelFromGPUSuccess,
			RefreshHook=self.UnloadModelFromGPURefresh
		)
		self.ThreadManager.EnqueueTask(myThread)
		self.Logger.Info('Unloading model from GPU.')
	
	def UnloadModelFromGPU(self) -> None:
		"""
		Unload the TensorRT model from the GPU and release associated resources.
		"""
		with self.EngineLock:
			self.engine = None
			self.context = None
			if self.stream:
				self.stream.synchronize()
			self.stream = None
			self.Accelerate.free_acc_mem()
	
	def UnloadModelFromGPUSuccess(self) -> None:
		"""
		Callback for when the model is successfully unloaded from the GPU.
		"""
		self.ownerComp.par.Modelstatus = 'None'
		self.Logger.Info('Accelerated model was unloaded from GPU.')

	def UnloadModelFromGPURefresh(self) -> None:
		"""
		Refresh callback for unloading the model from the GPU.
		"""
		self.Logger.Info('Unloading model from GPU.')

	def Reset(self) -> None:
		"""
		Reset the extension by clearing all resources, tensors, and GPU memory.
		"""
		with self.EngineLock:
			self.engine = None
			self.context = None
			if self.stream:
				self.stream.synchronize()
			self.stream = None

		self.trt_input = None
		self.trt_output = None
		self.to_tensor = None
		self.normalize = None

		self.ownerComp.par.Active = False
		self.ownerComp.par.Modelstatus = 'None'

		self.scriptBuffer.copyNumpyArray(
			np.random.randint(
				0,
				high=255,
				size=(
					int(self.ownerComp.par.Resolutionh.eval()),
					int(self.ownerComp.par.Resolutionw.eval()),
					4
				),
				dtype=np.uint16
			)
		)

		gc.collect()
		torch.cuda.empty_cache()

	"""
	Parameters Handlers
	"""
	def OnValueChangeModeltype(self, par: Any, prev: Any) -> None:
		"""
		Handle changes to the model type parameter.

		Args:
			par (Any): The parameter that changed.
			prev (Any): The previous value of the parameter.
		"""
		self.Accelerate.model_type = par.eval()
	
	def OnValueChangeResolutionw(self, par: Any, prev: Any) -> None:
		"""
		Handle changes to the resolution width parameter.

		Args:
			par (Any): The parameter that changed.
			prev (Any): The previous value of the parameter.
		"""
		self.Accelerate.width = par.eval()
		
	def OnValueChangeResolutionh(self, par: Any, prev: Any) -> None:
		"""
		Handle changes to the resolution height parameter.

		Args:
			par (Any): The parameter that changed.
			prev (Any): The previous value of the parameter.
		"""
		self.Accelerate.height = par.eval()
	
	def OnValueChangeCheckpointdir(self, par: Any, prev: Any) -> None:
		"""
		Handle changes to the checkpoint directory parameter.

		Args:
			par (Any): The parameter that changed.
			prev (Any): The previous value of the parameter.
		"""
		self.Accelerate.checkpoints_dir = par.eval()
		
	def OnValueChangeActive(self, par: Any, prev: Any) -> None:
		"""
		Handle changes to the active parameter.

		Args:
			par (Any): The parameter that changed.
			prev (Any): The previous value of the parameter.
		"""
		if par.eval():
			if self.engine is None:
				self.UploadModelToGPUThreaded()
			else:
				self.setupTensor()
		else:
			self.trt_input = None
			self.trt_output = None
			self.to_tensor = None
			self.normalize = None
	
	def OnValueChangeNormalize(self, par: Any, prev: Any) -> None:
		"""
		Handle changes to the normalize parameter.

		Args:
			par (Any): The parameter that changed.
			prev (Any): The previous value of the parameter.
		"""
		pass
	
	def OnPulseDownloadmodel(self, par: Any) -> None:
		"""
		Handle the pulse to download the model.

		Args:
			par (Any): The parameter that triggered the pulse.
		"""
		self.LoadModelThreaded()
	
	def OnPulseAccelerate(self, par: Any) -> None:
		"""
		Handle the pulse to accelerate the model.

		Args:
			par (Any): The parameter that triggered the pulse.
		"""
		self.AccelerateModelThreaded()
	
	def OnPulseUploadmodeltogpu(self, par: Any) -> None:
		"""
		Handle the pulse to upload the model to the GPU.

		Args:
			par (Any): The parameter that triggered the pulse.
		"""
		self.UploadModelToGPUThreaded()

	def OnPulseUnloadmodelfromgpu(self, par: Any) -> None:
		"""
		Handle the pulse to unload the model from the GPU.

		Args:
			par (Any): The parameter that triggered the pulse.
		"""
		self.UnloadModelFromGPUThreaded()

	def OnPulseReset(self, par: Any) -> None:
		"""
		Handle the pulse to reset the extension.

		Args:
			par (Any): The parameter that triggered the pulse.
		"""
		self.Reset()

"""
TopArray from IntentDev - License for https://github.com/IntentDev/TopArray
"""
class TopCUDAInterface:
	def __init__(self, width, height, num_comps, dtype):
		self.mem_shape = CUDAMemoryShape()
		self.mem_shape.width = width
		self.mem_shape.height = height
		self.mem_shape.numComps = num_comps
		self.mem_shape.dataType = dtype
		self.bytes_per_comp = np.dtype(dtype).itemsize
		self.size = width * height * num_comps * self.bytes_per_comp

class TopArrayInterface:
	def __init__(self, top, streamHandleID):
		self.top = top
		mem = top.cudaMemory(stream=streamHandleID)
		self.w, self.h = mem.shape.width, mem.shape.height
		self.num_comps = mem.shape.numComps
		self.dtype = mem.shape.dataType
		shape = (mem.shape.numComps, self.h, self.w)
		dtype_info = {'descr': [('', '<f4')], 'num_bytes': 4}
		dtype_descr = dtype_info['descr']
		num_bytes = dtype_info['num_bytes']
		num_bytes_px = num_bytes * mem.shape.numComps
		
		self.__cuda_array_interface__ = {
			"version": 3,
			"shape": shape,
			"typestr": dtype_descr[0][1],
			"descr": dtype_descr,
			"stream": streamHandleID,
			"strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
			"data": (mem.ptr, False),
		}

	def update(self, streamHandleID):
		mem = self.top.cudaMemory(stream=streamHandleID)
		self.__cuda_array_interface__['stream'] = streamHandleID
		self.__cuda_array_interface__['data'] = (mem.ptr, False)
		return
