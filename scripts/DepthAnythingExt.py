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

import pathlib
import types
from typing import Type
import tensorrt as trt
import torch
import numpy as np
import torchvision.transforms as transforms
from TDDepthAnythingAccelerate import TDDepthAnythingAccelerate

class DepthAnythingExt:
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.Logger = self.ownerComp.op('logger')
		self.SafeLogger = self.Logger.Logger
		self.ThreadManager = op.TDResources.ThreadManager

		self.Accelerate = TDDepthAnythingAccelerate(
			width=self.ownerComp.par.Resolutionw.eval(),
			height=self.ownerComp.par.Resolutionh.eval(),
			model_type=self.ownerComp.par.Modeltype.eval(),
			checkpoints_dir=self.ownerComp.par.Checkpointsdir.eval(),
		)

		"""Initialize TensorRT plugins, engine and conetxt."""
		self.trt_path = self.Accelerate.engine_path
		self.device = "cuda"
		self.trt_logger = trt.Logger(trt.Logger.INFO)
		
		self.engine = None
		self.context = None
		self.stream = None
		
		self.source = op('inputImage')
		self.trt_input = None
		self.trt_output = None
		self.expanded_output = None
		self.to_tensor = None
		self.normalize = None
		self.scriptBuffer = op('script1')

	def setupTensor(self):
		self.trt_input = torch.zeros((self.source.height, self.source.width), device=self.device)
		self.trt_output = torch.zeros((self.source.height, self.source.width), device=self.device)
		self.expanded_output = torch.zeros((4, self.source.height, self.source.width), device=self.device, dtype=self.trt_output.dtype)
		self.to_tensor = TopArrayInterface(self.source)
		self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
		return

	def _load_engine(self):
		"""Load TensorRT engine."""
		TRTbin = self.trt_path
		with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
			return runtime.deserialize_cuda_engine(f.read())
	
	def infer(self, img, output):
		"""Run inference on TensorRT engine."""
		self.bindings = [img.data_ptr()] + [output.data_ptr()]

		for i in range(self.engine.num_io_tensors):
			self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

		# Run inference
		self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

		self.stream.synchronize()
			
	def run(self):
		if self.ownerComp.par.Active.eval() and self.engine:
			self.to_tensor.update(self.stream.cuda_stream)
			self.trt_input = torch.as_tensor(self.to_tensor, device=self.device)
			self.trt_input = self.normalize(self.trt_input[1:, :, :]).ravel()

			self.infer(self.trt_input, self.trt_output)
			
			if self.ownerComp.par.Normalize == 'normal':
				tensor_min = self.trt_output.min()
				tensor_max = self.trt_output.max()
				self.trt_output = (self.trt_output - tensor_min) / (tensor_max - tensor_min)

			output = TopCUDAInterface(
				self.source.width,
				self.source.height,				
				1,
				np.float32
			)
			
			self.expanded_output[0, :, :] = self.trt_output

			self.scriptBuffer.copyCUDAMemory(self.expanded_output.data_ptr(), output.size, output.mem_shape)

	"""
	Setup methods
	"""
	def LoadModelThreaded(self):
		myThread = self.ThreadManager.TDTask(
				target=self.Accelerate.load_model,
				SuccessHook=self.LoadModelSuccess,
				ExceptHook=self.LoadModelExcept,
				RefreshHook=self.LoadModelRefresh
			)
		self.ThreadManager.EnqueueTask(myThread)		

		self.Logger.Info(f'Loading model from HuggingFace.')

	def LoadModelSuccess(self):
		self.Logger.Info(f'Model loaded successfully.')
		return
	
	def LoadModelExcept(self, *args: tuple[Type[BaseException], BaseException, types.TracebackType]):
		self.Logger.Error(f'An error occured when trying to load the model from HuggingFace. {args}')
		# Extract the traceback object
		exc_type, exc_value, tb = args[0]
		formatted_tb = traceback.format_tb(tb)
		self.Logger.Error("".join(formatted_tb))		
	
	def LoadModelRefresh(self):
		self.Logger.Info(f'Loading model from HuggingFace.')
		return
	
	def AccelerateModelThreaded(self):
		if self.Accelerate.model:
			myThread = self.ThreadManager.TDTask(
					target=self.Accelerate.accelerate,
					SuccessHook=self.AccelerateModelSuccess,
					ExceptHook=self.AccelerateModelExcept,
					RefreshHook=self.AccelerateModelRefresh
				)
			self.ThreadManager.EnqueueTask(myThread)
			self.Logger.Info(f'Accelerating model.')
		
		else:
			self.Logger.Error('The model was not downloaded or was not loaded to be used for acceleration. Click "Download Model" first.')


	def AccelerateModelSuccess(self):
		self.Logger.Info(f'Model accelerated successfully.')
		return
	
	def AccelerateModelExcept(self, *args: tuple[Type[BaseException], BaseException, types.TracebackType]):
		self.Logger.Error(f'An error occured when trying to accelerate the model. {args}')
		# Extract the traceback object
		exc_type, exc_value, tb = args[0]
		formatted_tb = traceback.format_tb(tb)
		self.Logger.Error("".join(formatted_tb))

	def AccelerateModelRefresh(self):
		self.Logger.Info(f'Accelerating model.')
		return
	
	def UploadModelToGPUThreaded(self):
		myThread = self.ThreadManager.TDTask(
				target=self.UplopadModelToGPU,
				SuccessHook=self.UploadModelToGPUSuccess,
				ExceptHook=self.UploadModelToGPUExcept,
				RefreshHook=self.UploadModelToGPURefresh
		)
		self.ThreadManager.EnqueueTask(myThread)

		self.Logger.Info(f'Uploading model to GPU.')

	def UplopadModelToGPU(self):
		if pathlib.Path(self.trt_path).exists():
			self.engine = self._load_engine()
			self.context = self.engine.create_execution_context()
			self.stream = torch.cuda.current_stream(device=self.device)		
		
	def UploadModelToGPUSuccess(self):
		self.Logger.Info(f'Model uploaded to GPU successfully.')
		if self.ownerComp.par.Active.eval():
			self.setupTensor()
			self.Logger.Info('Active was on, setting up tensors.')
		return

	def UploadModelToGPUExcept(self, *args: tuple[Type[BaseException], BaseException, types.TracebackType]):
		self.Logger.Error(f'An error occured when trying to upload the model to GPU. {args}')
		# Extract the traceback object
		exc_type, exc_value, tb = args[0]
		formatted_tb = traceback.format_tb(tb)
		self.Logger.Error("".join(formatted_tb))

	def UploadModelToGPURefresh(self):
		self.Logger.Info(f'Uploading model to GPU.')
		return

	"""
	Parameters Handlers
	"""
	def OnValueChangeModeltype(self, par, prev):
		"""
		When changing the model type, and if the model is not accelerated yet, we need to:
		- Update the model complete name
		- Update the model path
		- Update the ONNX path
		- Update the engine path

		Args:
			par (_type_): _description_
			prev (_type_): _description_
		"""
		self.Accelerate.model_type = par.eval()
		return
	
	def OnValueChangeResolutionw(self, par, prev):
		"""
		When changing the resolution width, and if the model is not accelerated yet, we need to:
		- Update the image shape
		- Update the ONNX path
		- Update the engine path

		Args:
			par (_type_): _description_
			prev (_type_): _description_
		"""
		self.Accelerate.width = par.eval()
		return
	
	def OnValueChangeResolutionh(self, par, prev):
		"""
		When changing the resolution height, and if the model is not accelerated yet, we need to:
		- Update the image shape
		- Update the ONNX path
		- Update the engine path

		Args:
			par (_type_): _description_
			prev (_type_): _description_
		"""
		self.Accelerate.height = par.eval()
		return
	
	def OnValueChangeCheckpointdir(self, par, prev):
		"""
		When changing the checkpoint directory, and if the model is not accelerated yet, we need to:
		- Update the checkpoint directory
		- Update the ONNX path
		- Update the engine path

		Args:
			par (_type_): _description_
			prev (_type_): _description_
		"""
		self.Accelerate.checkpoints_dir = par.eval()
		return
	
	def OnValueChangeActive(self, par, prev):
		"""
		When changing the active parameter, we need to:
		- Check that the accelerated model was loaded
		- Setup tensor for inference

		Args:
			par (_type_): _description_
			prev (_type_): _description_
		"""
		if par.eval():
			if self.engine is None:
				self.UploadModelToGPUThreaded()
				self.setupTensor()
			else:
				self.setupTensor()

		return
	
	def OnValueChangeNormalize(self, par, prev):
		"""
		When changing the normalize parameter, we need to:
		- None

		Args:
			par (_type_): _description_
			prev (_type_): _description_
		"""
		return
	
	def OnPulseDownloadmodel(self, par):
		"""
		When pressing the download model button, we need to:
		- Download the model from HuggingFace in a a threaded method

		Args:
			par (_type_): _description_
		"""
		self.LoadModelThreaded()
		return
	
	def OnPulseAccelerate(self, par):
		"""
		When pressing the accelerate model button, we need to:
		- Check that the model is not already accelerated
		- Accelerate the model using the TDDepthAnythingAccelerate class, in a threaded method

		Args:
			par (_type_): _description_
		"""
		self.AccelerateModelThreaded()
		return
	
	def OnPulseUploadmodeltogpu(self, par):
		"""
		When pressing the upload model to GPU button, we need to:
		- Upload the accelerated model to GPU in a threaded method

		Args:
			par (_type_): _description_
		"""
		self.UploadModelToGPUThreaded()
		return


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
	def __init__(self, top, stream=0):
		self.top = top
		mem = top.cudaMemory(stream=stream)
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
			"stream": stream,
			"strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
			"data": (mem.ptr, False),
		}

	def update(self, stream=0):
		mem = self.top.cudaMemory(stream=stream)
		self.__cuda_array_interface__['stream'] = stream
		self.__cuda_array_interface__['data'] = (mem.ptr, False)
		return
