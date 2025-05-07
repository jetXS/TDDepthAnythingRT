import gc
import os
import pathlib
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import torch.onnx
import tensorrt as trt
from polygraphy.backend.trt import (
	CreateConfig,
	Profile,
	engine_from_network,
	network_from_onnx_path,
	save_engine,
)
torch.hub.set_dir('torchhub')

class TDDepthAnythingAccelerate:
	def __init__(self, width=518, height=518, model_name='Depth-Anything-V2', model_type='Base', checkpoints_dir=f'{os.getcwd()}/checkpoints'):
		self._width = self.adjust_image_size(width)
		self._height = self.adjust_image_size(height)
		self.image_shape = (3, self.height, self.width)
		self.model_name = model_name
		self._model_type = model_type
		self.model_complete_name = f'{self.model_name}-{self.model_type}-hf'
		self.model_path = f'depth-anything/{self.model_complete_name}'
		self.model = None
		self._checkpoints_dir = checkpoints_dir
		self.onnx_path = f"{self.checkpoints_dir}/onnx_models/{self.get_output_name()}.onnx"
		self.engine_path = f"{self.checkpoints_dir}/engines/{self.get_output_name()}.engine"
		self.ensure_directories_exist()

	def ensure_directories_exist(self):
		"""
		Ensure that the required directories for ONNX models and TensorRT engines exist.
		"""
		os.makedirs(f"{self.checkpoints_dir}/onnx_models", exist_ok=True)
		os.makedirs(f"{self.checkpoints_dir}/engines", exist_ok=True)

	@property
	def model_type(self):
		return self._model_type

	@model_type.setter
	def model_type(self, value):
		self._model_type = value
		self.model_complete_name = f'{self.model_name}-{self.model_type}'
		self.model_path = f'depth-anything/{self.model_complete_name}'
		self.onnx_path = f"{self.checkpoints_dir}/onnx_models/{self.get_output_name()}.onnx"
		self.engine_path = f"{self.checkpoints_dir}/engines/{self.get_output_name()}.engine"

	@property
	def width(self):
		return self._width
	
	@width.setter
	def width(self, value):
		self._width = self.adjust_image_size(value)
		self.image_shape = (3, self.height, self.width)
		self.onnx_path = f"{self.checkpoints_dir}/onnx_models/{self.get_output_name()}.onnx"
		self.engine_path = f"{self.checkpoints_dir}/engines/{self.get_output_name()}.engine"

	@property
	def height(self):
		return self._height
	
	@height.setter
	def height(self, value):
		self._height = self.adjust_image_size(value)
		self.image_shape = (3, self.height, self.width)
		self.onnx_path = f"{self.checkpoints_dir}/onnx_models/{self.get_output_name()}.onnx"
		self.engine_path = f"{self.checkpoints_dir}/engines/{self.get_output_name()}.engine"

	@property
	def checkpoints_dir(self):
		return self._checkpoints_dir
	
	@checkpoints_dir.setter
	def checkpoints_dir(self, value):
		self._checkpoints_dir = value
		self.onnx_path = f"{self.checkpoints_dir}/onnx_models/{self.get_output_name()}.onnx"
		self.engine_path = f"{self.checkpoints_dir}/engines/{self.get_output_name()}.engine"	
		self.ensure_directories_exist()

	def adjust_image_size(self, image_size):
		patch_size = 14
		adjusted_size = (image_size // patch_size) * patch_size
		if image_size % patch_size != 0:
			adjusted_size += patch_size
		return int(adjusted_size)

	def get_output_name(self):
		return f"{self.model_name}_{self.width}x{self.height}"

	def load_model(self):
		"""
		Load the model using the specified model name and path.
		"""
		print(f"Loading model: {self.model_name} from {self.model_path}")
		self.model = AutoModelForDepthEstimation.from_pretrained(self.model_path, cache_dir=self.checkpoints_dir)
		print(f"Model loaded successfully.")

	def unload_model(self):
		self.model = None
		gc.collect()
		torch.cuda.empty_cache()

	def accelerate(self, model=None):
		print(f'Image shape is {self.width}x{self.height}')
		if model is None:
			model = self.model if self.model else self.load_model()

		model.eval()

		# Define dummy input data
		dummy_input = torch.ones(self.image_shape).unsqueeze(0)

		if not pathlib.Path(self.onnx_path).exists():
		# Export the PyTorch model to ONNX format
			if not os.path.exists(os.path.dirname(self.onnx_path)):
				os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
			
			torch.onnx.export(
					model, 
					dummy_input, 
					self.onnx_path, 
					opset_version=14, 
					input_names=["input"], 
					output_names=["output"], 
					verbose=True
				)
			print(f"Model exported to {self.onnx_path}")

			# Clear memory after ONNX export
			del dummy_input
			gc.collect()
			torch.cuda.empty_cache()

		# Build TensorRT engine

		if not pathlib.Path(self.engine_path).exists():
			if not os.path.exists(os.path.dirname(self.engine_path)):
				os.makedirs(os.path.dirname(self.engine_path), exist_ok=True)
			
			print(f"Building TensorRT engine for {self.onnx_path}: {self.engine_path}")
			
			p = Profile()
			config_kwargs = {}

			engine = engine_from_network(
				network_from_onnx_path(self.onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
				config=CreateConfig(
					fp16=True, refittable=False, profiles=[p], load_timing_cache=None, **config_kwargs
				),
				save_timing_cache=None,
			)
			save_engine(engine, path=self.engine_path)

			# Clear memory after TensorRT engine creation
			del engine
			gc.collect()
			torch.cuda.empty_cache()

		print(f"Finished building TensorRT engine: {self.engine_path}")