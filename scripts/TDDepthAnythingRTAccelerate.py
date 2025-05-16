import gc
import os
import pathlib
import logging
import traceback
logger = logging.getLogger('TDAppLogger')

try:
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

except Exception as e:
	logger.error(f'TDDepthAnythingRT - An error occured trying to import some of the required libraries. Make sure that the environment is setup properly.')
	logger.error(f'TDDepthAnythingRT - {e}\n{traceback.format_exc()}')

class TDDepthAnythingRTAccelerate:
	"""_summary_
	"""
	def __init__(self, width: int=518, height: int=518, model_name:str='Depth-Anything-V2', model_type:str='Base', checkpoints_dir:str=f'{os.getcwd()}/checkpoints'):
		"""_summary_

		Args:
			width (int, optional): _description_. Defaults to 518.
			height (int, optional): _description_. Defaults to 518.
			model_name (str, optional): _description_. Defaults to 'Depth-Anything-V2'.
			model_type (str, optional): _description_. Defaults to 'Base'.
			checkpoints_dir (str, optional): _description_. Defaults to f'{os.getcwd()}/checkpoints'.
		"""
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
	
	def free_acc_mem(self):
		"""_summary_
		"""
		if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0:
			logger.info("Clearing GPU memory cache.")
			torch.cuda.empty_cache()

		if gc.isenabled():
			logger.info("Performing garbage collection.")
			gc.collect()


	def update_engine_paths(self):
		"""_summary_
		"""
		self.onnx_path = f"{self.checkpoints_dir}/onnx_models/{self.get_output_name()}.onnx"
		self.engine_path = f"{self.checkpoints_dir}/engines/{self.get_output_name()}.engine"

	@property
	def model_type(self):
		"""_summary_

		Returns:
			_type_: _description_
		"""
		return self._model_type

	@model_type.setter
	def model_type(self, value):
		"""_summary_

		Args:
			value (_type_): _description_
		"""
		self._model_type = value
		self.model_complete_name = f'{self.model_name}-{self.model_type}-hf'
		self.model_path = f'depth-anything/{self.model_complete_name}'
		self.update_engine_paths()

	@property
	def width(self):
		"""_summary_

		Returns:
			_type_: _description_
		"""
		return self._width
	
	@width.setter
	def width(self, value):
		"""_summary_

		Args:
			value (_type_): _description_
		"""
		self._width = self.adjust_image_size(value)
		self.image_shape = (3, self.height, self.width)
		self.update_engine_paths()

	@property
	def height(self):
		"""_summary_

		Returns:
			_type_: _description_
		"""
		return self._height
	
	@height.setter
	def height(self, value):
		"""_summary_

		Args:
			value (_type_): _description_
		"""
		self._height = self.adjust_image_size(value)
		self.image_shape = (3, self.height, self.width)
		self.update_engine_paths()

	@property
	def checkpoints_dir(self):
		"""_summary_

		Returns:
			_type_: _description_
		"""
		return self._checkpoints_dir
	
	@checkpoints_dir.setter
	def checkpoints_dir(self, value):
		"""_summary_

		Args:
			value (_type_): _description_
		"""
		self._checkpoints_dir = value
		self.update_engine_paths()	
		self.ensure_directories_exist()

	def adjust_image_size(self, image_size):
		"""_summary_

		Args:
			image_size (_type_): _description_

		Returns:
			_type_: _description_
		"""
		patch_size = 14
		adjusted_size = (image_size // patch_size) * patch_size
		
		if image_size % patch_size != 0:
			adjusted_size += patch_size

		logger.info(f"Adjusted image size from {image_size} to {adjusted_size}")
		return int(adjusted_size)

	def get_output_name(self):
		"""_summary_

		Returns:
			_type_: _description_
		"""
		return f"{self.model_complete_name}_{self.width}x{self.height}"

	def load_model(self):
		"""
		Load the model using the specified model name and path.
		"""
		logger.info(f"Loading model: {self.model_name} from {self.model_path}")
		try:
			self.model = AutoModelForDepthEstimation.from_pretrained(self.model_path, cache_dir=self.checkpoints_dir)
			logger.info(f"Model loaded successfully.")
		except Exception as e:
			logger.error(f"Failed to load model: {e}\n{traceback.format_exc()}")
			self.model = None

	def unload_model(self):
		"""_summary_
		"""
		self.model = None
		self.free_acc_mem()

	def accelerate(self, model=None):
		"""_summary_

		Args:
			model (_type_, optional): _description_. Defaults to None.
		"""
		logger.info(f'Accelerating, image shape is {self.width}x{self.height}')
		try:
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
				del dummy_input
				logger.info(f"Model exported to {self.onnx_path}")

			# Build TensorRT engine

			if not pathlib.Path(self.engine_path).exists():
				if not os.path.exists(os.path.dirname(self.engine_path)):
					os.makedirs(os.path.dirname(self.engine_path), exist_ok=True)
				
				logger.info(f"Building TensorRT engine for {self.onnx_path}: {self.engine_path}")
				
				p = Profile()
				config_kwargs = {}

				logger.info(f"Created engine profile.")

				engine = engine_from_network(
					network_from_onnx_path(self.onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
					config=CreateConfig(
						fp16=True, refittable=False, profiles=[p], load_timing_cache=None, **config_kwargs
					),
					save_timing_cache=None,
				)

				logger.info(f"Built TensorRT engine from ONNX file.")

				save_engine(engine, path=self.engine_path)

				logger.info(f"Saved TensorRT engine to file.")

				# Clear memory after TensorRT engine creation
				del engine
				self.model = None

			self.free_acc_mem()

			logger.info(f"Finished building TensorRT engine: {self.engine_path}")

		except Exception as e:
			logger.error(f"Error during acceleration: {e}\n{traceback.format_exc()}")