import types
from utils.generic_utils import has_arg


class BaseWrapper(object):
	"""Base class for LIME Scikit-Image wrapper


	Args:
		target_fn: callable function or class instance
		**target_params: parameters to pass to the target_fn


	'target_params' takes parameters required to instanciate the desired Scikit-Image class/model
	"""

	def __init__(self, target_fn=None, **target_params):
		self.target_fn = target_fn
		self.target_params = target_params
		self.check_params(target_params)

	def check_params(self, obj, parameters):
		"""Checks for mistakes in 'parameters'

		Args :
			obj: callable object to analyze
			parameters: dict, parameters to be checked

		Raises :
			ValueError: if any parameter is not a valid argument for the target function
		"""

		a_valid_fn = []
		if self.target_fn is None:
			a_valid_fn.append(self.__call__)
		elif (not isinstance(self.target_fn, types.FunctionType) and not isinstance(self.target_fn, types.MethodType)):
			a_valid_fn.append(self.target_fn.__call__)
		else:
			a_valid_fn.append(self.target_fn)

		for p in parameters:
			for fn in a_valid_fn:
				if has_arg(fn, p):
					pass
				else:
					raise ValueError('{} is not a valid parameter'.format(p))

	def set_params(self, **params):
		"""Sets the parameters of this estimator.
		Args:
			**params: Dictionary of parameter names mapped to their values.
		Returns:
			self
		"""
		self.check_params(params)
		self.target_params.update(params)
		return self

	def filter_params(self, fn, override=None):
		"""Filters `target_params` and return those in `fn`'s arguments.
		Args:
			fn : arbitrary function
			override: dict, values to override sk_params
		Returns:
			d_result : dict, dictionary containing variables
			in both target_params and fn's arguments.
		"""
		d_override = override or {}
		d_result = {}
		for name, value in self.target_params.items():
			if has_arg(fn, name):
				d_result.update({name: value})
		d_result.update(d_override)
		return d_result
