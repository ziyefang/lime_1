import unittest
from lime.wrappers.scikit_image import BaseWrapper
from lime.wrappers.scikit_image import SegmentationAlgorithm

class TestBaseWrapper(unittest.TestCase):

	def test_base_wrapper(self):
        obj_with_params = BaseWrapper(a=10, b='message')
        obj_without_params = BaseWrapper()
        
        def foo_fn():
            return 'bar'
        obj_with_fn = BaseWrapper(foo_fn)
        
        self.assertEqual(obj_with_params.target_params, {'a':10,'b':'message'})
        self.assertEqual(obj_without_params.target_params, {})
        self.assertEqual(obj_with_fn.target_fn(), 'bar')

	def test_check_params(self):
        
        def bar_fn():
            return 'foo'
        
        
        class Pipo():
    
            def __init__(self):
                self.name = 'pipo'

            def __call__(self, message):
                return message
        pipo = Pipo()
        
        obj_with_valid_fn = BaseWrapper(bar_fn, a=10, b='message')
        obj_with_valid_callable_fn = BaseWrapper(pipo, c=10, d='message')
        obj_with_invalid_fn = BaseWrapper([1,2,3], fn_name='invalid')
        
        # target_fn is not a callable or function/method
        with self.assertRaises(AttributeError):
            obj_with_invalid_fn.check_params('fn_name')
            
        # parameters is not in target_fn args
        with self.assertRaises(ValueError):
            obj_with_valid_fn.check_params(['c'])
            obj_with_valid_callable_fn.check_params(['e'])
        
        # params is in target_fn args
        try:
            obj_with_valid_fn.check_params(['a','b'])
            obj_with_valid_callable_fn.check_params(['c', 'd'])
        except Exception:
            self.fail("check_params() raised an unexpected exception")
        
        # params is not a dict or list
        with self.assertRaises(TypeError):
            obj_with_valid_fn.check_params(None)
        with self.assertRaises(TypeError):
            obj_with_valid_fn.check_params('param_name')

	def test_set_params(self):
        class Pipo():
    
            def __init__(self):
                self.name = 'pipo'

            def __call__(self, message):
                return message
        pipo = Pipo()
        obj = BaseWrapper(pipo)
        
        # argument is set accordingly
        obj.set_params(message='OK')
        self.assertEqual(obj.target_params, {'message':'OK'})
        self.assertEqual(obj.target_fn(obj.target_params), 'OK')
        
        # invalid argument is passed
        try:
            obj = BaseWrapper(Pipo())
            obj.set_params(invalid='KO')
        except:
            self.assertEqual(obj.target_params, {})

	def test_filter_params(self):
		pass


class TestSegmentationAlgorithm(unittest.TestCase):

	def test_instanciate_quickshift(self):
		pass

	def test_instanciate_slic(self):
		pass

	def test_instanciate_felzenszwalb(self):
		pass

if __name__ == '__main__':
    unittest.main()
