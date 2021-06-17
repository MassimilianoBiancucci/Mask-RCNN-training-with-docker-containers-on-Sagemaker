from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmenters.meta import Sometimes
from imgaug.augmenters.size import Scale

class aug_presets():
	"""
	Collection of various styles of augmentations
	"""

	def __init__(self):

		self.presets_list = [
			self.preset_0(),
			self.preset_1(),
			self.preset_2()
		]

	######################################################################################################
	#   AUGMENTATIONS PRESETS  ###########################################################################
	######################################################################################################

	@staticmethod
	def preset_0(severity=1.0):
		"""[summary]
			Apply a large variety of augmentation, included the most disruptives,
			standard augmentations is applayed in the same way but th heaviest augmentations
			can be dimmered in frequency.

			Strong augmentations are also time consuming.

			Returns:
			[type]: [description]
		"""   

		aug = iaa.Sequential([
					aug_presets.geometric_aug(sets=[0, 1]).seq(),
					iaa.SomeOf((1, 2), [
							aug_presets.aritmetic_aug(severity=0.3, sets=[0, 1]).maybe_some(p=0.95, n=(1, 2)),
							aug_presets.contrast_aug().maybe_some(p=0.8, n=(1, 3)),
						],
						random_order=True
					)
				],
				random_order=True
			)

		return aug

	@staticmethod
	def preset_1(severity=1.0):
		"""[summary]
			Apply a large variety of augmentation, included the most disruptives,
			standard augmentations is applayed in the same way but th heaviest augmentations
			can be dimmered in frequency.

			Strong augmentations are also time consuming.

			Returns:
			[type]: [description]
		"""   

		aug = iaa.Sequential([
					aug_presets.geometric_aug(sets=[0, 1]).seq(),
					iaa.SomeOf((1, 2), [
							aug_presets.geometric_aug(sets=2).maybe_one(p=0.6),
							aug_presets.aritmetic_aug(sets=[0, 1]).maybe_some(p=0.6, n=(1, 2)),
							aug_presets.aritmetic_aug(sets=2).maybe_one(p=0.35),
							aug_presets.contrast_aug().maybe_some(p=0.4, n=(1, 3)),
							aug_presets.blend_aug(sets=[0, 1, 3]).maybe_one(p=0.4)
						],
						random_order=True
					)
				],
				random_order=True
			)

		return aug


	@staticmethod
	def preset_2(severity=1.0):
		"""[summary]
			Apply a large variety of augmentation, included the most disruptives,
			standard augmentations is applayed in the same way but th heaviest augmentations
			can be dimmered in frequency.

			Strong augmentations are also time consuming.

			Returns:
			[type]: [description]
		"""   

		aug = iaa.Sequential([
					aug_presets.geometric_aug(sets=[0, 1]).seq(),
					iaa.SomeOf((1, 4), [
							aug_presets.geometric_aug(sets=2).maybe_one(),
							aug_presets.aritmetic_aug(sets=[0, 1]).maybe_some(p=0.95, n=(1, 2)),
							aug_presets.aritmetic_aug(sets=2).maybe_one(p=0.15),
							aug_presets.contrast_aug().maybe_some(p=0.8, n=(1, 3)),
							aug_presets.blend_aug(sets=[0, 1, 3]).maybe_one(p=0.8),
							aug_presets.color_aug().maybe_one(p=0.8)
						],
						random_order=True
					)
				],
				random_order=True
			)

		return aug


	######################################################################################################
	#   AUGMENTATIONS SETS DIVIDED BY TYPE  ##############################################################
	######################################################################################################
	"""
		Notes about sets of augmentations.
		each sets is marked with one number, that identify the heaviness of the augmentation, (1 is lower),
		remember more is heavy more is time consuming during the training.
	"""

	class base_aug():
		
		aug_list = []
		aug_lists = {}
		n_aug = 0 # number of augmentors in the class list

		def seq(self, rand = True):
			"""
			return: the sequence of selected lists
			"""
			return iaa.Sequential(self.aug_list, random_order=rand)

		def some(self, n = 0, rand = True):
			"""
			return augmenter that apply a subset of augmentations
			default interval of aplayed augmentations (0, max)
			"""
			n = (0, self.n_aug) if n == 0 else \
				n if isinstance(n, tuple) else \
				n if isinstance(n, list) else (0, self.n_aug)

			return iaa.SomeOf(n, self.aug_list, random_order=rand)

		def one(self):
			"""
			return augmentor that applay one augmentations
			each times taken from the set
			"""
			return iaa.OneOf(self.aug_list)

		def maybe_all(self, p=0.5, rand = True):
			"""
			return augmentor that if applayed (with probability p) apply all the augmentations in the given set
			"""
			return iaa.Sometimes(p, then_list= 
							iaa.Sequential(self.aug_list, random_order=rand))

		def maybe_some(self, p=0.5, n=0,rand = True):
			"""
			return augmentor that if applayed (with probability p) apply a subset of augmentations
			the default interval of aplayed augmentations (0, max)
			"""
			n = (0, self.n_aug) if n == 0 else \
				n if isinstance(n, tuple) else \
				n if isinstance(n, list) else (0, self.n_aug)

			return iaa.Sometimes(p, then_list= 
							iaa.SomeOf(n, self.aug_list, random_order=rand))

		def maybe_one(self, p=0.5, rand = True):
			"""
			return augmentor that if applayed (with probability p) apply one augmentation in the given set
			"""
			return iaa.Sometimes(p, then_list= 
							iaa.OneOf(self.aug_list))
		

	# ARITMETIC ##########################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html

	class aritmetic_aug(base_aug):
		
		def __init__(self, severity=1.0, sets=[0, 1, 2]):
			
			s = severity
			self.aug_lists = {
				0 : [
					iaa.OneOf([
							iaa.Add((int(-50*s), int(50*s)), per_channel=True),
							iaa.AddElementwise((int(-50*s), int(50*s)), per_channel=True)
						]
					)
				],
				1 : [ 
					iaa.OneOf([
							iaa.AdditiveGaussianNoise(scale=iap.Clip(iap.Poisson((-20*s, 20*s)), 0, 255), per_channel=True),
							iaa.Multiply((max(1.0 - (0.2*s), 0.2), min(1.0 + (0.2*s), 1.0))),
							iaa.Multiply((max(1.0 - (0.2*s), 0.2), min(1.0 + (0.2*s), 1.0)), per_channel=True)
						]
					)
				],
				2 : [
					iaa.OneOf([
							iaa.OneOf([
									iaa.ImpulseNoise((0.015, 0.05)),
									iaa.Dropout(p=iap.Uniform((0.5, 0.7*s), 0.9*s))
								]
							),
							iaa.OneOf([
									iaa.OneOf([
										iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1)),
										iaa.WithPolarWarping(iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1))),
										]
									),
									iaa.OneOf([
										iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1), per_channel=True),
										iaa.WithPolarWarping(iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1), per_channel=True))
										]
									)
								]
							),
							iaa.OneOf([
									iaa.OneOf([
											iaa.Cutout(
												nb_iterations=(5, 10),
												size = (0.05, 0.1),
												fill_mode="gaussian",
												fill_per_channel=True,
												squared=False
											),
											iaa.WithPolarWarping(iaa.Cutout(
												nb_iterations=(5, 10),
												size = (0.05, 0.1),
												fill_mode="gaussian",
												fill_per_channel=True,
												squared=False
											))
										]
									),
									iaa.OneOf([
											iaa.Cutout(
												nb_iterations=(5, 10),
												size=(0.05, 0.1),
												cval=(0, 255),
												fill_per_channel=0.5,
												squared=False
											),
											iaa.WithPolarWarping(iaa.Cutout(
												nb_iterations=(5, 10),
												size=(0.05, 0.1),
												cval=(0, 255),
												fill_per_channel=0.5,
												squared=False
											))
										]
									)
								]
							)
						]
					)
				]
			}
			
			if not isinstance(sets, list):
				sets = [sets]

			for list_ in sets:
				self.aug_list += self.aug_lists[list_]
			
			self.n_aug = len(self.aug_list)
			
		
	# GEOMETRIC ##########################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html

	class geometric_aug(base_aug):
		
		def __init__(self, severity=1.0, sets=[0, 1, 2]):
			
			s = severity
			self.aug_lists = {
				0 : [
					iaa.Sequential([
						iaa.Fliplr(0.5), # horizontaly flip with probability
						iaa.Flipud(0.5), # vertical flip with probability
						]
					)
				],
				1 : [ 
					iaa.Sometimes(p=0.95, 
						then_list=iaa.SomeOf((1, 2), [
								iaa.ScaleX((0.9, 1.1), mode="constant", cval= 0),
								iaa.ScaleY((0.9, 1.1), mode="constant", cval= 0),
								iaa.TranslateX(percent=(-0.15, 0.15), mode="constant", cval= 0),
								iaa.TranslateY(percent=(-0.15, 0.15), mode="constant", cval= 0),
								iaa.Rotate((-30, 30), mode="constant", cval= 0),
								iaa.ShearX((-5, 5), mode="constant", cval= 0),
								iaa.ShearY((-5, 5), mode="constant", cval= 0)
							],
							random_order=True
						),
						else_list=iaa.SomeOf((2, 4), [
								iaa.ScaleX((0.8, 1.1), mode="constant", cval= 0),
								iaa.ScaleY((0.8, 1.1), mode="constant", cval= 0),
								iaa.TranslateX(percent=(-0.1, 0.1), mode="constant", cval= 0),
								iaa.TranslateY(percent=(-0.1, 0.1), mode="constant", cval= 0),
								iaa.Rotate((-30, 30), mode="constant", cval= 0),
								iaa.ShearX((-10, 10), mode="constant", cval= 0),
								iaa.ShearY((-10, 10), mode="constant", cval= 0)
							],
							random_order=True
						)
					)
				],
				2 : [
					iaa.OneOf([
						iaa.PiecewiseAffine(scale=(0.01, 0.05), mode="constant", cval= 0),
						#iaa.ElasticTransformation(alpha=(2.0, 10.0), sigma=(0.1, 1.0), mode="constant", cval= 0),
						iaa.PerspectiveTransform(scale=(0.08, 0.10), mode="constant", cval= 0)
					])
				]
			}
			
			if not isinstance(sets, list):
				sets = [sets]

			for list_ in sets:
				self.aug_list += self.aug_lists[list_]
			
			self.n_aug = len(self.aug_list)


	# CONTRAST ###########################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/contrast.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_contrast.html
	
	class contrast_aug(base_aug):

		def __init__(self, severity=1.0, sets=[0, 1, 2]):
			
			s = severity
			self.aug_lists = {
				0: [
					iaa.OneOf([
						iaa.GammaContrast((0.6, 3.0)),
						iaa.GammaContrast((0.6, 2.0), per_channel=True)
					]
					)
				],
				1: [
					iaa.OneOf([
						iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
						iaa.SigmoidContrast(
							gain=(3, 10),
							cutoff=(0.4, 0.6),
							per_channel=True
						),
						iaa.LogContrast(gain=(0.6, 1.4)),
						iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
						iaa.LinearContrast((0.4, 1.6)),
						iaa.LinearContrast((0.4, 1.6), per_channel=True)
					]
					)
				],
				2: [
					iaa.OneOf([
						iaa.AllChannelsCLAHE(clip_limit=(1, 15)),
						iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
						iaa.CLAHE(clip_limit=(1, 10)),
						iaa.AllChannelsHistogramEqualization()
					]
					)
				]
			}

			if not isinstance(sets, list):
				sets = [sets]

			for list_ in sets:
				self.aug_list += self.aug_lists[list_]
			
			self.n_aug = len(self.aug_list)

	# COLOR ##############################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/color.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_color.html
	class color_aug(base_aug):
		def __init__(self, severity=1.0, sets=[0, 1, 2, 3]):
			
			s = severity
			self.aug_lists = {
				0: [
					iaa.WithColorspace(
						to_colorspace="RGB",
						from_colorspace="HSV",
						children=iaa.Grayscale(alpha=(0.0, 1.0))
					)
				],  
				1: [
					iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
				],
				2: [
					iaa.Sequential([
						iaa.ChangeColorspace(
							from_colorspace="HSV", to_colorspace="RGB"),
						iaa.WithChannels(children=iaa.Add((-10, 10))),
						iaa.ChangeColorspace(
							from_colorspace="RGB", to_colorspace="HSV")
					])
				],
				3: [
					iaa.ChangeColorTemperature((1100, 10000))
				]
			}

			if not isinstance(sets, list):
				sets = [sets]

			for list_ in sets:
				self.aug_list += self.aug_lists[list_]

			self.n_aug = len(self.aug_list)

	# BLEND ##############################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/blend.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blend.html
	class blend_aug(base_aug):
		
		def __init__(self, severity=1.0, sets=[0, 1, 2, 3]):
			
			s = severity
			self.aug_lists = {
				0: [
					iaa.BlendAlphaRegularGrid(
						nb_rows=(15, 40), 
						nb_cols=(15, 40),
						foreground=iaa.MotionBlur(k=(3, 5))
					)
				],
				1: [
					iaa.BlendAlphaFrequencyNoise(foreground=iaa.Add((int(-50*s), int(50*s)), per_channel=True),
												 iterations=(1, 3),
												 upscale_method='linear',
												 size_px_max=(2, 8),
												 sigmoid=0.2)
				],
				2: [
					iaa.BlendAlphaSimplexNoise(
								iaa.EdgeDetect(iap.Clip(iap.Absolute(iap.Normal(0.4, 0.18)), 0, 1)),
								upscale_method="linear",
						per_channel=True)
				],
				3: [
					iaa.BlendAlphaSimplexNoise(
						iaa.AdditiveGaussianNoise(
							scale=iap.Clip(iap.Poisson(
								(0, int(60*s))), 0, 255),
							per_channel=True),
						upscale_method="linear",
						per_channel=True)
				]
			}

			if not isinstance(sets, list):
				sets = [sets]

			for list_ in sets:
				self.aug_list += self.aug_lists[list_]
			
			self.n_aug = len(self.aug_list)

	# BLUR ###############################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/blur.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html
	class blur_aug(base_aug):
		
		def __init__(self, severity=1.0, sets=[0, 1, 2, 3]):
			
			s = severity
			self.aug_lists = {
				0: [
					iaa.GaussianBlur(sigma=(0, 3.0*s)),
				],
				1: [
					iaa.OneOf([
						iaa.AverageBlur(k=((15*s, 25*s), (0, 2))),
						iaa.AverageBlur(k=((0, 2), (15*s, 25*s))),
					])
				],
				2: [
					iaa.BilateralBlur(
						d=(4, int(15*s)), sigma_color=(20, 250), sigma_space=(20, 250))
				],
				3: [
					iaa.MotionBlur(k=int(10*s))
				]
			}

			if not isinstance(sets, list):
				sets = [sets]

			for list_ in sets:
				self.aug_list += self.aug_lists[list_]
			
			self.n_aug = len(self.aug_list)

	# CONVOLUTIONAL ######################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/convolutional.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_convolutional.html


	# POOLING ############################################################################
	# overview: https://imgaug.readthedocs.io/en/latest/source/overview/pooling.html
	# docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_pooling.html

	

	######################################################################################
	#   ???
	######################################################################################
