from imgaug import augmenters as iaa
from imgaug import parameters as iap

class aug_presets():
    """
    Collection of various styles of augmentations
    """

    ######################################################################################################
    #   AUGMENTATIONS PRESETS  ###########################################################################
    ######################################################################################################

    @staticmethod
    def psychedelic(self):
        """[summary]
        First exagerate attempt of augmentation, it isn't reccomanded,
        that would be good for a laugh.

        Returns:
            [type]: [description]
        """        
        psychedelic_ = iaa.Sequential([
                iaa.SomeOf((0,1),[
                        iaa.BlendAlphaFrequencyNoise(
                            foreground=iaa.EdgeDetect(1.0),
                            per_channel=True
                        ),
                        iaa.ElasticTransformation(alpha=50, sigma=5),  # apply water effect (affects segmaps)
                        
                        iaa.ReplaceElementwise(
                            iap.FromLowerResolution(
                                iap.Binomial(0.1), size_px=8
                            ),
                            iap.Normal(128, 0.4*128),
                            per_channel=0.5
                        )
                    ]
                ),
                iaa.PiecewiseAffine(scale=iap.Absolute(iap.Normal(0, 0.1))),
                iaa.Sharpen((0.0, 1.0)),       # sharpen the image
                iaa.Affine(
                    rotate=(-45, 45),
                    mode="edge"
                )  # rotate by -45 to 45 degrees (affects segmaps)

            ], random_order=True)
        return psychedelic_


    def preset_1(self, severity=1.0):
        """[summary]
            Apply a large variety of augmentation, included the most disruptives,
            standard augmentations is applayed in the same way but th heaviest augmentations
            can be dimmered in frequency.

            Strong augmentations are also time consuming.

            Returns:
            [type]: [description]
        """   

        various = iaa.Sequential([

            ]
        )


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
                (0, n) if isinstance(n, tuple) else \
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

        def maybe_some(self, p=0.5, n = 0, rand = True):
            """
            return augmentor that if applayed (with probability p) apply a subset of augmentations
            the default interval of aplayed augmentations (0, max)
            """
            n = (0, self.n_aug) if n == 0 else \
                (0, n) if isinstance(n, tuple) else \
                n if isinstance(n, list) else (0, self.n_aug)

            return iaa.Sometimes(p, then_list= 
                            iaa.SomeOf(n, self.aug_list, random_order=rand))

        def maybe_one(self, p=0.5, rand = True):
            """
            return augmentor that if applayed (with probability p) apply one augmentation in the given set
            """
            return iaa.Sometimes(p, then_list= 
                            iaa.OneOf(self.aug_list, random_order=rand))
        

    # ARITMETIC ##########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html

    class aritmetic_aug(base_aug):

        def __init__(self, severity=1.0, sets=[0, 1, 2]):
            
            s = severity
            self.aug_lists = {
                0 : [
                    iaa.OneOf([
                            iaa.Add((int(-30*s), int(30*s)), per_channel=True),
                            iaa.AddElementwise((int(-30*s), int(30*s)), per_channel=True)
                        ]
                    )
                ],
                1 : [ 
                    iaa.OneOf([
                            iaa.AdditiveGaussianNoise(scale=iap.Clip(iap.Poisson((-30*s, 30*s)), 0, 255), per_channel=True),
                            iaa.OneOf([
                                    iaa.Multiply((0.9*s, 1.2*s)),
                                    iaa.Multiply((0.9*s, 1.2*s), per_channel=True)
                                ]
                            )
                        ]
                    )
                ],
                2 : [
                    iaa.OneOf([
                            iaa.OneOf([
                                    iaa.ImpulseNoise((0.6, 0.8)),
                                    iaa.Dropout(p=iap.Uniform((0.4, 0.7*s), 0.8*s))
                                ]
                            ),
                            iaa.OneOf([
                                iaa.CoarseSaltAndPepper(0.1*s, size_percent=(0.03, 0.1)),
                                iaa.CoarseSaltAndPepper(0.1*s, size_percent=(0.03, 0.1), per_channel=True)
                                ]
                            ),
                            iaa.OneOf([
                                    iaa.Cutout(
                                        nb_iterations=(5, 10),
                                        size = (0.05, 0.1),
                                        fill_mode="gaussian",
                                        fill_per_channel=True,
                                        squared=False
                                    ),
                                    iaa.Cutout(
                                        nb_iterations=(5, 10),
                                        size=(0.05, 0.1),
                                        cval=(0, 255),
                                        fill_per_channel=0.5,
                                        squared=False
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
        
        def __init__(self, severity=1.0, lists=[0, 1, 2, 3]):
            
            s = severity
            self.aug_lists = {
                0 : [
                    iaa.Fliplr(0.5), # horizontaly flip with probability
                    iaa.Flipud(0.5), # vertical flip with probability
                    iaa.Affine(	# rotation with edge filling
                        rotate=(-25, 25), # rotation between interval (degrees)
                        mode="constant", # filler type (new pixels are generated based on edge pixels)
                        cval=0
                    )
                ],
                1 : [ 
                    iaa.AdditiveGaussianNoise(scale=iap.Clip(iap.Poisson((0, int(50*s))), 0, 255), per_channel=True),
                    iaa.Multiply((0.5*s, 1.5*s)),
                    iaa.Multiply((0.5*s, 1.5*s), per_channel=True)
                ],
                2 : [
                    iaa.ImpulseNoise(0.4*s),
                    iaa.Dropout(p=(0, 0.6*s)),
                    iaa.CoarseSaltAndPepper(0.2*s, size_percent=(0.01, 0.1)),
                    iaa.CoarseSaltAndPepper(0.2*s, size_percent=(0.01, 0.1), per_channel=True)
                ]
            }
            
            if not isinstance(lists, list):
                lists = [lists]

            for list_ in lists:
                self.aug_list += self.aug_lists[list_]

            self.n_aug = len(self.aug_list)


    # CONTRAST ###########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/contrast.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_contrast.html
    


    # COLOR ##############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/color.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_color.html


    # BLEND ##############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/blend.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blend.html


    # BLUR ###############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/blur.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html


    # CONVOLUTIONAL ######################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/convolutional.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_convolutional.html


    # POOLING ############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/pooling.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_pooling.html


    ######################################################################################
    #   ???
    ######################################################################################
