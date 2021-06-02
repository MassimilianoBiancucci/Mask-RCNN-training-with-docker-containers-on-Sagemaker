from imgaug import augmenters as iaa
from imgaug import parameters as iap

class aug_presets():
    """
    Collection of various styles of augmentations
    """

    ######################################################################################################
    #   AUGMENTATIONS PRESETS  ###########################################################################
    ######################################################################################################

    def psychedelic(self):
        """[summary]

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


    ######################################################################################################
    #   AUGMENTATIONS SETS DIVIDED BY TYPE  ##############################################################
    ######################################################################################################
    """
        Notes about sets of augmentations.
        each sets is marked with one number, that identify the heaviness of the augmentation, (1 is lower),
        remember more is heavy more is time consuming during the training.
    """

    # ARITMETIC ##########################################################################
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html

    aug_aritmetic_1 = iaa.SomeOf((0, 3), [
            iaa.Fliplr(0.5), # horizontaly flip with probability
			iaa.Flipud(0.5), # vertical flip with probability
            iaa.Affine(	# rotation with edge filling
                rotate=(-25, 25), # rotation between interval (degrees)
                mode="edge" # filler type (new pixels are generated based on edge pixels)
            )
        ]
    )

    aug_aritmetic_2 = iaa.Sequential([
            
        ]
    )

    aug_aritmetic_3 = iaa.Sequential([
            
        ]
    )

    # GEOMETRIC ##########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html
    # docs: 


    # CONTRAST ###########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/contrast.html
    # docs: 
    

    # COLOR ##############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/color.html
    # docs: 


    # BLEND ##############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/blend.html
    # docs: 


    # BLUR ###############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/blur.html
    # docs: 


    # CONVOLUTIONAL ######################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/convolutional.html
    # docs: 


    # POOLING ############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/pooling.html
    # docs: 


    # ARITMETIC ##########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html

    # Add

    # AddElementWise

    # 

    # 

    # 

    ######################################################################################
    #   ???
    ######################################################################################
