from imgaug import augmenters as iaa
from imgaug import parameters as iap

class aug_presets():
    """
    Collection of various styles of augmentations
    """

    ######################################################################################################
    #   AUGMENTATIONS PRESETS  ###########################################################################
    ######################################################################################################

    psychedelic = iaa.Sequential([
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
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html



    # CONTRAST ###########################################################################
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/contrast.html

    

    # COLOR ##############################################################################
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/color.html



    # BLEND ##############################################################################
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/blend.html



    # BLUR ###############################################################################
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/blur.html



    # CONVOLUTIONAL ######################################################################
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/convolutional.html



    # POOLING ############################################################################
    # docs: https://imgaug.readthedocs.io/en/latest/source/overview/pooling.html



    ######################################################################################
    #   ???
    ######################################################################################

    def __init__():
        """
        
        """