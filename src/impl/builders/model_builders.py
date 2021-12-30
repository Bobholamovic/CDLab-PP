# Custom model builders

from core.misc import MODELS


@MODELS.register_func('Unet_model')
def build_unet_model(C):
    from models.unet import Unet
    return Unet(6, 2)


@MODELS.register_func('Unet_OSCD_model')
def build_unet_oscd_model(C):
    from models.unet import Unet
    return Unet(26, 2)


@MODELS.register_func('SiamUnet-diff_model')
def build_siamunet_diff_model(C):
    from models.siamunet_diff import SiamUnet_diff
    return SiamUnet_diff(3, 2)


@MODELS.register_func('SiamUnet-diff_OSCD_model')
def build_siamunet_diff_oscd_model(C):
    from models.siamunet_diff import SiamUnet_diff
    return SiamUnet_diff(13, 2)


@MODELS.register_func('SiamUnet-conc_model')
def build_siamunet_conc_model(C):
    from models.siamunet_conc import SiamUnet_conc
    return SiamUnet_conc(3, 2)


@MODELS.register_func('SiamUnet-conc_OSCD_model')
def build_siamunet_conc_oscd_model(C):
    from models.siamunet_conc import SiamUnet_conc
    return SiamUnet_conc(13, 2)


@MODELS.register_func('CDNet_model')
def build_cdnet_model(C):
    from models.cdnet import CDNet
    return CDNet(6, 2)


@MODELS.register_func('IFN_model')
def build_ifn_model(C):
    from models.ifn import DSIFN
    return DSIFN()


@MODELS.register_func('SNUNet_model')
def build_snunet_model(C):
    from models.snunet import SNUNet
    return SNUNet(3, 2, 32)


@MODELS.register_func('STANet_model')
def build_stanet_model(C):
    from models.stanet import STANet
    return STANet(**C['stanet_model'])