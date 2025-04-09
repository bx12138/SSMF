from dataset.uci_har import UCI_HAR
from dataset.usc_had import USC_HAD
from dataset.pamap2 import PAMAP2
from dataset.opportunity import OPPORTUNITY
from dataset.mhealth import MHEALTH
from dataset.mobiact import MobiAct


if __name__ == "__main__":
    dataset = UCI_HAR()
    dataset.dataset_verbose()
    dataset.save_split()
    
    dataset = PAMAP2(clean=False, include_null=True)
    dataset.save_split('splits_Xclean')

    dataset = MHEALTH(clean=False, include_null=True)
    dataset.dataset_verbose()
    dataset.save_split('splits_Xclean')
