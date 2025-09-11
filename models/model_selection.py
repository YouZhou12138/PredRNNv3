from models.PredFormer_TS.predformer import PredFormer_Model
from models.RNN.ct_rnn import CT_RNN
from models.onlly_vit.r_vit import V_RNN

MODELS = {
    "PredRNNv3": CT_RNN,
    "R_ViT": V_RNN,
    "PredFormer": PredFormer_Model,
}
