import torch.nn
from layers.transformer import Transformer, UniversalTransformer, \
    RelativeTransformer, UniversalRelativeTransformer, CompositionalTransformer
from models import TransformerEncDecModel
from interfaces import TransformerEncDecInterface


class TransformerMixin:
    def create_model(self) -> torch.nn.Module:
        rel_args = dict(pos_embeddig=(lambda x, offset: x), embedding_init="xavier")
        trafos = {
            "scaledinit": (Transformer, dict(embedding_init="kaiming", scale_mode="down")),
            "opennmt": (Transformer, dict(embedding_init="xavier", scale_mode="opennmt")),
            "noscale": (Transformer, {}),
            "universal_noscale": (UniversalTransformer, {}),
            "universal_scaledinit": (UniversalTransformer, dict(embedding_init="kaiming", scale_mode="down")),
            "universal_opennmt": (UniversalTransformer, dict(embedding_init="xavier", scale_mode="opennmt")),
            "relative": (RelativeTransformer, rel_args),
            "relative_universal": (UniversalRelativeTransformer, rel_args),
            "compositional_8r": (CompositionalTransformer, dict(nrules=8, qk_dim=32)),
            "compositional_2r": (CompositionalTransformer, dict(nrules=2, qk_dim=32)),
            "compositional_4r": (CompositionalTransformer, dict(nrules=4, qk_dim=32)),
            "compositional_1r": (CompositionalTransformer, dict(nrules=1, qk_dim=32)),
            "compositional_8r_16": (CompositionalTransformer, dict(nrules=8, qk_dim=16)),
            "compositional_2r_16": (CompositionalTransformer, dict(nrules=2, qk_dim=16)),
            "compositional_4r_16": (CompositionalTransformer, dict(nrules=4, qk_dim=16)),
            "compositional_1r_16": (CompositionalTransformer, dict(nrules=1, qk_dim=16)),

        }

        constructor, args = trafos[self.helper.args.transformer.variant]

        return TransformerEncDecModel(len(self.train_set.in_vocabulary),
                                      len(self.train_set.out_vocabulary), self.helper.args.state_size,
                                      nhead=self.helper.args.transformer.n_heads,
                                      num_encoder_layers=self.helper.args.transformer.encoder_n_layers,
                                      num_decoder_layers=self.helper.args.transformer.decoder_n_layers or \
                                                         self.helper.args.transformer.encoder_n_layers,
                                      ff_multiplier=self.helper.args.transformer.ff_multiplier,
                                      transformer=constructor,
                                      tied_embedding=self.helper.args.transformer.tied_embedding, **args)

    def create_model_interface(self):
        self.model_interface = TransformerEncDecInterface(self.model, label_smoothing=self.helper.args.label_smoothing)
