import torch
import pandas as pd
import numpy as np
import random
from torch import nn
from fuxictr.utils import not_in_whitelist
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2
from sklearn.cluster import KMeans

class ResidualQuantizer:
    def __init__(self, num_clusters=5, num_layers=3):
        self.num_clusters = num_clusters
        self.num_layers = num_layers
        self.codebooks = []
        self.layer_all_embeddings = []
        self.layer_all_item_ids = []
        
        self.codebooks_tensor = []
        self.layer_all_embeddings_tensor = []  
        self.layer_all_item_ids_tensor = []  

    def fit(self, X, item_ids):
        residual = X.copy()
        current_item_ids = item_ids.copy()
        
        self.codebooks = []
        self.layer_all_embeddings = []
        self.layer_all_item_ids = []
        
        for layer in range(self.num_layers):
            kmeans = KMeans(n_clusters=self.num_clusters, n_init='auto', random_state=20242025)
            kmeans.fit(residual)
            centers = kmeans.cluster_centers_  # [num_clusters, D]
            labels = kmeans.labels_            # [N, ]
            self.codebooks.append(centers)
            
            embeddings_list = []
            item_ids_list = []
            for cluster_id in range(self.num_clusters):
                indices = np.where(labels == cluster_id)[0]
                if len(indices) > 0:
                    embeddings_list.append(residual[indices])   # [M, D]
                    item_ids_list.append(current_item_ids[indices])  # [M, ]

            layer_embeddings = np.concatenate(embeddings_list, axis=0)  # [total_items, D]
            layer_item_ids = np.concatenate(item_ids_list, axis=0)        # [total_items, ]
            self.layer_all_embeddings.append(layer_embeddings)
            self.layer_all_item_ids.append(layer_item_ids)
            
            residual = residual - centers[labels]

        self.codebooks_tensor = [torch.tensor(cb, dtype=torch.float32) for cb in self.codebooks]
        self.layer_all_embeddings_tensor = [torch.tensor(emb, dtype=torch.float32) for emb in self.layer_all_embeddings]
        self.layer_all_item_ids_tensor = [torch.tensor(ids, dtype=torch.long) for ids in self.layer_all_item_ids]

    def quantize(self, X):
        with torch.no_grad():
            residual = X
            quantized_ids = []
            for layer in range(self.num_layers):
                all_emb = self.layer_all_embeddings_tensor[layer].to(X.device)  # [total_items, D]
                all_item_ids = self.layer_all_item_ids_tensor[layer].to(X.device)  # [total_items]

                dists = torch.cdist(residual.float(), all_emb.float())  # [B, total_items]
                min_indices = torch.argmin(dists, dim=1)  # [B]

                selected_item_ids = all_item_ids[min_indices]  # [B]
                quantized_ids.append(selected_item_ids)
                
                centers = self.codebooks_tensor[layer].to(X.device)  # [num_clusters, D]
                candidate = all_emb[min_indices]  # [B, D]

                d_cluster = torch.cdist(candidate.float().unsqueeze(0), centers.float().unsqueeze(0)).squeeze(0)  # [B, num_clusters]
                selected_cluster = torch.argmin(d_cluster, dim=1)  # [B]
                selected_center = centers[selected_cluster]  # [B, D]
                residual = residual - selected_center
            return quantized_ids

class Transformer(nn.Module):
    def __init__(self,
                 transformer_in_dim,
                 dim_feedforward=256,
                 num_heads=1,
                 dropout=0.2,
                 transformer_layers=2,
                 first_k_cols=16,
                 concat_max_pool=True):
        super(Transformer, self).__init__()
        self.concat_max_pool = concat_max_pool
        self.first_k_cols = first_k_cols
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_in_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        if self.concat_max_pool:
            self.out_linear = nn.Linear(transformer_in_dim, transformer_in_dim)

    def forward(self, target_emb, sequence_emb, mask=None):
        # concat action sequence emb with target emb
        seq_len = sequence_emb.size(1)
        concat_seq_emb = torch.cat([sequence_emb,
                                    target_emb.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        # get sequence mask (1's are masked)
        key_padding_mask = self.adjust_mask(mask).bool()  # keep the last dim
        tfmr_out = self.transformer_encoder(src=concat_seq_emb,
                                            src_key_padding_mask=key_padding_mask)
        tfmr_out = tfmr_out.masked_fill(
            key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), 0.
        )
        # process the transformer output
        output_concat = []
        output_concat.append(tfmr_out[:, -self.first_k_cols:].flatten(start_dim=1))
        if self.concat_max_pool:
            # Apply max pooling to the transformer output
            tfmr_out = tfmr_out.masked_fill(
                key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), -1e9
            )
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        return torch.cat(output_concat, dim=-1)

    def adjust_mask(self, mask):
        # make sure not all actions in the sequence are masked
        fully_masked = mask.all(dim=-1)
        mask[fully_masked, -1] = 0
        return mask

class Transformer_DCN_Quant(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="Transformer_DCN_Quant",
                 gpu=-1,
                 hidden_activations="ReLU",
                 dcn_cross_layers=3,
                 dcn_hidden_units=[1024, 512, 256],
                 mlp_hidden_units=[64, 32],
                 num_heads=1,
                 transformer_layers=2,
                 transformer_dropout=0.2,
                 dim_feedforward=256,
                 learning_rate=0.0005,
                 embedding_dim=64,
                 net_dropout=0.2,
                 batch_norm=False,
                 first_k_cols=16,
                 concat_max_pool=True,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 top_k=3,
                 codebook_num=3,
                 default_vocab=5,
                 global_item_info=None,
                 **kwargs):
        super().__init__(feature_map,
                         model_id=model_id,
                         gpu=gpu,
                         embedding_regularizer=embedding_regularizer,
                         net_regularizer=net_regularizer,
                         **kwargs)
        self.feature_map = feature_map
        self.top_k = top_k
        self.codebook_num = codebook_num

        self.raw_item_info_dim = sum(
            spec.get("embedding_dim", embedding_dim)
            for feat, spec in self.feature_map.features.items()
            if spec.get("source") == "item" and feat not in ["quanid", "item_emb_d128"]
        )
        specified_feats = ["likes_level", "views_level"]
        self.batch_info_dim = sum(
            self.feature_map.features[feat].get("embedding_dim", embedding_dim)
            for feat in specified_feats
            if feat in self.feature_map.features and self.feature_map.features[feat].get("active", True)
        )

        if "quanid" not in self.feature_map.features:
            self.feature_map.features["quanid"] = {
                "dtype": "int",
                "vocab_size": default_vocab,
                "embedding_dim": embedding_dim,
                "active": True,
                "type": "categorical",
                "source": "item"
            }
        self.fused_item_dim = self.raw_item_info_dim + (self.top_k + self.codebook_num) * self.feature_map.features["quanid"]["embedding_dim"]

        self.use_global_codebook = False
        if global_item_info is not None:
            item_df = pd.read_parquet(global_item_info)
            if "item_id" in item_df.columns and "item_emb_d128" in item_df.columns:
                emb_list = np.stack(item_df["item_emb_d128"].values, axis=0)

                global_item_ids = item_df["item_id"].values
                self.rq = ResidualQuantizer(num_clusters=5, num_layers=self.codebook_num)
                self.rq.fit(emb_list, global_item_ids)
                global_item_ids_tensor = torch.from_numpy(global_item_ids).long()
                global_emb_tensor = torch.from_numpy(emb_list).float()

                self.register_buffer("global_item_ids", global_item_ids_tensor)
                self.register_buffer("global_item_embeddings", global_emb_tensor)
                self.register_buffer("global_codebook", global_emb_tensor)
                self.use_global_codebook = True
                self.feature_map.features["quanid"]["vocab_size"] = int(global_item_ids_tensor.max().item()) + 1
            else:
                print("Warning: global_item_info provided but missing 'item_id' or 'item_emb_d128' columns.")

        self.accumulation_steps = accumulation_steps

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, not_required_feature_columns=["quanid", "item_emb_d128"])
        self.quan_embedding = FeatureEmbedding(feature_map, embedding_dim, required_feature_columns=["quanid"])

        transformer_in_dim = self.fused_item_dim * 2
        self.transformer_encoder = Transformer(
            transformer_in_dim,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            dropout=transformer_dropout,
            transformer_layers=transformer_layers,
            first_k_cols=first_k_cols,
            concat_max_pool=concat_max_pool
        )
        seq_out_dim = (first_k_cols + int(concat_max_pool)) * transformer_in_dim

        # dcn_in_dim = feature_map.sum_emb_out_dim() + seq_out_dim
        dcn_in_dim = self.batch_info_dim + seq_out_dim + self.fused_item_dim
        self.crossnet = CrossNetV2(dcn_in_dim, dcn_cross_layers)
        self.parallel_dnn = MLP_Block(
            input_dim=dcn_in_dim,
            output_dim=None,
            hidden_units=dcn_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        )
        dcn_out_dim = dcn_in_dim + dcn_hidden_units[-1]
        self.mlp = MLP_Block(
            input_dim=dcn_out_dim,
            output_dim=1,
            hidden_units=mlp_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=self.output_activation
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs

        X_dict = {}
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)

        for k, v in item_dict.items():
            item_dict[k] = v.to(self.device)

        if "item_id" in item_dict and "item_emb_d128" in item_dict:
            item_emb_d128 = item_dict.pop("item_emb_d128")
            self.add_quanid_as_feature(item_dict, item_emb_d128)
        return X_dict, item_dict, mask.to(self.device)

    def add_quanid_as_feature(self, item_dict, item_emb_d128):
        item_ids = item_dict["item_id"]
        norm_emb = item_emb_d128.float() / (item_emb_d128.float().norm(dim=1, keepdim=True) + 1e-8)

        rq_ids = self.rq.quantize(norm_emb)
        rq_ids_tensor = torch.stack(rq_ids, dim=1)

        if self.use_global_codebook:
            sim_matrix = torch.matmul(norm_emb, self.global_codebook.transpose(0, 1))
            _, topk_idx = torch.topk(sim_matrix, k=self.top_k, dim=1)
            vq_ids = self.global_item_ids[topk_idx]  # [B*seq_len, top_k]
        else:
            sim_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
            _, topk_idx = torch.topk(sim_matrix, k=self.top_k, dim=1)
            vq_ids = item_ids[topk_idx]  # [B*seq_len, top_k]

        quanid = torch.cat([rq_ids_tensor, vq_ids], dim=1)  # [B*seq_len, codebook_num + top_k]
        item_dict["quanid"] = quanid

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)

        if batch_dict:
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        else:
            batch_size = mask.shape[0]
            feature_emb = torch.zeros(batch_size, self.feature_map.sum_emb_out_dim(), device=self.device)

        if "quanid" in item_dict:
            quan_ids = item_dict.pop("quanid")
            quan_emb = self.quan_embedding({"quanid": quan_ids})
            quan_emb = quan_emb.view(quan_emb.size(0), -1)
        else:
            batch_item_dim = next(iter(item_dict.values())).shape[0]
            quan_emb = torch.zeros(batch_item_dim, (self.codebook_num + self.top_k) * self.quan_embedding.embedding_dim, device=self.device)

        other_item_emb = self.embedding_layer(item_dict, flatten_emb=True)
        fused_emb = torch.cat([other_item_emb, quan_emb], dim=-1)
        batch_size = mask.shape[0]
        seq_len = fused_emb.shape[0] // batch_size
        item_feat_emb = fused_emb.view(batch_size, seq_len, -1)
        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, :-1, :]
        transformer_emb = self.transformer_encoder(target_emb, sequence_emb, mask=mask)
        dcn_in_emb = torch.cat([feature_emb, target_emb, transformer_emb], dim=-1)
        cross_out = self.crossnet(dcn_in_emb)
        dnn_out = self.parallel_dnn(dcn_in_emb)
        y_pred = self.mlp(torch.cat([cross_out, dnn_out], dim=-1))
        return {"y_pred": y_pred}

    def concat_embedding(self, field, feature_emb_dict):
        if isinstance(field, tuple):
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]
