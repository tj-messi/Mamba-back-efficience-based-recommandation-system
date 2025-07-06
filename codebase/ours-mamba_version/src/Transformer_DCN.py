import torch
from fuxictr.utils import not_in_whitelist
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.layers.activations import Dice
import torch.nn.functional as F
from mamba_ssm import Mamba2


class Transformer_DCN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="Transformer_DCN",
                 gpu=-1,
                 num_heads=1,
                 transformer_layers=2,
                 transformer_dropout=0.2,
                 dim_feedforward=256,
                 learning_rate=5e-4,
                 embedding_dim=64,
                 net_dropout=0.2,
                 first_k_cols=16,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 num_layers=3,
                 num_row=2,
                 accumulation_steps=1,
                 din_use_softmax=True,
                 batch_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super().__init__(feature_map,
                         model_id=model_id,
                         gpu=gpu,
                         embedding_regularizer=embedding_regularizer,
                         net_regularizer=net_regularizer,
                         **kwargs)
        self.accumulation_steps = accumulation_steps
        self.first_k_cols = first_k_cols
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)

        # Feature embedding layer (learnable weight matrix)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Transformer backbone for users' self-interest representation
        transformer_in_dim = self.item_info_dim
        # self.transformer_encoder = SelfAttentionTransformer(
        #     transformer_in_dim,
        #     dim_feedforward=dim_feedforward,
        #     num_heads=num_heads,
        #     dropout=transformer_dropout,
        #     transformer_layers=transformer_layers,
        #     first_k_cols=first_k_cols
        # )

        # Mamba block 
        batch, length, dim = 128, 64, 16*16
        self.mamba_block = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        
        
        # AUGRU backbone for users' interactive-interest representation
        self.attention_layers = DotProductAttention(
            hidden_dim=self.item_info_dim, 
            ad_emb_dim=self.item_info_dim
        )
        
        self.augru_encoder = IIN(
            attention_layers=self.attention_layers,
            input_dim=self.item_info_dim,
            hidden_dim=self.item_info_dim,
            first_k_cols=first_k_cols
        )
        
        # QNN backbone for CTR binary prediction
        qnn_in_dim = sum([128, self.item_info_dim * 3])
        self.qnn = QuadraticNeuralNetworks(input_dim=qnn_in_dim,
                                    num_layers=num_layers,
                                    net_dropout=net_dropout,
                                    num_row=num_row,
                                    batch_norm=batch_norm)
        
        # self.dnn = MLP_Block(input_dim=qnn_in_dim,
        #         output_dim=1,
        #         hidden_units=dnn_hidden_units,
        #         hidden_activations=dnn_activations,
        #         output_activation=self.output_activation, 
        #         dropout_rates=net_dropout,
        #         batch_norm=batch_norm)       
                               
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def adjust_mask(self, mask):
        # make sure not all actions in the sequence are masked
        fully_masked = mask.all(dim=-1)
        mask[fully_masked, -1] = 0
        return mask

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        
        # inputs输入数据格式：
        # 1. batch_dict : ['likes_level', 'view_level']   (item info / context info)
        #    Descrption :   item info : item分级特征信息(即task2的Item Feature)
        # 2. item_dict  : ['item_id', 'item_tags', 'item_embed_d128']
        #    Descrption :   user history sequence : 多模态特征的item交互历史记录(即task1的Input)
        #    共64历史交互序列 + 1目标查询item = 65条, 特征包含tags和图片(from Clip)与文本(from Bert)嵌入
        # 3. mask       : tensor(B, 64)
        #    Descrption :   padding : 1为有效位
        
        # ================= Task1 =================
        # multimodal item feature embedding
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)  # （B, 65, 256）
        target_emb = item_feat_emb[:, -1, :]     # （B, 256)
        sequence_emb = item_feat_emb[:, 0:-1, :] # （B, 64, 256)
        # item info embedding
        emb_list = []
        if batch_dict:  # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)    # (B, 128)
            emb_list.append(feature_emb)
        feat_emb = torch.cat(emb_list, dim=-1)
        

        # ================= Task2 =================
        # Stage1 : extracting users' self-interest representation from history sequence 


        # seq_transformer_emb, mask = self.transformer_encoder(sequence_emb, mask=mask)   # (B, first_k_cols=16, 256), (B, first_k_cols=16)
        seq_mamba_emb = self.mamba_block(sequence_emb) # seq_mamba_emb (B,64,256)
        key_part_mask = self.adjust_mask(mask).bool() # mask (B,64)

        temp_mask = (key_part_mask == 0).unsqueeze(-1) 
        mamba_out = seq_mamba_emb.masked_fill(temp_mask, 0.)

        mamba_out = mamba_out[:, -self.first_k_cols:]
        mask = key_part_mask[:, -self.first_k_cols:]

        positive_sequence_emb, negative_sample_emb = self.get_auxilary_behaviour_sequence(sequence_emb, 
                                                                                          item_dict['item_id'].view(batch_size,-1, 1),
                                                                                          mask, 
                                                                                          self.first_k_cols)



        # Stage2 : generating users' interactive-interest representation by interacting with target embedding 
        augru_pooling_emb = self.augru_encoder(mamba_out, target_emb, mask=mask)  # (b, 16, 256), (B, 256+256)
        # Output : concat User Interest Feature(augru_pooling_emb), Multimodal Embedding(target_emb) and
        #          Feature Embedding(feat_emb) for CTR prediction
        dnn_in_emb = torch.cat([feat_emb, target_emb, augru_pooling_emb], dim=-1)   # (B, 128 + 256 + 256 + 256) = (B, 896)
        y_pred = self.qnn(dnn_in_emb)

        # y_pred (128,1) -> sigmoid
        # y_pred = F.sigmoid(y_pred)

        return_dict = {
            "y_pred": self.output_activation(y_pred),
            "positive_embedding": positive_sequence_emb,  # (B, 15, 256)
            "negative_embedding": negative_sample_emb,  # (B, 15, 256)
            "hidden_state": mamba_out,  # (B, 16, 256)
            "mask": mask,  # (B, 16)
            }
        
        return return_dict

    def get_auxilary_behaviour_sequence(self, sequence_emb, item_id, mask, max_len):
        mask = (mask == 0).unsqueeze(-1)  # [B, 16, 1]
        sequence_emb_filtered = sequence_emb[:, -max_len:].masked_fill(mask, 0.)  # (B, 16, 256)
        # construct example:
        # for i-th user whose hidden_state is:  ( h^(i)[1],  h^(i)[2],  ... ,  h^(i)[t-1] , h^(i)[t])
        # auxilary sequence embed(x is p or n): (e_x^(i)[2], e_x^(i)[3], ... , e_x^(i)[t] ,    0    )
        
        # obtain positive_sequence directly from user click behaviour
        # since the last part is  h^(i)[t] without gt embed guiding, so just remove it when calculating loss
        positive_sequence_emb = sequence_emb_filtered[:, -(max_len-1):, :]            # (B, 15, 256)
        # sample negative sequence from user history
        history_behaviour_id = item_id[:, 0:-1, :]  # (b, 64, 1)
        history_behaviour_id = history_behaviour_id[:, -max_len:].masked_fill(mask, 0.) # (B, 16, 1)
        
        B, T, D = sequence_emb_filtered.shape  # e.g., (B, 16, 256)
        device = sequence_emb.device

        # Step 1: 获取待采样id(第 i+1 个位置)
        pos_ids = history_behaviour_id[:, 1:, 0]     # (B, 15)

        valid_mask = (pos_ids != 0)                  # (B, 15) → True 表示有效位置

        # Step 2: 全局候选池（从非 padding 的位置中采样）
        all_ids = history_behaviour_id.reshape(-1)   # (B*16,)
        all_embs = sequence_emb_filtered.reshape(-1, D)       # (B*16, 256)
 
        # 筛掉 padding id=0
        candidate_mask = (all_ids != 0)
        candidate_ids = all_ids[candidate_mask]      # (N_pool,)
        candidate_embs = all_embs[candidate_mask]    # (N_pool, 256)
        N_pool = candidate_embs.shape[0]

        # Step 3: 对每个 pos_id 构造 N_pool 个候选，用广播做对比
        # pos_ids: (B, 15) → (B, 15, 1)
        # candidate_ids: (N_pool,) → (1, 1, N_pool)
        pos_ids_exp = pos_ids.unsqueeze(-1)                   # (B, 15, 1)
        candidate_ids_exp = candidate_ids.view(1, 1, -1)      # (1, 1, N_pool)

        # 得到 mask：哪些 candidate 是负样本（≠ pos_id）
        neg_mask = (candidate_ids_exp != pos_ids_exp)         # (B, 15, N_pool)

        # Step 4: 从每个位置上随机选择一个负样本（soft mask with Gumbel）
        # 生成 logits: 随机值 + mask，保证不会选到无效位置
        gumbel_noise = -torch.empty_like(neg_mask, dtype=torch.float).exponential_().log()  # Gumbel noise
        masked_logits = gumbel_noise.masked_fill(~neg_mask, float('-inf'))  # (B, 15, N_pool)

        # 找出最大值对应的位置（即为采样）
        sample_idx = masked_logits.argmax(dim=-1)  # (B, 15)

        # Step 5: 根据 sample_idx 从 candidate_embs 中 gather
        # sample_idx: (B, 15) → 展平后 gather
        flat_idx = sample_idx.view(-1)  # (B*15,)
        selected_embs = candidate_embs[flat_idx]  # (B*15, 256)
        negative_sample_emb = selected_embs.view(B, 15, D)  # (B, 15, 256)

        # Step 6: 对无效位置用 0 填充
        negative_sample_emb = negative_sample_emb * valid_mask.unsqueeze(-1)  # (B, 15, 256)
        
        return positive_sequence_emb, negative_sample_emb
        
    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]
    
    def compute_auxilary_loss(self, return_dict, eps=1e-8):
        h = return_dict["hidden_state"][:, -self.first_k_cols:-1, :]  # (B, 15, 256)
        mask = return_dict["mask"][:, -(self.first_k_cols-1):]  # (B, 15)
        positive_embedding = return_dict["positive_embedding"]  # (B, 15, 256)
        negative_embedding = return_dict["negative_embedding"]  # (B, 15, 256)
        pos_score = torch.sum(h * positive_embedding, dim=-1)   # (B, T)
        neg_score = torch.sum(h * negative_embedding, dim=-1)   # (B, T)
        
        pos_prob = torch.sigmoid(pos_score)
        neg_prob = torch.sigmoid(neg_score)

        loss1 = torch.log(pos_prob + eps) 
        loss2 = torch.log(1. - neg_prob + eps)   # (B, T)

        if mask is not None:
            loss1 = loss1 * mask
            loss2 = loss2 * mask
            # 对每个用户的 loss 求和，再平均（符合论文）
            loss1 = -loss1.sum(dim=1).mean()
            loss2 = -loss2.sum(dim=1).mean()
            # cos_sim, angle = self.compute_gradient_cosine_similarity(loss1, loss2, retain_graph=True)
            # print(f"Cosine Similarity: {cos_sim:.4f}, Angle: {angle:.4f} degrees")
            return loss1# + loss2
        else:
            loss1 = -loss1.sum(dim=1).mean()
            loss2 = -loss2.sum(dim=1).mean()
            return loss1# + loss2
    
    def _flatten_grads(self):
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads)

    @torch.no_grad()
    def compute_gradient_cosine_similarity(self, loss1, loss2, retain_graph=False):
        # 保存当前梯度状态
        grads_backup = [p.grad.clone() if p.grad is not None else None for p in self.parameters()]

        self.zero_grad()
        loss1.backward(retain_graph=True)
        grads1 = self._flatten_grads()

        self.zero_grad()
        loss2.backward(retain_graph=True if retain_graph else False)
        grads2 = self._flatten_grads()

        # 恢复原始梯度状态（非必须，如果你只在分析时调用）
        for p, g in zip(self.parameters(), grads_backup):
            p.grad = g

        # 计算 cosine similarity 和夹角
        cos_sim = F.cosine_similarity(grads1.unsqueeze(0), grads2.unsqueeze(0)).item()
        angle = torch.acos(torch.clamp(torch.tensor(cos_sim), -1.0, 1.0)) * 180 / torch.pi
        return cos_sim, angle.item()
        
    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        #TODO: a auxliary loss for the transformer encoder is needed
        loss_target = self.compute_loss(return_dict, y_true)
        loss_aux = self.compute_auxilary_loss(return_dict)
        loss = (loss_target + 0.01 * loss_aux) / self.accumulation_steps
        print(f"Loss Target: {loss_target.item():.4f}, Loss_aux:{loss_aux.item():.4f}, Total Loss: {loss.item():.4f}")
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim, ad_emb_dim=None):
        super().__init__()
        if ad_emb_dim is None:
            ad_emb_dim = hidden_dim
        self.W = nn.Linear(ad_emb_dim, hidden_dim, bias=False)  # W ∈ R^{ad_dim × hidden_dim}

    def forward(self, target_item, history_hidden, mask=None):
        """
        target_item: b x emd
        history_hidden: b x len x emb
        mask: mask of history_hidden, 0 for masked positions
        """
        query = self.W(target_item)         # (B, H)
        query = query.unsqueeze(1)           # (B, 1, H)

        att_score = torch.bmm(query, history_hidden.transpose(1, 2))  # (B, 1, T)
        att_score = att_score.squeeze(1)      # (B, T)

        # mask
        if mask is not None:
            att_score = att_score.masked_fill(mask == 0, float('-inf'))

        att_weight = torch.softmax(att_score, dim=-1)  # (B, T)
        return att_weight  # 后续可以加权 history_hidden 得到 context 向量


class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AUGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, att_score_t):
        # x_t: (batch, input_dim)
        # h_prev: (batch, hidden_dim)
        # att_score_t: (batch, 1)

        combined = torch.cat([x_t, h_prev], dim=-1)

        r = torch.sigmoid(self.W_r(combined))         # Reset gate
        u = torch.sigmoid(self.W_u(combined))         # Update gate
        combined_reset = torch.cat([x_t, r * h_prev], dim=-1)
        c = torch.tanh(self.W_c(combined_reset))      # Candidate
        u_ = att_score_t.unsqueeze(-1) * u            # Modulated update gate
        h = (1 - u_) * h_prev + u_ * c                # Final hidden state
        return h
    
class SelfAttentionTransformer(nn.Module):
    def __init__(self,
                 transformer_in_dim,
                 dim_feedforward=64,
                 num_heads=1,
                 dropout=0,
                 transformer_layers=1,
                 first_k_cols=16,
                ):
        
        super(SelfAttentionTransformer, self).__init__()
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
        
        self.first_k_cols = first_k_cols
        
    def forward(self, sequence_emb, mask=None):
        # get sequence mask (1's are masked)
        key_padding_mask = self.adjust_mask(mask).bool()  # keep the last dim
        tfmr_out = self.transformer_encoder(src=sequence_emb,
                                            src_key_padding_mask=key_padding_mask)
        
        # (B, 64, 256)
        mask = (key_padding_mask == 0).unsqueeze(-1)  # [B, T, 1]
        tfmr_out = tfmr_out.masked_fill(mask, 0.)

        return tfmr_out[:, -self.first_k_cols:], key_padding_mask[:, -self.first_k_cols:]

    def adjust_mask(self, mask):
        # make sure not all actions in the sequence are masked
        fully_masked = mask.all(dim=-1)
        mask[fully_masked, -1] = 0
        return mask
    
# Interest Interaction Network (IIN) with AUGRUCell
class IIN(nn.Module):
    def __init__(self, 
                 attention_layers,
                 input_dim=256, 
                 hidden_dim=256,
                 first_k_cols=32):
        super(IIN, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.att_unit = attention_layers
        self.first_k_cols = first_k_cols
        self.augru_cell = AUGRUCell(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, hist_emb, target_emb, mask=None):
        # hist_emb: (batch, seq_len, input_dim)
        # target_emb: (batch, input_dim)
        # mask: (batch, seq_len), optional

        att_scores = self.att_unit(target_emb, hist_emb, mask=mask)  # (batch, seq_len, 1)

        batch_size, seq_len, hidden_dim = hist_emb.size()
        h = torch.zeros(batch_size, hidden_dim, device=hist_emb.device)

        h_list = []
        for t in range(seq_len):
            x_t = hist_emb[:, t, :]             # (batch, hidden_dim)
            att_t = att_scores[:, t]            # (batch, 1)
            h = self.augru_cell(x_t, h, att_t)  # (batch, hidden_dim)
            h_list.append(h.unsqueeze(1))       # Store the hidden state for each time step
            
        h = torch.cat(h_list, dim=1)  # (b, 16, 256)
        
        # process the transformer output(concat(last, max pooling))
        output_concat = []
        output_concat.append(h[:, -1:].flatten(start_dim=1))
        # Apply max pooling to the transformer output
        mask = (mask == 0).unsqueeze(-1)  # [B, T, 1]
        h = h.masked_fill(mask, 0.)
        pooled_out = self.out_linear(h.max(dim=1).values)
        output_concat.append(pooled_out)
        out = torch.cat(output_concat, dim=-1) # (B, 256)
        
        return out  # (B, 256+256)

    
class QuadraticNeuralNetworks(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers=3,
                 net_dropout=0.1,
                 num_row=2,
                 batch_norm=False):
        super(QuadraticNeuralNetworks, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.layer = nn.ModuleList()
        self.activation = nn.ModuleList()
        for i in range(num_layers):
            self.layer.append(QuadraticLayer(input_dim, num_row=num_row, net_dropout=net_dropout))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(input_dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            self.activation.append(nn.PReLU())
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        for i in range(self.num_layers):
            residual = x
            x = self.layer[i](x)
            if len(self.norm) > i:
                x = self.norm[i](x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
            x = x + residual
        logit = self.fc(x)
        return logit

class QuadraticLayer(nn.Module):
    def __init__(self, input_dim, num_row=2, net_dropout=0.1):
        super(QuadraticLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim * num_row),
                                    nn.Dropout(net_dropout))
        self.num_row = num_row
        self.input_dim = input_dim

    def forward(self, x):  # Khatri-Rao Product
        h = self.linear(x).view(-1, self.num_row, self.input_dim)  # B × R × D
        x = torch.einsum("bd,brd->bd", x, h)
        return x