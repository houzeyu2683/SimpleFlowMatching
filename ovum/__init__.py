from ._model_ import *
from ._framework_ import *
from ._distribution_ import *
__all__ = ['Model', 'Discriminator', 'Framework']


# model = Model(device='cpu')
# model.activateLayer()
# reference = torch.randn((1, 3, 64, 64))
# step = torch.randint(0, 999, (1,))
# model.getInaccuracy(reference, step)

    # def getIndication(
    #     self, 
    #     # condition: torch.Tensor, # B, C, H, W
    #     # reference: torch.Tensor, # B, L, C, H, W
    #     noise: torch.Tensor, # B, C, H, W
    #     step: torch.Tensor, # B
    #     # length: torch.Tensor # B
    # ) -> torch.Tensor:
    #     noise = noise.to(self.device, non_blocking=True)
    #     step = step.to(self.device, non_blocking=True)
    #     projection = self.layer['latent projection'](noise.flatten(1, -1)) # B, E
    #     index = diffusers.models.embeddings.get_timestep_embedding(step, 2048) # B, E
    #     indication = projection + index

    #     return(indication)

    # def getCausation(self, length: torch.Tensor) -> torch.Tensor:
    #     length = length.to(self.device, non_blocking=True)
    #     #
    #     size = (length, length)
    #     # matrix = torch.ones(size, device=length.device)
    #     matrix = torch.ones_like(length).expand(size)
    #     causation = torch.triu(
    #         matrix, 
    #         diagonal=1,
    #     )
    #     causation = (causation==1)
    #     return(causation)
            # 'latent projection': torch.nn.Sequential(
            #     torch.nn.Linear(2048, 2048),
            #     # torch.nn.LayerNorm(512),
            # ),
            # # 'latent step': torch.nn.Sequential(
            # #     torch.nn.Linear(2048, 2048),
            # #     # torch.nn.SiLU(),
            # #     # torch.nn.Linear(512, 512),
            # # ),
            # # 'latent attention': torch.nn.TransformerEncoder(
            # #     torch.nn.TransformerEncoderLayer(
            # #         2048,
            # #         32,
            # #         2048,
            # #         batch_first=True
            # #     ),
            # #     num_layers=4
            # # ),
            # 'restoration': torch.nn.Sequential(
            #     torch.nn.Linear(2048, 2048),
            #     torch.nn.SiLU(),
            #     torch.nn.Linear(2048, 2048),
            # )
        #
        # condition = batch['condition']
        # step = batch['step']
        # noise = batch['noise']
        # reference = batch['reference']
        # length = batch['length']
        # padding = batch['padding']
        # direction = batch['direction']
        #
        # causation = self.getCausation(length.max())
        # indication = self.getIndication(condition, reference, step)
        # attention = self.layer['latent attention'](
        #     src=indication,
        #     mask=causation,
        #     src_key_padding_mask=padding,
        #     is_causal=True
        # )
        # chunk = self.layer['restoration'](attention)
        # restoration = torch.reshape(chunk[:, 1:, :], noise.shape)
        # margin = torch.pow(restoration - noise, 2).mean([2, 3, 4])
        # #
        # activity = ~padding[:, 1:]
        # error = margin.where(activity, 0).sum() / torch.sum(activity)
        #
# model = Model(device='cuda')
# model.activateLayer()
# sequence = torch.randn((1, 2, 32, 8, 8)).cuda()
# step = 20
# window = 5
# model.getInference(sequence, step, window)

        # torch.ones_like(length.max())
        # vector = tunnel.flatten(2, -1) # b, l, d
        # _, _, dimension = vector.shape
        # position = diffusers.models.embeddings.get_timestep_embedding(
        #     torch.ones_like(timestep)[0].cumsum(0), 
        #     dimension
        # )[None, :, :]
        #
        # projection = self.layer['projection'](vector)
        # expression
        # embedding = self.layer['embedding'](symbol)
        # #
        # condition = (symbol != 1).unsqueeze(-1)  # <CLS>
        # projection = torch.where(condition, projection, embedding)
        # #
        # condition = (symbol != 2).unsqueeze(-1)  # <BOS>
        # projection = torch.where(condition, projection, embedding)
        # #
        # condition = (symbol != 3).unsqueeze(-1)  # <EOS>
        # projection = torch.where(condition, projection, embedding)
        #
        #
        # index = torch.cumsum(torch.ones_like(symbol), dim=1)[0]
        # position = diffusers.models.embeddings.get_timestep_embedding(
        #     index, 
        #     projection.size(-1)
        # )
        # notation = projection + position


        #
        # torch.cumsum(direction, dim=2)
        #
        # area = mask['padding'][:, 1:]
        # bias = fragment[:, 1:, :, :, :] - fragment[:, :-1, :, :, :]
        # similarity = torch.nn.functional.cosine_similarity(
        #     direction[:, :-1, :], 
        #     bias.flatten(2, -1), 
        #     dim=-1
        # ) # b, l
        # similarity = similarity.where(~area, 0)
        # consistency = 1 - (similarity.sum() / torch.sum(~area))
        #
        # digit = fragment.flatten(2, -1)[:, :-1, :] + sketch[:, :-1, :]
        # target = fragment.flatten(2, -1)[:, 1:, :]
        # #
        # measurement = torch.pow(target - digit, 2)#.flatten(2, -1)
        # measurement = measurement.where(~area[:, :, None], 0)
        # error = measurement.mean([2]).sum() / torch.sum(~area)
        #
        # similarity = torch.nn.functional.cosine_similarity(
        #     digit, 
        #     target, 
        #     dim=-1
        # ) # b, l
        # similarity = similarity.where(~area, 0)
        # consistency = 1 - (similarity.sum() / torch.sum(~area))

        #
        # tunnel = batch['tunnel']
        # target = batch['target']
        # timestep = batch['timestep']
        # padding = batch['padding']
        # causation = batch['causation']
        # reference = batch['reference']

        # block = diffusers.models.unets.unet_2d_blocks.DownEncoderBlock2D
        # torch.nn.Linear(2048, 1024)

    # def getCriteria_v0(
    #     self,
    #     batch: tensordict.TensorDict,
    # ) -> tensordict.TensorDict:
    #     batch = batch.to(self.device, non_blocking=True)
    #     #
    #     target = batch['target'] # b, l, c, h, w
    #     timestep = batch['timestep'] # b, l
    #     confusion = batch['confusion'] # b, l, c, h, w
    #     padding = batch['padding'] # b, l
    #     field = batch['field'] # b, l
    #     length = batch['length'] # b
    #     #
    #     expression = self.getExpression(confusion, timestep, length)
    #     expression.shape

    #     #
    #     causation = self.getCausation(length)
    #     attention = self.layer['attention'](
    #         src=expression,
    #         mask=causation,
    #         src_key_padding_mask=padding,
    #         is_causal=True
    #     )
    #     # length.max()
    #     sketch = self.layer['sketch'](attention)
    #     prediction = torch.reshape(sketch, target.shape)
    #     #
    #     gap = torch.pow(target - prediction, 2).mean([2, 3, 4])
    #     #
    #     error = gap.where(field, 0).sum() / torch.sum(field)

    #     #
    #     total = error
    #     # total = stress
    #     criteria = tensordict.TensorDict(device=self.device)
    #     # criteria.set('error', error)
    #     criteria.set('error', error)
    #     criteria.set('total', total)
    #     # criteria=0
    #     return(criteria)

        # condition = condition.to(self.device, non_blocking=True)
        # reference = reference.to(self.device, non_blocking=True)
        # step = step.to(self.device, non_blocking=True)
        # #
        # condition = self.layer['latent projection'](condition.flatten(1, -1))
        # reference = self.layer['latent projection'](reference.flatten(2, -1))
        # #
        # condition = condition[:, None, :]
        # index = diffusers.models.embeddings.get_timestep_embedding(step, 2048)
        # step = self.layer['latent step'](index)
        # reference = reference + step[:, None, :]
        # #
        # indication = torch.cat([condition, reference], dim=1)
        # projection = self.layer['projection'](reference.flatten(2, -1))
        # indication = projection
    # def getCriteria(
    #     self,
    #     batch: tensordict.TensorDict,
    #     regression: float,
    #     exposure: float
    # ) -> tensordict.TensorDict:
    #     batch = batch.to(self.device, non_blocking=True)
    #     #
    #     target = batch['target'] # b, l, c, h, w
    #     step = batch['step'] # b, l
    #     phase = batch['phase'] # b, l, c, h, w
    #     padding = batch['padding'] # b, l
    #     length = batch['length'] # b
    #     head = batch['head'] # b, l, c, h, w
    #     tail = batch['tail'] # b, l, c, h, w
    #     if(random.random()<regression):
    #         with torch.no_grad():
    #             concept = self.getConcept(phase, step, length)
    #             causation = self.getCausation(length)
    #             attention = self.layer['attention'](
    #                 src=concept,
    #                 mask=causation,
    #                 src_key_padding_mask=padding,
    #                 is_causal=True
    #             )
    #             restoration = self.layer['restoration'](attention)
    #             velocity = torch.reshape(restoration, target.shape)
    #             pass
    #         size = len(length), length.max()
    #         replacement = torch.rand(size, device=length.device) < exposure
    #         replacement[:, 0] = False  # position 0 永遠用 GT, (B, L)
    #         reservation = ~replacement
    #         #
    #         reference = head + velocity
    #         head[:, 1:] = head[:, 1:].where(
    #                 reservation[:, 1:, None, None, None],
    #                 reference[:, :-1]
    #             )
    #         #
    #         target = tail - head
    #         #
    #         decay = step[:, :, None, None, None]
    #         phase = (1 - decay) * head + decay * tail
    #         pass
    #     #
    #     concept = self.getConcept(phase, step, length)
    #     causation = self.getCausation(length)
    #     attention = self.layer['attention'](
    #         src=concept,
    #         mask=causation,
    #         src_key_padding_mask=padding,
    #         is_causal=True
    #     )
    #     restoration = self.layer['restoration'](attention)
    #     velocity = torch.reshape(restoration, target.shape)
    #     #
    #     disparity = torch.pow(target - velocity, 2).mean([2, 3, 4])
    #     #
    #     activity = ~padding
    #     error = disparity.where(activity, 0).sum() / torch.sum(activity)
    #     #
    #     total = error
    #     criteria = tensordict.TensorDict(device=self.device)
    #     criteria.set('error', error)
    #     criteria.set('total', total)
    #     return(criteria)

        # index = torch.arange(length.max(), device=length.device)
        # position = diffusers.models.embeddings.get_timestep_embedding(
        #     index, 
        #     2048
        # )
        # #
        # digit = self.layer['digit'](step[:, :, None])
        # #
        # projection = self.layer['projection'](phase.flatten(2, -1))
        # concept = projection + position[None, :, :] + digit





