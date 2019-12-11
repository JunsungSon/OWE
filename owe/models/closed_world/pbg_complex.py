import torch
import pickle
import logging

from owe.models import KGCBase

import csv
import numpy as np

logger = logging.getLogger("owe")


class PBGComplEx(KGCBase):

    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embedding_r = torch.nn.Embedding(self.E, self.D)
        self.entity_embedding_i = torch.nn.Embedding(self.E, self.D)
        self.relation_embedding_r = torch.nn.Embedding(self.R, self.D)
        self.relation_embedding_i = torch.nn.Embedding(self.R, self.D)

    @staticmethod
    def _score(heads_r, heads_i, relations_r, relations_i, tails_r, tails_i):
        return ((heads_r * relations_r * tails_r) +
                (heads_i * relations_r * tails_i) +
                (heads_r * relations_i * tails_i) -
                (heads_i * relations_i * tails_r)).sum(dim=-1)

    def init_embeddings(self, dataset, emb_dir, entity2id="entity2id.txt",
                        relation2id="relation2id.txt"):
        """
        Initializes the pytorch biggraph complex model with embeddings from previous runs.

        :param dataset:
        :param emb_dir:
        :param entity2id:
        :param relation2id:
        :return:
        """
        logger.info("Loading pretrained embeddings from %s into Pytorch BigGraph ComplEx model" % str(emb_dir))

        entity_file = emb_dir / "entity_embeddings.tsv"
        relation_file = emb_dir / "relation_types_parameters.tsv"

        if not (entity_file.exists() or relation_file.exists()):
            m = ("Trying to load pretrained embeddings (Config setting:"
                 "InitializeWithPretrainedComplexEmbedding ). Not all files"
                 "found under %s" % str(emb_dir))
            logger.error(m)
            raise ValueError(m)

        external_entity2id = {}
        entity_emb = []
        with open(entity_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for index, line in enumerate(reader):
                external_entity2id[line[0]] = index
                emb = line[1:].copy()
                assert len(emb) == 300
                for i in range(len(emb)):
                    emb[i] = float(emb[i])
                entity_emb.append(emb)

        external_relation2id = {}
        relation_emb_r = []
        relation_emb_i = []
        with open(relation_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                name = line[0]
                if name not in external_relation2id:
                    index = len(relation_emb_r)
                    external_relation2id[name] = index
                    relation_emb_r.append(list(range(300)))
                    relation_emb_i.append(list(range(300)))
                else:
                    index = external_relation2id[name]
                direction = line[1]
                type = line[3]
                emb = line[5:].copy()
                assert len(emb) == 150

                for i in range(len(emb)):
                    emb[i] = float(emb[i])

                if type == "real":
                    if direction == "lhs":
                        relation_emb_r[index][:150] = emb
                    else:
                        relation_emb_r[index][150:] = emb
                else:
                    if direction == "lhs":
                        relation_emb_i[index][:150] = emb
                    else:
                        relation_emb_i[index][150:] = emb

        entity_emb_r = np.array(entity_emb.copy())
        entity_emb_i = np.array(entity_emb.copy())
        relation_emb_r = np.array(relation_emb_r)
        relation_emb_i = np.array(relation_emb_i)

        our_entity2id = {e.entity_id: i for e, i in dataset.vocab.entity2id.items()}
        our_relation2id = dataset.vocab.relation2id

        # logger.info("First entity embedding row: %s" % model.model.module.entity_embedding.weight.data[0])
        self.copy_embeddings(self.entity_embedding_r, entity_emb_r, external_entity2id, our_entity2id)
        # logger.info("First entity embedding row: %s" % model.model.module.entity_embedding.weight.data[0])

        # model.model.module.entity_embedding.weight.data.copy_(torch.from_numpy(emb))
        self.copy_embeddings(self.entity_embedding_i, entity_emb_i, external_entity2id, our_entity2id)

        # model.model.module.weight.data.copy_(torch.from_numpy(emb))
        self.copy_embeddings(self.relation_embedding_r, relation_emb_r, external_relation2id, our_relation2id)

        self.copy_embeddings(self.relation_embedding_i, relation_emb_i, external_relation2id, our_relation2id)
        # model.model.module.relation_embedding_i.weight.data.copy_(torch.from_numpy(emb))

        # logger.info("Loaded embeddings from %s to complex model." % (emb_dir))

    def score(self, heads, relations, tails):
        """
                :param relations: [B] Tensor of relations
                :param heads: [B], or [E] if predicting tails
                :param tails: [B], or [E] if predicting tails
                :return: [B] or [B,E] Tensor of predictions.
                """

        tail_pred = tails.size(0) == self.E
        assert (not tail_pred) == (heads.size(0) == self.E)

        if tail_pred:
            tails_r = self.entity_embedding_r(tails)  # [E] -> [E,D]
            tails_i = self.entity_embedding_i(tails)  # [E] -> [E,D]
            tails_r = tails_r.unsqueeze(0)  # [E,D] -> [1,E,D]
            tails_i = tails_i.unsqueeze(0)  # [E,D] -> [1,E,D]

            if self.embeddings is None:  # the head entity is known (part of the training graph) # TODO remove self.embedding
                heads_r = self.entity_embedding_r(heads)  # [B] -> [B,D]
                heads_i = self.entity_embedding_i(heads)  # [B] -> [B,D]
            else:  # Use the projected head embeddings
                heads_r, heads_i = self.embeddings  # [B,D]
            heads_r = heads_r.unsqueeze(1)  # [B,D] -> [B,1,D]
            heads_i = heads_i.unsqueeze(1)  # [B,D] -> [B,1,D]

        else:  # Head prediction
            heads_r = self.entity_embedding_r(heads)  # [E] -> [E,D]
            heads_i = self.entity_embedding_i(heads)  # [E] -> [E,D]
            heads_r = heads_r.unsqueeze(0)  # [E,D] -> [1,E,D]
            heads_i = heads_i.unsqueeze(0)  # [E,D] -> [1,E,D]

            if self.embeddings is None:  # We know the tail entity  # TODO remove self.embedding
                tails_r = self.entity_embedding_r(tails)  # [B] -> [B,D]
                tails_i = self.entity_embedding_i(tails)  # [B] -> [B,D]
            else:  # Use the projected tail embeddings
                tails_r, tails_i = self.embeddings  # [B,D]
            tails_r = tails_r.unsqueeze(1)  # [B,D] -> [B,1,D]
            tails_i = tails_i.unsqueeze(1)  # [B,D] -> [B,1,D]

        relations_r = self.relation_embedding_r(relations)  # [B] -> [B,D]
        relations_i = self.relation_embedding_i(relations)  # [B] -> [B,D]
        relations_r = relations_r.unsqueeze(1)  # [B,D] -> [B,1,D]
        relations_i = relations_i.unsqueeze(1)  # [B,D] -> [B,1,D]

        # 1. Head Prediction: [1,E, D] * [B,1,D] * [B,1,D] -> [B, E]
        # 2. Tail Prediction: [B,1,D] * [B,1,D] * [1,E,D] -> [B, E]
        scores = self._score(heads_r, heads_i,
                             relations_r, relations_i,
                             tails_r, tails_i)

        return scores
