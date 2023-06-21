from typing import NamedTuple, List
import torch
from transformers import PreTrainedTokenizerBase
from diffusers import DiffusionPipeline


def is_eng(s: str) -> bool:
    """source: https://stackoverflow.com/questions/27084617/detect-strings-with-non-english-characters-in-python"""
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_valid_token_ids(tokenizer: PreTrainedTokenizerBase,
                        min_length: int = 3):


    valid_token_ids = []
    unvalid_token_ids = []
    for token_id in range(tokenizer.vocab_size):
        token: str = tokenizer.convert_ids_to_tokens(token_id)
        if (token.endswith("</w>")
            and is_eng(token.replace("</w>", ""))
            and len(token.replace("</w>", "")) > min_length) :
            valid_token_ids.append(token_id)
        else:
            unvalid_token_ids.append(token_id)
    return (valid_token_ids,
            unvalid_token_ids)



class Similarity:
    def __init__(self,
                 embed_tables: torch.Tensor,
                 tokenizer: PreTrainedTokenizerBase ) -> None:
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        self.embed_tables = embed_tables
        self.tokenizer = tokenizer
        (self.valid_token_ids,
         self.unvalid_token_ids,
         ) = get_valid_token_ids(tokenizer)
        self.vocab_size, self.emb_dim = self.embed_tables.size()
        assert tokenizer.vocab_size == self.vocab_size

    @staticmethod
    def from_diffusers_pipeline(pipeline: DiffusionPipeline):
        assert isinstance(pipeline, DiffusionPipeline)

        embeding_layer_parmas = [*(
            pipeline.text_encoder
                .text_model
                .embeddings
                .token_embedding
                .parameters()
        )]
        assert len(embeding_layer_parmas) == 1
        embed_tables = embeding_layer_parmas[0].detach()

        return Similarity(embed_tables,
                          pipeline.tokenizer)

    def get_embed(self, word: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(word)
        assert len(encoded) == 3, (
            f"The token must be single, but your token ('{word}') seems NOT single"
        )

        _, token_id , _ = encoded

        return self.embed_tables[token_id]

    def search(self,
                word: str,
                               k: int = 10):
        embed: torch.Tensor = self.get_embed(word)
        similarities = (
            (embed[None, :] * self.embed_tables).sum(dim=1)/
            ((embed**2).sum()*(self.embed_tables**2).sum(1))**(0.5)
        )
        similarities[self.unvalid_token_ids] = 0.
        token_ids = similarities.argsort(descending=True)[:k]
        return SearchResult(self.embed_tables[token_ids],
                token_ids,
                self.tokenizer.convert_ids_to_tokens(token_ids),
                similarities[token_ids])

    def get_sim_dict(self,
                           word: str,
                           k: int = 10):
        _, __, tokens, sims = self.search(word, k)
        return {
            token.replace("</w>", "") : sim.item()
            for token, sim in zip(tokens, sims)
        }


class SearchResult(NamedTuple):
    embed_table: torch.Tensor
    token_ids: torch.Tensor
    tokens: List[str]
    Similarity: torch.Tensor