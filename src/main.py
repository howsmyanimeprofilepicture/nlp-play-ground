
from diffusers import DiffusionPipeline
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
from utils import *



def create_images(word: str, pipeline: DiffusionPipeline):
    return pipeline(word).images[0]



def main():
    if not torch.cuda.is_available():
        print("This is only availiable in the env that supports the CUDA! However, your env doesn't support CUDA.... T.T ;;;")
        return None
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipeline.to("cuda")



    simil = Similarity.from_diffusers_pipeline(pipeline)

    # embs, ids, tokens, sims = simil.search("music", 5000)

    print(simil.get_sim_dict("rust", 50))
    create_images("ghost likes you", pipeline)

if __name__ == "__main__":
    main()