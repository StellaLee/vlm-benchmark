"""Direct-answer baseline. If the backend exposes a sequence logprob, we record
it as a token-probability confidence signal (works on vLLM; usually None on the
Gemini free tier)."""

import math

from avbench.inference.client import VLMClient
from avbench.inference.parsing import extract_answer, is_abstention
from avbench.inference.strategies.base import (
    PromptStrategy, answer_instruction, register, render_question)
from avbench.schema import Prediction, Sample


@register("direct")
class DirectAnswer(PromptStrategy):
    def build_prompt(self, sample: Sample) -> str:
        return "{}\n\n{}\nAnswer:".format(render_question(sample), answer_instruction(sample))

    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        imgs = self.images_for(sample)
        res = (await client.generate(self.build_prompt(sample), imgs, n=1))[0]
        token_conf = None
        if res.avg_logprob is not None:
            token_conf = math.exp(res.avg_logprob)  # mean-token prob proxy
        return Prediction(
            sample_id=sample.sample_id,
            model=client.model,
            strategy=self.name,
            raw_text=res.text,
            answer=extract_answer(res.text, sample),
            token_logprob=res.avg_logprob,
            verbal_confidence=token_conf,
            abstained=is_abstention(res.text),
            usage=res.usage,
            condition=self.condition(),
        )
